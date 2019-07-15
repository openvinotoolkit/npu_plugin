//
// Copyright 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <gtest/gtest.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <memory>

#include "NNFlicPlg.h"
#include "GraphManagerPlg.h"
#include "PlgTensorSource.h"
#include "PlgStreamResult.h"

#include "vpusmm.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>


#ifdef INTEGRATION_TESTS_ENABLE_IE
#include "test_model_path.hpp"
#endif

using namespace testing;

using kmbVPUALHostIntegrationTests = ::testing::Test;

const uint32_t INPUT_WIDTH = 800, INPUT_HEIGHT = 480;
const uint32_t FRAME_SIZE = INPUT_WIDTH*INPUT_HEIGHT*3/2;
const uint32_t POOL_SIZE = 2 * FRAME_SIZE + 1024;
const uint32_t XLINK_INPUT_CHANNEL = 3, XLINK_OUTPUT_CHANNEL = 4;

static uint32_t roundUp(uint32_t x, uint32_t a) {
    return ((x + a - 1) / a) * a;
}

static uint32_t calculateRequiredSize(uint32_t blobSize, int pageSize) {
    uint32_t blobSizeRem = blobSize % pageSize;
    uint32_t requiredSize = (blobSize / pageSize) * pageSize;
    if (blobSizeRem) {
        requiredSize += pageSize;
    }
    return requiredSize;
}

std::string readFromFile(const std::string & filePath) {
    std::ifstream fileStreamHandle(filePath, std::ios::binary);
    if (!fileStreamHandle.good()) {
        return "";
    }
    std::ostringstream fileContentStream;
    fileContentStream << fileStreamHandle.rdbuf();
    fileStreamHandle.close();
    return fileContentStream.str();
}

class IMemoryAllocator {
public:
    virtual ~IMemoryAllocator() {}
    virtual unsigned char* getVirtualAddress() = 0;
    virtual unsigned long getPhysicalAddress() = 0;
};

class UdmaAllocator : public IMemoryAllocator {
public:
    UdmaAllocator(size_t requestedSize);
    virtual ~UdmaAllocator();
    unsigned char* getVirtualAddress() { return _buf; }
    unsigned long getPhysicalAddress() { return _phys_addr; }
private:
    int _fd;
    unsigned char* _buf;
    unsigned long _phys_addr;
    unsigned int _size;
    static int _bufferCount;
};

int UdmaAllocator::_bufferCount = 0;

UdmaAllocator::UdmaAllocator(size_t requestedSize) {
    std::ostringstream bufferNameStream;
    bufferNameStream << "udmabuf" << _bufferCount;
    const std::string bufname = bufferNameStream.str();
    const std::string udmabufdevname = "/dev/" + bufname;
    const std::string udmabufsize = "/sys/class/udmabuf/" +  bufname + "/size";
    const std::string udmabufphysaddr = "/sys/class/udmabuf/" + bufname + "/phys_addr";
    const std::string udmabufclassname = "/sys/class/udmabuf/" + bufname + "/sync_mode";

    // Set the sync mode.
    const std::string SYNC_MODE_STR = "3";
    int devFileDesc = -1;
    if ((devFileDesc  = open(udmabufclassname.c_str(), O_WRONLY)) != -1) {
        size_t bytesWritten = write(devFileDesc, SYNC_MODE_STR.c_str(), SYNC_MODE_STR.size());
        close(devFileDesc);
    } else {
        throw std::runtime_error("UdmaAllocator::UdmaAllocator No Device: " + udmabufclassname);
    }

    _size = requestedSize;
    // Get the size of the region.
    int bufSizeFileDesc = -1;
    if ((bufSizeFileDesc  = open(udmabufsize.c_str(), O_RDONLY)) != -1) {
        const std::size_t maxRegionSizeLength = 1024;
        std::string regionSizeString(maxRegionSizeLength, 0x0);

        size_t bytesRead = read(bufSizeFileDesc, &regionSizeString[0], maxRegionSizeLength);
        std::istringstream regionStringToInt(regionSizeString);
        regionStringToInt >> _size;
        close(bufSizeFileDesc);
    } else {
        throw std::runtime_error("UdmaAllocator::UdmaAllocator No Device: " + udmabufsize);
    }

    // Get the physical address of the region.
    int physAddrFileDesc = -1;
    if ((physAddrFileDesc  = open(udmabufphysaddr.c_str(), O_RDONLY)) != -1) {
        const std::size_t maxPhysAddrLength = 1024;
        std::string physAddrString(maxPhysAddrLength, 0x0);

        size_t bytesRead = read(physAddrFileDesc, &physAddrString[0], maxPhysAddrLength);
        std::istringstream physAddrToHex(physAddrString);
        physAddrToHex >> std::hex >> _phys_addr;
        close(physAddrFileDesc);
    } else {
        throw std::runtime_error("UdmaAllocator::UdmaAllocator No Device: " + udmabufphysaddr);
    }

    // Map a virtual address which we can use to the region.
    // O_SYNC is important to ensure our data is written through the cache.
    if ((_fd  = open(udmabufdevname.c_str(), O_RDWR | O_SYNC)) != -1) {
        _buf = static_cast<unsigned char*>(mmap(nullptr, _size, PROT_READ|PROT_WRITE, MAP_SHARED, _fd, 0));
    } else {
        throw std::runtime_error("UdmaAllocator::UdmaAllocator No Device: " + udmabufdevname);
    }

    _bufferCount++;
}

UdmaAllocator::~UdmaAllocator() {
    munmap(_buf, _size);
    close(_fd);
}

class VpusmmAllocator : public IMemoryAllocator {
public:
    VpusmmAllocator(size_t requestedSize);
    virtual ~VpusmmAllocator();
    unsigned char* getVirtualAddress() { return _virtAddr; }
    unsigned long getPhysicalAddress() { return _physAddr; }
private:
    int _fileDesc;
    unsigned char* _virtAddr;
    unsigned long _physAddr;
    unsigned int _allocSize;
    static int _pageSize;
};

int VpusmmAllocator::_pageSize = getpagesize();

VpusmmAllocator::VpusmmAllocator(size_t requestedSize) {
    const uint32_t requiredBlobSize = calculateRequiredSize(requestedSize, _pageSize);
    _fileDesc = vpusmm_alloc_dmabuf(requiredBlobSize, VPUSMMTYPE_NON_COHERENT);
    if (_fileDesc < 0) {
        throw std::runtime_error("VpusmmAllocator::VpusmmAllocator: vpusmm_alloc_dmabuf failed");
    }

    _physAddr = vpusmm_import_dmabuf(_fileDesc, DMA_BIDIRECTIONAL);
    if (_physAddr == 0) {
        throw std::runtime_error("VpusmmAllocator::VpusmmAllocator: vpusmm_import_dmabuf failed");
    }

    _virtAddr = static_cast<unsigned char*>(mmap(0, requiredBlobSize, PROT_READ|PROT_WRITE, MAP_SHARED, _fileDesc, 0));
    if (_virtAddr == MAP_FAILED) {
        throw std::runtime_error("VpusmmAllocator::VpusmmAllocator: mmap failed");
    }
    _allocSize = requiredBlobSize;
}

VpusmmAllocator::~VpusmmAllocator() {
    vpusmm_unimport_dmabuf(_fileDesc);
    munmap(_virtAddr, _allocSize);
    close(_fileDesc);
}

enum memoryAllocatorType {
    MA_UDMA, MA_VPUSMM
};

std::shared_ptr<IMemoryAllocator> buildMemoryAllocator(memoryAllocatorType type, size_t memSize) {
    std::shared_ptr<IMemoryAllocator> resultAllocator(nullptr);
    switch (type) {
    case MA_UDMA:
        resultAllocator = std::make_shared<UdmaAllocator>(memSize);
        break;
    case MA_VPUSMM:
        resultAllocator = std::make_shared<VpusmmAllocator>(memSize);
        break;
    default:
        throw std::runtime_error("buildMemoryAllocator: invalid allocator type");
    }
    return resultAllocator;
}

template<class T>
class kmbVPUALTestsWithParam : public testing::Test,
                               public testing::WithParamInterface<T>
{};

typedef kmbVPUALTestsWithParam<memoryAllocatorType> kmbVPUALAllocTests;

TEST_P(kmbVPUALAllocTests, sendBlobToLeonViaDifferentAllocators) {
    memoryAllocatorType allocatorType = GetParam();
    const int nThreads = 4, nShaves = 16;
    RgnAllocator RgnAlloc;

#if (!defined INTEGRATION_TESTS_ENABLE_IE && defined INTEGRATION_TESTS_BLOB_FILE)
    std::string blobFilePath = INTEGRATION_TESTS_BLOB_FILE;
#else
    std::string blobFilePath = ModelsPath() + "/KMB_models/BLOBS/TwoFramesConvolution/conv.blob";
#endif
    std::string blobRawData = readFromFile(blobFilePath);
    ASSERT_NE(blobRawData, "");

#if (!defined INTEGRATION_TESTS_ENABLE_IE && defined INTEGRATION_TESTS_INPUT_FILE)
    std::string inputTensorFilePath = INTEGRATION_TESTS_INPUT_FILE;
#else
    std::string inputTensorFilePath = ModelsPath() + "/KMB_models/BLOBS/TwoFramesConvolution/input.dat";
#endif
    std::string inputTensorRawData = readFromFile(inputTensorFilePath);
    ASSERT_NE(inputTensorRawData, "");

    const uint32_t blobSize = blobRawData.size();
    BlobHandle_t BHandle = {
        1,              // ID of graph
        0x00000000,     // P-address of graph (will be filled in shortly).
        blobSize,      // Length of graph
        0,              // Ref count of graph
    };

    // ########################################################################
    // Try and get some CMA allocations.
    // ########################################################################
    std::shared_ptr<IMemoryAllocator> blob_file(nullptr);
    ASSERT_NO_THROW(blob_file = buildMemoryAllocator(allocatorType, blobSize));

    unsigned long blobPhysAddr = blob_file->getPhysicalAddress();
    void * blobFileData = blob_file->getVirtualAddress();

    const uint32_t tensorInSize = inputTensorRawData.size();
    std::shared_ptr<IMemoryAllocator> input_tensor(nullptr);
    ASSERT_NO_THROW(input_tensor = buildMemoryAllocator(allocatorType, tensorInSize));

    unsigned long tensorInPhysAddr = input_tensor->getPhysicalAddress();
    void * inputTensorData = input_tensor->getVirtualAddress();

    const uint32_t tensorOutSize = inputTensorRawData.size();
    std::shared_ptr<IMemoryAllocator> output_tensor(nullptr);
    ASSERT_NO_THROW(output_tensor = buildMemoryAllocator(allocatorType, tensorOutSize));

    unsigned long tensorOutPhysAddr = output_tensor->getPhysicalAddress();
    void * outputTensorData = output_tensor->getVirtualAddress();

    // ########################################################################
    // Load the input files
    // ########################################################################

    // Load the blob file:
    std::memcpy(blobFileData, blobRawData.c_str(), blobSize);

    // Point Blob Handle to the newly loaded graph file.
    BHandle.graphBuff = blobPhysAddr; // Only lower 32-bits

    // Load the input tensor
    std::memcpy(inputTensorData, inputTensorRawData.c_str(), tensorInSize);

    // ########################################################################
    // Create and use the FLIC Pipeline.
    // ########################################################################

    GraphManagerPlg gg;
    gg.Create();

    GraphStatus status = gg.NNGraphCheckAvailable(BHandle.graphid);
    ASSERT_TRUE(status == Success || status == No_GraphId_Found);
    if (Success == status) {
        status = gg.NNGraphAllocateExistingBlob(&BHandle);
    } else if (No_GraphId_Found == status) {
        status = gg.NNGraphAllocate(&BHandle);
    }

    // Plugins:
    PlgTensorSource plgTensorInput;
    PlgStreamResult plgTensorOutput;
    NNFlicPlg nnPl;

    // Pool plugins (to allocate memory for the plugins which require some):
    PlgPool<TensorMsg> plgPoolA;
    PlgPool<TensorMsg> plgPoolB;

    // FLIC Pipeline:
    Pipeline pipe;

    // Region Allocator
    RgnAlloc.Create(tensorOutPhysAddr, POOL_SIZE);

    //Setting number of threads for NNPlugin
    nnPl.SetNumberOfThreads(nThreads);
    nnPl.SetNumberOfShaves(nShaves);

    nnPl.Create(&BHandle);

    NNPlgState state = nnPl.GetLatestState();
    ASSERT_EQ(state, SUCCESS);

    flicTensorDescriptor_t descOut = nnPl.GetOutputTensorDescriptor(0);
    flicTensorDescriptor_t  descIn = nnPl.GetInputTensorDescriptor(0);

    const unsigned int shavel2CacheLineSize = 64;
    const unsigned int sizeOfFrames = 32;
    const unsigned int outputTensorSize = roundUp(descOut.totalSize, shavel2CacheLineSize);

    plgPoolA.Create(&RgnAlloc, 1, sizeOfFrames);
    plgPoolB.Create(&RgnAlloc, 1, outputTensorSize);

    plgTensorInput.Create(tensorInSize, XLINK_INPUT_CHANNEL, descIn);
    plgTensorOutput.Create(tensorOutSize, XLINK_OUTPUT_CHANNEL, descOut);

    // Add the plugins to the pipeline:
    pipe.Add(&plgPoolA);
    pipe.Add(&plgPoolB);
    pipe.Add(&plgTensorInput);
    pipe.Add(&plgTensorOutput);
    pipe.Add(&nnPl);

    // Link the plugins' messages:
    plgPoolA.out.Link(&plgTensorInput.emptyTensor);
    plgPoolB.out.Link(&nnPl.resultInput);
    plgTensorInput.tensorOut.Link(&nnPl.tensorInput);
    nnPl.output.Link(&plgTensorOutput.dataIn);

    pipe.Start();

    // Send and receive tensors
    plgTensorInput.Push(tensorInPhysAddr, tensorInSize);

    // Pull from pipeline.
    uint32_t len = 0;
    uint32_t pAddr = 0;
    plgTensorOutput.Pull(&pAddr, &len);

    // Convert the physical address we received back to a virtual address we can use.
    uint32_t offset = pAddr - tensorOutPhysAddr;
    unsigned char *data = static_cast<uint8_t *>(outputTensorData) + offset;
    ASSERT_NE(data, nullptr);

    // ########################################################################
    // Finish the application (stop FLIC pipeline)
    // ########################################################################

    // Some cleanup.
    pipe.Stop();
    pipe.Delete();
    RgnAlloc.Delete();
}

const static std::vector<memoryAllocatorType> allocatorTypes = {
    MA_UDMA, MA_VPUSMM
};

INSTANTIATE_TEST_CASE_P(sendBlob, kmbVPUALAllocTests,
    ::testing::ValuesIn(allocatorTypes)
);

#ifndef INTEGRATION_TESTS_ENABLE_IE
int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
