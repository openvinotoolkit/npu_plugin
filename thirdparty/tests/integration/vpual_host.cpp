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
#include "XPool.h"

#include "vpusmm/vpusmm.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>


#ifdef INTEGRATION_TESTS_ENABLE_IE
#include "test_model_path.hpp"
#endif

using namespace testing;

using kmbVPUALHostIntegrationTests = ::testing::Test;

const uint32_t POOL_SIZE = 30 * 1024 * 1024;
const uint32_t XLINK_INPUT_CHANNEL = 3, XLINK_OUTPUT_CHANNEL = 4, XPOOL_CHANNEL = 5;

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
    _fileDesc = vpusmm_alloc_dmabuf(requiredBlobSize, VPUSMMTYPE_COHERENT);
    if (_fileDesc < 0) {
        throw std::runtime_error("VpusmmAllocator::VpusmmAllocator: vpusmm_alloc_dmabuf failed");
    }

    _physAddr = vpusmm_import_dmabuf(_fileDesc, VPU_DEFAULT);
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
    std::string blobFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";
#endif
    std::string blobRawData = readFromFile(blobFilePath);
    ASSERT_NE(blobRawData, "");

#if (!defined INTEGRATION_TESTS_ENABLE_IE && defined INTEGRATION_TESTS_INPUT_FILE)
    std::string inputTensorFilePath = INTEGRATION_TESTS_INPUT_FILE;
#else
    std::string inputTensorFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/input.bin";
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

    // Looks weird but the same thing is done in the SimpleNN sample
    // TODO: rename to outputPoolBufferSize
    const uint32_t tensorOutSize = POOL_SIZE;
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
    PlgPool<TensorMsg> plgPoolOutputs;

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
    std::cout << "Parsed output descriptor: (" << descOut.n << ", " << descOut.c << ", "
              << descOut.h << "," << descOut.w << ")\n";

    flicTensorDescriptor_t  descIn = nnPl.GetInputTensorDescriptor(0);
    std::cout << "Parsed input descriptor: (" << descIn.n << ", " << descIn.c << ", "
              << descIn.h << "," << descIn.w << ")\n";

    const unsigned int shavel2CacheLineSize = 64;
    const unsigned int outputTensorSize = roundUp(descOut.totalSize, shavel2CacheLineSize);

    plgPoolOutputs.Create(&RgnAlloc, 1, 3 * outputTensorSize);
    std::cout << "Created plgPoolOutputs\n";

    plgTensorInput.Create(descIn.totalSize, XLINK_INPUT_CHANNEL, descIn);
    std::cout << "Created plgTensorInput\n";

    plgTensorOutput.Create(outputTensorSize, XLINK_OUTPUT_CHANNEL, descOut);
    std::cout << "Created plgTensorOutput\n";

    // Add the plugins to the pipeline:
    pipe.Add(&plgPoolOutputs);
    pipe.Add(&plgTensorInput);
    pipe.Add(&plgTensorOutput);
    pipe.Add(&nnPl);
    // Link the plugins' messages:
    plgPoolOutputs.out.Link(&nnPl.resultInput);
    plgTensorInput.tensorOut.Link(&nnPl.tensorInput);
    nnPl.output.Link(&plgTensorOutput.dataIn);

    pipe.Start();
    std::cout << "Started pipeline\n";

    plgTensorInput.Push(tensorInPhysAddr, tensorInSize);
    std::cout << "Pushed input\n";

    uint32_t len = 0;
    uint32_t pAddr = 0;
    plgTensorOutput.Pull(&pAddr, &len);
    std::cerr << "Pulled output\n";
    // Convert the physical address we received back to a virtual address we can use.
    uint32_t offset = pAddr - tensorOutPhysAddr;
    unsigned char *data = static_cast<uint8_t *>(outputTensorData) + offset;
    ASSERT_NE(data, nullptr);

    pipe.Stop();
    pipe.Delete();
    RgnAlloc.Delete();
}

static std::map< DevicePtr, std::shared_ptr<VpusmmAllocator> > bufferMap;

static DevicePtr allocateTensor(uint32_t size) {
    std::shared_ptr<VpusmmAllocator> allocator = std::make_shared<VpusmmAllocator>(size);
    DevicePtr physAddress = allocator->getPhysicalAddress();
    bufferMap[physAddress] = allocator;
    return physAddress;
}

static void freeTensor(DevicePtr paddr) {
    std::map< DevicePtr, std::shared_ptr<VpusmmAllocator> >::iterator bufferMapIter = bufferMap.find(paddr);
    assert(bufferMapIter != bufferMap.end());
    bufferMap.erase(bufferMapIter);
}

template<>
XPool<TensorMsg>::XPool(uint32_t deviceId)
  : PluginStub("XPoolTensorMsg", deviceId), out(deviceId) {
    std::cout << "XPool constructor is called" << std::endl;
}

std::vector<size_t> yieldTopClasses(const std::vector<uint8_t> & unsortedRawData, size_t maxClasses) {
    // map key is a byte from raw data (quantized probability)
    // map value is the index of that byte (class id)
    std::multimap<uint8_t, size_t> sortedClassMap;
    for (size_t classIndex = 0; classIndex < unsortedRawData.size(); classIndex++) {
        uint8_t classProbability = unsortedRawData.at(classIndex);
        std::pair<uint8_t, size_t> mapItem(classProbability, classIndex);
        sortedClassMap.insert(mapItem);
    }

    std::vector<size_t> topClasses;
    for (size_t classCounter = 0; classCounter < maxClasses; classCounter++) {
        std::multimap<uint8_t, size_t>::reverse_iterator classIter = sortedClassMap.rbegin();
        std::advance(classIter, classCounter);
        topClasses.push_back(classIter->second);
        std::cout << "index: " << classIter->second << " value: " << (int) classIter->first << std::endl;
    }

    return topClasses;
}

TEST_P(kmbVPUALAllocTests, xPoolTest) {
    memoryAllocatorType allocatorType = GetParam();
    const int nThreads = 1, nShaves = 16;

#if (!defined INTEGRATION_TESTS_ENABLE_IE && defined INTEGRATION_TESTS_BLOB_FILE)
    std::string blobFilePath = INTEGRATION_TESTS_BLOB_FILE;
#else
    std::string blobFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";
#endif
    std::string blobRawData = readFromFile(blobFilePath);
    ASSERT_NE(blobRawData, "");

#if (!defined INTEGRATION_TESTS_ENABLE_IE && defined INTEGRATION_TESTS_INPUT_FILE)
    std::string inputTensorFilePath = INTEGRATION_TESTS_INPUT_FILE;
#else
    std::string inputTensorFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/input.bin";
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

    // Looks weird but the same thing is done in the SimpleNN sample
    // TODO: rename to outputPoolBufferSize
    const uint32_t tensorOutSize = POOL_SIZE;
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
    constexpr uint32_t device_id = 0;
    XPool<TensorMsg> xPool(device_id);

    // FLIC Pipeline:
    Pipeline pipe;

    //Setting number of threads for NNPlugin
    nnPl.SetNumberOfThreads(nThreads);
    nnPl.SetNumberOfShaves(nShaves);

    nnPl.Create(&BHandle);

    NNPlgState state = nnPl.GetLatestState();
    ASSERT_EQ(state, SUCCESS);

    flicTensorDescriptor_t descOut = nnPl.GetOutputTensorDescriptor(0);
    std::cout << "Parsed output descriptor: (" << descOut.n << ", " << descOut.c << ", "
              << descOut.h << "," << descOut.w << ")\n";

    flicTensorDescriptor_t  descIn = nnPl.GetInputTensorDescriptor(0);
    std::cout << "Parsed input descriptor: (" << descIn.n << ", " << descIn.c << ", "
              << descIn.h << "," << descIn.w << ")\n";

    const unsigned int shavel2CacheLineSize = 64;
    const unsigned int outputTensorSize = roundUp(descOut.totalSize, shavel2CacheLineSize);

    xPool.Create(2, POOL_SIZE, XPOOL_CHANNEL, allocateTensor, freeTensor);
    std::cout << "Created xPool\n";

    plgTensorInput.Create(descIn.totalSize, XLINK_INPUT_CHANNEL, descIn);
    std::cout << "Created plgTensorInput\n";

    plgTensorOutput.Create(outputTensorSize, XLINK_OUTPUT_CHANNEL, descOut);
    std::cout << "Created plgTensorOutput\n";

    // Add the plugins to the pipeline:
    pipe.Add(&xPool);
    pipe.Add(&plgTensorInput);
    pipe.Add(&plgTensorOutput);
    pipe.Add(&nnPl);
    // Link the plugins' messages:
    xPool.out.Link(&nnPl.resultInput);
    plgTensorInput.tensorOut.Link(&nnPl.tensorInput);
    nnPl.output.Link(&plgTensorOutput.dataIn);

    pipe.Start();
    std::cout << "Started pipeline\n";

    plgTensorInput.Push(tensorInPhysAddr, tensorInSize);
    std::cout << "Pushed input\n";

    uint32_t len = 0;
    uint32_t pAddr = 0;
    plgTensorOutput.Pull(&pAddr, &len);
    std::cout << "Pulled output with length " << len << std::endl;
    // Convert the physical address we received back to a virtual address we can use.
    uint32_t offset = pAddr - tensorOutPhysAddr;
    uint8_t *data = static_cast<uint8_t *>(outputTensorData) + offset;
    ASSERT_NE(data, nullptr);

    std::vector<uint8_t> unsortedRawData(data, data + len);

    const int MAX_TOP_CLASSES = 5;
    std::vector<size_t> topClasses = yieldTopClasses(unsortedRawData, MAX_TOP_CLASSES);
    std::cout << "Top " << MAX_TOP_CLASSES << " classification results: ";
    for(auto && x : topClasses) {
        std::cout << x << " ";
    }
    std::cout << std::endl;

    pipe.Stop();
    pipe.Delete();
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
