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

#include "NNFlicPlg.h"
#include "GraphManagerPlg.h"
#include "PlgTensorSource.h"
#include "PlgStreamResult.h"

#include "vpusmm.h"

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

TEST(kmbVPUALHostIntegrationTests, sendBlobToLeon) {
    const int nThreads = 4, nShaves = 16;

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

    const int pageSize = getpagesize();
    const uint32_t requiredBlobSize = calculateRequiredSize(blobSize, pageSize);
    int blobFileDesc = vpusmm_alloc_dmabuf(requiredBlobSize, VPUSMMTYPE_NON_COHERENT);
    ASSERT_GE(blobFileDesc, 0);

    unsigned long blobPhysAddr = vpusmm_import_dmabuf(blobFileDesc, DMA_BIDIRECTIONAL);
    ASSERT_NE(blobPhysAddr, 0);

    void * blobFileData = mmap(0, requiredBlobSize, PROT_READ|PROT_WRITE, MAP_SHARED, blobFileDesc, 0);
    ASSERT_NE(blobFileData, MAP_FAILED);

    const uint32_t tensorInSize = inputTensorRawData.size();
    const uint32_t requiredTensorInSize = calculateRequiredSize(tensorInSize, pageSize);

    int tensorInFileDesc = vpusmm_alloc_dmabuf(requiredTensorInSize, VPUSMMTYPE_NON_COHERENT);
    ASSERT_GE(tensorInFileDesc, 0);

    unsigned long tensorInPhysAddr = vpusmm_import_dmabuf(tensorInFileDesc, DMA_BIDIRECTIONAL);
    ASSERT_NE(tensorInPhysAddr, 0);

    void * inputTensorData = mmap(0, requiredTensorInSize, PROT_READ|PROT_WRITE, MAP_SHARED, tensorInFileDesc, 0);
    ASSERT_NE(inputTensorData, MAP_FAILED);

    const uint32_t tensorOutSize = inputTensorRawData.size();
    const uint32_t requiredTensorOutSize = calculateRequiredSize(tensorOutSize, pageSize);
    int tensorOutFileDesc = vpusmm_alloc_dmabuf(requiredTensorOutSize, VPUSMMTYPE_NON_COHERENT);
    ASSERT_GE(tensorOutFileDesc, 0);

    unsigned long tensorOutPhysAddr = vpusmm_import_dmabuf(tensorOutFileDesc, DMA_BIDIRECTIONAL);
    ASSERT_NE(tensorOutPhysAddr, 0);

    void * outputTensorData = mmap(0, requiredTensorOutSize, PROT_READ|PROT_WRITE, MAP_SHARED, tensorOutFileDesc, 0);
    ASSERT_NE(outputTensorData, MAP_FAILED);

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

    vpusmm_unimport_dmabuf(tensorOutFileDesc);
    munmap(outputTensorData, requiredTensorOutSize);
    close(tensorOutFileDesc);

    vpusmm_unimport_dmabuf(tensorInFileDesc);
    munmap(inputTensorData, requiredTensorInSize);
    close(tensorInFileDesc);

    vpusmm_unimport_dmabuf(blobFileDesc);
    munmap(blobFileData, requiredBlobSize);
    close(blobFileDesc);
}

#ifndef INTEGRATION_TESTS_ENABLE_IE
int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
