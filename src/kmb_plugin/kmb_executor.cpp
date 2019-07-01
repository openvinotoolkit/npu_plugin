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

#include <iostream>
#include <fstream>
#include <vector>
#include <mutex>
#include <map>
#include <algorithm>
#include <utility>
#include <cstring>

#include <fcntl.h>
#include <sys/stat.h>
#include <chrono>
#include <stdio.h>
#include <unistd.h>

#include <mvnc.h>
#include <ie_common.h>
#include <thread>

#include <vpu/kmb_plugin_config.hpp>
#include <vpu/utils/extra.hpp>
#include <vpu/utils/logger.hpp>

#include "kmb_executor.h"
#include "kmb_config.h"

#include <sys/mman.h>
#include "vpusmm.h"

#ifndef _WIN32
# include <libgen.h>
# include <dlfcn.h>
#endif

using namespace vpu::KmbPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::VPUConfigParams;
using namespace std;

#define TENSOR_MAX_SIZE (2 * 1024 * 1024)
#define BLOB_SIZE (30 * 1024 * 1024)


#define N_POOL_TENSORS  (4)
#define TENSOR_IN_SIZE  (896)
#define TENSOR_OUT_SIZE (896)

#define XLINK_INPUT_CHANNEL (3)
#define XLINK_OUTPUT_CHANNEL (4)

#define POOL_SIZE (4 * TENSOR_MAX_SIZE + 1024)

#ifdef ENABLE_VPUAL
KmbCmaData::~KmbCmaData() {
    if (fd >= 0) {
        vpusmm_unimport_dmabuf(fd);
        munmap(buf, size);
        close(fd);
    }
}

const int KmbCmaData::pageSize = getpagesize();

static uint32_t calculateRequiredSize(uint32_t blobSize, int pageSize) {
    uint32_t blobSizeRem = blobSize % pageSize;
    uint32_t requiredSize = (blobSize / pageSize) * pageSize;
    if (blobSizeRem) {
        requiredSize += pageSize;
    }
    return requiredSize;
}

int KmbCmaData::Create(uint32_t requested_size) {
    const int page_size = getPageSize();

    size = calculateRequiredSize(requested_size, page_size);
    fd = vpusmm_alloc_dmabuf(size, VPUSMMTYPE_NON_COHERENT);
    if (fd < 0) {
        int error_num = errno;
        std::cout << "vpusmm_alloc_dmabuf failed with " << error_num << std::endl;
        return -1;
    }

    phys_addr = vpusmm_import_dmabuf(fd, DMA_BIDIRECTIONAL);
    if (phys_addr == 0) {
        int error_num = errno;
        std::cout << "vpusmm_import_dmabuf failed with " << error_num << std::endl;
        return -2;
    }

    buf = static_cast<unsigned char *>(mmap(nullptr, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0));
    if (buf == MAP_FAILED) {
        int error_num = errno;
        std::cout << "mmap failed with " << error_num << std::endl;
        return -3;
    }

    return 0;
}
#endif

KmbExecutor::KmbExecutor(const Logger::Ptr& log, const std::shared_ptr<KmbConfig>& config)
            : _log(log), _config(config) {
    auto parsedConfig = _config->getParsedConfig();
    if (parsedConfig[VPU_KMB_CONFIG_KEY(KMB_EXECUTOR)] == "NO") {
        return;
    }

#ifdef ENABLE_VPUAL
    nnPl = make_shared<NNFlicPlg>();
    gg = make_shared<GraphManagerPlg>();
    plgTensorInput_ = make_shared<PlgTensorSource>();
    plgTensorOutput_ = make_shared<PlgStreamResult>();
    plgPoolA = make_shared<PlgPool<TensorMsg>>();
    plgPoolB = make_shared<PlgPool<TensorMsg>>();

    blob_file = make_shared<KmbCmaData>();
    input_tensor = make_shared<KmbCmaData>();
    output_tensor = make_shared<KmbCmaData>();
    BHandle = make_shared<BlobHandle_t>();
    pipe = make_shared<Pipeline>();
#endif
}

void KmbExecutor::allocateGraph(const std::vector<char> &graphFileContent, const char* networkName) {
    auto parsedConfig = _config->getParsedConfig();
    if (parsedConfig[VPU_KMB_CONFIG_KEY(KMB_EXECUTOR)] == "NO") {
        return;
    }

#ifdef ENABLE_VPUAL
    int graphId_main = 1;
    int nThreads = 4;
    int nShaves = 16;

    std::cout << "Initiating verification of use case 1" << std::endl;

    BHandle->graphid = graphId_main;
    BHandle->graphBuff = 0x00000000;
    BHandle->graphLen = graphFileContent.size();
    BHandle->refCount = 0;

    // ########################################################################
    // Try and get some CMA allocations.
    // ########################################################################

    if (blob_file->Create(graphFileContent.size())) {
        std::cout << "Error getting CMA " << std::endl;
        return;
    }

    // ########################################################################
    // Load the input files
    // ########################################################################

    std::copy(graphFileContent.begin(), graphFileContent.end(), blob_file->buf);
    // Point Blob Handle to the newly loaded graph file. Only allow 32-bit

    // Assigning physical address of Blob file

    BHandle->graphBuff = blob_file->phys_addr;  // Only lower 32-bits

    gg->Create();

    GraphStatus status = gg->NNGraphCheckAvailable(graphId_main);
    if (Success == status) {
        std::cout << "Blob available!" << std::endl;
        status = gg->NNGraphAllocateExistingBlob(BHandle.get());
        std::cout << "Allocated existing blob with status: " << status << std::endl;
    } else if (No_GraphId_Found == status) {
        std::cout << "Blob not found." << std::endl;
        status = gg->NNGraphAllocate(BHandle.get());
        std::cout << "Allocated new blob with status: " << status << std::endl;
    } else {
        std::cerr << "Error checking graph availability: " << status << std::endl;
        // TODO: error
    }

    // Plugins:


    // Pool plugins (to allocate memory for the plugins which require some):


    std::cout << "Instantiated Plugins..." << std::endl;

    // FLIC Pipeline:

    // Setting number of threads for NNPlugin

    nnPl->SetNumberOfThreads(nThreads);
    nnPl->SetNumberOfShaves(nShaves);

    nnPl->Create(BHandle.get());

    std::cout << "NN Plugin Create finished..." << std::endl;

    NNPlgState state = nnPl->GetLatestState();
    if (SUCCESS != state) {
        std::cerr << "Error, bad NN Plugin state: " << state << std::endl;
        return;
    }

    auto tensor_deserializer = [](const flicTensorDescriptor_t & descriptor)->void {
        std::cout << "{";
        std::cout << "n: " << descriptor.n << ", ";
        std::cout << "c: " << descriptor.c << ", ";
        std::cout << "h: " << descriptor.h << ", ";
        std::cout << "w: " << descriptor.w << ", ";
        std::cout << "totalSize: " << descriptor.totalSize << ", ";
        std::cout << "widthStride: " << descriptor.widthStride << ", ";
        std::cout << "heightStride: " << descriptor.heightStride << ", ";
        std::cout << "channelsStride: " << descriptor.channelsStride << "}" << std::endl;
    };

    flicTensorDescriptor_t descOut = nnPl->GetOutputTensorDescriptor(0);
    flicTensorDescriptor_t  descIn = nnPl->GetInputTensorDescriptor(0);
    std::cout << "Deserializing descriptors:" << std::endl;
    std::cout << "Input: ";
    tensor_deserializer(descIn);
    std::cout << "Output: ";
    tensor_deserializer(descOut);

    std::cout << "KmbExecutor::allocateGraph: calling input_tensor->Create" << std::endl;
    if (input_tensor->Create(descIn.totalSize)) {
        std::cout << "KmbExecutor::allocateGraph: Error getting CMA " << std::endl;
        return;
    }
    std::cout << "KmbExecutor::allocateGraph: calling output_tensor->Create" << std::endl;
    if (output_tensor->Create(descOut.totalSize)) {
        std::cout << "KmbExecutor::allocateGraph: Error getting CMA " << std::endl;
        return;
    }

    InferenceEngine::SizeVector inputDims({descIn.n, descIn.c, descIn.h, descIn.w});
    InferenceEngine::Layout inputLayout = InferenceEngine::Layout::NCHW;
    // TODO: add proper precision handling
    InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP16;
    InferenceEngine::TensorDesc inputDesc(inputPrecision, inputDims, inputLayout);
    InferenceEngine::Data inputData("input", inputDesc);

    InferenceEngine::InputInfo inputInfo;
    inputInfo.setInputData(std::make_shared<InferenceEngine::Data>(inputData));
    m_networkInputs[inputInfo.name()] = std::make_shared<InferenceEngine::InputInfo>(inputInfo);
    m_inputInfo.totalSize = descIn.totalSize;

    InferenceEngine::SizeVector outputDims({descOut.n, descOut.c, descOut.h, descOut.w});
    InferenceEngine::Layout outputLayout = InferenceEngine::Layout::NCHW;
    InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP16;
    InferenceEngine::TensorDesc outputDesc(outputPrecision, outputDims, outputLayout);
    InferenceEngine::Data outputData("output", outputDesc);

    m_networkOutputs[outputData.getName()] = std::make_shared<InferenceEngine::Data>(outputData);
    m_outputInfo.totalSize = descOut.totalSize;

    const unsigned int sizeOfFrames = 32;
    const unsigned int shavel2CacheLineSize = 64;
    unsigned int outputTensorSize = ROUND_UP(descOut.totalSize, shavel2CacheLineSize);
    RgnAlloc.Create(output_tensor->phys_addr, POOL_SIZE);

    // TODO - These
    std::cout << "Starting input output tensor plugin along with memory pool create..." << std::endl;
    plgPoolA->Create(&RgnAlloc, 1, sizeOfFrames);
    std::cout << "read memory pool finished..." << std::endl;
    plgPoolB->Create(&RgnAlloc, 1, outputTensorSize);
    std::cout << "write memory pool finished..." << std::endl;
    plgTensorInput_->Create(descIn.totalSize, XLINK_INPUT_CHANNEL, descIn);
    std::cout << "input tensor plugin finished..." << std::endl;
    plgTensorOutput_->Create(descOut.totalSize, XLINK_OUTPUT_CHANNEL, descOut);
    std::cout << "output tensor plugin finished..." << std::endl;
    std::cout << "'Created' all Plugins..." << std::endl;

    // Add the plugins to the pipeline:

    pipe->Add(plgPoolA.get());
    pipe->Add(plgPoolB.get());
    pipe->Add(plgTensorInput_.get());
    pipe->Add(plgTensorOutput_.get());
    pipe->Add(nnPl.get());

    std::cout << "Added Plugins to Pipeline..." << std::endl;

    // Link the plugins' messages:

    plgPoolA->out.Link(&plgTensorInput_->emptyTensor);
    plgPoolB->out.Link(&nnPl->resultInput);
    plgTensorInput_->tensorOut.Link(&nnPl->tensorInput);
    nnPl->output.Link(&plgTensorOutput_->dataIn);

    std::cout << "Linked Plugins..." << std::endl;

    pipe->Start();
    std::cout << "Started FLIC pipeline..." << std::endl;

    std::cout << "Fin" << std::endl;
#endif
}

void KmbExecutor::queueInference(void *input_data, size_t input_bytes,
                    void *result_data, size_t result_bytes) {
    auto parsedConfig = _config->getParsedConfig();
    if (parsedConfig[VPU_KMB_CONFIG_KEY(KMB_EXECUTOR)] == "NO") {
        return;
    }

#ifdef ENABLE_VPUAL
    std::cout << "KmbExecutor::queueInference: memcpy started" << std::endl;
    std::memcpy(input_tensor->buf, input_data, input_bytes);
    std::cout << "KmbExecutor::queueInference: memcpy done" << std::endl;
    plgTensorInput_->Push(input_tensor->phys_addr, input_bytes);
#endif
    return;
}

void KmbExecutor::getResult(void *result_data, unsigned int result_bytes) {
    auto parsedConfig = _config->getParsedConfig();
    if (parsedConfig[VPU_KMB_CONFIG_KEY(KMB_EXECUTOR)] == "NO") {
        return;
    }
    uint32_t len = 0;
    uint32_t pAddr = 0;

#ifdef ENABLE_VPUAL
    plgTensorOutput_->Pull(&pAddr, &len);

    std::cout << "Output tensor returned of length: " << std::dec << len << std::endl;

    // Convert the physical address we received back to a virtual address we can use.
    uint32_t offset = pAddr - output_tensor->phys_addr;
    unsigned char *data = output_tensor->buf + offset;

    // write to file
    // Open output file
    auto out_file = open("output.dat", O_WRONLY | O_CREAT, 0664);
    if (out_file <= 0) {
        std::cout << "Error opening output file" << std::endl;
        return;
    }
    // Write tensor output to file.
    if (write(out_file, data, len) != len) {
        std::cout << "Error writing tensor output to file..." << std::endl;
    }

    close(out_file);
    // Write tensor output to result_data.
    if (len > result_bytes) {
        std::cout << "Error: result_data buffer size less then output length." << std::endl;
    }
    std::cout << "KmbExecutor::getResult memcpy started" << std::endl;
    std::memcpy(result_data, data, len);
    std::cout << "KmbExecutor::getResult memcpy finished" << std::endl;
#endif
    return;
}

void KmbExecutor::deallocateGraph() {
    auto parsedConfig = _config->getParsedConfig();
    if (parsedConfig[VPU_KMB_CONFIG_KEY(KMB_EXECUTOR)] == "NO") {
        return;
    }
#ifdef ENABLE_VPUAL
    pipe->Stop();
    pipe->Delete();
    RgnAlloc.Delete();
#endif

    return;
}

