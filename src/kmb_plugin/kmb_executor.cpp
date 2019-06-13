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

#include <mvnc.h>
#include <ie_common.h>
#include <thread>

#include <vpu/kmb_plugin_config.hpp>
#include <vpu/utils/extra.hpp>
#include <vpu/utils/logger.hpp>

#include "kmb_executor.h"
#include "kmb_config.h"

#include <mvMacros.h>

#include "NNFlicPlg.h"
#include "MemAllocator.h"
#include "Pool.h"

#ifndef _WIN32
# include <libgen.h>
# include <dlfcn.h>
#endif

using namespace vpu::KmbPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::VPUConfigParams;
using namespace std;

static std::mutex device_mutex;

VPU_PACKED(bin_header {
    int32_t  magic;
    uint32_t frequency;
};)

#define INPUTSIZE (2 * 1024 * 1024)
#define Blob_Size (30 * 1024 * 1024)

#define N_POOL_TENSORS  (4)
#define TENSOR_IN_SIZE  (INPUTSIZE)
#define TENSOR_OUT_SIZE (TENSOR_IN_SIZE)

int KmbExecutor::flic_pipeline(int graphId, BlobHandle_t* BHandle, int nThreads, int nShaves) {
    int returnVal;
    // Let's try and use the plugins in a FLIC context:
    std::cout << "FLIC Testing:" << std::endl;

    NNFlicPlg* nnPl = new NNFlicPlg();

    // Pool plugins (to allocate memory for the plugins which require some):
    PlgPool<TensorMsg> plgPoolA;
    PlgPool<TensorMsg> plgPoolB;
    std::cout << "Instantiated Plugins..." << std::endl;

    // FLIC Pipeline:
    Pipeline pipe;
    // "Create" the plugins:
    // Setting number of threads for NNPlugin
    nnPl->SetNumberOfThreads(nThreads);
    nnPl->SetNumberOfShaves(nShaves);
    nnPl->Create(BHandle->graphBuff, BHandle->graphLen);

    if (nnPl->GetLatestState() != 0) {
        returnVal = nnPl->GetLatestState();
        // gg.NNDeallocateGraph(graphId);
        std::cout << "network status not okay" << std::endl;
    }
    fathomTensorDescriptor_t descOut = nnPl->GetOutputTensorDescriptor(0);
    fathomTensorDescriptor_t  descIn = nnPl->GetInputTensorDescriptor(0);

    const unsigned int shavel2CacheLineSize = 64;
    unsigned int outputTensorSize = ROUND_UP(descOut.totalSize, shavel2CacheLineSize);

    plgPoolB.Create(&(*HeapAlloc), 10, outputTensorSize);
    plgPoolA.Create(&(*HeapAlloc), 32, 32);

    plgTensorInput_->Create(const_cast<char*>("TensorIStream"), descIn);
    plgTensorOutput_->Create(const_cast<char*>("TensorOStream"), descOut, 4);

    std::cout << "'Created' all Plugins..." << std::endl;

    // Add the plugins to the pipeline:
    pipe.Add(&plgPoolA);
    pipe.Add(&plgPoolB);
    pipe.Add(&(*plgTensorInput_));
    pipe.Add(&(*plgTensorOutput_));
    pipe.Add(nnPl);

    std::cout << "Added Plugins to Pipeline..." << std::endl;

    // Link the plugins' messages:
    plgPoolA.out.Link(&(plgTensorInput_->emptyTensor));
    plgPoolB.out.Link(&nnPl->resultInput);
    plgTensorInput_->tensorOut.Link(&(nnPl->tensorInput));
    nnPl->output.Link(&(plgTensorOutput_->dataIn));

    std::cout << "Linked Plugins..." << std::endl;

    nnPl->SetConfig(1);

    pipe.Start();
    std::cout << "Started FLIC pipeline..." << std::endl;

    pipe.Wait();
    std::cout << "Waited for FLIC pipeline to end (should never end)..." << std::endl;

    pipe.Stop();

    std::cout<< "Finishes the pipeline with Design #2......" << std::endl;

    pipe.Delete();

    std::cout << "Inference ran successfully, proceeding to destroy all plugins and handlers" << std::endl;

    return 1;
}

void KmbExecutor::allocateGraph(DevicePtr &device,
                                GraphDesc &graphDesc,
                                const std::vector<char> &graphFileContent,
                                const MVCNN::SummaryHeader *graphHeaderDesc,
                                size_t numStages,
                                const char* networkName) {
    HeapAlloc = make_shared<HeapAllocator>();
    gg = make_shared<GraphManagerPlg>();
    plgTensorInput_ = make_shared<PlgTensorSource>();
    plgTensorOutput_ = make_shared<PlgStreamResult>();
    gg->Create();

    int graphId_main = 1;
    bool graphAvail_main = false;
    BlobHandle_t* BHandle_main;

    int nThreads_main = 4;
    int nShaves_main = 16;
    int status;

    int useCase = 1;

    std::cout << "Selected use case for testing " << useCase << std::endl;
    switch (useCase) {
    case 1:
    {
        std::cout << "Initiating verification of use case 1" << std::endl;
        graphAvail_main = gg->GraphCheckAvailable(graphId_main);

        if (!(graphAvail_main)) {
            // Allocating Graph with GraphId from main.cpp
            BHandle_main = gg->NNGraphAllocate(NULL, 0, graphId_main);
        } else {
            BHandle_main = gg->NNGraphAllocateExistingBlob(graphId_main);
        }
        status = flic_pipeline(graphId_main, BHandle_main, nThreads_main, nShaves_main);
        if (status == 1) {
            std::cout << "Completed use case 1 successfully!!!" << std::endl;
        }
        break;
    }
    case 2:
    {
        std::cout << "Initiating verification of use case 2" << std::endl;

        for (int i = 0 ; i < 3; i ++) {
            graphAvail_main = gg->GraphCheckAvailable(graphId_main);

            if (!(graphAvail_main)) {
                // Allocating Graph with GraphId from main.cpp
                BHandle_main = gg->NNGraphAllocate(NULL, 0, graphId_main);
            } else {
                BHandle_main = gg->NNGraphAllocateExistingBlob(graphId_main);
            }
            status = flic_pipeline(graphId_main, BHandle_main, nThreads_main, nShaves_main);
            if (status == 1) {
                std::cout << "Completed use case 2 successfully!!!"<< std::endl;
            }
        }
        break;
    }
    default:
        std::cout << "Use Case value is not correct!!!" << std::endl;
        break;
    }
}

void KmbExecutor::queueInference(GraphDesc &graphDesc, void *input_data, size_t input_bytes,
                    void *result_data, size_t result_bytes) {
    /* plgTensorInput_.Push();*/
    return;
}

void KmbExecutor::getResult(GraphDesc &graphDesc, void *result_data, unsigned int result_bytes) {
    /* PlgStreamResult_.Pull(); */
    return;
}

void KmbExecutor::deallocateGraph(DevicePtr &device, GraphDesc &graphDesc) {
    std::lock_guard<std::mutex> lock(device_mutex);
    /* gg.NNDeallocateGraph(graphId); */
    return;
}
