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

#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>

#include <inference_engine.hpp>

#include <samples/args_helper.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>

#include "inference_sample.h"
#include "file_reader.h"

//HDDLUnite header files
#include "WorkloadContext.h"
#include "hddl2_params.hpp"

#include "creator_blob_nv12.h"
#include "helper_remote_memory.h"
#include "vpux/vpux_plugin_params.hpp"

#include <chrono>
using Time = std::chrono::high_resolution_clock::time_point;
Time (&Now)() = std::chrono::high_resolution_clock::now;

using namespace InferenceEngine;

RemoteMemory_Helper _remoteMemoryHelper;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

/**
* @brief The entry point the Inference Engine sample application
* @file classification_sample/main.cpp
* @example classification_sample/main.cpp
*/
int main(int argc, char *argv[]) {
    try {
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        WorkloadID workloadId = -1;
        int numberOfIterations = 1000;

        // -----------------------------------------------------------------------------------------------------

        // ---- Create workload context
        HddlUnite::WorkloadContext::Ptr context = HddlUnite::createWorkloadContext();
        if (context.get() == nullptr) {
            throw std::logic_error("Create workload context failed!");
        }
        context->setContext(workloadId);
        if (workloadId != context->getWorkloadContextID()) {
            throw std::logic_error("Invalid workloadId!");
        }
        if (HddlStatusCode::HDDL_OK != registerWorkloadContext(context)) {
            throw std::logic_error("registerWorkloadContext failed!");
        }

        // ---- Load frame to remote memory (emulate VAAPI result)
        // ----- Load NV12 input
        std::string inputNV12Path = FLAGS_i;
        const size_t inputWidth = 1920;
        const size_t inputHeight = 1080;
        const size_t nv12Size = inputWidth * inputHeight * 3 / 2;
        std::vector<uint8_t> nv12InputBlobMemory;
        nv12InputBlobMemory.resize(nv12Size);

        NV12Blob::Ptr inputNV12Blob = NV12Blob_Creator::createFromFile(
            inputNV12Path, inputWidth, inputHeight, nv12InputBlobMemory.data());

        // ----- Allocate memory with HddlUnite on device
        auto remoteMemoryFD = _remoteMemoryHelper.allocateRemoteMemory(workloadId, nv12Size, inputNV12Blob->y()->cbuffer().as<void*>());

        // ---- Load inference engine instance
        Core ie;

        // ---- Init context map and create context based on it
        ParamMap paramMap = {{HDDL2_PARAM_KEY(WORKLOAD_CONTEXT_ID), workloadId}};
        RemoteContext::Ptr contextPtr = ie.CreateContext("VPUX", paramMap);

        // ---- Import network providing context as input to bind to context
        const std::string& modelPath = FLAGS_m;

        std::filebuf blobFile;
        if (!blobFile.open(modelPath, std::ios::in | std::ios::binary)) {
            THROW_IE_EXCEPTION << "Could not open file: " << modelPath;
        }
        std::istream graphBlob(&blobFile);

        ExecutableNetwork executableNetwork = ie.ImportNetwork(graphBlob, contextPtr);
        blobFile.close();

        // ---- Create infer request
        InferRequest inferRequest;
        inferRequest = executableNetwork.CreateInferRequest();

        Time start_sync;
        Time end_sync;

        start_sync = Now();
        for (int i = 0; i < numberOfIterations; ++i) {
            ROI roi {0, 2, 2, 1900, 1000};

            // ---- Create remote NV12 blob by using already exists remote memory
            const size_t yPlanes = 1;
            const size_t uvPlanes = 2;
            ParamMap blobYParamMap = {{VPUX_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFD},
                                            {VPUX_PARAM_KEY(MEM_OFFSET), static_cast<size_t>(0)}};

            TensorDesc inputYTensor = TensorDesc(Precision::U8, {1, yPlanes, inputHeight, inputWidth}, Layout::NHWC);
            RemoteBlob::Ptr remoteYBlobPtr = contextPtr->CreateBlob(inputYTensor, blobYParamMap);

            ParamMap blobUVParamMap = {{VPUX_PARAM_KEY(REMOTE_MEMORY_FD), remoteMemoryFD},
                                            {VPUX_PARAM_KEY(MEM_OFFSET), static_cast<size_t>(inputWidth * inputHeight * yPlanes)}};

            TensorDesc inputUVTensor = TensorDesc(Precision::U8, {1, uvPlanes, inputHeight / 2, inputWidth / 2}, Layout::NHWC);
            RemoteBlob::Ptr remoteUVBlobPtr = contextPtr->CreateBlob(inputUVTensor, blobUVParamMap);

            NV12Blob::Ptr remoteNV12BlobPtr = make_shared_blob<NV12Blob>(remoteYBlobPtr, remoteUVBlobPtr);

            Blob::Ptr remoteROIBlobPtr = remoteNV12BlobPtr->createROI(roi);
            // Specify input
            auto inputsInfo = executableNetwork.GetInputsInfo();
            const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;

            // Since it 228x228 image on 224x224 network, resize preprocessing also required
            PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
            preprocInfo.setColorFormat(ColorFormat::NV12);

            // ---- Set remote NV12 blob with preprocessing information
            inferRequest.SetBlob(inputName, remoteROIBlobPtr, preprocInfo);

            // ---- Run the request synchronously
            inferRequest.Infer();
        }
        end_sync = Now();

        auto elapsedSync = std::chrono::duration_cast<std::chrono::milliseconds>(end_sync - start_sync);
        auto executionTimeMs = elapsedSync.count();

        std::cout << "Execution inference (ms): " << executionTimeMs << " on " << numberOfIterations << " iterations"
              << std::endl;
        std::cout << "One frame execution (ms): " << executionTimeMs / numberOfIterations << std::endl;
        const auto inferencePerSeconds = 1000 / (static_cast<float>(executionTimeMs) / numberOfIterations);
        std::cout << "Inference per seconds (fps): " << inferencePerSeconds << std::endl;
    }
    catch (const std::exception& error) {
        slog::err << "" << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
