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

#include <samples/common.hpp>
#include <samples/args_helper.hpp>
#include <samples/classification_results.h>
#include <samples/slog.hpp>

#include "classification_sample.h"
#include "infer_request_wrap.hpp"

#include <file_reader.h>

//HDDLUnite header files
#include "WorkloadContext.h"
#include "hddl2_params.hpp"

#include "creator_blob_nv12.h"
#include "helper_remote_memory.h"
#include "vpux/vpux_plugin_params.hpp"

#include <vpux/vpux_plugin_config.hpp>

using namespace InferenceEngine;

#define MAX_INFER_REQS 32

const auto DEFAULT_INFERENCE_SHAVES = "8";

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

std::vector<std::string> readLabelsFromFile(const std::string& labelFileName) {
    std::vector<std::string> labels;

    std::ifstream inputFile;
    inputFile.open(labelFileName, std::ios::in);
    if (inputFile.is_open()) {
        std::string strLine;
        while (std::getline(inputFile, strLine)) {
            trim(strLine);
            labels.push_back(strLine);
        }
    }
    return labels;
}

Blob::Ptr deQuantize(const Blob::Ptr &quantBlob, float scale, uint8_t zeroPoint) {
  const TensorDesc quantTensor = quantBlob->getTensorDesc();
  SizeVector dims = quantTensor.getDims();
  size_t batchSize = dims.at(0);
  slog::info << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3] << slog::endl;
  const size_t Count = quantBlob->size() / batchSize;
  const size_t ResultsCount = Count > 1000 ? 1000 : Count;
  dims[1] = ResultsCount;
  const TensorDesc outTensor = TensorDesc(
      InferenceEngine::Precision::FP32,
      dims,
      quantTensor.getLayout());
  slog::info << dims[0] << " " << dims[1] << " " << dims[2] << " " << dims[3] << slog::endl;
  Blob::Ptr outputBlob = make_shared_blob<float>(outTensor);
  outputBlob->allocate();
  float *outRaw = outputBlob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
  const uint8_t *quantRaw = quantBlob->cbuffer().as<const uint8_t *>();

  for (size_t pos = 0; pos < outputBlob->size(); pos++) {
    outRaw[pos] = (quantRaw[pos] - zeroPoint) * scale;
  }
  return outputBlob;
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

        // Number of requests
        uint32_t nireq = FLAGS_nireq;
        if (nireq > MAX_INFER_REQS)
            throw std::logic_error("Invalid nireq configuration!");

        size_t iteration = 0;
        uint32_t niter = FLAGS_niter;

        /** This vector stores paths to the processed images **/
        std::string inputNV12Path = FLAGS_i;

        int32_t image_width = FLAGS_iw;
        int32_t image_height = FLAGS_ih;
        slog::info << "Inference on image with resolution #" << image_width << "x" << image_height << slog::endl;

        // ----------------------------Call HDDLUnite to create workload context----------------------------------------------------
        WorkloadID workloadId = -1;
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
        const size_t inputWidth = static_cast<size_t>(image_width);
        const size_t inputHeight = static_cast<size_t>(image_height);
        const size_t nv12Size = inputWidth * inputHeight * 3 / 2;
        std::vector<uint8_t> nv12InputBlobMemory;
        nv12InputBlobMemory.resize(nv12Size);

        NV12Blob::Ptr inputNV12Blob = NV12Blob_Creator::createFromFile(
            inputNV12Path, inputWidth, inputHeight, nv12InputBlobMemory.data());

        // ----- Allocate memory with HddlUnite on device
        auto remoteMemoryFD = _remoteMemoryHelper.allocateRemoteMemory(workloadId, nv12Size, inputNV12Blob->y()->cbuffer().as<void*>());

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Creating Inference Engine" << slog::endl;
        Core ie;

        ie.SetConfig({{"VPUX_GRAPH_COLOR_FORMAT", "RGB"}}, "VPUX");

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

        std::map<std::string, std::string> config;
        config[VPUX_CONFIG_KEY(INFERENCE_SHAVES)] = DEFAULT_INFERENCE_SHAVES;

        ExecutableNetwork executableNetwork = ie.ImportNetwork(graphBlob, contextPtr, config);
        blobFile.close();

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

        PreProcPara desc;
        desc.aspect_ratio = 1;
        desc.align_center = 1;
        remoteNV12BlobPtr->y()->updatePreProcDesc(desc);

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------
        ConstInputsDataMap inputInfo = executableNetwork.GetInputsInfo();

        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Create infer request -------------------------------------------------
        InferRequestsQueue inferRequestsQueue(executableNetwork, nireq);
        slog::info << "CreateInferRequest completed successfully" << slog::endl;
        // -----------------------------------------------------------------------------------------------------

        // Specify input
        auto inputsInfo = executableNetwork.GetInputsInfo();
        const std::string inputName = executableNetwork.GetInputsInfo().begin()->first;

        // warming up - out of scope
        auto inferRequest = inferRequestsQueue.getIdleRequest();
        if (!inferRequest) {
            THROW_IE_EXCEPTION << "No idle Infer Requests!";
        }

        PreProcessInfo preprocInfo = inferRequest->getPreProcess(inputName);
        preprocInfo.setColorFormat(ColorFormat::NV12);

        inferRequest->setBlob(inputName, remoteNV12BlobPtr, preprocInfo);
        inferRequest->startAsync();
        inferRequestsQueue.waitAll();

        auto startTime = Time::now();
        while ((niter != 0LL && iteration < niter)) {
            auto inferRequest = inferRequestsQueue.getIdleRequest();
            if (!inferRequest) {
                THROW_IE_EXCEPTION << "No idle Infer Requests!";
            }

            preprocInfo = inferRequest->getPreProcess(inputName);
            preprocInfo.setColorFormat(ColorFormat::NV12);

            inferRequest->setBlob(inputName, remoteNV12BlobPtr, preprocInfo);
            inferRequest->startAsync();

            iteration++;
            if (iteration%100 == 0)
                slog::info << "inferRequest completed successfully #" << iteration << slog::endl;
        }
        // -----------------------------------------------------------------------------------------------------

        // wait the latest inference executions
        inferRequestsQueue.waitAll();
        auto execTime = std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - startTime).count();

        // --------------------------- 6. Process output -------------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;

        ConstOutputsDataMap outputInfo = executableNetwork.GetOutputsInfo();

        std::string firstOutputName = outputInfo.begin()->first;
        Blob::Ptr firstoutputBlob = inferRequestsQueue.requests.at(0)->getBlob(firstOutputName.c_str());

        size_t batchSize = 1;
        slog::info << "Output batch size " << batchSize << slog::endl;

        if (outputInfo.size() != 1) { // Multiple outputs
            int outputId = 0;
            for (const auto& output : outputInfo) {
                const auto outputBlobName = output.first;
                for (size_t requestId = 0; requestId < nireq; requestId++) {
                    Blob::Ptr outputBlob = inferRequestsQueue.requests.at(requestId)->getBlob(outputBlobName.c_str());
                    if (!outputBlob) {
                        throw std::logic_error("Cannot get output blob from inferRequest");
                    }

                    std::string outFilePath = "./output" + std::to_string(outputId) + "_"+ std::to_string(requestId) + ".dat";
                    std::ofstream outFile(outFilePath, std::ios::binary);
                    if (outFile.is_open()) {
                        outFile.write(outputBlob->buffer(), outputBlob->byteSize());
                    } else {
                        slog::warn << "Failed to open '" << outFilePath << "'" << slog::endl;
                    }
                    outFile.close();
                }
                outputId++;
            }
        } else { // Single output
            for (size_t requestId = 0; requestId < nireq; requestId++) {
                Blob::Ptr outputBlob = inferRequestsQueue.requests.at(requestId)->getBlob(firstOutputName.c_str());
                /** Read labels from file (e.x. AlexNet.labels) **/
                std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";
                std::vector<std::string> labels = readLabelsFromFile(labelFileName);

                std::vector<std::string> imageNames = { inputNV12Path };
                const size_t maxNumOfTop = 10;
                const size_t resultsCount = outputBlob->size() / batchSize;
                const size_t printedResultsCount = resultsCount > maxNumOfTop ? maxNumOfTop : resultsCount;
                slog::info << "resultsCount " << resultsCount << slog::endl;
                slog::info << "printedResultsCount " << printedResultsCount << slog::endl;

                // de-Quantization
                int zeroPoint = FLAGS_z;
                if (zeroPoint < std::numeric_limits<uint8_t>::min() || zeroPoint > std::numeric_limits<uint8_t>::max()) {
                    slog::warn << "zeroPoint value " << zeroPoint << " overflows byte. Setting default." << slog::endl;
                    zeroPoint = DEFAULT_ZERO_POINT;
                }
                float scale = static_cast<float>(FLAGS_s);
                slog::info<< "zeroPoint" << zeroPoint << slog::endl;
                slog::info<< "scale" << scale << slog::endl;

                Blob::Ptr classificationOut = nullptr;
                if (outputBlob->getTensorDesc().getPrecision() == InferenceEngine::Precision::U8) {
                    classificationOut = deQuantize(outputBlob, scale, zeroPoint);
                } else {
                    classificationOut = outputBlob;
                }

                ClassificationResult classificationResult(classificationOut, imageNames,
                                                  batchSize, printedResultsCount,
                                                  labels);
                classificationResult.print();

                std::string outFilePath = "./output_" + std::to_string(requestId) + ".dat";
                std::ofstream outFile(outFilePath, std::ios::binary);
                if (outFile.is_open()) {
                    outFile.write(outputBlob->buffer(), outputBlob->byteSize());
                } else {
                    slog::warn << "Failed to open '" << outFilePath << "'" << slog::endl;
                }
                outFile.close();
            }
        }
        double fps = batchSize * 1000.0 * 1000 * 1000 * iteration / execTime;
        auto double_to_string = [] (const double number) {
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(2) << number;
                    return ss.str();
        };
        std::cout << "Throughput: " << double_to_string(fps) << " FPS" << std::endl;
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
