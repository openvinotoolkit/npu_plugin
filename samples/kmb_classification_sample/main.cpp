//
// Copyright 2019 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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
// define for NV12Blob
#include <ie_compound_blob.h>

using namespace InferenceEngine;
#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080

static void setNV12Preproc(const std::string& inputName, const std::string& inputFilePath,
    std::vector<InferReqWrap::Ptr> requests, std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator>& allocator,
    size_t expectedWidth, size_t expectedHeight) {
    Blob::Ptr inputBlob;
    inputBlob = vpu::KmbPlugin::utils::fromNV12File(inputFilePath, expectedWidth, expectedHeight, allocator);
    for (size_t requestId = 0; requestId < requests.size(); requestId++) {
        PreProcessInfo preprocInfo = requests.at(requestId)->getPreProcess(inputName);
        preprocInfo.setResizeAlgorithm(RESIZE_BILINEAR);
        preprocInfo.setColorFormat(ColorFormat::NV12);
        requests.at(requestId)->setBlob(inputName, inputBlob, preprocInfo);
    }
}

std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> buildAllocator(const char* allocatorType) {
    if (allocatorType == nullptr) {
        return std::make_shared<vpu::KmbPlugin::utils::VPUSMMAllocator>();
    }

    std::string allocTypeStr(allocatorType);
    if (allocTypeStr == "NATIVE") {
        return std::make_shared<vpu::KmbPlugin::utils::NativeAllocator>();
    } else if (allocTypeStr == "UDMA") {
        throw std::runtime_error("buildAllocator: UDMA is not supported");
    }

    // VPUSMM is default
    return std::make_shared<vpu::KmbPlugin::utils::VPUSMMAllocator>();
}

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

        size_t iteration = 0;
        uint32_t niter = FLAGS_niter;


        /** This vector stores paths to the processed images **/
        std::string imageFileName = FLAGS_i;

        // -----------------------------------------------------------------------------------------------------
        std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> kmbAllocator =
            buildAllocator(std::getenv("IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE"));

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Creating Inference Engine" << slog::endl;
        Core ie;

        // --------------------------- 2. Read blob Generated by MCM Compiler ----------------------------------
        std::string binFileName = FLAGS_m;
        slog::info << "Loading blob:\t" << binFileName << slog::endl;

#if 1  //Load blob from memory buffer
        std::ifstream fileReader(binFileName, std::ios_base::ate | std::ios_base::binary);
        if (!fileReader.good()) {
            throw std::runtime_error("readNV12FileHelper: failed to open file " + binFileName);
        }
        const size_t fileSize = fileReader.tellg();
        char * blob_buf = reinterpret_cast<char *>(malloc(fileSize));
        fileReader.seekg(0, std::ios_base::beg);
        fileReader.read(blob_buf, fileSize);
        fileReader.close();
        ExecutableNetwork importedNetwork = ie.ImportNetwork(reinterpret_cast<uint8_t *>(blob_buf), fileSize, "VPUX", {});
        free(blob_buf);
#else
        ExecutableNetwork importedNetwork = ie.ImportNetwork(binFileName, "VPUX", {});
#endif
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------
        ConstInputsDataMap inputInfo = importedNetwork.GetInputsInfo();

        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Create infer request -------------------------------------------------
        InferRequestsQueue inferRequestsQueue(importedNetwork, nireq);
        slog::info << "CreateInferRequest completed successfully" << slog::endl;
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Prepare input --------------------------------------------------------
        /** Creating input blob **/
        std::string input_name = inputInfo.begin()->first;
        setNV12Preproc(input_name, imageFileName, inferRequestsQueue.requests, kmbAllocator, IMAGE_WIDTH, IMAGE_HEIGHT);

        auto startTime = Time::now();
        while ((niter != 0LL && iteration < niter)) {
            auto inferRequest = inferRequestsQueue.getIdleRequest();
            if (!inferRequest) {
                THROW_IE_EXCEPTION << "No idle Infer Requests!";
            }
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

        ConstOutputsDataMap outputInfo = importedNetwork.GetOutputsInfo();

        std::string firstOutputName = outputInfo.begin()->first;
        Blob::Ptr firstoutputBlob = inferRequestsQueue.requests.at(0)->getBlob(firstOutputName.c_str());
        const SizeVector outputDims = firstoutputBlob->getTensorDesc().getDims();
        size_t batchSize = outputDims.at(0);
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

                std::vector<std::string> imageNames = { imageFileName };
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
                slog::info<< "zeroPoint:" << zeroPoint << slog::endl;
                slog::info<< "scale:" << scale << slog::endl;

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
