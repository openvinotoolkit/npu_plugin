//
// Copyright 2021 Intel Corporation.
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
#include <thread>
#include <mutex>

#include <inference_engine.hpp>

#include <samples/common.hpp>
#include <samples/args_helper.hpp>
#include <samples/classification_results.h>
#include <samples/slog.hpp>

#include <format_reader_ptr.h>
#include <vpux/utils/IE/blob.hpp>
#include "multi_classification_sample.h"

using namespace InferenceEngine;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i1.empty()) {
        throw std::logic_error("Parameter -i1 is not set");
    }

    if (FLAGS_m1.empty()) {
        throw std::logic_error("Parameter -m1 is not set");
    }

    if (FLAGS_i2.empty()) {
        throw std::logic_error("Parameter -i2 is not set");
    }

    if (FLAGS_m2.empty()) {
        throw std::logic_error("Parameter -m2 is not set");
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

std::mutex mtx;

void runInfer(const std::string& binFileName, const std::string& imageFileName, Core& ie) {
try {
    mtx.lock();
    ExecutableNetwork importedNetwork = ie.ImportNetwork(binFileName, "VPUX", {});
    mtx.unlock();
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 3. Configure input & output ---------------------------------------------
    ConstInputsDataMap inputInfo = importedNetwork.GetInputsInfo();

    if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");

    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 4. Create infer request -------------------------------------------------
    InferenceEngine::InferRequest inferRequest = importedNetwork.CreateInferRequest();
    slog::info << "CreateInferRequest completed successfully" << slog::endl;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 5. Prepare input --------------------------------------------------------
    /** Creating input blob **/
    /** Filling input tensor with images. **/
    FormatReader::ReaderPtr image_reader(imageFileName.c_str());
    if (image_reader.get() == nullptr) {
        throw std::logic_error("Image " + imageFileName + " cannot be read!");
    }

    /** Image reader is expected to return interlaced (NHWC) BGR image **/
    TensorDesc inputDataDesc = inputInfo.begin()->second->getTensorDesc();
    std::vector<size_t> inputBlobDims = inputDataDesc.getDims();
    size_t imageWidth = inputBlobDims.at(3);
    size_t imageHeight = inputBlobDims.at(2);

    Blob::Ptr imageBlob = make_shared_blob<uint8_t>(TensorDesc(Precision::U8,
                                                               inputBlobDims,
                                                               Layout::NHWC), image_reader->getData(imageWidth, imageHeight).get());

    const auto firstInputName = inputInfo.begin()->first;
    inferRequest.SetBlob(firstInputName, imageBlob);

    inferRequest.Infer();
    slog::info << "inferRequest completed successfully" << slog::endl;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 6. Process output -------------------------------------------------------
    slog::info << "Processing output blobs" << slog::endl;

    ConstOutputsDataMap outputInfo = importedNetwork.GetOutputsInfo();
    if (outputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 output");

    const auto firstOutputName = outputInfo.begin()->first;

    Blob::Ptr outputBlob = inferRequest.GetBlob(firstOutputName);
    if (!outputBlob) {
        throw std::logic_error("Cannot get output blob from inferRequest");
    }

    outputBlob = vpux::toPrecision(InferenceEngine::as<InferenceEngine::MemoryBlob>(outputBlob),
                                   InferenceEngine::Precision::FP32);


    /** Read labels from file (e.x. AlexNet.labels) **/
    const auto labelFileName = fileNameNoExt(binFileName) + ".labels";
    std::vector<std::string> labels = readLabelsFromFile(labelFileName);

    auto inputInfoItem = *inputInfo.begin();
    Blob::Ptr inputBlob = inferRequest.GetBlob(inputInfoItem.first);

    const SizeVector inputDims = inputBlob->getTensorDesc().getDims();
    size_t batchSize = inputDims.at(0);

    std::vector<std::string> imageNames = { imageFileName };
    const size_t maxNumOfTop = 10;
    const size_t resultsCount = outputBlob->size() / batchSize;
    const size_t printedResultsCount = resultsCount > maxNumOfTop ? maxNumOfTop : resultsCount;

    // de-Quantization
    int zeroPoint = FLAGS_z;
    if (zeroPoint < std::numeric_limits<uint8_t>::min() || zeroPoint > std::numeric_limits<uint8_t>::max()) {
        slog::warn << "zeroPoint value " << zeroPoint << " overflows byte. Setting default." << slog::endl;
        zeroPoint = DEFAULT_ZERO_POINT;
    }
    auto scale = static_cast<float>(FLAGS_s);
    slog::info << "zeroPoint: " << zeroPoint << slog::endl;
    slog::info << "scale: " << scale << slog::endl;

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

    const auto outFilePath = "./output.dat";
    std::ofstream outFile(outFilePath, std::ios::binary);
    if (outFile.is_open()) {
        outFile.write(outputBlob->buffer(), outputBlob->byteSize());
    } else {
        slog::warn << "Failed to open '" << outFilePath << "'" << slog::endl;
    }
    outFile.close();
} catch (const std::exception& error) {
    slog::err << "" << error.what() << slog::endl;
    return;
} catch (...) {
    slog::err << "Unknown/internal exception happened." << slog::endl;
    return;
}
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

        const auto imageFileName1 = FLAGS_i1;
        const auto imageFileName2 = FLAGS_i2;

        // -----------------------------------------------------------------------------------------------------


        // --------------------------- 2. Read blob Generated by MCM Compiler ----------------------------------
        const auto binFileName1 = FLAGS_m1;
        slog::info << "Loading blob:\t" << binFileName1 << slog::endl;

        const auto binFileName2 = FLAGS_m1;
        slog::info << "Loading blob:\t" << binFileName2 << slog::endl;

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Creating Inference Engine" << slog::endl;
        Core ie;
        std::thread t1(runInfer, binFileName1, imageFileName1, std::ref(ie));
        std::thread t2(runInfer, binFileName2, imageFileName2, std::ref(ie));
        t1.join();
        t2.join();
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
