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

// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/classification_results.h"
#include "samples/slog.hpp"
#include "format_reader_ptr.h"
// clang-format on

#include <vpux/utils/IE/blob.hpp>
#include "multi_classification_sample.h"

using namespace ov;

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

ov::runtime::Tensor deQuantize(const ov::runtime::Tensor &quantTensor,
                               float scale,
                               uint8_t zeroPoint) {
    auto outputTensor = ov::runtime::Tensor(ov::element::f32, quantTensor.get_shape());
    const auto* quantRaw = quantTensor.data<const uint8_t>();
    float* outRaw = outputTensor.data<float>();

    for (size_t pos = 0; pos < quantTensor.get_size(); pos++) {
        outRaw[pos] = (quantRaw[pos] - zeroPoint) * scale;
    }

    return outputTensor;
}

ov::runtime::Tensor convertF16ToF32(const ov::runtime::Tensor& inputTensor) {
    auto outputTensor = ov::runtime::Tensor(ov::element::f32, inputTensor.get_shape());
    const auto* inRaw = inputTensor.data<float16>();
    float* outRaw = outputTensor.data<float>();

    for (size_t pos = 0; pos < outputTensor.get_size(); pos++) {
        outRaw[pos] = inRaw[pos];
    }

    return outputTensor;
}

std::mutex mtx;

void runInfer(const std::string& blob_path,
              const std::string& image_path,
              ov::runtime::Core& ie) {
    try {
        // -------- 3. Read pre-compiled blob --------
        ov::runtime::ExecutableNetwork executable_network = [&]() {
            const std::lock_guard<std::mutex> mtx_lock(mtx);

            slog::info << "Loading blob:\t" << blob_path << slog::endl;
            std::ifstream blob_stream(blob_path, std::ios::binary);
            return ie.import_model(blob_stream, "VPUX", {});
        }();

        OPENVINO_ASSERT(executable_network.inputs().size() == 1,
                        "Sample supports models with 1 input only");

        OPENVINO_ASSERT(executable_network.outputs().size() == 1,
                        "Sample supports models with 1 output only");

        // -------- 4. Set up input --------
        slog::info << "Processing input tensor" << slog::endl;

        // Read input image to a tensor and set it to an infer request
        // without resize and layout conversions
        FormatReader::ReaderPtr reader(image_path.c_str());
        if (reader.get() == nullptr) {
            throw std::logic_error("Image " + image_path + " cannot be read!");
        }

        /** Image reader is expected to return interlaced (NHWC) BGR image **/
        auto input = *executable_network.inputs().begin();
        ov::element::Type input_type = input.get_tensor().get_element_type();
        ov::Shape input_shape = input.get_shape();

        size_t image_width = input_shape.at(3);
        size_t image_height = input_shape.at(2);
        std::shared_ptr<unsigned char> input_data = reader->getData(image_width, image_height);

        // just wrap image data by ov::runtime::Tensor without allocating of new memory
        ov::runtime::Tensor input_tensor = ov::runtime::Tensor(input_type,
                                                               input_shape,
                                                               input_data.get());

        // -------- 5. Create an infer request --------
        ov::runtime::InferRequest infer_request = executable_network.create_infer_request();
        slog::info << "CreateInferRequest completed successfully" << slog::endl;

        // -------- 6. Prepare input --------
        infer_request.set_input_tensor(input_tensor);

        // -------- 7. Do inference synchronously --------
        infer_request.infer();
        slog::info << "inferRequest completed successfully" << slog::endl;

        // -------- 8. Process output --------
        slog::info << "Processing output tensor" << slog::endl;

        ov::runtime::Tensor output_tensor = infer_request.get_output_tensor();
        if (!output_tensor) {
            throw std::logic_error("Cannot get output tensor from infer_request!");
        }
        // Read labels from file (e.x. AlexNet.labels)
        const auto labels_path = fileNameNoExt(blob_path) + ".labels";
        std::vector<std::string> labels = readLabelsFromFile(labels_path);

        size_t batchSize = input_shape[0];

        std::vector<std::string> imageNames = { image_path };
        const size_t maxNumOfTop = 10;
        const size_t resultsCount = output_tensor.get_size() / batchSize;
        const size_t printedResultsCount = std::min(resultsCount, maxNumOfTop);

        // de-Quantization
        int zeroPoint = FLAGS_z;
        if (zeroPoint < std::numeric_limits<uint8_t>::min() || zeroPoint > std::numeric_limits<uint8_t>::max()) {
            slog::warn << "zeroPoint value " << zeroPoint << " overflows byte. Setting default." << slog::endl;
            zeroPoint = DEFAULT_ZERO_POINT;
        }
        float scale = static_cast<float>(FLAGS_s);
        slog::info << "zeroPoint: " << zeroPoint << slog::endl;
        slog::info << "scale: " << scale << slog::endl;

        ov::runtime::Tensor classificationOut;
        if (output_tensor.get_element_type() == ov::element::u8) {
            classificationOut = deQuantize(output_tensor, scale, zeroPoint);
        } else {
            classificationOut = convertF16ToF32(output_tensor);
        }

        ClassificationResult classificationResult(classificationOut, imageNames,
                                                  batchSize, printedResultsCount,
                                                  labels);
        classificationResult.show();

        const auto outFilePath = "./output.dat";
        std::ofstream outFile(outFilePath, std::ios::binary);
        if (outFile.is_open()) {
            outFile.write(reinterpret_cast<const char*>(classificationOut.data()),
                          classificationOut.get_byte_size());
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
        // -------- Get OpenVINO runtime version --------
        slog::info << ov::get_openvino_version() << slog::endl;

        // -------- Parsing and validation of input arguments --------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        const std::string image_path_1 = FLAGS_i1;
        const std::string image_path_2 = FLAGS_i2;

        const auto blob_path_1 = FLAGS_m1;
        const auto blob_path_2 = FLAGS_m1;

        // -------- 1. Initialize OpenVINO Runtime Core --------
        slog::info << "Creating Inference Engine" << slog::endl;
        ov::runtime::Core core;

        // -------- 2. Run inference --------
        std::thread t1(runInfer, blob_path_1, image_path_1, std::ref(core));
        std::thread t2(runInfer, blob_path_1, image_path_2, std::ref(core));
        t1.join();
        t2.join();
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
