//
// Copyright 2020 Intel Corporation.
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

#include "vpux/utils/IE/blob.hpp"
#include "yolo_helpers.hpp"

#include <inference_engine.hpp>
#include <blob_factory.hpp>
#include <caseless.hpp>
#include <file_utils.h>
#include <precision_utils.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <gflags/gflags.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace ie = InferenceEngine;

ie::details::CaselessEq<std::string> strEq;

//
// Command line options
//

DEFINE_string(network, "", "Network file (either XML or pre-compiled blob)");
DEFINE_string(input, "", "Input file(s)");
DEFINE_string(device, "", "Device to use");
DEFINE_string(config, "", "Path to the configuration file (optional)");
DEFINE_string(ip, "", "Input precision (default FP32)");
DEFINE_string(op, "", "Input precision (default FP32)");
DEFINE_string(il, "", "Input layout (default NCHW)");
DEFINE_string(ol, "", "Input layout (default NCHW)");

DEFINE_bool(run_test, false, "Run the test (compare current results with previously dumped)");
DEFINE_string(mode, "", "Comparison mode to use");

DEFINE_uint32(top_k, 1, "Top K parameter for 'classification' mode");
DEFINE_double(prob_tolerance, 1e-4, "Probability tolerance for 'classification/ssd/yolo' mode");

DEFINE_double(raw_tolerance, 1e-4, "Tolerance for 'raw' mode (absolute diff)");
DEFINE_double(confidence_threshold, 1e-4, "Confidence threshold for Detection mode");
DEFINE_double(box_tolerance, 1e-4, "Box tolerance for 'detection' mode");

DEFINE_string(log_level, "", "IE logger level (optional)");
DEFINE_string(color_format, "BGR", "Color format for input: RGB or BGR");

// for Yolo
DEFINE_bool(is_tiny_yolo, false, "Is it Tiny Yolo or not (true or false)?");

std::vector<std::string> splitStringList(const std::string& str, char delim) {
    std::vector<std::string> out;

    if (str.empty()) {
        return out;
    }

    std::istringstream istr(str);

    std::string elem;
    while (std::getline(istr, elem, delim)) {
        if (elem.empty()) {
            continue;
        }

        out.push_back(std::move(elem));
    }

    return out;
}

void parseCommandLine(int argc, char* argv[]) {
    std::ostringstream usage;
    usage << "Usage: " << argv[0] << "[<options>]";
    gflags::SetUsageMessage(usage.str());

    std::ostringstream version;
    version << ie::GetInferenceEngineVersion();
    gflags::SetVersionString(version.str());

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::cout << "Parameters:" << std::endl;
    std::cout << "    Network file:     " << FLAGS_network << std::endl;
    std::cout << "    Input file(s):    " << FLAGS_input << std::endl;
    std::cout << "    Color format:     " << FLAGS_color_format << std::endl;
    std::cout << "    Input precision:  " << FLAGS_ip << std::endl;
    std::cout << "    Output precision: " << FLAGS_op << std::endl;
    std::cout << "    Input layout:     " << FLAGS_il << std::endl;
    std::cout << "    Output layout:    " << FLAGS_ol << std::endl;
    std::cout << "    Device:           " << FLAGS_device << std::endl;
    std::cout << "    Config file:      " << FLAGS_config << std::endl;
    std::cout << "    Run test:         " << FLAGS_run_test << std::endl;
    if (FLAGS_run_test) {
        std::cout << "    Mode:             " << FLAGS_mode << std::endl;
        if (strEq(FLAGS_mode, "classification")) {
            std::cout << "    Top K:            " << FLAGS_top_k << std::endl;
            std::cout << "    Tolerance:        " << FLAGS_prob_tolerance << std::endl;
        } else if (strEq(FLAGS_mode, "raw")) {
            std::cout << "    Tolerance:        " << FLAGS_raw_tolerance << std::endl;
        }
    }
    std::cout << "    Log level:        " << FLAGS_log_level << std::endl;
    std::cout << std::endl;
}

//
// OpenCV to InferenceEngine conversion
//

bool isImage(const ie::TensorDesc& desc) {
    const auto& dims = desc.getDims();

    if (dims.size() == 4) {
        const auto numChannels = dims[1];
        return (numChannels == 3) || (numChannels == 4);
    }

    return false;
}

std::vector<cv::Mat> ieToCv(const ie::MemoryBlob::Ptr& blob, size_t batchInd = 0, size_t depthInd = 0) {
    IE_ASSERT(blob != nullptr);

    const auto& tensDesc = blob->getTensorDesc();
    const auto layout = tensDesc.getLayout();
    const auto& precision = tensDesc.getPrecision();
    const auto& dims = tensDesc.getDims();

    IE_ASSERT(layout == ie::Layout::NCHW || layout == ie::Layout::NCDHW);

    IE_ASSERT(precision == ie::Precision::U8   ||
              precision == ie::Precision::FP32 ||
              precision == ie::Precision::FP16);

    int cvType = 0;
    size_t elemSize = 0;

    if (precision == ie::Precision::U8) {
        cvType = CV_8UC1;
        elemSize = sizeof(uint8_t);
    } else if (precision == ie::Precision::FP32) {
        cvType = CV_32FC1;
        elemSize = sizeof(float);
    } else if (precision == ie::Precision::FP16) {
        cvType = CV_16SC1;
        elemSize = sizeof(ie::ie_fp16);
    }

    std::vector<cv::Mat> out;

    if (layout == ie::Layout::NCHW) {
        const auto N = dims[0];
        const auto C = dims[1];
        const auto H = dims[2];
        const auto W = dims[3];

        IE_ASSERT(batchInd < N);
        IE_ASSERT(C == 3 || C == 4);

        const auto blobMem = blob->wmap();
        const auto blobPtr = blobMem.as<uint8_t *>();

        out.resize(C);
        for (size_t c = 0; c < C; ++c) {
            out[c] = cv::Mat(H, W, cvType, blobPtr + (batchInd * C + c) * W * H * elemSize);
        }
    } else if (layout == ie::Layout::NCDHW) {
        const auto N = dims[0];
        const auto C = dims[1];
        const auto D = dims[2];
        const auto H = dims[3];
        const auto W = dims[4];

        const auto& strides = tensDesc.getBlockingDesc().getStrides();
        const auto strideN = strides[0];
        const auto strideC = strides[1];
        const auto strideD = strides[2];

        IE_ASSERT(batchInd < N);
        IE_ASSERT(depthInd < D);
        IE_ASSERT(C == 3 || C == 4);

        const auto blobMem = blob->wmap();
        const auto blobPtr = blobMem.as<uint8_t *>();

        out.resize(C);
        for (size_t c = 0; c < C; ++c) {
            out[c] = cv::Mat(H, W, cvType, blobPtr + (strideN * batchInd + strideC * c + strideD * depthInd) * elemSize);
        }
    }

    return out;
}

void cvToIe(const cv::Mat& cvImg, const ie::MemoryBlob::Ptr& ieBlob, const std::string& colorFormat) {
    const auto& tensDesc = ieBlob->getTensorDesc();
    const auto& precision = tensDesc.getPrecision();
    const auto layout = tensDesc.getLayout();
    const auto& dims = tensDesc.getDims();

    IE_ASSERT(layout == ie::Layout::NHWC || layout == ie::Layout::NCHW) << "Unsupported layout " << layout;

    const auto N = dims[0];
    const auto C = dims[1];
    const auto H = dims[2];
    const auto W = dims[3];

    IE_ASSERT(C == 3 || C == 4) << "Unsupported number of channels " << C;

    int cvType = 0;

    if (precision == ie::Precision::U8) {
        cvType = CV_8UC(C);
    } else if (precision == ie::Precision::FP32) {
        cvType = CV_32FC(C);
    } else if (precision == ie::Precision::FP16) {
        cvType = CV_16SC(C);
    } else {
        IE_ASSERT(precision == ie::Precision::U8   ||
                  precision == ie::Precision::FP32 ||
                  precision == ie::Precision::FP16) << "Unsupported precision " << precision;
    }

    cv::Mat in;

    if (C == 3) {
        if (colorFormat == "RGB") {
            cv::cvtColor(cvImg, in, cv::COLOR_BGR2RGB);
        } else {
            in = cvImg;
        }
    } else {
        if (colorFormat == "RGB") {
            cv::cvtColor(cvImg, in, cv::COLOR_BGR2RGBA);
        } else {
            cv::cvtColor(cvImg, in, cv::COLOR_BGR2BGRA);
        }
    }


    if (precision != ie::Precision::U8) {
        in.convertTo(in, CV_32F);
    }

    const auto pictureArea = static_cast<size_t>(in.size().area());

    if (W * H > pictureArea) {
        cv::resize(in, in, cv::Size(W, H), 0.0, 0.0, cv::INTER_AREA);
    } else {
        cv::resize(in, in, cv::Size(W, H), 0.0, 0.0, cv::INTER_LINEAR);
    }

    if (layout == ie::Layout::NHWC) {
        const auto blobMem = ieBlob->wmap();
        const auto blobPtr = blobMem.as<uint8_t *>();

        cv::Mat out(H, W, cvType, blobPtr);

        if (precision != ie::Precision::FP16) {
            in.copyTo(out);
        } else {
            const auto inPtr = in.ptr<float>();
            const auto outPtr = out.ptr<ie::ie_fp16>();
            ie::PrecisionUtils::f32tof16Arrays(outPtr, inPtr, out.size().area()*C);
        }

        for (size_t n = 1; n < N; ++n) {
            cv::Mat batch(H, W, cvType, blobPtr + out.size().area() * out.elemSize());
            out.copyTo(batch);
        }
    } else if (layout == ie::Layout::NCHW) {
        auto blobPlanes = ieToCv(ieBlob, 0);

        if (precision != ie::Precision::FP16) {
            cv::split(in, blobPlanes);
        } else {
            std::vector<cv::Mat> inPlanes;
            cv::split(in, inPlanes);

            IE_ASSERT(blobPlanes.size() == inPlanes.size());

            for (size_t i = 0; i < blobPlanes.size(); ++i) {
                const auto inPtr = inPlanes[i].ptr<float>();
                const auto outPtr = blobPlanes[i].ptr<ie::ie_fp16>();
                ie::PrecisionUtils::f32tof16Arrays(outPtr, inPtr, inPlanes[i].size().area());
            }
        }

        for (size_t n = 1; n < N; ++n) {
            const auto batchPlanes = ieToCv(ieBlob, n);

            IE_ASSERT(batchPlanes.size() == blobPlanes.size());

            for (size_t i = 0; i < blobPlanes.size(); ++i) {
                blobPlanes[i].copyTo(batchPlanes[i]);
            }
        }
    }
}

//
// File utils
//

std::string cleanName(std::string name) {
    std::replace_if(
        name.begin(), name.end(),
        [](char c) {
            return !std::isalnum(c);
        },
        '_');
    return name;
}

ie::MemoryBlob::Ptr loadImage(const ie::TensorDesc& desc, const std::string& filePath, const std::string& colorFormat) {
    const auto frame = cv::imread(filePath, cv::IMREAD_COLOR);
    IE_ASSERT(!frame.empty()) << "Failed to open input image file " << filePath;

    const auto blob = ie::as<ie::MemoryBlob>(make_blob_with_precision(desc));
    blob->allocate();

    cvToIe(frame, blob, colorFormat);

    return blob;
}

ie::MemoryBlob::Ptr loadBinary(const ie::TensorDesc& desc, const std::string& filePath) {
    const auto blob = ie::as<ie::MemoryBlob>(make_blob_with_precision(desc));

    std::ifstream binaryFile(filePath, std::ios_base::binary | std::ios_base::ate);

    IE_ASSERT(binaryFile) << "Failed to open input binary file";

    const int fileSize = binaryFile.tellg();
    binaryFile.seekg(0, std::ios_base::beg);
    const int expectedSize = static_cast<int>(blob->byteSize());
    IE_ASSERT(fileSize == expectedSize) << "File contains " << fileSize << " bytes, but " << expectedSize << " expected";

    IE_ASSERT(binaryFile.good()) << "While reading a file an error is encountered";

    blob->allocate();
    const auto blobMem = blob->wmap();
    const auto blobPtr = blobMem.as<char *>();
    binaryFile.read(blobPtr, static_cast<std::streamsize>(expectedSize));

    return blob;
}

ie::MemoryBlob::Ptr loadInput(const ie::TensorDesc& desc, const std::string& filePath, const std::string& colorFormat) {
    if (isImage(desc)) {
        return loadImage(desc, filePath, colorFormat);
    } else {
        return loadBinary(desc, filePath);
    }
}

ie::MemoryBlob::Ptr loadBlob(const ie::TensorDesc& desc, const std::string& filePath) {
    const auto blob = ie::as<ie::MemoryBlob>(make_blob_with_precision(desc));
    blob->allocate();

    std::ifstream file(filePath, std::ios_base::in | std::ios_base::binary);
    IE_ASSERT(file.is_open()) << "Can't open file " << filePath << " for read";

    const auto blobMem = blob->wmap();
    const auto blobPtr = blobMem.as<char*>();
    file.read(blobPtr, static_cast<std::streamsize>(blob->byteSize()));

    return blob;
}

void dumpBlob(const ie::MemoryBlob::Ptr& blob, const std::string& filePath) {
    std::ofstream file(filePath, std::ios_base::out | std::ios_base::binary);
    IE_ASSERT(file.is_open()) << "Can't open file " << filePath << " for write";

    const auto blobMem = blob->rmap();
    const auto blobPtr = blobMem.as<const char*>();
    file.write(blobPtr, static_cast<std::streamsize>(blob->byteSize()));
}

//
// Inference Engine
//

ie::Core ieCore;

void setupInferenceEngine() {
    if (!FLAGS_log_level.empty()) {
        ieCore.SetConfig({{CONFIG_KEY(LOG_LEVEL), FLAGS_log_level}}, FLAGS_device);
    }

    if (FLAGS_device == "CPU") {
        ieCore.SetConfig({{"LP_TRANSFORMS_MODE", CONFIG_VALUE(NO)}}, FLAGS_device);
    }

    if (!FLAGS_config.empty()) {
        std::ifstream file(FLAGS_config);
        IE_ASSERT(file.is_open()) << "Can't open file " << FLAGS_config << " for read";

        std::string key, value;
        while (file >> key >> value) {
            if (key.empty() || key[0] == '#') {
                continue;
            }

            ieCore.SetConfig({{key, value}}, FLAGS_device);
        }
    }
}

ie::ExecutableNetwork importNetwork(const std::string& filePath) {
    std::ifstream file(filePath, std::ios_base::in | std::ios_base::binary);
    IE_ASSERT(file.is_open()) << "Can't open file " << filePath << " for read";

    return ieCore.ImportNetwork(file, FLAGS_device);
}

ie::BlobMap runInfer(ie::ExecutableNetwork& exeNet, const ie::BlobMap& inputs) {
    auto inferRequest = exeNet.CreateInferRequest();

    for (const auto& p : inputs) {
        inferRequest.SetBlob(p.first, p.second);
    }

    inferRequest.Infer();

    ie::BlobMap out;

    for (const auto& p : exeNet.GetOutputsInfo()) {
        out.insert({p.first, inferRequest.GetBlob(p.first)});
    }

    return out;
}

//
// SSD_Detection mode
//

static std::vector<utils::BoundingBox> SSDBoxExtractor(float threshold, std::vector<float>& net_out, size_t imgWidth,
                                                       size_t imgHeight) {
    std::vector<utils::BoundingBox> boxes_result;

    if (net_out.empty()) {
        return boxes_result;
    }
    size_t oneDetectionSize = 7;

    IE_ASSERT(net_out.size() % oneDetectionSize == 0);

    for (size_t i = 0; i < net_out.size() / oneDetectionSize; ++i) {
        if (net_out[i * oneDetectionSize + 2] > threshold) {
            boxes_result.emplace_back(net_out[i * oneDetectionSize + 1], net_out[i * oneDetectionSize + 3] * imgWidth,
                                      net_out[i * oneDetectionSize + 4] * imgHeight,
                                      net_out[i * oneDetectionSize + 5] * imgWidth,
                                      net_out[i * oneDetectionSize + 6] * imgHeight, net_out[i * oneDetectionSize + 2]);
        }
    }

    return boxes_result;
}

bool checkBBoxOutputs(std::vector<utils::BoundingBox> &actualOutput,
                                               std::vector<utils::BoundingBox> &refOutput,
                                               const size_t imgWidth,
                                               const size_t imgHeight,
                                               const float boxTolerance,
                                               const float probTolerance) {
    std::cout << "Ref Top:" << std::endl;
    for (size_t i = 0; i < refOutput.size(); ++i) {
        const auto& bb = refOutput[i];
        std::cout << i << " : " << bb.idx
                  << " : [("
                  << bb.left << " " << bb.top << "), ("
                  << bb.right << " " << bb.bottom
                  << ")] : "
                  << bb.prob * 100 << "%"
                  << std::endl;
    }

    std::cout << "Actual top:" << std::endl;
    for (size_t i = 0; i < actualOutput.size(); ++i) {
        const auto& bb = actualOutput[i];
        std::cout << i << " : " << bb.idx
                  << " : [("
                  << bb.left << " " << bb.top << "), ("
                  << bb.right << " " << bb.bottom
                  << ")] : "
                  << bb.prob * 100 << "%" << std::endl;
    }

    for (const auto& refBB : refOutput) {
        bool found = false;

        float maxBoxError = 0.0f;
        float maxProbError = 0.0f;

        for (const auto& actualBB : actualOutput) {
            if (actualBB.idx != refBB.idx) {
                continue;
            }

            const utils::Box actualBox {
                    actualBB.left / imgWidth,
                    actualBB.top / imgHeight,
                    (actualBB.right - actualBB.left) / imgWidth,
                    (actualBB.bottom - actualBB.top) / imgHeight
            };
            const utils::Box refBox {
                    refBB.left / imgWidth,
                    refBB.top / imgHeight,
                    (refBB.right - refBB.left) / imgWidth,
                    (refBB.bottom - refBB.top) / imgHeight
            };

            const auto boxIntersection = boxIntersectionOverUnion(actualBox, refBox);
            const auto boxError = 1.0f - boxIntersection;
            maxBoxError = std::max(maxBoxError, boxError);

            const auto probError = std::fabs(actualBB.prob - refBB.prob);
            maxProbError = std::max(maxProbError, probError);

            if (boxError > boxTolerance) {
                continue;
            }

            if (probError > probTolerance) {
                continue;
            }

            found = true;
            break;
        }
        if (!found) {
            std::cout << "maxBoxError=" << maxBoxError << " " << "maxProbError=" << maxProbError << std::endl;
            return false;
        }
    }
    return true;
}

bool testSSDDetection(const ie::BlobMap& outputs, const ie::BlobMap& refOutputs, const ie::ConstInputsDataMap& inputsDesc) {
    IE_ASSERT(outputs.size() == 1 && refOutputs.size() == 1);
    const auto& outBlob = outputs.begin()->second;
    const auto& refOutBlob = refOutputs.begin()->second;
    IE_ASSERT(refOutBlob->getTensorDesc().getPrecision() == ie::Precision::FP32);
    IE_ASSERT(outBlob->getTensorDesc().getPrecision() == ie::Precision::FP32);
    IE_ASSERT(!inputsDesc.empty());

    const auto& inputDesc = inputsDesc.begin()->second->getTensorDesc();

    const auto imgWidth = inputDesc.getDims().at(3);
    const auto imgHeight = inputDesc.getDims().at(2);

    auto confThresh = FLAGS_confidence_threshold;
    auto probTolerance = FLAGS_prob_tolerance;
    auto boxTolerance = FLAGS_box_tolerance;

    auto actualOutput = utils::parseSSDOutput(ie::as<ie::MemoryBlob>(outBlob), imgWidth, imgHeight, confThresh);
    auto refOutput = utils::parseSSDOutput(ie::as<ie::MemoryBlob>(refOutBlob), imgWidth, imgHeight, confThresh);

    auto result = checkBBoxOutputs(actualOutput, refOutput, imgWidth, imgHeight, boxTolerance, probTolerance);

    return result;
}

//
// Classification mode
//

std::vector<std::pair<int, float>> parseClassification(const ie::MemoryBlob::Ptr& blob) {
    IE_ASSERT(blob->getTensorDesc().getPrecision() == ie::Precision::FP32);

    std::vector<std::pair<int, float>> res(blob->size());

    const auto blobMem = blob->rmap();
    const auto blobPtr = blobMem.as<const float*>();
    IE_ASSERT(blobPtr != nullptr);

    for (size_t i = 0; i < blob->size(); ++i) {
        res[i].first = static_cast<int>(i);
        res[i].second = blobPtr[i];
    }

    std::sort(res.begin(), res.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second > b.second;
    });

    return res;
}

bool testClassification(const ie::BlobMap& outputs, const ie::BlobMap& refOutputs) {
    IE_ASSERT(outputs.size() == 1 && refOutputs.size() == 1);

    const auto& outBlob = outputs.begin()->second;
    const auto& refOutBlob = refOutputs.begin()->second;

    IE_ASSERT(outBlob->getTensorDesc() == refOutBlob->getTensorDesc());
    IE_ASSERT(refOutBlob->getTensorDesc().getPrecision() == ie::Precision::FP32);

    auto probs = parseClassification(ie::as<ie::MemoryBlob>(outBlob));
    auto refProbs = parseClassification(ie::as<ie::MemoryBlob>(refOutBlob));

    IE_ASSERT(probs.size() >= FLAGS_top_k);
    probs.resize(FLAGS_top_k);

    IE_ASSERT(refProbs.size() >= FLAGS_top_k);
    refProbs.resize(FLAGS_top_k);

    std::cout << "Actual top:" << std::endl;
    for (size_t i = 0; i < probs.size(); ++i) {
        std::cout << "    " << i << " : " << probs[i].first << " : " << probs[i].second << std::endl;
    }

    std::cout << "Ref Top:" << std::endl;
    for (size_t i = 0; i < refProbs.size(); ++i) {
        std::cout << "    " << i << " : " << refProbs[i].first << " : " << refProbs[i].second << std::endl;
    }

    for (const auto& refElem : refProbs) {
        const auto actualIt = std::find_if(
            probs.cbegin(), probs.cend(),
            [&refElem](const std::pair<int, float>& arg) {
                return refElem.first == arg.first;
            });
        if (actualIt == probs.end()) {
            std::cout << "Ref result " << refElem.first << " was not found in actual results" << std::endl;
            return false;
        }

        const auto& actualElem = *actualIt;

        if (refElem.second > actualElem.second) {
            const auto probDiff = std::fabs(refElem.second - actualElem.second);
            if (probDiff > FLAGS_prob_tolerance) {
                std::cout << "Probability value mismatch for " << refElem.first << " : " << refElem.second << " vs " << actualElem.second;
                return false;
            }
        }
    }

    return true;
}

//
// RAW mode
//

bool compareBlobs(
        const ie::MemoryBlob::Ptr& actualOutput,
        const ie::MemoryBlob::Ptr& refOutput) {
    const auto& actualDesc = actualOutput->getTensorDesc();
    const auto& refDesc = refOutput->getTensorDesc();

    if (actualDesc.getDims() != refDesc.getDims()) {
        std::cout << "Actual and reference blobs has different shape" << std::endl;
        return false;
    }

    const auto actualFP32 = vpux::toFP32(vpux::toDefLayout(actualOutput));
    const auto refFP32 = vpux::toFP32(vpux::toDefLayout(refOutput));

    const auto actualMem = actualFP32->rmap();
    const auto refMem = refFP32->rmap();

    const auto actualPtr = actualMem.as<const float*>();
    const auto refPtr = refMem.as<const float*>();

    const auto totalCount = refOutput->size();
    const auto printCount = std::min<size_t>(totalCount, 10);

    for (size_t i = 0; i < totalCount; ++i) {
        const auto refVal = refPtr[i];
        const auto actualVal = actualPtr[i];

        const auto absDiff = std::fabs(refVal - actualVal);

        if (i < printCount) {
            std::cout << "        " << i << " :"
                      << " ref : " << std::setw(10) << refVal
                      << " actual : " << std::setw(10) << actualVal
                      << " absdiff : " << std::setw(10) << absDiff
                      << std::endl;
        }

        if (absDiff > FLAGS_raw_tolerance) {
            std::cout
                    << "Absolute difference between actual value " << actualVal
                    << " and reference value " << refVal
                    << " at index " << i
                    << " larger then tolerance" << std::endl;
            return false;
        }
    }

    return true;
}

bool testRAW(const ie::BlobMap& outputs, const ie::BlobMap& refOutputs) {
    if (outputs.size() != refOutputs.size()) {
        std::cout << "Actual and reference has different number of output blobs" << std::endl;
        return false;
    }

    for (const auto& actualBlob : outputs) {
        auto ref_it = refOutputs.find(actualBlob.first);
        IE_ASSERT(ref_it != refOutputs.end());

        std::cout << "Compare " << actualBlob.first << " with reference" << std::endl;
        if (!compareBlobs(ie::as<ie::MemoryBlob>(actualBlob.second), ie::as<ie::MemoryBlob>(ref_it->second))) {
            return false;
        }
    }

    return true;
}

//
// Yolo V2 mode
//
bool testYoloV2 (const ie::BlobMap& actualBlobs,
                 const ie::BlobMap& refBlobs,
                 const ie::ConstInputsDataMap& inputsDesc) {

    IE_ASSERT(inputsDesc.size() == 1);
    IE_ASSERT(actualBlobs.size() == 1u && actualBlobs.size() == refBlobs.size());
    auto actualBlob = actualBlobs.begin()->second;
    auto refBlob    = refBlobs.begin()->second;

    const auto& inputDesc = inputsDesc.begin()->second->getTensorDesc();

    const auto imgWidth = inputDesc.getDims().at(3);
    const auto imgHeight = inputDesc.getDims().at(2);
    double confThresh = FLAGS_confidence_threshold;
    double probTolerance = FLAGS_prob_tolerance;
    double boxTolerance = FLAGS_box_tolerance;
    bool isTiny = FLAGS_is_tiny_yolo;

    auto actualOutput = utils::parseYoloOutput(vpux::toFP32(InferenceEngine::as<InferenceEngine::MemoryBlob>(actualBlob)),
                                               imgWidth, imgHeight, confThresh, isTiny);
    auto refOutput = utils::parseYoloOutput(vpux::toFP32(InferenceEngine::as<InferenceEngine::MemoryBlob>(refBlob)),
                                            imgWidth, imgHeight, confThresh, isTiny);

    bool result = checkBBoxOutputs(actualOutput, refOutput, imgWidth, imgHeight, boxTolerance, probTolerance);
    return result;
}

//
// main
//

int main(int argc, char* argv[]) {
    try {
        parseCommandLine(argc, argv);

        const std::unordered_set<std::string> allowedPrecision = {"U8", "FP16", "FP32"};
        if (!FLAGS_ip.empty()) {
            // input precision is U8, FP16 or FP32 only
            std::transform(FLAGS_ip.begin(),FLAGS_ip.end(),FLAGS_ip.begin(), ::toupper);
            if (allowedPrecision.count(FLAGS_ip) == 0)
                throw std::logic_error("Parameter -ip " + FLAGS_ip + " is not supported");
        }
        if (!FLAGS_op.empty()) {
            // input precision is U8, FP16 or FP32 only
            std::transform(FLAGS_op.begin(),FLAGS_op.end(),FLAGS_op.begin(), ::toupper);
            if (allowedPrecision.count(FLAGS_op) == 0)
                throw std::logic_error("Parameter -op " + FLAGS_op + " is not supported");
        }

        std::vector<std::string> inputFilesPerCase;
        std::vector<std::vector<std::string>> inputFilesForOneInfer;
        inputFilesPerCase = splitStringList(FLAGS_input, ';');
        for (const auto& images : inputFilesPerCase) {
            inputFilesForOneInfer.push_back(splitStringList(images, ','));
        }

        if (FLAGS_network.empty()) {
            std::cout << "Not enough parameters. Check help." << std::endl;
            return EXIT_FAILURE;
        }

        setupInferenceEngine();

        ie::ExecutableNetwork exeNet;

        if (strEq(FileUtils::fileExt(FLAGS_network), "xml")) {
            std::cout << "Load network " << FLAGS_network << std::endl;

            auto cnnNet = ieCore.ReadNetwork(FLAGS_network);

            // Input precision
            ie::InputsDataMap inputInfo(cnnNet.getInputsInfo());
            if (!FLAGS_ip.empty()) {
                ie::Precision prc_in = ie::Precision::U8;
                if (FLAGS_ip == "FP16")
                    prc_in = ie::Precision::FP16;
                else if (FLAGS_ip == "FP32")
                    prc_in = ie::Precision::FP32;
                else
                    prc_in = ie::Precision::U8;

                for (auto inputInfoIt=inputInfo.begin(); inputInfoIt!=inputInfo.end(); ++inputInfoIt){
                    inputInfoIt->second->setPrecision(prc_in);
                }
            }
            // Input layout
            if (!FLAGS_il.empty()) {
                const ie::Layout layout = FLAGS_il == "NCHW" ? ie::Layout::NCHW : ie::Layout::NHWC;
                for (auto & info: inputInfo) info.second->setLayout(layout);
            }
            // Output precision
            ie::OutputsDataMap outputInfo(cnnNet.getOutputsInfo());
            if (!FLAGS_op.empty()) {
                ie::Precision prc_out = ie::Precision::U8;
                if (FLAGS_op == "FP16")
                    prc_out = ie::Precision::FP16;
                else if (FLAGS_op == "FP32")
                    prc_out = ie::Precision::FP32;
                else
                    prc_out = ie::Precision::U8;

                // possibly multiple outputs
                for (auto outputInfoIt=outputInfo.begin(); outputInfoIt!=outputInfo.end(); ++outputInfoIt){
                    outputInfoIt->second->setPrecision(prc_out);
                }
            }
            // Output layout
            if (!FLAGS_ol.empty()) {
                for (auto outputInfoIt=outputInfo.begin(); outputInfoIt!=outputInfo.end(); ++outputInfoIt){
                    if (outputInfoIt->second->getDims().size() == 2) {
                            outputInfoIt->second->setLayout(ie::Layout::NC);
                    } else {
                        if (FLAGS_ol == "NCHW"){
                            outputInfoIt->second->setLayout(ie::Layout::NCHW);
                        }
                        else{
                            outputInfoIt->second->setLayout(ie::Layout::NHWC);
                        }
                    }
                }
            }

            exeNet = ieCore.LoadNetwork(cnnNet, FLAGS_device);
        } else {
            std::cout << "Import network " << FLAGS_network << std::endl;

            exeNet = importNetwork(FLAGS_network);
        }

        std::string netFileName;
        {
            auto startPos = FLAGS_network.rfind('/');
            if (startPos == std::string::npos) {
                startPos = FLAGS_network.rfind('\\');
                if (startPos == std::string::npos) {
                    startPos = 0;
                }
            }

            auto endPos = FLAGS_network.rfind('.');
            if (endPos == std::string::npos) {
                endPos = FLAGS_network.size();
            }

            IE_ASSERT(endPos > startPos);
            netFileName = cleanName(FLAGS_network.substr(startPos, endPos - startPos));
        }

        for (size_t numberOfTestCase = 0; numberOfTestCase < inputFilesPerCase.size(); ++numberOfTestCase) {
            const auto inputsInfo = exeNet.GetInputsInfo();
            const auto outputsInfo = exeNet.GetOutputsInfo();
            std::vector<std::string> inputFiles = inputFilesForOneInfer[numberOfTestCase];
            IE_ASSERT(inputFiles.size() == inputsInfo.size())
                        << "Number of input files " << inputFiles.size()
                        << " doesn't match network configuration " << inputsInfo.size();

            ie::BlobMap inputs;
            size_t inputInd = 0;
            for (const auto &p : inputsInfo) {
                std::cout << "Load input #" << inputInd << " from " << inputFiles[inputInd] << std::endl;
                const auto blob = loadInput(p.second->getTensorDesc(), inputFiles[inputInd], FLAGS_color_format);
                inputs.emplace(p.first, blob);

                std::ostringstream ostr;
                ostr << netFileName << "_input_" << inputInd << "_case_" << numberOfTestCase << ".blob";
                const auto blobFileName = ostr.str();

                std::cout << "Dump input #" << inputInd << "_case_" << numberOfTestCase << " to " << blobFileName << std::endl;
                dumpBlob(blob, blobFileName);

                ++inputInd;
            }

            std::cout << "Run inference on " << FLAGS_device << std::endl;
            const auto outputs = runInfer(exeNet, inputs);

            if (FLAGS_run_test) {
                ie::BlobMap refOutputs;
                size_t outputInd = 0;
                for (const auto &p : outputsInfo) {
                    std::ostringstream ostr;
                    ostr << netFileName << "_ref_out_" << outputInd << "_case_" << numberOfTestCase << ".blob";
                    const auto blobFileName = ostr.str();

                    std::cout << "Load reference output #" << outputInd << " from " << blobFileName << std::endl;
                    const auto blob = loadBlob(p.second->getTensorDesc(), blobFileName);
                    refOutputs.emplace(p.first, blob);

                    ++outputInd;
                }

                outputInd = 0;
                for (const auto &p : outputs) {
                    std::ostringstream ostr;
                    ostr << netFileName << "_kmb_out_" << outputInd << "_case_" << numberOfTestCase << ".blob";
                    const auto blobFileName = ostr.str();

                    std::cout << "Dump device output #" << outputInd << "_case_" << numberOfTestCase << " to " << blobFileName << std::endl;
                    dumpBlob(ie::as<ie::MemoryBlob>(p.second), blobFileName);

                    ++outputInd;
                }

                if (strEq(FLAGS_mode, "classification")) {
                    if (testClassification(outputs, refOutputs)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "raw")) {
                    if (testRAW(outputs, refOutputs)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "ssd")) {
                    if (testSSDDetection(outputs, refOutputs, inputsInfo)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "yolo_v2")) {
                    if (testYoloV2(outputs, refOutputs, inputsInfo)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else {
                    std::cout << "Unknown mode " << FLAGS_mode << std::endl;
                    return EXIT_FAILURE;
                }
            } else {
                size_t outputInd = 0;
                for (const auto &p : outputs) {
                    std::ostringstream ostr;
                    ostr << netFileName << "_ref_out_" << outputInd << "_case_" << numberOfTestCase << ".blob";
                    const auto blobFileName = ostr.str();

                    std::cout << "Dump reference output #" << outputInd << " to " << blobFileName << std::endl;
                    dumpBlob(ie::as<ie::MemoryBlob>(p.second), blobFileName);

                    ++outputInd;
                }
            }
        }
    } // try
    catch (const std::exception &ex) {
        std::cerr << "exception: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
