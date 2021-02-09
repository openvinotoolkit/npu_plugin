//
// Copyright 2020 Intel Corporation.
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

DEFINE_bool(run_test, false, "Run the test (compare current results with previously dumped)");
DEFINE_string(mode, "", "Comparison mode to use");

DEFINE_uint32(top_k, 1, "Top K parameter for 'classification' mode");
DEFINE_double(prob_tolerance, 1e-4, "Probability tolerance for 'classification' mode");

DEFINE_string(log_level, "", "IE logger level (optional)");

std::vector<std::string> inputFiles;

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
    std::cout << "    Device:           " << FLAGS_device << std::endl;
    std::cout << "    Run test:         " << FLAGS_run_test << std::endl;
    if (FLAGS_run_test) {
        std::cout << "    Mode:             " << FLAGS_mode << std::endl;
        if (strEq(FLAGS_mode, "classification")) {
            std::cout << "    Top K:            " << FLAGS_top_k << std::endl;
            std::cout << "    Tolerance:        " << FLAGS_prob_tolerance << std::endl;
        }
    }
    std::cout << "    Log level:        " << FLAGS_log_level << std::endl;
    std::cout << std::endl;

    inputFiles = splitStringList(FLAGS_input, ',');
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

std::vector<cv::Mat> ieToCv(const ie::Blob::Ptr& blob, size_t batchInd = 0, size_t depthInd = 0) {
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

        const auto blobPtr = blob->buffer().as<uint8_t *>();

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

        const auto blobPtr = blob->buffer().as<uint8_t *>();

        out.resize(C);
        for (size_t c = 0; c < C; ++c) {
            out[c] = cv::Mat(H, W, cvType, blobPtr + (strideN * batchInd + strideC * c + strideD * depthInd) * elemSize);
        }
    }

    return out;
}

void cvToIe(const cv::Mat& cvImg, const ie::Blob::Ptr& ieBlob) {
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
        in = cvImg;
    } else {
        cv::cvtColor(cvImg, in, cv::COLOR_BGR2BGRA);
    }

    if (precision != ie::Precision::U8) {
        in.convertTo(in, CV_32F);
    }

    const auto pictureArea = in.size().area();

    if (W * H > pictureArea) {
        cv::resize(in, in, cv::Size(W, H), 0.0, 0.0, cv::INTER_AREA);
    } else {
        cv::resize(in, in, cv::Size(W, H), 0.0, 0.0, cv::INTER_LINEAR);
    }

    if (layout == ie::Layout::NHWC) {
        cv::Mat out(H, W, cvType, ieBlob->buffer().as<uint8_t *>());

        if (precision != ie::Precision::FP16) {
            in.copyTo(out);
        } else {
            const auto inPtr = in.ptr<float>();
            const auto outPtr = out.ptr<ie::ie_fp16>();
            ie::PrecisionUtils::f32tof16Arrays(outPtr, inPtr, out.size().area());
        }

        for (size_t n = 1; n < N; ++n) {
            cv::Mat batch(H, W, cvType, ieBlob->buffer().as<uint8_t *>() + out.size().area() * out.elemSize());
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

ie::Blob::Ptr loadInput(const ie::TensorDesc& desc, const std::string& filePath) {
    IE_ASSERT(isImage(desc)) << "Only image inputs are supported";

    const auto frame = cv::imread(filePath, cv::IMREAD_COLOR);
    IE_ASSERT(!frame.empty()) << "Failed to open input image file " << filePath;

    const auto blob = make_blob_with_precision(desc);
    blob->allocate();

    cvToIe(frame, blob);

    return blob;
}

ie::Blob::Ptr loadBlob(const ie::TensorDesc& desc, const std::string& filePath) {
    const auto blob = make_blob_with_precision(desc);
    blob->allocate();

    std::ifstream file(filePath, std::ios_base::in | std::ios_base::binary);
    IE_ASSERT(file.is_open()) << "Can't open file " << filePath << " for read";

    file.read(blob->buffer().as<char*>(), static_cast<std::streamsize>(blob->byteSize()));

    return blob;
}

void dumpBlob(const ie::Blob::Ptr& blob, const std::string& filePath) {
    std::ofstream file(filePath, std::ios_base::out | std::ios_base::binary);
    IE_ASSERT(file.is_open()) << "Can't open file " << filePath << " for write";

    file.write(blob->cbuffer().as<const char*>(), static_cast<std::streamsize>(blob->byteSize()));
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
// Main
//

std::vector<std::pair<int, float>> parseClassification(const ie::Blob::Ptr& blob) {
    IE_ASSERT(blob->getTensorDesc().getPrecision() == ie::Precision::FP32);

    std::vector<std::pair<int, float>> res(blob->size());

    const auto blobPtr = blob->cbuffer().as<const float*>();
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

    auto probs = parseClassification(outBlob);
    auto refProbs = parseClassification(refOutBlob);

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

int main(int argc, char* argv[]) {
    try {
        parseCommandLine(argc, argv);

        if (FLAGS_network.empty()) {
            std::cout << "Not enough parameters. Check help." << std::endl;
            return EXIT_FAILURE;
        }

        setupInferenceEngine();

        ie::ExecutableNetwork exeNet;

        if (strEq(FileUtils::fileExt(FLAGS_network), "xml")) {
            std::cout << "Load network " << FLAGS_network << std::endl;

            auto cnnNet = ieCore.ReadNetwork(FLAGS_network);
            exeNet = ieCore.LoadNetwork(cnnNet, FLAGS_device);
        } else {
            std::cout << "Import network " << FLAGS_network << std::endl;

            exeNet = importNetwork(FLAGS_network);
        }

        const auto inputsInfo = exeNet.GetInputsInfo();
        const auto outputsInfo = exeNet.GetOutputsInfo();

        IE_ASSERT(inputFiles.size() == inputsInfo.size())
            << "Number of input files " << inputFiles.size()
            << " doesn't match network configuration " << inputsInfo.size();

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

        ie::BlobMap inputs;
        size_t inputInd = 0;
        for (const auto& p : inputsInfo) {
            std::cout << "Load input #" << inputInd << " from " << inputFiles[inputInd] << std::endl;
            const auto blob = loadInput(p.second->getTensorDesc(), inputFiles[inputInd]);
            inputs.emplace(p.first, blob);

            std::ostringstream ostr;
            ostr << netFileName << "_input_" << inputInd << ".blob";
            const auto blobFileName = ostr.str();

            std::cout << "Dump input #" << inputInd << " to " << blobFileName << std::endl;
            dumpBlob(blob, blobFileName);

            ++inputInd;
        }

        std::cout << "Run inference on " << FLAGS_device << std::endl;
        const auto outputs = runInfer(exeNet, inputs);

        if (FLAGS_run_test) {
            ie::BlobMap refOutputs;
            size_t outputInd = 0;
            for (const auto& p : outputsInfo) {
                std::ostringstream ostr;
                ostr << netFileName << "_ref_out_" << outputInd << ".blob";
                const auto blobFileName = ostr.str();

                std::cout << "Load reference output #" << outputInd << " from " << blobFileName << std::endl;
                const auto blob = loadBlob(p.second->getTensorDesc(), blobFileName);
                refOutputs.emplace(p.first, blob);

                ++outputInd;
            }

            outputInd = 0;
            for (const auto& p : outputs) {
                std::ostringstream ostr;
                ostr << netFileName << "_kmb_out_" << outputInd << ".blob";
                const auto blobFileName = ostr.str();

                std::cout << "Dump device output #" << outputInd << " to " << blobFileName << std::endl;
                dumpBlob(p.second, blobFileName);

                ++outputInd;
            }

            if (strEq(FLAGS_mode, "classification")) {
                if (testClassification(outputs, refOutputs)) {
                    std::cout << "PASSED" << std::endl;
                } else {
                    std::cout << "FAILED" << std::endl;
                }
            } else {
                std::cout << "Unsupported mode " << FLAGS_mode << std::endl;
            }
        } else {
            size_t outputInd = 0;
            for (const auto& p : outputs) {
                std::ostringstream ostr;
                ostr << netFileName << "_ref_out_" << outputInd << ".blob";
                const auto blobFileName = ostr.str();

                std::cout << "Dump reference output #" << outputInd << " to " << blobFileName << std::endl;
                dumpBlob(p.second, blobFileName);

                ++outputInd;
            }
        }

    } // try
    catch (const std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
