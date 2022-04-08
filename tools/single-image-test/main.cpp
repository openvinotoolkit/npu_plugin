//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "image_quality_helper.hpp"
#include "semantic_segmentation_helpers.hpp"
#include "vpux/utils/IE/blob.hpp"
#include "yolo_helpers.hpp"

#include <file_utils.h>
#include <precision_utils.h>
#include <blob_factory.hpp>
#include <caseless.hpp>
#include <inference_engine.hpp>
#include <openvino/openvino.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <gflags/gflags.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace ie = InferenceEngine;

ie::details::CaselessEq<std::string> strEq;

//
// Command line options
//

DEFINE_bool(ov_api_1_0, false, "Optional. Use legacy Inference Engine API (default: false)");
DEFINE_string(network, "", "Network file (either XML or pre-compiled blob)");
DEFINE_string(input, "", "Input file(s)");
DEFINE_string(device, "", "Device to use");
DEFINE_string(config, "", "Path to the configuration file (optional)");
DEFINE_string(ip, "", "Input precision (default: U8)");
DEFINE_string(op, "", "Output precision (default: FP32)");
DEFINE_string(il, "", "Input layout");
DEFINE_string(ol, "", "Output layout");
DEFINE_string(iml, "", "Model input layout");
DEFINE_string(oml, "", "Model output layout");
DEFINE_bool(img_as_bin, false, "Force binary input even if network expects an image");
DEFINE_bool(pc, false, "Report performance counters");
DEFINE_string(img_bin_precision, "", "Specify the precision of the binary input files. Eg: 'FP32,FP16;FP32,U8'");

DEFINE_bool(run_test, false, "Run the test (compare current results with previously dumped)");
DEFINE_string(mode, "", "Comparison mode to use");

DEFINE_uint32(top_k, 1, "Top K parameter for 'classification' mode");
DEFINE_double(prob_tolerance, 1e-4, "Probability tolerance for 'classification/ssd/yolo' mode");

DEFINE_double(raw_tolerance, 1e-4, "Tolerance for 'raw' mode (absolute diff)");
DEFINE_double(cosim_threshold, 0.90, "Threshold for 'cosim' mode");
DEFINE_double(confidence_threshold, 1e-4, "Confidence threshold for Detection mode");
DEFINE_double(box_tolerance, 1e-4, "Box tolerance for 'detection' mode");

DEFINE_double(psnr_reference, 30.0, "PSNR reference value in dB");
DEFINE_double(psnr_tolerance, 1e-4, "Tolerance for 'psnr' mode");

DEFINE_string(log_level, "", "IE logger level (optional)");
DEFINE_string(color_format, "BGR", "Color format for input: RGB or BGR");
DEFINE_uint32(scale_border, 4, "Scale border");
DEFINE_bool(normalized_image, false, "Images in [0, 1] range or not");

// for Yolo
DEFINE_bool(is_tiny_yolo, false, "Is it Tiny Yolo or not (true or false)?");
DEFINE_int32(classes, 80, "Number of classes for Yolo V3");
DEFINE_int32(coords, 4, "Number of coordinates for Yolo V3");
DEFINE_int32(num, 3, "Number of scales for Yolo V3");

typedef std::chrono::high_resolution_clock Time;
// for Semantic Segmentation
DEFINE_uint32(sem_seg_classes, 12, "Number of classes for semantic segmentation");
DEFINE_double(sem_seg_threshold, 0.98, "Threshold for 'semantic segmentation' mode");
DEFINE_uint32(sem_seg_ignore_label, std::numeric_limits<uint32_t>::max(), "The number of the label to be ignored");
DEFINE_string(dataset, "NONE",
              "The dataset used to train the model. Useful for instances such as semantic segmentation to visualize "
              "the accuracy per-class");
std::vector<std::string> camVid12 = {"Sky",        "Building", "Pole", "Road",       "Pavement",  "Tree",
                                     "SignSymbol", "Fence",    "Car",  "Pedestrian", "Bicyclist", "Unlabeled"};

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
    std::cout << "    Network file:             " << FLAGS_network << std::endl;
    std::cout << "    Input file(s):            " << FLAGS_input << std::endl;
    std::cout << "    Color format:             " << FLAGS_color_format << std::endl;
    std::cout << "    Input precision:          " << FLAGS_ip << std::endl;
    std::cout << "    Output precision:         " << FLAGS_op << std::endl;
    std::cout << "    Input layout:             " << FLAGS_il << std::endl;
    std::cout << "    Output layout:            " << FLAGS_ol << std::endl;
    std::cout << "    Model input layout:       " << FLAGS_iml << std::endl;
    std::cout << "    Model output layout:      " << FLAGS_oml << std::endl;
    std::cout << "    Img as binary:            " << FLAGS_img_as_bin << std::endl;
    std::cout << "    Bin input file precision: " << FLAGS_img_bin_precision << std::endl;
    std::cout << "    Device:                   " << FLAGS_device << std::endl;
    std::cout << "    Config file:              " << FLAGS_config << std::endl;
    std::cout << "    Run test:                 " << FLAGS_run_test << std::endl;
    std::cout << "    Performance counters:     " << FLAGS_pc << std::endl;
    if (FLAGS_run_test) {
        std::cout << "    Mode:             " << FLAGS_mode << std::endl;
        if (strEq(FLAGS_mode, "classification")) {
            std::cout << "    Top K:            " << FLAGS_top_k << std::endl;
            std::cout << "    Tolerance:        " << FLAGS_prob_tolerance << std::endl;
        } else if (strEq(FLAGS_mode, "raw")) {
            std::cout << "    Tolerance:        " << FLAGS_raw_tolerance << std::endl;
        } else if (strEq(FLAGS_mode, "cosim")) {
            std::cout << "    Threshold:        " << FLAGS_cosim_threshold << std::endl;
        }
        if (strEq(FLAGS_mode, "psnr")) {
            std::cout << "    Reference:        " << FLAGS_psnr_reference << std::endl;
            std::cout << "    Tolerance:        " << FLAGS_psnr_tolerance << std::endl;
            std::cout << "    Scale_border:     " << FLAGS_scale_border << std::endl;
            std::cout << "    Normalized_image: " << FLAGS_normalized_image << std::endl;
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

    IE_ASSERT(precision == ie::Precision::U8 || precision == ie::Precision::FP32 || precision == ie::Precision::FP16);

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
        const auto blobPtr = blobMem.as<uint8_t*>();

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
        const auto blobPtr = blobMem.as<uint8_t*>();

        out.resize(C);
        for (size_t c = 0; c < C; ++c) {
            out[c] =
                    cv::Mat(H, W, cvType, blobPtr + (strideN * batchInd + strideC * c + strideD * depthInd) * elemSize);
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
        IE_ASSERT(precision == ie::Precision::U8 || precision == ie::Precision::FP32 ||
                  precision == ie::Precision::FP16)
                << "Unsupported precision " << precision;
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
        const auto blobPtr = blobMem.as<uint8_t*>();

        cv::Mat out(H, W, cvType, blobPtr);

        if (precision != ie::Precision::FP16) {
            in.copyTo(out);
        } else {
            const auto inPtr = in.ptr<float>();
            const auto outPtr = out.ptr<ie::ie_fp16>();
            ie::PrecisionUtils::f32tof16Arrays(outPtr, inPtr, out.size().area() * C);
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

ie::MemoryBlob::Ptr loadBinary(const ie::TensorDesc& desc, const std::string& filePath,
                               const ie::Precision& inputPrecision) {
    const auto blob = ie::as<ie::MemoryBlob>(make_blob_with_precision(desc));
    IE_ASSERT(blob != nullptr) << "Can't create MemoryBlob for binary file";

    std::ifstream binaryFile(filePath, std::ios_base::binary | std::ios_base::ate);

    IE_ASSERT(binaryFile) << "Failed to open input binary file";

    const int fileSize = binaryFile.tellg();
    binaryFile.seekg(0, std::ios_base::beg);
    const int expectedSize = static_cast<int>(blob->byteSize());
    IE_ASSERT(binaryFile.good()) << "While reading a file an error is encountered";
    blob->allocate();
    const auto blobMem = blob->wmap();

    const auto inTensorPrec = desc.getPrecision();
    bool implicitPrecision = false;
    switch (inputPrecision) {
    case ie::Precision::FP32: {
        switch (inTensorPrec) {
        case ie::Precision::FP32:
            implicitPrecision = true;
            break;
        case ie::Precision::FP16: {
            std::cout << "Converting " << filePath << " input from FP32 to FP16\n";
            IE_ASSERT(fileSize == expectedSize * 2)
                    << "File contains " << fileSize << " bytes, but " << expectedSize * 2 << " expected";

            std::vector<uint8_t> input(fileSize);
            binaryFile.read(reinterpret_cast<char*>(input.data()), static_cast<std::streamsize>(fileSize));
            const auto blobPtr = blobMem.as<ie::ie_fp16*>();
            ie::PrecisionUtils::f32tof16Arrays(blobPtr, reinterpret_cast<float*>(input.data()), blob->size());
            break;
        }
        case ie::Precision::U8: {
            std::cout << "Converting " << filePath << " input from FP32 to U8\n";
            std::cout << "WARNING: FP32->U8 conversion is currently just a type-cast\n";
            IE_ASSERT(fileSize == expectedSize * 4)
                    << "File contains " << fileSize << " bytes, but " << expectedSize * 4 << " expected";

            std::vector<uint8_t> input(fileSize);
            binaryFile.read(reinterpret_cast<char*>(input.data()), static_cast<std::streamsize>(fileSize));
            const auto blobPtr = blobMem.as<uint8_t*>();
            const float* srcPtr = reinterpret_cast<float*>(input.data());

            // naive conversion, to be improved
            for (std::size_t i = 0; i < expectedSize; ++i) {
                blobPtr[i] = static_cast<uint8_t>(srcPtr[i]);
            }
            break;
        }
        default:
            implicitPrecision = true;
            break;
        }
        break;
    }
    case ie::Precision::FP16: {
        switch (inTensorPrec) {
        case ie::Precision::FP32: {
            std::cout << "Converting " << filePath << " input from FP16 to FP32\n";
            IE_ASSERT(fileSize * 2 == expectedSize)
                    << "File contains " << fileSize * 2 << " bytes, but " << expectedSize << " expected";

            std::vector<uint8_t> input(fileSize);
            binaryFile.read(reinterpret_cast<char*>(input.data()), static_cast<std::streamsize>(fileSize));
            const auto blobPtr = blobMem.as<float*>();
            ie::PrecisionUtils::f16tof32Arrays(blobPtr, reinterpret_cast<ie::ie_fp16*>(input.data()), blob->size());
            break;
        }
        case ie::Precision::FP16:
            implicitPrecision = true;
            break;
        case ie::Precision::U8: {
            std::cout << "Converting " << filePath << " input from FP16 to U8\n";
            std::cout << "WARNING: FP16->U8 conversion is currently just a type-cast\n";
            IE_ASSERT(fileSize == expectedSize * 2)
                    << "File contains " << fileSize << " bytes, but " << expectedSize * 2 << " expected";

            std::vector<uint8_t> input(fileSize);
            std::vector<uint8_t> intermediaryFP32(fileSize * 2);
            binaryFile.read(reinterpret_cast<char*>(input.data()), static_cast<std::streamsize>(fileSize));
            const auto blobPtr = blobMem.as<uint8_t*>();
            ie::PrecisionUtils::f16tof32Arrays(reinterpret_cast<float*>(intermediaryFP32.data()),
                                               reinterpret_cast<ie::ie_fp16*>(input.data()), blob->size());
            const float* srcPtr = reinterpret_cast<float*>(intermediaryFP32.data());

            // naive conversion, to be improved
            for (std::size_t i = 0; i < expectedSize; ++i) {
                blobPtr[i] = static_cast<uint8_t>(srcPtr[i]);
            }
            break;
        }
        default:
            implicitPrecision = true;
            break;
        }
        break;
    }
    case ie::Precision::U8: {
        switch (inTensorPrec) {
        case ie::Precision::FP32: {
            std::cout << "Converting " << filePath << " input from U8 to FP32\n";
            std::cout << "WARNING: U8->FP32 conversion is currently just a type-cast\n";
            IE_ASSERT(fileSize * 4 == expectedSize)
                    << "File contains " << fileSize * 4 << " bytes, but " << expectedSize << " expected";

            std::vector<uint8_t> input(fileSize);
            binaryFile.read(reinterpret_cast<char*>(input.data()), static_cast<std::streamsize>(fileSize));
            const auto blobPtr = blobMem.as<float*>();

            // naive conversion, to be improved
            for (std::size_t i = 0; i < fileSize; ++i) {
                blobPtr[i] = static_cast<float>(input[i]);
            }
            break;
        }
        case ie::Precision::FP16: {
            std::cout << "Converting " << filePath << " input from U8 to FP16\n";
            std::cout << "WARNING: U8->FP16 conversion is currently just a type-cast\n";
            IE_ASSERT(fileSize * 2 == expectedSize)
                    << "File contains " << fileSize * 2 << " bytes, but " << expectedSize << " expected";

            std::vector<uint8_t> input(fileSize);
            std::vector<float> intermediaryFP32(fileSize);
            binaryFile.read(reinterpret_cast<char*>(input.data()), static_cast<std::streamsize>(fileSize));
            const auto blobPtr = blobMem.as<ie::ie_fp16*>();

            // naive conversion, to be improved
            for (std::size_t i = 0; i < fileSize; ++i) {
                intermediaryFP32[i] = static_cast<float>(input[i]);
            }
            ie::PrecisionUtils::f32tof16Arrays(blobPtr, intermediaryFP32.data(), blob->size());

            break;
        }
        case ie::Precision::U8:
            implicitPrecision = true;
            break;
        }
        break;
    }
    default:
        implicitPrecision = true;
        break;
    }

    if (implicitPrecision == true) {
        IE_ASSERT(fileSize == expectedSize)
                << "File contains " << fileSize << " bytes, but " << expectedSize << " expected";

        const auto blobPtr = blobMem.as<char*>();
        binaryFile.read(blobPtr, static_cast<std::streamsize>(expectedSize));
    }

    return blob;
}

ie::MemoryBlob::Ptr loadInput(const ie::TensorDesc& desc, const std::string& filePath, const std::string& colorFormat,
                              const ie::Precision& inputPrecision = ie::Precision::UNSPECIFIED) {
    if (isImage(desc) && !FLAGS_img_as_bin) {
        return loadImage(desc, filePath, colorFormat);
    } else {
        return loadBinary(desc, filePath, inputPrecision);
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

std::weak_ptr<ie::Core> ieCore;

void setupInferenceEngine() {
    auto flagDevice = FLAGS_device;
    auto ieCoreShared = ieCore.lock();
    IE_ASSERT(!ieCore.expired()) << "ieCore object expired";

    if (!FLAGS_log_level.empty()) {
        ieCoreShared->SetConfig({{CONFIG_KEY(LOG_LEVEL), FLAGS_log_level}}, flagDevice);
    }

    if (FLAGS_device == "CPU") {
        ieCoreShared->SetConfig({{"LP_TRANSFORMS_MODE", CONFIG_VALUE(NO)}}, flagDevice);
    }

    if (FLAGS_pc) {
        ieCoreShared->SetConfig({{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)}}, flagDevice);
    }

    if (!FLAGS_config.empty()) {
        std::ifstream file(FLAGS_config);
        IE_ASSERT(file.is_open()) << "Can't open file " << FLAGS_config << " for read";

        std::string key, value;
        while (file >> key >> value) {
            if (key.empty() || key[0] == '#') {
                continue;
            }

            ieCoreShared->SetConfig({{key, value}}, flagDevice);
        }
    }
}

ie::ExecutableNetwork importNetwork(const std::string& filePath) {
    std::ifstream file(filePath, std::ios_base::in | std::ios_base::binary);
    IE_ASSERT(file.is_open()) << "Can't open file " << filePath << " for read";
    if (auto ieCoreShared = ieCore.lock()) {
        return ieCoreShared->ImportNetwork(file, FLAGS_device);
    }
    THROW_IE_EXCEPTION << "ieCore object expired";
}

// This function formats performance counters in a same way as benchmark_app -pc does.
// It is a copy-paste from $OPENVINO_HOME/samples/cpp/common/utils/include/samples/common.hpp
using ProfVec = std::vector<ov::ProfilingInfo>;
static void printPerformanceCounts(ProfVec performanceData, std::ostream& stream, std::string deviceName,
                                   bool bshowHeader = true) {
    std::chrono::microseconds totalTime = std::chrono::microseconds::zero();
    // Print performance counts
    if (bshowHeader) {
        stream << std::endl << "performance counts:" << std::endl << std::endl;
    }
    std::ios::fmtflags fmt(std::cout.flags());
    for (const auto& it : performanceData) {
        std::string toPrint(it.node_name);
        const int maxLayerName = 30;

        if (it.node_name.length() >= maxLayerName) {
            toPrint = it.node_name.substr(0, maxLayerName - 4);
            toPrint += "...";
        }

        stream << std::setw(maxLayerName) << std::left << toPrint;
        switch (it.status) {
        case ov::ProfilingInfo::Status::EXECUTED:
            stream << std::setw(15) << std::left << "EXECUTED";
            break;
        case ov::ProfilingInfo::Status::NOT_RUN:
            stream << std::setw(15) << std::left << "NOT_RUN";
            break;
        case ov::ProfilingInfo::Status::OPTIMIZED_OUT:
            stream << std::setw(15) << std::left << "OPTIMIZED_OUT";
            break;
        }
        stream << std::setw(30) << std::left << "layerType: " + std::string(it.node_type) + " ";
        stream << std::setw(20) << std::left << "realTime: " + std::to_string(it.real_time.count());
        stream << std::setw(20) << std::left << "cpu: " + std::to_string(it.cpu_time.count());
        stream << " execType: " << it.exec_type << std::endl;
        if (it.real_time.count() > 0) {
            totalTime += it.real_time;
        }
    }
    stream << std::setw(20) << std::left << "Total time: " + std::to_string(totalTime.count()) << " microseconds"
           << std::endl;
    std::cout << std::endl;
    std::cout << "Full device name: " << deviceName << std::endl;
    std::cout << std::endl;
    std::cout.flags(fmt);
}

// This is a copy-paste of openvino/src/inference/src/cpp/ie_infer_request.cpp get_profiling_info func
static ProfVec get_profiling_info(InferenceEngine::InferRequest& req) {
    auto ieInfos = req.GetPerformanceCounts();
    ProfVec infos;
    infos.reserve(ieInfos.size());
    while (!ieInfos.empty()) {
        auto itIeInfo = std::min_element(
                std::begin(ieInfos), std::end(ieInfos),
                [](const decltype(ieInfos)::value_type& lhs, const decltype(ieInfos)::value_type& rhs) {
                    return lhs.second.execution_index < rhs.second.execution_index;
                });
        IE_ASSERT(itIeInfo != ieInfos.end());
        auto& ieInfo = itIeInfo->second;
        infos.push_back(ov::ProfilingInfo{});
        auto& info = infos.back();
        switch (ieInfo.status) {
        case ie::InferenceEngineProfileInfo::NOT_RUN:
            info.status = ov::ProfilingInfo::Status::NOT_RUN;
            break;
        case ie::InferenceEngineProfileInfo::OPTIMIZED_OUT:
            info.status = ov::ProfilingInfo::Status::OPTIMIZED_OUT;
            break;
        case ie::InferenceEngineProfileInfo::EXECUTED:
            info.status = ov::ProfilingInfo::Status::EXECUTED;
            break;
        }
        info.real_time = std::chrono::microseconds{ieInfo.realTime_uSec};
        info.cpu_time = std::chrono::microseconds{ieInfo.cpu_uSec};
        info.node_name = itIeInfo->first;
        info.exec_type = std::string{ieInfo.exec_type};
        info.node_type = std::string{ieInfo.layer_type};
        ieInfos.erase(itIeInfo);
    }
    return infos;
}

std::pair<ie::BlobMap, ProfVec> runInfer(ie::ExecutableNetwork& exeNet, const ie::BlobMap& inputs,
                                         const std::vector<std::string>& dumpedInputsPaths) {
    auto inferRequest = exeNet.CreateInferRequest();

    for (const auto& p : inputs) {
        inferRequest.SetBlob(p.first, p.second);
    }

    inferRequest.Infer();

    ie::BlobMap out;

    for (const auto& p : exeNet.GetOutputsInfo()) {
        out.insert({p.first, inferRequest.GetBlob(p.first)});
    }

    ProfVec profData{};

    if (FLAGS_pc) {
        profData = get_profiling_info(inferRequest);
    }

    return std::make_pair(out, profData);
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
            boxes_result.emplace_back(
                    static_cast<int>(net_out[i * oneDetectionSize + 1]), net_out[i * oneDetectionSize + 3] * imgWidth,
                    net_out[i * oneDetectionSize + 4] * imgHeight, net_out[i * oneDetectionSize + 5] * imgWidth,
                    net_out[i * oneDetectionSize + 6] * imgHeight, net_out[i * oneDetectionSize + 2]);
        }
    }

    return boxes_result;
}

bool checkBBoxOutputs(std::vector<utils::BoundingBox>& actualOutput, std::vector<utils::BoundingBox>& refOutput,
                      const size_t imgWidth, const size_t imgHeight, const float boxTolerance,
                      const float probTolerance) {
    std::cout << "Ref Top:" << std::endl;
    for (size_t i = 0; i < refOutput.size(); ++i) {
        const auto& bb = refOutput[i];
        std::cout << i << " : " << bb.idx << " : [(" << bb.left << " " << bb.top << "), (" << bb.right << " "
                  << bb.bottom << ")] : " << bb.prob * 100 << "%" << std::endl;
    }

    std::cout << "Actual top:" << std::endl;
    for (size_t i = 0; i < actualOutput.size(); ++i) {
        const auto& bb = actualOutput[i];
        std::cout << i << " : " << bb.idx << " : [(" << bb.left << " " << bb.top << "), (" << bb.right << " "
                  << bb.bottom << ")] : " << bb.prob * 100 << "%" << std::endl;
    }

    for (const auto& refBB : refOutput) {
        bool found = false;

        float maxBoxError = 0.0f;
        float maxProbError = 0.0f;

        for (const auto& actualBB : actualOutput) {
            if (actualBB.idx != refBB.idx) {
                continue;
            }

            const utils::Box actualBox{actualBB.left / imgWidth, actualBB.top / imgHeight,
                                       (actualBB.right - actualBB.left) / imgWidth,
                                       (actualBB.bottom - actualBB.top) / imgHeight};
            const utils::Box refBox{refBB.left / imgWidth, refBB.top / imgHeight, (refBB.right - refBB.left) / imgWidth,
                                    (refBB.bottom - refBB.top) / imgHeight};

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
            std::cout << "maxBoxError=" << maxBoxError << " "
                      << "maxProbError=" << maxProbError << std::endl;
            return false;
        }
    }
    return true;
}

bool testSSDDetection(const ie::BlobMap& outputs, const ie::BlobMap& refOutputs,
                      const ie::ConstInputsDataMap& inputsDesc) {
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

    const auto& outBlob = vpux::toFP32(InferenceEngine::as<InferenceEngine::MemoryBlob>(outputs.begin()->second));
    const auto& refOutBlob = vpux::toFP32(InferenceEngine::as<InferenceEngine::MemoryBlob>(refOutputs.begin()->second));

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
        const auto actualIt = std::find_if(probs.cbegin(), probs.cend(), [&refElem](const std::pair<int, float>& arg) {
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
                std::cout << "Probability value mismatch for " << refElem.first << " : " << refElem.second << " vs "
                          << actualElem.second;
                return false;
            }
        }
    }

    return true;
}

//
// RAW mode
//

bool compareBlobs(const ie::MemoryBlob::Ptr& actualOutput, const ie::MemoryBlob::Ptr& refOutput) {
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
                      << " ref : " << std::setw(10) << refVal << " actual : " << std::setw(10) << actualVal
                      << " absdiff : " << std::setw(10) << absDiff << std::endl;
        }

        if (absDiff > FLAGS_raw_tolerance) {
            std::cout << "Absolute difference between actual value " << actualVal << " and reference value " << refVal
                      << " at index " << i << " larger then tolerance" << std::endl;
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
// Cosine-Similarity mode
// (using 'cosim_threshold' flag, with expected value in range [0.0 -> 1.0])
// e.g. '--mode cosim --cosim_threshold 0.98'
//

bool compareCoSim(const ie::MemoryBlob::Ptr actOutput, const ie::MemoryBlob::Ptr refOutput) {
    const auto& actDesc = actOutput->getTensorDesc();
    const auto& refDesc = refOutput->getTensorDesc();

    if (actDesc.getDims() != refDesc.getDims()) {
        std::cout << "Actual and reference blobs has different shape" << std::endl;
        return false;
    }

    const auto actFP32 = vpux::toFP32(vpux::toDefLayout(actOutput));
    const auto refFP32 = vpux::toFP32(vpux::toDefLayout(refOutput));

    const auto actMem = actFP32->rmap();
    const auto refMem = refFP32->rmap();

    const auto act = actMem.as<const float*>();
    const auto ref = refMem.as<const float*>();

    const auto size = refOutput->size();

    double numr = 0.0, denA = 0.0, denB = 0.0;
    for (size_t i = 0; i < size; ++i) {
        numr += act[i] * ref[i];
        denA += act[i] * act[i];
        denB += ref[i] * ref[i];
    }

    if (denA == 0 || denB == 0) {
        std::cout << "Div by ZERO. Cannot compute CoSim metric" << std::endl;
        return false;
    }

    const auto similarity = numr / (sqrt(denA) * sqrt(denB));
    const double eps = 0.0000001;
    // Some experiments revealed that when applying the CoSim metric to large buffers it could provide
    // similarity values that are outside the [-1:1] interval due the big number of operations done on
    // floating point value. A small epsilon value was added to extend the interval to [-(1+eps):1+eps]
    // to ensure that the above check is not failing.
    if (similarity > (1.0 + eps) || similarity < -(1.0 + eps)) {
        std::cout << "Invalid result " << similarity << " (valid range [-1 : +1])" << std::endl;
        return false;
    }

    std::cout << "Cosine similarity : " << similarity * 100 << "%" << std::endl;
    return similarity > FLAGS_cosim_threshold;
}

bool testCoSim(const ie::BlobMap& outputs, const ie::BlobMap& refOutputs) {
    if (outputs.size() != refOutputs.size()) {
        std::cout << "Actual and reference has different number of output blobs" << std::endl;
        return false;
    }

    for (const auto& actualBlob : outputs) {
        auto ref_it = refOutputs.find(actualBlob.first);
        IE_ASSERT(ref_it != refOutputs.end());

        std::cout << "Compare " << actualBlob.first << " with reference" << std::endl;
        if (!compareCoSim(ie::as<ie::MemoryBlob>(actualBlob.second), ie::as<ie::MemoryBlob>(ref_it->second))) {
            return false;
        }
    }

    return true;
}

//
// Yolo V2 mode
//
bool testYoloV2(const ie::BlobMap& actualBlobs, const ie::BlobMap& refBlobs, const ie::ConstInputsDataMap& inputsDesc) {
    IE_ASSERT(inputsDesc.size() == 1);
    IE_ASSERT(actualBlobs.size() == 1u && actualBlobs.size() == refBlobs.size());
    auto actualBlob = actualBlobs.begin()->second;
    auto refBlob = refBlobs.begin()->second;

    const auto& inputDesc = inputsDesc.begin()->second->getTensorDesc();

    const auto imgWidth = inputDesc.getDims().at(3);
    const auto imgHeight = inputDesc.getDims().at(2);
    double confThresh = FLAGS_confidence_threshold;
    double probTolerance = FLAGS_prob_tolerance;
    double boxTolerance = FLAGS_box_tolerance;
    bool isTiny = FLAGS_is_tiny_yolo;

    auto actualOutput =
            utils::parseYoloOutput(vpux::toFP32(InferenceEngine::as<InferenceEngine::MemoryBlob>(actualBlob)), imgWidth,
                                   imgHeight, confThresh, isTiny);
    auto refOutput = utils::parseYoloOutput(vpux::toFP32(InferenceEngine::as<InferenceEngine::MemoryBlob>(refBlob)),
                                            imgWidth, imgHeight, confThresh, isTiny);

    bool result = checkBBoxOutputs(actualOutput, refOutput, imgWidth, imgHeight, boxTolerance, probTolerance);
    return result;
}

//
// Yolo V3 mode
//
bool testYoloV3(const ie::BlobMap& actBlobs, const ie::BlobMap& refBlobs, const ie::ConstInputsDataMap& inputsDesc) {
    IE_ASSERT(inputsDesc.size() == 1);
    IE_ASSERT(actBlobs.size() == 3);
    IE_ASSERT(actBlobs.size() == refBlobs.size());

    const auto& inputDesc = inputsDesc.begin()->second->getTensorDesc();
    const auto imgWidth = inputDesc.getDims().at(3);
    const auto imgHeight = inputDesc.getDims().at(2);

    double confThresh = FLAGS_confidence_threshold;
    double probTolerance = FLAGS_prob_tolerance;
    double boxTolerance = FLAGS_box_tolerance;
    int classes = FLAGS_classes;
    int coords = FLAGS_coords;
    int num = FLAGS_num;
    std::vector<float> anchors = {10.0, 13.0, 16.0,  30.0,  33.0, 23.0,  30.0,  61.0,  62.0,
                                  45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};

    auto actOutput = utils::parseYoloV3Output(actBlobs, imgWidth, imgHeight, classes, coords, num, anchors, confThresh,
                                              InferenceEngine::NCHW);
    auto refOutput = utils::parseYoloV3Output(refBlobs, imgWidth, imgHeight, classes, coords, num, anchors, confThresh,
                                              refBlobs.begin()->second->getTensorDesc().getLayout());

    bool result = checkBBoxOutputs(actOutput, refOutput, imgWidth, imgHeight, boxTolerance, probTolerance);
    return result;
};

//
// Yolo V4 mode
// Ref link: https://docs.openvino.ai/latest/omz_models_model_yolo_v4_tiny_tf.html
//
bool testYoloV4(const ie::BlobMap& actBlobs, const ie::BlobMap& refBlobs, const ie::ConstInputsDataMap& inputsDesc) {
    IE_ASSERT(inputsDesc.size() == 1);
    IE_ASSERT(actBlobs.size() == 2);
    IE_ASSERT(actBlobs.size() == refBlobs.size());

    const auto& inputDesc = inputsDesc.begin()->second->getTensorDesc();
    const auto imgWidth = inputDesc.getDims().at(3);
    const auto imgHeight = inputDesc.getDims().at(2);

    double confThresh = FLAGS_confidence_threshold;
    double probTolerance = FLAGS_prob_tolerance;
    double boxTolerance = FLAGS_box_tolerance;
    int classes = FLAGS_classes;
    int coords = FLAGS_coords;
    int num = FLAGS_num;
    std::vector<float> anchors = {10.0, 14.0, 23.0, 27.0, 37.0, 58.0, 81.0, 82.0, 135.0, 169.0, 344.0, 319.0};
    std::vector<std::vector<float>> anchor_mask{{3, 4, 5}, {1, 2, 3}};
    std::vector<float> masked_anchors{};
    for (auto& it : anchor_mask) {
        int index = 0;
        for (auto& anchorIndex : it) {
            if (index >= num)
                break;

            index++;
            masked_anchors.push_back(anchors[2 * anchorIndex]);
            masked_anchors.push_back(anchors[2 * anchorIndex + 1]);
        }
    }

    auto refOutput = utils::parseYoloV4Output(refBlobs, imgWidth, imgHeight, classes, coords, num, masked_anchors,
                                              confThresh, refBlobs.begin()->second->getTensorDesc().getLayout());
    auto actOutput = utils::parseYoloV4Output(actBlobs, imgWidth, imgHeight, classes, coords, num, masked_anchors,
                                              confThresh, actBlobs.begin()->second->getTensorDesc().getLayout());
    bool result = checkBBoxOutputs(actOutput, refOutput, imgWidth, imgHeight, boxTolerance, probTolerance);
    return result;
};

//
// PSNR mode
// using psnr_reference and psnr_tolerance flags for validation
// e.g. '--mode psnr --psnr_reference <value> --psnr_tolerance <value>'
// Direction of metric’s growth is higher-better. If the images are identical, the PSNR is infinite.
//

bool testPSNR(const ie::BlobMap& actBlobs, const ie::BlobMap& refBlobs, const int dstHeight, const int dstWidth) {
    IE_ASSERT(actBlobs.size() == refBlobs.size());

    int scaleBorder = FLAGS_scale_border;
    bool normalizedImage = FLAGS_normalized_image;

    auto refOutput = utils::parseBlobAsFP32(refBlobs);
    auto actOutput = utils::parseBlobAsFP32(actBlobs);

    auto result = utils::runPSNRMetric(actOutput, refOutput, dstHeight, dstWidth, scaleBorder, normalizedImage);

    if (std::fabs(result - FLAGS_psnr_reference) > FLAGS_psnr_tolerance) {
        std::cout << "Absolute difference between actual value " << result << " and reference value "
                  << FLAGS_psnr_reference << " larger then tolerance " << FLAGS_psnr_tolerance << std::endl;
        return false;
    }

    return true;
}

static void printPerformanceCountsAndLatency(size_t numberOfTestCase, const ProfVec& profilingData,
                                             std::chrono::duration<double, std::milli> duration) {
    auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(duration);

    if (!profilingData.empty()) {
        std::cout << "Performance counts for " << numberOfTestCase << "-th infer request:" << std::endl;
        printPerformanceCounts(profilingData, std::cout, FLAGS_device, false);
    }

    std::cout << "Latency: " << std::fixed << std::setprecision(2) << durationMs.count() << " ms" << std::endl;
}

bool compare_mean_IoU(std::vector<float> iou, float semSegThreshold, uint32_t classes) {
    float threshold = semSegThreshold * 100;
    float ma = 0.0f;
    bool stateValue = true;

    if (FLAGS_sem_seg_ignore_label != std::numeric_limits<uint32_t>::max()) {
        classes--;
    }

    for (size_t i = 0; i < classes; i++) {
        if (FLAGS_dataset == "camVid12") {
            std::cout << "mean_iou@" << camVid12[i].c_str() << ": " << std::fixed << std::setprecision(2) << iou[i]
                      << "%" << std::endl;
        } else {
            std::cout << "mean_iou@class" << i << ": " << std::fixed << std::setprecision(2) << iou[i] << "%"
                      << std::endl;
        }

        if (iou[i] < threshold) {
            std::cout << "Threshold smaller than " << threshold << "%" << std::endl;
            stateValue = false;
        }
        ma += iou[i];
    }
    std::cout << "mean_iou@:mean " << std::fixed << std::setprecision(2) << (ma / classes) << "%" << std::endl;

    return stateValue;
}

//
// MeanIoU mode
// Using sem_seg_classes, sem_seg_threshold flags and optionally sem_seg_ignore_label and dataset flags for validation
// e.g. '--mode mean_iou --sem_seg_classes 12 --sem_seg_threshold 0.98 --sem_seg_ignore_label 11 --dataset camVid12'
//
bool testMeanIoU(const ie::BlobMap& actBlobs, const ie::BlobMap& refBlobs, const ie::ConstInputsDataMap& inputsDesc) {
    IE_ASSERT(inputsDesc.size() == 1);
    IE_ASSERT(actBlobs.size() == 1);
    IE_ASSERT(actBlobs.size() == refBlobs.size());

    unsigned int classes = FLAGS_sem_seg_classes;
    float semSegThreshold = FLAGS_sem_seg_threshold;

    std::vector<uint8_t> refOutput;
    std::vector<uint8_t> actOutput;
    std::vector<float> iou(classes, 0.0f);

    utils::argMax_channels(ie::as<ie::MemoryBlob>(refBlobs.begin()->second), refOutput);
    utils::argMax_channels(ie::as<ie::MemoryBlob>(actBlobs.begin()->second), actOutput);

    if (refOutput.size() != actOutput.size()) {
        std::cout << "Ref size and Act size are different" << std::endl;
        return false;
    }
    iou = utils::mean_IoU(actOutput, refOutput, classes, FLAGS_sem_seg_ignore_label);

    return compare_mean_IoU(iou, semSegThreshold, classes);
};

static int runSingleImageTestOV10() {
    std::cout << "Run single image test with OV 1.0 API" << std::endl;
    try {
        std::shared_ptr<ie::Core> ieCoreShared = std::make_shared<ie::Core>();
        ieCore = ieCoreShared;

        const std::unordered_set<std::string> allowedPrecision = {"U8", "FP16", "FP32"};
        if (!FLAGS_ip.empty()) {
            // input precision is U8, FP16 or FP32 only
            std::transform(FLAGS_ip.begin(), FLAGS_ip.end(), FLAGS_ip.begin(), ::toupper);
            if (allowedPrecision.count(FLAGS_ip) == 0)
                throw std::logic_error("Parameter -ip " + FLAGS_ip + " is not supported");
        }
        if (!FLAGS_op.empty()) {
            // output precision is U8, FP16 or FP32 only
            std::transform(FLAGS_op.begin(), FLAGS_op.end(), FLAGS_op.begin(), ::toupper);
            if (allowedPrecision.count(FLAGS_op) == 0)
                throw std::logic_error("Parameter -op " + FLAGS_op + " is not supported");
        }

        std::vector<std::string> inputFilesPerCase;
        std::vector<std::vector<std::string>> inputFilesForOneInfer;
        inputFilesPerCase = splitStringList(FLAGS_input, ';');
        for (const auto& images : inputFilesPerCase) {
            inputFilesForOneInfer.push_back(splitStringList(images, ','));
        }

        std::vector<std::string> inputBinPrecisionStrPerCase;
        std::vector<std::vector<ie::Precision>> inputBinPrecisionForOneInfer(inputFilesForOneInfer.size());
        if (FLAGS_img_as_bin) {
            for (std::size_t i = 0; i < inputFilesForOneInfer.size(); ++i) {
                inputBinPrecisionForOneInfer[i] =
                        std::vector<ie::Precision>(inputFilesForOneInfer[i].size(), ie::Precision::UNSPECIFIED);
            }
            inputBinPrecisionStrPerCase = splitStringList(FLAGS_img_bin_precision, ';');
            std::size_t inferIdx = 0;
            for (const auto& precisions : inputBinPrecisionStrPerCase) {
                std::vector<std::string> inputBinPrecisionsStrThisInfer = splitStringList(precisions, ',');
                std::size_t precisionIdx = 0;
                for (const auto& precision : inputBinPrecisionsStrThisInfer) {
                    if (strEq(precision, "FP32")) {
                        inputBinPrecisionForOneInfer[inferIdx][precisionIdx] = ie::Precision::FP32;
                    } else if (strEq(precision, "FP16")) {
                        inputBinPrecisionForOneInfer[inferIdx][precisionIdx] = ie::Precision::FP16;
                    } else if (strEq(precision, "U8")) {
                        inputBinPrecisionForOneInfer[inferIdx][precisionIdx] = ie::Precision::U8;
                    } else {
                        std::cout << "WARNING: Unhandled precision '" << precision
                                  << "'! Only FP32, FP16 and U8 can be currently converted to the network's input "
                                     "tensor precision.";
                    }
                    ++precisionIdx;
                }
                ++inferIdx;
            }
        }

        if (FLAGS_network.empty()) {
            std::cout << "Not enough parameters. Check help." << std::endl;
            return EXIT_FAILURE;
        }

        setupInferenceEngine();

        ie::ExecutableNetwork exeNet;

        if (strEq(FileUtils::fileExt(FLAGS_network), "xml")) {
            std::cout << "Load network " << FLAGS_network << std::endl;

            auto cnnNet = ieCoreShared->ReadNetwork(FLAGS_network);

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

                for (auto inputInfoIt = inputInfo.begin(); inputInfoIt != inputInfo.end(); ++inputInfoIt) {
                    inputInfoIt->second->setPrecision(prc_in);
                }
            }
            // Input layout
            if (!FLAGS_il.empty()) {
                for (auto& info : inputInfo) {
                    if (info.second->getLayout() == InferenceEngine::C)
                        continue;
                    else if (info.second->getTensorDesc().getDims().size() == 2) {
                        info.second->setLayout(ie::Layout::NC);
                    } else if (info.second->getTensorDesc().getDims().size() == 3) {
                        if (FLAGS_il == "NCHW") {
                            info.second->setLayout(ie::Layout::CHW);
                        } else {
                            info.second->setLayout(ie::Layout::HWC);
                        }
                    } else {
                        if (FLAGS_il == "NCHW") {
                            info.second->setLayout(ie::Layout::NCHW);
                        } else {
                            info.second->setLayout(ie::Layout::NHWC);
                        }
                    }
                }
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
                for (auto outputInfoIt = outputInfo.begin(); outputInfoIt != outputInfo.end(); ++outputInfoIt) {
                    outputInfoIt->second->setPrecision(prc_out);
                }
            }
            // Output layout
            if (!FLAGS_ol.empty()) {
                for (auto outputInfoIt = outputInfo.begin(); outputInfoIt != outputInfo.end(); ++outputInfoIt) {
                    if (outputInfoIt->second->getDims().size() == 2) {
                        outputInfoIt->second->setLayout(ie::Layout::NC);
                    } else if (outputInfoIt->second->getDims().size() == 3) {
                        if (FLAGS_ol == "NCHW") {
                            outputInfoIt->second->setLayout(ie::Layout::CHW);
                        } else {
                            outputInfoIt->second->setLayout(ie::Layout::HWC);
                        }
                    } else {
                        if (FLAGS_ol == "NCHW") {
                            outputInfoIt->second->setLayout(ie::Layout::NCHW);
                        } else {
                            outputInfoIt->second->setLayout(ie::Layout::NHWC);
                        }
                    }
                }
            }

            exeNet = ieCoreShared->LoadNetwork(cnnNet, FLAGS_device);
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
                    << "Number of input files " << inputFiles.size() << " doesn't match network configuration "
                    << inputsInfo.size();

            ie::BlobMap inputs;
            size_t inputInd = 0;
            std::vector<std::string> dumpedInputsPaths;
            for (const auto& p : inputsInfo) {
                std::cout << "Load input #" << inputInd << " from " << inputFiles[inputInd] << " as "
                          << p.second->getTensorDesc().getPrecision() << std::endl;
                const auto blob =
                        !FLAGS_img_as_bin
                                ? loadInput(p.second->getTensorDesc(), inputFiles[inputInd], FLAGS_color_format)
                                : loadInput(p.second->getTensorDesc(), inputFiles[inputInd], FLAGS_color_format,
                                            inputBinPrecisionForOneInfer[numberOfTestCase][inputInd]);
                inputs.emplace(p.first, blob);

                std::ostringstream ostr;
                ostr << netFileName << "_input_" << inputInd << "_case_" << numberOfTestCase << ".blob";
                const auto blobFileName = ostr.str();

                std::cout << "Dump input #" << inputInd << "_case_" << numberOfTestCase << " to " << blobFileName
                          << std::endl;
                dumpBlob(blob, blobFileName);

                ++inputInd;

                dumpedInputsPaths.push_back(blobFileName);
            }

            std::cout << "Run inference on " << FLAGS_device << std::endl;
            const auto startTime = Time::now();
            const auto inferenceOutput = runInfer(exeNet, inputs, dumpedInputsPaths);
            const auto endTime = Time::now();

            const ie::BlobMap outputs = inferenceOutput.first;

            printPerformanceCountsAndLatency(numberOfTestCase, inferenceOutput.second, endTime - startTime);

            if (FLAGS_run_test) {
                ie::BlobMap refOutputs;
                size_t outputInd = 0;
                for (const auto& p : outputsInfo) {
                    std::ostringstream ostr;
                    ostr << netFileName << "_ref_out_" << outputInd << "_case_" << numberOfTestCase << ".blob";
                    const auto blobFileName = ostr.str();

                    auto refTensorDesc = p.second->getTensorDesc();

                    std::cout << "Load reference output #" << outputInd << " from " << blobFileName << " as "
                              << refTensorDesc.getPrecision() << std::endl;

                    const auto blob = loadBlob(refTensorDesc, blobFileName);
                    refOutputs.emplace(p.first, blob);

                    ++outputInd;
                }

                outputInd = 0;
                for (const auto& p : outputs) {
                    std::ostringstream ostr;
                    ostr << netFileName << "_kmb_out_" << outputInd << "_case_" << numberOfTestCase << ".blob";
                    const auto blobFileName = ostr.str();

                    std::cout << "Dump device output #" << outputInd << "_case_" << numberOfTestCase << " to "
                              << blobFileName << std::endl;
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
                } else if (strEq(FLAGS_mode, "cosim")) {
                    if (testCoSim(outputs, refOutputs)) {
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
                } else if (strEq(FLAGS_mode, "yolo_v3")) {
                    if (testYoloV3(outputs, refOutputs, inputsInfo)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "yolo_v4")) {
                    if (testYoloV4(outputs, refOutputs, inputsInfo)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "psnr")) {
                    const auto& outputDesc = outputsInfo.begin()->second->getTensorDesc();
                    const auto dstHeight = outputDesc.getDims().at(2);
                    const auto dstWidth = outputDesc.getDims().at(3);

                    if (testPSNR(outputs, refOutputs, dstHeight, dstWidth)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "mean_iou")) {
                    if (testMeanIoU(outputs, refOutputs, inputsInfo)) {
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
                for (const auto& p : outputs) {
                    std::ostringstream ostr;
                    ostr << netFileName << "_ref_out_" << outputInd << "_case_" << numberOfTestCase << ".blob";
                    const auto blobFileName = ostr.str();

                    std::cout << "Dump reference output #" << outputInd << " to " << blobFileName << std::endl;
                    // WA: By some CPU Plugin returns FP32 blob although user specified FP16 for the output.
                    // This issue is reproducible only with OpenVINO 1.0 API.
                    // In that case, perform conversion on application side.
                    if (FLAGS_device == "CPU" &&
                        outputsInfo.at(p.first)->getTensorDesc().getPrecision() == ie::Precision::FP16) {
                        dumpBlob(vpux::toFP16(ie::as<ie::MemoryBlob>(p.second)), blobFileName);
                    } else {
                        dumpBlob(ie::as<ie::MemoryBlob>(p.second), blobFileName);
                    }

                    ++outputInd;
                }
            }
        }
    }  // try
    catch (const std::exception& ex) {
        std::cerr << "exception: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

//
// OpenVINO 2.0
//

void setupOVCore(ov::Core& core) {
    auto flagDevice = FLAGS_device;

    if (!FLAGS_log_level.empty()) {
        core.set_property(flagDevice, {{CONFIG_KEY(LOG_LEVEL), FLAGS_log_level}});
    }

    if (FLAGS_device == "CPU") {
        core.set_property(flagDevice, {{"LP_TRANSFORMS_MODE", CONFIG_VALUE(NO)}});
    }

    if (FLAGS_pc) {
        core.set_property(flagDevice, {{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)}});
    }

    if (!FLAGS_config.empty()) {
        std::ifstream file(FLAGS_config);
        IE_ASSERT(file.is_open()) << "Can't open file " << FLAGS_config << " for read";

        std::string key, value;
        while (file >> key >> value) {
            if (key.empty() || key[0] == '#') {
                continue;
            }

            core.set_property(flagDevice, {{key, value}});
        }
    }
}

using TensorMap = std::map<std::string, ov::Tensor>;
std::pair<TensorMap, ProfVec> runInfer(ov::CompiledModel& compiledModel, const TensorMap& inputs,
                                       const std::vector<std::string>& dumpedInputsPaths) {
    auto inferRequest = compiledModel.create_infer_request();

    for (const auto& p : inputs) {
        inferRequest.set_tensor(p.first, p.second);
    }

    inferRequest.infer();

    TensorMap out;
    for (const auto& outputInfo : compiledModel.outputs()) {
        const std::string layer_name = outputInfo.get_any_name();
        out.insert({layer_name, inferRequest.get_tensor(layer_name)});
    }

    ProfVec profData{};

    if (FLAGS_pc) {
        profData = inferRequest.get_profiling_info();
    }

    return std::make_pair(out, profData);
}

static ie::Precision toIE(const ov::element::Type& type) {
    switch (type) {
    case ov::element::u8:
        return ie::Precision::U8;
    case ov::element::f32:
        return ie::Precision::FP32;
    case ov::element::f16:
        return ie::Precision::FP16;
    case ov::element::undefined:
        return ie::Precision::UNSPECIFIED;
    }
    std::stringstream ss;
    ss << "Failed to convert ov::Precision: " << type << " to IE::Precision";
    throw std::logic_error(ss.str());
}

static ie::Layout toIE(const ov::Layout layout) {
    if (layout == ov::Layout("NCHW")) {
        return ie::Layout::NCHW;
    } else if (layout == ov::Layout("NHWC")) {
        return ie::Layout::NHWC;
    } else if (layout == ov::Layout("CHW")) {
        return ie::Layout::CHW;
    } else if (layout == ov::Layout("HWC")) {
        return ie::Layout::HWC;
    } else if (layout == ov::Layout("NC")) {
        return ie::Layout::NC;
    } else if (layout == ov::Layout("HW")) {
        return ie::Layout::HW;
    } else if (layout == ov::Layout("CN")) {
        return ie::Layout::CN;
    } else if (layout == ov::Layout("C")) {
        return ie::Layout::C;
    }
    std::stringstream ss;
    ss << "Failed to convert ov::Layout(" << layout.to_string() << ") to IE::Layout";
    throw std::logic_error(ss.str());
}

// NB: Need to keep dims in "NCHW" order as it was for OV 1.0 to keep backward compatibility for ie::TensorDesc.
static ie::SizeVector toIE(const ov::Shape& shape, const ov::Layout& layout) {
    if (shape.size() == 4u) {
        return ie::SizeVector{shape[ov::layout::batch_idx(layout)], shape[ov::layout::channels_idx(layout)],
                              shape[ov::layout::height_idx(layout)], shape[ov::layout::width_idx(layout)]};
    }
    return ie::SizeVector(shape);
}

// FIXME: User must provide layout explicitly.
// No "default" layout for IRv11 models.
static ov::Layout getLayoutByRank(const int rank) {
    switch (rank) {
    case 1:
        return ov::Layout("C");
    case 2:
        return ov::Layout("NC");
    case 3:
        return ov::Layout("CHW");
    case 4:
        return ov::Layout("NCHW");
    case 5:
        return ov::Layout("NCDHW");
    }
    throw std::logic_error("Failed to get layout for rank equal to " + std::to_string(rank));
}

static std::string toString(const std::vector<size_t>& vec) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < vec.size() - 1; ++i) {
        ss << vec[i] << ",";
    }
    ss << vec[vec.size() - 1] << "]";
    return ss.str();
}

ie::Blob::Ptr toIE(const ov::Tensor& tensor, const ov::Layout& layout) {
    ie::TensorDesc desc(toIE(tensor.get_element_type()), toIE(tensor.get_shape(), layout), toIE(layout));
    return make_blob_with_precision(desc, tensor.data());
}

using TensorDescMap = std::unordered_map<std::string, ie::TensorDesc>;

bool testSSDDetection(const ie::BlobMap& outputs, const ie::BlobMap& refOutputs, const TensorDescMap& inputsDesc) {
    IE_ASSERT(outputs.size() == 1 && refOutputs.size() == 1);
    const auto& outBlob = outputs.begin()->second;
    const auto& refOutBlob = refOutputs.begin()->second;
    IE_ASSERT(refOutBlob->getTensorDesc().getPrecision() == ie::Precision::FP32);
    IE_ASSERT(outBlob->getTensorDesc().getPrecision() == ie::Precision::FP32);
    IE_ASSERT(!inputsDesc.empty());

    const auto& inputDesc = inputsDesc.begin()->second;

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
// Yolo V2 mode
//
bool testYoloV2(const ie::BlobMap& actualBlobs, const ie::BlobMap& refBlobs, const TensorDescMap& inputsDesc) {
    IE_ASSERT(inputsDesc.size() == 1);
    IE_ASSERT(actualBlobs.size() == 1u && actualBlobs.size() == refBlobs.size());
    auto actualBlob = actualBlobs.begin()->second;
    auto refBlob = refBlobs.begin()->second;

    const auto& inputDesc = inputsDesc.begin()->second;

    const auto imgWidth = inputDesc.getDims().at(3);
    const auto imgHeight = inputDesc.getDims().at(2);
    double confThresh = FLAGS_confidence_threshold;
    double probTolerance = FLAGS_prob_tolerance;
    double boxTolerance = FLAGS_box_tolerance;
    bool isTiny = FLAGS_is_tiny_yolo;

    auto actualOutput =
            utils::parseYoloOutput(vpux::toFP32(InferenceEngine::as<InferenceEngine::MemoryBlob>(actualBlob)), imgWidth,
                                   imgHeight, confThresh, isTiny);
    auto refOutput = utils::parseYoloOutput(vpux::toFP32(InferenceEngine::as<InferenceEngine::MemoryBlob>(refBlob)),
                                            imgWidth, imgHeight, confThresh, isTiny);

    bool result = checkBBoxOutputs(actualOutput, refOutput, imgWidth, imgHeight, boxTolerance, probTolerance);
    return result;
}

//
// Yolo V3 mode
//
bool testYoloV3(const ie::BlobMap& actBlobs, const ie::BlobMap& refBlobs, const TensorDescMap& inputsDesc) {
    IE_ASSERT(inputsDesc.size() == 1);
    IE_ASSERT(actBlobs.size() == 3);
    IE_ASSERT(actBlobs.size() == refBlobs.size());

    const auto& inputDesc = inputsDesc.begin()->second;
    const auto imgWidth = inputDesc.getDims().at(3);
    const auto imgHeight = inputDesc.getDims().at(2);

    double confThresh = FLAGS_confidence_threshold;
    double probTolerance = FLAGS_prob_tolerance;
    double boxTolerance = FLAGS_box_tolerance;
    int classes = FLAGS_classes;
    int coords = FLAGS_coords;
    int num = FLAGS_num;
    std::vector<float> anchors = {10.0, 13.0, 16.0,  30.0,  33.0, 23.0,  30.0,  61.0,  62.0,
                                  45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};

    auto actOutput = utils::parseYoloV3Output(actBlobs, imgWidth, imgHeight, classes, coords, num, anchors, confThresh,
                                              InferenceEngine::NCHW);
    auto refOutput = utils::parseYoloV3Output(refBlobs, imgWidth, imgHeight, classes, coords, num, anchors, confThresh,
                                              refBlobs.begin()->second->getTensorDesc().getLayout());

    bool result = checkBBoxOutputs(actOutput, refOutput, imgWidth, imgHeight, boxTolerance, probTolerance);
    return result;
};

//
// Yolo V4 mode
// Ref link: https://docs.openvino.ai/latest/omz_models_model_yolo_v4_tiny_tf.html
//
bool testYoloV4(const ie::BlobMap& actBlobs, const ie::BlobMap& refBlobs, const TensorDescMap& inputsDesc) {
    IE_ASSERT(inputsDesc.size() == 1);
    IE_ASSERT(actBlobs.size() == 2);
    IE_ASSERT(actBlobs.size() == refBlobs.size());

    const auto& inputDesc = inputsDesc.begin()->second;
    const auto imgWidth = inputDesc.getDims().at(3);
    const auto imgHeight = inputDesc.getDims().at(2);

    double confThresh = FLAGS_confidence_threshold;
    double probTolerance = FLAGS_prob_tolerance;
    double boxTolerance = FLAGS_box_tolerance;
    int classes = FLAGS_classes;
    int coords = FLAGS_coords;
    int num = FLAGS_num;
    std::vector<float> anchors = {10.0, 14.0, 23.0, 27.0, 37.0, 58.0, 81.0, 82.0, 135.0, 169.0, 344.0, 319.0};
    std::vector<std::vector<float>> anchor_mask{{3, 4, 5}, {1, 2, 3}};
    std::vector<float> masked_anchors{};
    for (auto& it : anchor_mask) {
        int index = 0;
        for (auto& anchorIndex : it) {
            if (index >= num)
                break;

            index++;
            masked_anchors.push_back(anchors[2 * anchorIndex]);
            masked_anchors.push_back(anchors[2 * anchorIndex + 1]);
        }
    }

    auto refOutput = utils::parseYoloV4Output(refBlobs, imgWidth, imgHeight, classes, coords, num, masked_anchors,
                                              confThresh, refBlobs.begin()->second->getTensorDesc().getLayout());
    auto actOutput = utils::parseYoloV4Output(actBlobs, imgWidth, imgHeight, classes, coords, num, masked_anchors,
                                              confThresh, actBlobs.begin()->second->getTensorDesc().getLayout());
    bool result = checkBBoxOutputs(actOutput, refOutput, imgWidth, imgHeight, boxTolerance, probTolerance);
    return result;
};

//
// MeanIoU mode
// Using sem_seg_classes, sem_seg_threshold flags and optionally sem_seg_ignore_label and dataset flags for validation
// e.g. '--mode mean_iou --sem_seg_classes 12 --sem_seg_threshold 0.98 --sem_seg_ignore_label 11 --dataset camVid12'
//
bool testMeanIoU(const ie::BlobMap& actBlobs, const ie::BlobMap& refBlobs, const TensorDescMap& inputsDesc) {
    IE_ASSERT(inputsDesc.size() == 1);
    IE_ASSERT(actBlobs.size() == 1);
    IE_ASSERT(actBlobs.size() == refBlobs.size());

    unsigned int classes = FLAGS_sem_seg_classes;
    float semSegThreshold = FLAGS_sem_seg_threshold;

    std::vector<uint8_t> refOutput;
    std::vector<uint8_t> actOutput;
    std::vector<float> iou(classes, 0.0f);

    utils::argMax_channels(ie::as<ie::MemoryBlob>(refBlobs.begin()->second), refOutput);
    utils::argMax_channels(ie::as<ie::MemoryBlob>(actBlobs.begin()->second), actOutput);

    if (refOutput.size() != actOutput.size()) {
        std::cout << "Ref size and Act size are different" << std::endl;
        return false;
    }
    iou = utils::mean_IoU(actOutput, refOutput, classes, FLAGS_sem_seg_ignore_label);

    return compare_mean_IoU(iou, semSegThreshold, classes);
};

static int runSingleImageTestOV20() {
    std::cout << "Run single image test with OV 2.0 API" << std::endl;
    try {
        const std::unordered_set<std::string> allowedPrecision = {"U8", "FP16", "FP32"};
        if (!FLAGS_ip.empty()) {
            // input precision is U8, FP16 or FP32 only
            std::transform(FLAGS_ip.begin(), FLAGS_ip.end(), FLAGS_ip.begin(), ::toupper);
            if (allowedPrecision.count(FLAGS_ip) == 0)
                throw std::logic_error("Parameter -ip " + FLAGS_ip + " is not supported");
        }
        if (!FLAGS_op.empty()) {
            // output precision is U8, FP16 or FP32 only
            std::transform(FLAGS_op.begin(), FLAGS_op.end(), FLAGS_op.begin(), ::toupper);
            if (allowedPrecision.count(FLAGS_op) == 0)
                throw std::logic_error("Parameter -op " + FLAGS_op + " is not supported");
        }

        ov::Layout inUserLayout(FLAGS_il);
        ov::Layout outUserLayout(FLAGS_ol);
        ov::Layout inModelLayout(FLAGS_iml);
        ov::Layout outModelLayout(FLAGS_oml);

        std::vector<std::string> inputFilesPerCase;
        std::vector<std::vector<std::string>> inputFilesForOneInfer;
        inputFilesPerCase = splitStringList(FLAGS_input, ';');
        for (const auto& images : inputFilesPerCase) {
            inputFilesForOneInfer.push_back(splitStringList(images, ','));
        }

        std::vector<std::string> inputBinPrecisionStrPerCase;
        std::vector<std::vector<ov::element::Type>> inputBinPrecisionForOneInfer(inputFilesForOneInfer.size());
        if (FLAGS_img_as_bin) {
            for (std::size_t i = 0; i < inputFilesForOneInfer.size(); ++i) {
                inputBinPrecisionForOneInfer[i] =
                        std::vector<ov::element::Type>(inputFilesForOneInfer[i].size(), ov::element::undefined);
            }
            inputBinPrecisionStrPerCase = splitStringList(FLAGS_img_bin_precision, ';');
            std::size_t inferIdx = 0;
            for (const auto& precisions : inputBinPrecisionStrPerCase) {
                std::vector<std::string> inputBinPrecisionsStrThisInfer = splitStringList(precisions, ',');
                std::size_t precisionIdx = 0;
                for (const auto& precision : inputBinPrecisionsStrThisInfer) {
                    if (strEq(precision, "FP32")) {
                        inputBinPrecisionForOneInfer[inferIdx][precisionIdx] = ov::element::f32;
                    } else if (strEq(precision, "FP16")) {
                        inputBinPrecisionForOneInfer[inferIdx][precisionIdx] = ov::element::f16;
                    } else if (strEq(precision, "U8")) {
                        inputBinPrecisionForOneInfer[inferIdx][precisionIdx] = ov::element::u8;
                    } else {
                        std::cout << "WARNING: Unhandled precision '" << precision
                                  << "'! Only FP32, FP16 and U8 can be currently converted to the network's input "
                                     "tensor precision.";
                    }
                    ++precisionIdx;
                }
                ++inferIdx;
            }
        }

        if (FLAGS_network.empty()) {
            std::cout << "Not enough parameters. Check help." << std::endl;
            return EXIT_FAILURE;
        }

        ov::Core core;
        setupOVCore(core);

        ov::CompiledModel compiledModel;
        if (strEq(FileUtils::fileExt(FLAGS_network), "xml")) {
            std::cout << "Load network " << FLAGS_network << std::endl;

            auto model = core.read_model(FLAGS_network);

            ov::preprocess::PrePostProcessor ppp(model);

            // Input precision
            const auto inputInfo = model->inputs();
            if (!FLAGS_ip.empty()) {
                ov::element::Type prc_in = ov::element::u8;
                if (FLAGS_ip == "FP16")
                    prc_in = ov::element::f16;
                else if (FLAGS_ip == "FP32")
                    prc_in = ov::element::f32;
                else
                    prc_in = ov::element::u8;

                for (size_t i = 0; i < inputInfo.size(); ++i) {
                    ppp.input(i).tensor().set_element_type(prc_in);
                }
            }

            // Input layout
            if (!inUserLayout.empty()) {
                for (size_t i = 0; i < inputInfo.size(); ++i) {
                    ov::Layout inLayerModelLayout;
                    if (inModelLayout.empty()) {
                        const auto shape = inputInfo[i].get_shape();
                        inLayerModelLayout = getLayoutByRank(shape.size());
                        std::cout << "WARNING: Configuring preprocessing. Since --iml option isn't set, input model "
                                     "layout for layer \""
                                  << inputInfo[i].get_any_name() << "\" is infered from shape: " << toString(shape)
                                  << " rank (" << shape.size() << ") as " << inLayerModelLayout.to_string()
                                  << std::endl;
                    } else {
                        inLayerModelLayout = inModelLayout;
                    }
                    ppp.input(i).model().set_layout(inLayerModelLayout);
                    ppp.input(i).tensor().set_layout(inUserLayout);
                }
            }

            // Output precision
            const auto outputInfo = model->outputs();
            if (!FLAGS_op.empty()) {
                ov::element::Type prc_out = ov::element::u8;
                if (FLAGS_op == "FP16")
                    prc_out = ov::element::f16;
                else if (FLAGS_op == "FP32")
                    prc_out = ov::element::f32;
                else
                    prc_out = ov::element::u8;

                for (size_t i = 0; i < outputInfo.size(); ++i) {
                    ppp.output(i).tensor().set_element_type(prc_out);
                }
            }

            // Output layout
            if (!outUserLayout.empty()) {
                for (size_t i = 0; i < outputInfo.size(); ++i) {
                    ov::Layout outLayerModelLayout;
                    if (outModelLayout.empty()) {
                        const auto shape = outputInfo[i].get_shape();
                        outLayerModelLayout = getLayoutByRank(shape.size());
                        std::cout << "WARNING: Configuring preprocessing. Since --oml option isn't set, output model "
                                     "layout for layer \""
                                  << outputInfo[i].get_any_name() << "\" is infered from shape: " << toString(shape)
                                  << " rank (" << shape.size() << ") as " << outLayerModelLayout.to_string()
                                  << std::endl;
                    } else {
                        outLayerModelLayout = outModelLayout;
                    }
                    ppp.output(i).model().set_layout(outLayerModelLayout);
                    ppp.output(i).tensor().set_layout(outUserLayout);
                }
            }

            compiledModel = core.compile_model(ppp.build(), FLAGS_device);
        } else {
            std::cout << "Import network " << FLAGS_network << std::endl;
            std::ifstream file(FLAGS_network, std::ios_base::in | std::ios_base::binary);
            IE_ASSERT(file.is_open()) << "Can't open file " << FLAGS_network << " for read";
            compiledModel = core.import_model(file, FLAGS_device);
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
            const auto inputsInfo = compiledModel.inputs();
            const auto outputsInfo = compiledModel.outputs();
            std::vector<std::string> inputFiles = inputFilesForOneInfer[numberOfTestCase];
            IE_ASSERT(inputFiles.size() == inputsInfo.size())
                    << "Number of input files " << inputFiles.size() << " doesn't match network configuration "
                    << inputsInfo.size();

            TensorMap in_tensors;
            ie::BlobMap inputs, outputs;
            size_t inputInd = 0;
            std::vector<std::string> dumpedInputsPaths;
            TensorDescMap inDescMap;

            for (const auto& inputInfo : inputsInfo) {
                const auto shape = inputInfo.get_shape();
                const auto prec = inputInfo.get_element_type();

                ov::Layout inLayerUserLayout;
                if (inUserLayout.empty()) {
                    ov::Layout inLayerModelLayout;
                    if (inModelLayout.empty()) {
                        inLayerModelLayout = getLayoutByRank(shape.size());
                        std::cout << "WARNING: Loading input data. Since --iml option isn't set, input model layout "
                                     "for layer \""
                                  << inputInfo.get_any_name() << "\" is infered from shape: " << toString(shape)
                                  << " rank (" << shape.size() << ") as " << inLayerModelLayout.to_string()
                                  << std::endl;
                    } else {
                        inLayerModelLayout = inModelLayout;
                    }
                    inLayerUserLayout = inLayerModelLayout;
                } else {
                    inLayerUserLayout = inUserLayout;
                }

                ie::TensorDesc desc(toIE(prec), toIE(shape, inLayerUserLayout), toIE(inLayerUserLayout));

                inDescMap.emplace(inputInfo.get_any_name(), desc);

                std::cout << "Load input #" << inputInd << " from " << inputFiles[inputInd] << " as "
                          << desc.getPrecision() << " " << desc.getLayout() << " " << toString(desc.getDims())
                          << std::endl;

                const auto blob = !FLAGS_img_as_bin
                                          ? loadInput(desc, inputFiles[inputInd], FLAGS_color_format)
                                          : loadInput(desc, inputFiles[inputInd], FLAGS_color_format,
                                                      toIE(inputBinPrecisionForOneInfer[numberOfTestCase][inputInd]));
                inputs.emplace(inputInfo.get_any_name(), blob);

                std::ostringstream ostr;
                ostr << netFileName << "_input_" << inputInd << "_case_" << numberOfTestCase << ".blob";
                const auto blobFileName = ostr.str();

                std::cout << "Dump input #" << inputInd << "_case_" << numberOfTestCase << " to " << blobFileName
                          << std::endl;
                dumpBlob(blob, blobFileName);

                ++inputInd;

                dumpedInputsPaths.push_back(blobFileName);

                in_tensors.emplace(inputInfo.get_any_name(), ov::Tensor(prec, shape, blob->buffer().as<void*>()));
            }

            std::cout << "Run inference on " << FLAGS_device << std::endl;

            const auto startTime = Time::now();
            const auto outInference = runInfer(compiledModel, in_tensors, dumpedInputsPaths);
            const auto endTime = Time::now();

            const TensorMap& outTensors = outInference.first;

            printPerformanceCountsAndLatency(numberOfTestCase, outInference.second, endTime - startTime);

            // NB: Make a view over ov::Tensor
            for (const auto& p : outTensors) {
                const auto shape = p.second.get_shape();
                ov::Layout outLayerUserLayout;
                if (outUserLayout.empty()) {
                    ov::Layout outLayerModelLayout;
                    if (outModelLayout.empty()) {
                        outLayerModelLayout = getLayoutByRank(shape.size());
                        std::cout << "WARNING: Casting output ov::Tensor to ie::Blob. Since --oml option isn't set, "
                                     "output model layout for layer \""
                                  << p.first << "\" is infered from shape: " << toString(shape) << " rank ("
                                  << shape.size() << ") as " << outLayerModelLayout.to_string() << std::endl;
                    } else {
                        outLayerModelLayout = outModelLayout;
                    }
                    outLayerUserLayout = outLayerModelLayout;
                } else {
                    outLayerUserLayout = outUserLayout;
                }

                outputs.emplace(p.first, toIE(p.second, outLayerUserLayout));
            }

            if (FLAGS_run_test) {
                ie::BlobMap refOutputs;
                size_t outputInd = 0;
                for (const auto& p : outputs) {
                    std::ostringstream ostr;
                    ostr << netFileName << "_ref_out_" << outputInd << "_case_" << numberOfTestCase << ".blob";
                    const auto blobFileName = ostr.str();
                    const auto& refTensorDesc = p.second->getTensorDesc();

                    std::cout << "Load reference output #" << outputInd << " from " << blobFileName << " as "
                              << refTensorDesc.getPrecision() << std::endl;

                    const auto blob = loadBlob(refTensorDesc, blobFileName);
                    refOutputs.emplace(p.first, blob);

                    ++outputInd;
                }

                outputInd = 0;
                for (const auto& p : outputs) {
                    std::ostringstream ostr;
                    ostr << netFileName << "_kmb_out_" << outputInd << "_case_" << numberOfTestCase << ".blob";
                    const auto blobFileName = ostr.str();

                    std::cout << "Dump device output #" << outputInd << "_case_" << numberOfTestCase << " to "
                              << blobFileName << std::endl;

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
                } else if (strEq(FLAGS_mode, "cosim")) {
                    if (testCoSim(outputs, refOutputs)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "ssd")) {
                    if (testSSDDetection(outputs, refOutputs, inDescMap)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "yolo_v2")) {
                    if (testYoloV2(outputs, refOutputs, inDescMap)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "yolo_v3")) {
                    if (testYoloV3(outputs, refOutputs, inDescMap)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "yolo_v4")) {
                    if (testYoloV4(outputs, refOutputs, inDescMap)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "psnr")) {
                    const auto shape = outTensors.begin()->second.get_shape();
                    const auto dstHeight = shape[2];
                    const auto dstWidth = shape[3];

                    if (testPSNR(outputs, refOutputs, dstHeight, dstWidth)) {
                        std::cout << "PASSED" << std::endl;
                    } else {
                        std::cout << "FAILED" << std::endl;
                        return EXIT_FAILURE;
                    }
                } else if (strEq(FLAGS_mode, "mean_iou")) {
                    if (testMeanIoU(outputs, refOutputs, inDescMap)) {
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
                for (const auto& p : outputs) {
                    std::ostringstream ostr;
                    ostr << netFileName << "_ref_out_" << outputInd << "_case_" << numberOfTestCase << ".blob";
                    const auto blobFileName = ostr.str();

                    std::cout << "Dump reference output #" << outputInd << " to " << blobFileName << std::endl;
                    dumpBlob(ie::as<ie::MemoryBlob>(p.second), blobFileName);

                    ++outputInd;
                }
            }
        }
    }  // try
    catch (const std::exception& ex) {
        std::cerr << "exception: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

//
// main
//

int main(int argc, char* argv[]) {
    parseCommandLine(argc, argv);
    if (FLAGS_ov_api_1_0) {
        return runSingleImageTestOV10();
    } else {
        return runSingleImageTestOV20();
    }
}
