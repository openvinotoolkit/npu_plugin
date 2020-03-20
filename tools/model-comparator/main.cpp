//
// Copyright (C) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include <ie_utils.hpp>

#include <inference_engine.hpp>
#include <blob_factory.hpp>
#include <precision_utils.h>

#include <ngraph/function.hpp>

#include <boost/filesystem.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <gflags/gflags.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>

namespace fs = boost::filesystem;

namespace {

//
// Command line options
//

DEFINE_string(network_name, "", "Network file name without extension");

DEFINE_string(input_dir, "", "Input directory");
DEFINE_string(output_dir, "", "Output directory");

DEFINE_string(input_file, "", "Input file");

DEFINE_string(ref_device, "CPU", "Reference device");
DEFINE_string(actual_device, "KMB", "Actual device");

DEFINE_string(black_list, "", "Black list");
DEFINE_string(white_list, "", "While list");

DEFINE_bool(run_compile, false, "Run compile");
DEFINE_bool(run_ref, false, "Run reference");
DEFINE_bool(run_infer, false, "Run infer");

DEFINE_bool(raw_export, false, "Raw export");

DEFINE_string(input_precision, "", "Input precision (optional)");
DEFINE_string(input_layout, "", "Input layout (optional)");
DEFINE_string(output_precision, "", "Output precision (optional)");
DEFINE_string(output_layout, "", "Output layout (optional)");

DEFINE_string(log_level, "", "Log level (optional)");

std::unordered_set<std::string> blackList;
std::unordered_set<std::string> whiteList;

bool checkFilter(const std::string& layerName, const std::string& layerType) {
    if (!blackList.empty()) {
        if (blackList.count(layerName) != 0 || blackList.count(layerType) != 0) {
            return false;
        }
    }

    if (!whiteList.empty()) {
        if (whiteList.count(layerName) == 0 && whiteList.count(layerType) == 0) {
            return false;
        }
    }

    return true;
}

std::unordered_set<std::string> splitStringList(const std::string& str, char delim) {
    std::unordered_set<std::string> out;

    if (str.empty()) {
        return out;
    }

    std::istringstream istr(str);

    std::string elem;
    while (std::getline(istr, elem, delim)) {
        if (elem.empty()) {
            continue;
        }

        out.emplace(std::move(elem));
    }

    return out;
}

void parseCommandLine(int argc, char* argv[]) {
    std::ostringstream usage;
    usage << "Usage: " << (*argv)[0] << "[<options>]";
    gflags::SetUsageMessage(usage.str());

    std::ostringstream version;
    version << ie::GetInferenceEngineVersion();
    gflags::SetVersionString(version.str());

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::cout << "Parameters:" << std::endl;
    std::cout << "    Network name: " << FLAGS_network_name << std::endl;
    std::cout << "    Input directory: " << FLAGS_input_dir << std::endl;
    std::cout << "    Output directory: " << FLAGS_output_dir << std::endl;
    std::cout << "    Input file: " << FLAGS_input_file << std::endl;
    std::cout << "    Reference device: " << FLAGS_ref_device << std::endl;
    std::cout << "    Actual device: " << FLAGS_actual_device << std::endl;
    std::cout << "    Black list: " << FLAGS_black_list << std::endl;
    std::cout << "    While list: " << FLAGS_white_list << std::endl;
    std::cout << "    Run compile:" << FLAGS_run_compile << std::endl;
    std::cout << "    Run ref:" << FLAGS_run_ref << std::endl;
    std::cout << "    Run infer:" << FLAGS_run_infer << std::endl;
    std::cout << "    Raw export: " << FLAGS_raw_export << std::endl;
    std::cout << "    Input precision: " << FLAGS_input_precision << std::endl;
    std::cout << "    Input layout: " << FLAGS_input_layout << std::endl;
    std::cout << "    Output precision: " << FLAGS_output_precision << std::endl;
    std::cout << "    Output layout: " << FLAGS_output_layout << std::endl;
    std::cout << "    Log level: " << FLAGS_log_level << std::endl;
    std::cout << std::endl;

    blackList = splitStringList(FLAGS_black_list, ',');
    whiteList = splitStringList(FLAGS_white_list, ',');
}

//
// Directories
//

fs::path inputBaseDir;
fs::path inputSubNetworksDir;
fs::path inputBlobsDir;

fs::path outputBaseDir;
fs::path outputSubNetworksDir;
fs::path outputBlobsDir;
fs::path outputReportDir;
fs::path outputReportLayersDir;

void setupDirectories() {
    IE_ASSERT(!FLAGS_input_dir.empty());
    inputBaseDir = FLAGS_input_dir;
    IE_ASSERT(fs::exists(inputBaseDir) && fs::is_directory(inputBaseDir));

    if (!FLAGS_run_compile) {
        inputSubNetworksDir = inputBaseDir / "sub-networks";
    }
    if (!FLAGS_run_ref) {
        inputBlobsDir = inputBaseDir / "blobs";
    }

    IE_ASSERT(!FLAGS_output_dir.empty());
    outputBaseDir = FLAGS_output_dir;
    fs::create_directories(outputBaseDir);

    if (FLAGS_run_compile) {
        outputSubNetworksDir = outputBaseDir / "sub-networks";
        fs::create_directories(outputSubNetworksDir);
    }

    if (FLAGS_run_ref && !FLAGS_run_infer) {
        outputBlobsDir = outputBaseDir / "blobs";
        fs::create_directories(outputBlobsDir);
    }

    if (FLAGS_run_infer) {
        outputReportDir = outputBaseDir / "report";
        fs::create_directories(outputReportDir);

        outputReportLayersDir = outputReportDir / "layers";
        fs::create_directories(outputReportLayersDir);
    }
}

//
// Utility
//

struct RangeFP32 final {
    float min;
    float max;
};

std::string cleanName(std::string name) {
    std::replace_if(name.begin(), name.end(), [](char c) { return !std::isalnum(c); }, '_');
    return name;
}

std::string getLayerFileBaseName(size_t layerInd, const std::string& layerName) {
    std::ostringstream strm;
    strm << std::setw(4) << std::setfill('0') << layerInd << "-" << cleanName(layerName);
    return strm.str();
}

//
// BMP diff map
//

float clamp(float val, RangeFP32 range = {0.0f, 1.0f}) {
    return std::max(range.min, std::min(range.max, val));
}

float colorMapRed(float x) {
    if (x < 0.7f) {
        return 4.0f * x - 1.5f;
    } else {
        return -4.0f * x + 4.5f;
    }
}

float colorMapGreen(float x) {
    if (x < 0.5f) {
        return 4.0f * x - 0.5f;
    } else {
        return -4.0f * x + 3.5f;
    }
}

float colorMapBlue(float x) {
    if (x < 0.3f) {
        return 4.0f * x + 0.5f;
    } else {
        return -4.0f * x + 2.5f;
    }
}

std::tuple<float, float, float> colorMap(float x) {
    return std::make_tuple(
        clamp(colorMapRed(x)) * 255.0f,
        clamp(colorMapGreen(x)) * 255.0f,
        clamp(colorMapBlue(x)) * 255.0f
    );
}

void dumpDiffMaps(const ie::Blob::Ptr& diffBlob, const std::string& baseName, std::ostream& html) {
    const auto& dims = diffBlob->getTensorDesc().getDims();
    if (dims.size() == 1) {
        std::cout << "    Only one dimension: " << dims[0] << std::endl;
        return;
    }

    const auto planeWidth = dims[dims.size() - 1];
    const auto planeHeight = dims[dims.size() - 2];
    const auto planeSize = planeWidth * planeHeight;
    const auto numPlanes = std::accumulate(dims.begin(), dims.end() - 2, size_t{1}, std::multiplies<size_t>{});
    const auto diffPtr = diffBlob->cbuffer().as<const float*>();

    if (planeSize < 25) {
        std::cout << "    Plane size is too small" << std::endl;
        return;
    } else if (planeWidth < 5 || planeHeight < 5 || planeWidth > 2048 || planeHeight > 2048) {
        std::cout << "    Dimensions are unapropriate to dump : [" << planeWidth << "x" << planeHeight << "]" << std::endl;
        return;
    } else {
        double ratio = static_cast<double>(planeWidth) / static_cast<double>(planeHeight);
        if (ratio < 1.0) {
            ratio = 1.0 / ratio;
        }

        if (ratio > 8.0) {
            std::cout << "    Suspicious aspect ratio : " << ratio << std::endl;
            return;
        }
    }

    html << "         <h2>Difference Maps</h2>" << std::endl;

    for (size_t planeInd = 0; planeInd < numPlanes; planeInd++) {
        cv::Mat planeDiffImg(static_cast<int>(planeHeight), static_cast<int>(planeWidth), CV_8UC3);

        auto lMinDiff = std::numeric_limits<float>::max();
        auto lMaxDiff = std::numeric_limits<float>::lowest();

        for (size_t i = 0; i < planeSize; ++i) {
            const auto val = diffPtr[planeInd * planeSize + i];
            lMaxDiff = std::max(lMaxDiff, val);
            lMinDiff = std::min(lMinDiff, val);
        }

        const auto mult = 1.0f / std::max(lMaxDiff - lMinDiff, std::numeric_limits<float>::epsilon());
        const auto base = lMinDiff;

        for (size_t i = 0; i < planeSize; ++i) {
            const auto diffVal = diffPtr[planeInd * planeSize + i];

            const auto pixelColor = mult * (diffVal - base);
            const auto rgb = colorMap(pixelColor);

            auto& bgr = planeDiffImg.at<cv::Vec3b>(static_cast<int>(i));
            bgr[0] = cv::saturate_cast<uchar>(std::get<2>(rgb));
            bgr[1] = cv::saturate_cast<uchar>(std::get<1>(rgb));
            bgr[2] = cv::saturate_cast<uchar>(std::get<0>(rgb));
        }

        const auto planeDiffFileName = baseName + "_" + std::to_string(planeInd) + ".bmp";
        const auto planeDiffFilePath = outputReportLayersDir / planeDiffFileName;
        cv::imwrite(planeDiffFilePath.string(), planeDiffImg);

        const int htmlWidth = 640;
        const int htmlHeight = static_cast<int>(static_cast<double>(planeHeight) / planeWidth * htmlWidth);

        html << "<h3>Plane #" << planeInd << "</h3>" << std::endl;
        html << "<p>Min: " << lMinDiff << ", Max: " << lMaxDiff << "</p>" << std::endl;
        html << "<img src=\"" << planeDiffFileName << "\" width=\"" << htmlWidth << "\" height=\"" << htmlHeight << "\">" << std::endl;
    }
}

//
// Layer comparison
//

const RangeFP32 initialRange = {std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()};

void printStats(const RangeFP32& range, std::ofstream& html) {
    html << "<ul>" << std::endl;

    html << "<li>Min: " << range.min << "</li>" << std::endl;
    html << "<li>Max: " << range.max << "</li>" << std::endl;

    html << "</ul>" << std::endl;
}

struct DiffStats final {
    RangeFP32 abs = initialRange;
    RangeFP32 rel = initialRange;
};

void printStats(const DiffStats& stats, std::ofstream& html) {
    html << "<ul>" << std::endl;

    html << "<li>Absolute:" << std::endl;
    printStats(stats.abs, html);
    html << "</li>" << std::endl;

    html << "<li>Relative:" << std::endl;
    printStats(stats.rel, html);
    html << "</li>" << std::endl;

    html << "</ul>" << std::endl;
}

struct ValueStats final {
    size_t numNans = 0;
    size_t numInfs = 0;

    RangeFP32 range = initialRange;
    RangeFP32 absRange = initialRange;
};

void printStats(const ValueStats& stats, std::ofstream& html) {
    html << "<ul>" << std::endl;

    html << "<li>Number of NaNs: " << stats.numNans << "</li>" << std::endl;
    html << "<li>Number of Infs: " << stats.numInfs << "</li>" << std::endl;

    html << "<li>Values range:" << std::endl;
    printStats(stats.range, html);
    html << "</li>" << std::endl;

    html << "<li>Abs values range:" << std::endl;
    printStats(stats.absRange, html);
    html << "</li>" << std::endl;

    html << "</ul>" << std::endl;
}

struct LayerStats final {
    ValueStats ref;
    ValueStats actual;
    DiffStats diff;
};

void printStats(const LayerStats& stats, std::ofstream& html) {
    html << "<ul>" << std::endl;

    html << "<li>Reference:" << std::endl;
    printStats(stats.ref, html);
    html << "</li>" << std::endl;

    html << "<li>Actual:" << std::endl;
    printStats(stats.actual, html);
    html << "</li>" << std::endl;

    html << "<li>Difference:" << std::endl;
    printStats(stats.diff, html);
    html << "</li>" << std::endl;

    html << "</ul>" << std::endl;
}

std::tuple<LayerStats, ie::Blob::Ptr> compare(const ie::Blob::Ptr& ref, const ie::Blob::Ptr& actual) {
    IE_ASSERT(ref->getTensorDesc().getPrecision() == ie::Precision::FP32);
    IE_ASSERT(actual->getTensorDesc().getPrecision() == ie::Precision::FP32);

    const auto& dims = ref->getTensorDesc().getDims();
    IE_ASSERT(actual->getTensorDesc().getDims() == dims);

    const auto defLayout = ie::TensorDesc::getLayoutByDims(dims);

    IE_ASSERT(ref->getTensorDesc().getLayout() == defLayout);
    IE_ASSERT(actual->getTensorDesc().getLayout() == defLayout);

    const auto diffBlob = make_blob_with_precision(ie::TensorDesc(ie::Precision::FP32, dims, defLayout));
    diffBlob->allocate();

    const auto refPtr = ref->cbuffer().as<const float*>();
    const auto actualPtr = actual->cbuffer().as<const float*>();
    const auto diffPtr = diffBlob->buffer().as<float*>();

    LayerStats stats;

    for (size_t i = 0; i < ref->size(); ++i) {
        const auto refVal = refPtr[i];
        const auto actualVal = actualPtr[i];

        if (std::isnan(refVal)) {
            ++stats.ref.numNans;
        } else if (std::isinf(refVal)) {
            ++stats.ref.numInfs;
        } else {
            stats.ref.range.min = std::min(stats.ref.range.min, refVal);
            stats.ref.range.max = std::max(stats.ref.range.max, refVal);

            stats.ref.absRange.min = std::min(stats.ref.absRange.min, std::fabs(refVal));
            stats.ref.absRange.max = std::max(stats.ref.absRange.max, std::fabs(refVal));
        }

        if (std::isnan(actualVal)) {
            ++stats.actual.numNans;
        } else if (std::isinf(actualVal)) {
            ++stats.actual.numInfs;
        } else {
            stats.actual.range.min = std::min(stats.actual.range.min, actualVal);
            stats.actual.range.max = std::max(stats.actual.range.max, actualVal);

            stats.actual.absRange.min = std::min(stats.actual.absRange.min, std::fabs(actualVal));
            stats.actual.absRange.max = std::max(stats.actual.absRange.max, std::fabs(actualVal));
        }

        const auto absDiff = std::fabs(actualVal - refVal);

        const auto maxAbsVal = std::max(std::fabs(refVal), std::fabs(actualVal));
        const auto relDiff = absDiff / std::max(maxAbsVal, std::numeric_limits<float>::epsilon());

        diffPtr[i] = absDiff;

        stats.diff.abs.min = std::min(stats.diff.abs.min, absDiff);
        stats.diff.abs.max = std::max(stats.diff.abs.max, absDiff);

        stats.diff.rel.min = std::min(stats.diff.rel.min, relDiff);
        stats.diff.rel.max = std::max(stats.diff.rel.max, relDiff);
    }

    return std::make_tuple(stats, diffBlob);
}

//
// Inference Engine
//

ie::Core ieCore;

void setupInferenceEngine() {
    if (!FLAGS_log_level.empty()) {
        ieCore.SetConfig({{CONFIG_KEY(LOG_LEVEL), FLAGS_log_level}}, FLAGS_actual_device);
    }
}

ie::Layout layoutFromStr(const std::string& str) {
    static std::unordered_map<std::string, ie::Layout> map {
        {"NCHW",  ie::Layout::NCHW},
        {"NHWC",  ie::Layout::NHWC},
        {"NCDHW", ie::Layout::NCDHW},
        {"NDHWC", ie::Layout::NDHWC},
    };

    return map.at(str);
}

void setInputOutputInfo(ie::CNNNetwork& net) {
    for (const auto& inputInfo : net.getInputsInfo()) {
        const auto& input = inputInfo.second;

        if (!FLAGS_input_precision.empty() && input->getPrecision() == ie::Precision::FP32) {
            input->setPrecision(ie::Precision::FromStr(FLAGS_input_precision));
        }

        if (!FLAGS_input_layout.empty() && input->getTensorDesc().getDims().size() == FLAGS_input_layout.size()) {
            input->setLayout(layoutFromStr(FLAGS_input_layout));
        }
    }

    for (const auto& outputInfo : net.getOutputsInfo()) {
        const auto& output = outputInfo.second;

        if (!FLAGS_output_precision.empty() && output->getPrecision() == ie::Precision::FP32) {
            output->setPrecision(ie::Precision::FromStr(FLAGS_output_precision));
        }

        if (!FLAGS_output_layout.empty() && output->getTensorDesc().getDims().size() == FLAGS_output_layout.size()) {
            output->setLayout(layoutFromStr(FLAGS_output_layout));
        }
    }
}

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

    IE_ASSERT(layout == ie::Layout::NHWC || layout == ie::Layout::NCHW);

    const auto N = dims[0];
    const auto C = dims[1];
    const auto H = dims[2];
    const auto W = dims[3];

    IE_ASSERT(C == 3 || C == 4);

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
                  precision == ie::Precision::FP16);
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

ie::Blob::Ptr loadInput(const ie::TensorDesc& desc) {
    IE_ASSERT(isImage(desc));

    const auto frame = cv::imread(FLAGS_input_file, cv::IMREAD_COLOR);
    IE_ASSERT(!frame.empty());

    const auto blob = make_blob_with_precision(desc);
    blob->allocate();

    cvToIe(frame, blob);

    return blob;
}

ie::BlobMap runInfer(ie::ExecutableNetwork& exeNet, const ie::BlobMap& inputs) {
    auto inferRequest = exeNet.CreateInferRequest();

    // TODO: remove this code once KMB plugin fixed import/export missing information
    const auto inputsInfo = exeNet.GetInputsInfo();
    if (inputsInfo.size() == 1) {
        IE_ASSERT(inputs.size() == 1);
        const auto& inputName = inputsInfo.begin()->first;
        const auto& inputBlob = inputs.begin()->second;
        inferRequest.SetBlob(inputName, inputBlob);
    } else {
        for (const auto& p : inputs) {
            inferRequest.SetBlob(p.first, p.second);
        }
    }

    inferRequest.Infer();

    ie::BlobMap out;

    for (const auto& p : exeNet.GetOutputsInfo()) {
        out.insert({p.first, inferRequest.GetBlob(p.first)});
    }

    return out;
}

void serializeNetwork(ie::CNNNetwork& net, const std::string& baseFileName) {
    const auto xmlPath = outputSubNetworksDir / (baseFileName + ".xml");
    const auto binPath = outputSubNetworksDir / (baseFileName + ".bin");

    net.serialize(xmlPath.string(), binPath.string());
}

void exportNetwork(ie::ExecutableNetwork& exeNet, const std::string& fileName) {
    const auto filePath = outputSubNetworksDir / fileName;

    if (FLAGS_raw_export) {
        exeNet.Export(filePath.string());
    } else {
        std::ofstream file(filePath.string(), std::ios_base::out | std::ios_base::binary);
        IE_ASSERT(file.is_open());

        exeNet.Export(file);
    }
}

ie::ExecutableNetwork importNetwork(const std::string& fileName) {
    const auto filePath = inputSubNetworksDir / fileName;

    if (FLAGS_raw_export) {
        return ieCore.ImportNetwork(filePath.string(), FLAGS_actual_device);
    } else {
        std::ifstream file(filePath.string(), std::ios_base::in | std::ios_base::binary);
        IE_ASSERT(file.is_open());

        return ieCore.ImportNetwork(file, FLAGS_actual_device);
    }
}

void dumpBlob(const ie::Blob::Ptr& blob, const std::string& fileName) {
    const auto filePath = outputBlobsDir / fileName;

    std::ofstream file(filePath.string(), std::ios_base::out | std::ios_base::binary);
    IE_ASSERT(file.is_open());

    file.write(blob->cbuffer().as<const char*>(), static_cast<std::streamsize>(blob->byteSize()));
}

ie::Blob::Ptr importBlob(const ie::TensorDesc& desc, const std::string& fileName) {
    const auto filePath = inputBlobsDir / fileName;

    const auto blob = make_blob_with_precision(desc);
    blob->allocate();

    std::ifstream file(filePath.string(), std::ios_base::in | std::ios_base::binary);
    IE_ASSERT(file.is_open());

    file.read(blob->buffer().as<char*>(), static_cast<std::streamsize>(blob->byteSize()));

    return blob;
}

std::shared_ptr<ngraph::Function> buildFunc(
        const std::string& name, const ngraph::ParameterVector& baseParameters, const ngraph::NodeVector& baseNodes) {
    IE_ASSERT(!baseNodes.empty());

    std::unordered_map<std::shared_ptr<ngraph::Node>, std::shared_ptr<ngraph::Node>> nodeMap;

    ngraph::ParameterVector newParameters;
    newParameters.reserve(baseParameters.size());

    for (const auto& baseParam : baseParameters) {
        const auto newParam = std::static_pointer_cast<ngraph::op::Parameter>(baseParam->copy_with_new_inputs({}));
        newParam->set_friendly_name(baseParam->get_friendly_name());

        newParameters.push_back(newParam);
        nodeMap.insert({baseParam, newParam});
    }

    ngraph::NodeVector newNodes;
    newNodes.reserve(baseNodes.size());

    for (const auto& baseNode : baseNodes) {
        ngraph::OutputVector newInputs;
        newInputs.reserve(baseNode->get_input_size());

        for (const auto& baseInput : baseNode->inputs()) {
            const auto baseSourceOutput = baseInput.get_source_output();
            const auto newSourceNode = nodeMap.at(baseSourceOutput.get_node_shared_ptr());
            newInputs.push_back(newSourceNode->output(baseSourceOutput.get_index()));
        }

        const auto newNode = baseNode->copy_with_new_inputs(newInputs);
        newNode->set_friendly_name(baseNode->get_friendly_name());

        newNodes.push_back(newNode);
        nodeMap.insert({baseNode, newNode});
    }

    const auto lastNode = newNodes.back();

    ngraph::ResultVector newResults;
    newResults.reserve(lastNode->get_output_size());

    for (const auto& output : lastNode->outputs()) {
        const auto newResult = std::make_shared<ngraph::op::Result>(output);

        if (lastNode->get_output_size() == 1) {
            newResult->set_friendly_name(lastNode->get_friendly_name());
        } else {
            newResult->set_friendly_name(lastNode->get_friendly_name() + "_" + std::to_string(output.get_index()));
        }

        newResults.push_back(std::make_shared<ngraph::op::Result>(output));
    }

    return std::make_shared<ngraph::Function>(newResults, newParameters, name);
}

std::ostream& operator<<(std::ostream& os, const ie::SizeVector& sz) {
    os << "[";

    size_t ind = 0;

    for (auto s : sz) {
        if (ind > 0) {
            os << ", ";
        }

        os << s;

        ++ind;
    }

    os << "]";

    return os;
}

}  // namespace

//
// Main
//

int main(int argc, char* argv[]) {
    parseCommandLine(argc, argv);

    setupDirectories();
    setupInferenceEngine();

    const auto baseNetFileName = FLAGS_network_name + ".xml";
    const auto baseBinFileName = FLAGS_network_name + ".bin";

    const auto baseNetFilePath = inputBaseDir / baseNetFileName;
    const auto baseBinFilePath = inputBaseDir / baseBinFileName;

    if (outputBaseDir != inputBaseDir) {
        fs::copy_file(baseNetFilePath, outputBaseDir / baseNetFileName, fs::copy_option::overwrite_if_exists);
        fs::copy_file(baseBinFilePath, outputBaseDir / baseBinFileName, fs::copy_option::overwrite_if_exists);
    }

    std::cout << "Load base network " << baseNetFileName << std::endl;
    auto baseNet = ieCore.ReadNetwork(baseNetFilePath.string());
    setInputOutputInfo(baseNet);
    std::cout << std::endl;

    const auto baseInputInfo = baseNet.getInputsInfo();
    const auto baseOutputInfo = baseNet.getOutputsInfo();

    ie::BlobMap inputs;
    if (FLAGS_run_ref || FLAGS_run_infer) {
        // TODO: support multiple inputs

        std::cout << "Load input file " << FLAGS_input_file << std::endl;
        IE_ASSERT(baseInputInfo.size() == 1);
        const auto inputName = baseInputInfo.begin()->first;
        const auto inputDesc = baseInputInfo.begin()->second->getTensorDesc();
        const auto inputBlob = loadInput(inputDesc);
        std::cout << std::endl;

        inputs.emplace(inputName, inputBlob);
    }

    std::ofstream htmlNet;
    fs::path htmlLayersBaseFilePath;

    if (FLAGS_run_infer) {
        const auto htmlNetFilePath = outputReportDir / (FLAGS_network_name + ".html");
        htmlNet.open(htmlNetFilePath.string());
        IE_ASSERT(htmlNet.is_open());

        htmlNet << "<html>" << std::endl;
        htmlNet << "    <head>" << std::endl;
        htmlNet << "        <title>" << FLAGS_network_name << "</title>" << std::endl;
        htmlNet << "    </head>" << std::endl;
        htmlNet << "    <body>" << std::endl;

        htmlNet << "         <h1>" << FLAGS_network_name << "</h1>" << std::endl;

        htmlNet << "         <h2>Network inputs</h2>" << std::endl;
        htmlNet << "         <table border=\"1\">" << std::endl;
        htmlNet << "             <tr><th>Name</th><th>Precision</th><th>Dims</th><th>Layout</th></tr>" << std::endl;
        for (const auto& p : baseInputInfo) {
            const auto& desc = p.second->getTensorDesc();

            htmlNet << "             <tr>" << std::endl;
            htmlNet << "                 <td>" << p.first << "</td>" << std::endl;
            htmlNet << "                 <td>" << desc.getPrecision() << "</td>" << std::endl;
            htmlNet << "                 <td>" << desc.getDims() << "</td>" << std::endl;
            htmlNet << "                 <td>" << desc.getLayout() << "</td>" << std::endl;
            htmlNet << "             </tr>" << std::endl;
        }
        htmlNet << "         </table>" << std::endl;

        htmlNet << "         <h2>Network outputs</h2>" << std::endl;
        htmlNet << "         <table border=\"1\">" << std::endl;
        htmlNet << "             <tr><th>Name</th><th>Precision</th><th>Dims</th><th>Layout</th></tr>" << std::endl;
        for (const auto& p : baseOutputInfo) {
            const auto& desc = p.second->getTensorDesc();

            htmlNet << "             <tr>" << std::endl;
            htmlNet << "                 <td>" << p.first << "</td>" << std::endl;
            htmlNet << "                 <td>" << desc.getPrecision() << "</td>" << std::endl;
            htmlNet << "                 <td>" << desc.getDims() << "</td>" << std::endl;
            htmlNet << "                 <td>" << desc.getLayout() << "</td>" << std::endl;
            htmlNet << "             </tr>" << std::endl;
        }
        htmlNet << "         </table>" << std::endl;

        htmlNet << "         <h2>Network layers</h2>" << std::endl;
        htmlNet << "         <table border=\"1\">" << std::endl;
        htmlNet << "             <tr><th>Name</th><th>Max Abs Diff</th><th>Max Rel Diff</th><th>Results</th></tr>" << std::endl;
    }

    const auto baseFunc = baseNet.getFunction();
    IE_ASSERT(baseFunc != nullptr);

    const auto& baseParameters = baseFunc->get_parameters();

    ngraph::NodeVector baseNodes;

    for (const auto& baseNode : baseFunc->get_ordered_ops()) {
        if (baseNode->is_parameter() || baseNode->is_output()) {
            continue;
        }

        baseNodes.push_back(baseNode);

        const auto& layerName = baseNode->get_friendly_name();

        if (!checkFilter(layerName, baseNode->get_type_name())) {
            continue;
        }

        // TODO: support multiple outputs
        if (baseNode->get_output_size() != 1) {
            continue;
        }

        std::cout << "Build sub-network up to layer " << layerName << std::endl;

        const auto layerBaseName = getLayerFileBaseName(baseNodes.size() - 1, baseNode->get_friendly_name());
        const auto layerNetName = FLAGS_network_name + "_" + layerBaseName;

        const auto layerFunc = buildFunc(layerNetName, baseParameters, baseNodes);

        try {
            auto layerNet = ie::CNNNetwork(layerFunc);
            serializeNetwork(layerNet, layerNetName);

            setInputOutputInfo(layerNet);

            ie::ExecutableNetwork actualExeNet;
            if (FLAGS_run_compile) {
                std::cout << "    Compile sub-network for " << FLAGS_actual_device << std::endl;

                actualExeNet = ieCore.LoadNetwork(layerNet, FLAGS_actual_device);

                if (!FLAGS_run_infer)
                {
                    exportNetwork(actualExeNet, layerNetName + ".compiled");
                }
            } else if (FLAGS_run_infer) {
                std::cout << "    Import sub-network for " << FLAGS_actual_device << std::endl;

                actualExeNet = importNetwork(layerNetName + ".compiled");
            }

            ie::BlobMap refOutputs;
            if (FLAGS_run_ref) {
                std::cout << "    Calc reference with " << FLAGS_ref_device << std::endl;

                std::map<std::string, std::string> refConfig;

                if (FLAGS_ref_device == "CPU") {
                    refConfig.emplace("LP_TRANSFORMS_MODE", CONFIG_VALUE(NO));
                }

                auto refExeNet = ieCore.LoadNetwork(layerNet, FLAGS_ref_device, refConfig);
                refOutputs = runInfer(refExeNet, inputs);

                if (!FLAGS_run_infer) {
                    for (const auto& p : refOutputs) {
                        dumpBlob(p.second, cleanName(p.first) + ".blob");
                    }
                }
            } else if (FLAGS_run_infer) {
                std::cout << "    Import reference" << std::endl;

                for (const auto& p : layerNet.getOutputsInfo()) {
                    const auto blob = importBlob(p.second->getTensorDesc(), cleanName(p.first) + ".blob");
                    refOutputs.insert({p.first, blob});
                }
            }

            if (FLAGS_run_infer) {
                std::cout << "    Run infer on " << FLAGS_actual_device << std::endl;
                const auto actualOutputs = runInfer(actualExeNet, inputs);

                const auto htmlLayerFileName = layerBaseName + ".html";
                const auto htmlLayerFilePath = outputReportLayersDir / htmlLayerFileName;

                std::ofstream htmlLayer(htmlLayerFilePath.string());
                IE_ASSERT(htmlLayer.is_open());

                htmlLayer << "<html>" << std::endl;
                htmlLayer << "    <head>" << std::endl;
                htmlLayer << "        <title>Layer " << layerName << "</title>" << std::endl;
                htmlLayer << "    </head>" << std::endl;
                htmlLayer << "    <body>" << std::endl;
                htmlLayer << "         <h1>Layer " << layerName << "</h1>" << std::endl;

                std::cout << "    Compare with reference" << std::endl;

                // TODO: support multiple outputs
                IE_ASSERT(refOutputs.size() == 1);
                IE_ASSERT(actualOutputs.size() == 1);

                const auto refOutput = toDefLayout(toPrecision(refOutputs.begin()->second, ie::Precision::FP32));
                const auto actualOutput = toDefLayout(toPrecision(actualOutputs.begin()->second, ie::Precision::FP32));

                LayerStats stats;
                ie::Blob::Ptr diffBlob;
                std::tie(stats, diffBlob) = compare(refOutput, actualOutput);

                htmlLayer << "         <h2>Statistics</h2>" << std::endl;
                printStats(stats, htmlLayer);

                dumpDiffMaps(diffBlob, "Output 0", htmlLayer);

                htmlLayer << "    </body>" << std::endl;
                htmlLayer << "</html>" << std::endl;

                htmlNet << "             <tr>" << std::endl;
                htmlNet << "                 <td>" << layerName << "</td>" << std::endl;
                htmlNet << "                 <td>" << stats.diff.abs.max << "</td>" << std::endl;
                htmlNet << "                 <td>" << stats.diff.rel.max << "</td>" << std::endl;
                htmlNet << "                 <td><a href=\"layers/" << htmlLayerFileName << "\">RESULTS</a></td>" << std::endl;
                htmlNet << "             </tr>" << std::endl;
            }
        } catch (const std::exception& err) {
            std::cerr << "    Failed to cut network on layer " << layerName << std::endl;
            std::cerr << "    " << err.what() << std::endl;

            if (FLAGS_run_infer) {
                htmlNet << "             <tr>" << std::endl;
                htmlNet << "                 <td>" << layerName << "</td>" << std::endl;
                htmlNet << "                 <td>NONE</td>" << std::endl;
                htmlNet << "                 <td>NONE</td>" << std::endl;
                htmlNet << "                 <td><span style=\"color:red;font-weight:bold\">FATAL ERROR</span></td>" << std::endl;
                htmlNet << "             </tr>" << std::endl;
            }
        }

        std::cout << std::endl;
    }

    if (FLAGS_run_ref) {
        htmlNet << "         </table>" << std::endl;
        htmlNet << "    </body>" << std::endl;
        htmlNet << "</html>" << std::endl;
    }

    return EXIT_SUCCESS;
}
