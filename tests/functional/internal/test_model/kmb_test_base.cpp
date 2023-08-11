//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "kmb_test_base.hpp"

#include <format_reader_ptr.h>
#include <blob_factory.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "common/functions.h"
#include "functional_test_utils/plugin_cache.hpp"
#include "vpux/utils/IE/config.hpp"
#include "vpux/vpux_metrics.hpp"
#include "vpux_private_config.hpp"

//
// KmbTestBase
//

namespace {

std::string cleanName(std::string name) {
    std::replace_if(
            name.begin(), name.end(),
            [](char c) {
                return !std::isalnum(c);
            },
            '_');
    return name;
}

}  // namespace

const std::string KmbTestBase::DEVICE_NAME = []() -> std::string {
    if (const auto var = std::getenv("IE_KMB_TESTS_DEVICE_NAME")) {
        return var;
    }

    return "VPUX";
}();

const std::string KmbTestBase::REF_DEVICE_NAME = []() -> std::string {
    if (const auto var = std::getenv("IE_KMB_TESTS_REF_DEVICE_NAME")) {
        return var;
    }

    return "CPU";
}();

const bool KmbTestBase::RUN_COMPILER = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_RUN_COMPILER")) {
        return vpux::envVarStrToBool("IE_KMB_TESTS_RUN_COMPILER", var);
    }

    if (KmbTestBase::DEVICE_NAME == "CPU") {
        return true;
    }

#if defined(__aarch64__)
    return false;
#else
    return true;
#endif
}();

const bool KmbTestBase::RUN_REF_CODE = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_RUN_REF_CODE")) {
        return vpux::envVarStrToBool("IE_KMB_TESTS_RUN_REF_CODE", var);
    }

#ifdef __aarch64__
    return false;
#else
    return true;
#endif
}();

const std::string KmbTestBase::DUMP_PATH = []() -> std::string {
    if (const auto var = std::getenv("IE_KMB_TESTS_DUMP_PATH")) {
        return var;
    }

    return std::string();
}();

const bool KmbTestBase::EXPORT_NETWORK = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_EXPORT_NETWORK")) {
        return vpux::envVarStrToBool("IE_KMB_TESTS_EXPORT_NETWORK", var);
    }

    if (KmbTestBase::DEVICE_NAME == "CPU") {
        return false;
    }

    return KmbTestBase::RUN_COMPILER && !KmbTestBase::DUMP_PATH.empty();
}();

const bool KmbTestBase::RAW_EXPORT = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_RAW_EXPORT")) {
        return vpux::envVarStrToBool("IE_KMB_TESTS_RAW_EXPORT", var);
    }

    if (KmbTestBase::DEVICE_NAME != "VPUX" || !KmbTestBase::EXPORT_NETWORK) {
        return false;
    }

    return false;
}();

const bool KmbTestBase::GENERATE_BLOBS = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_GENERATE_BLOBS")) {
        return vpux::envVarStrToBool("IE_KMB_TESTS_GENERATE_BLOBS", var);
    }

    return KmbTestBase::RUN_REF_CODE;
}();

const bool KmbTestBase::EXPORT_BLOBS = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_EXPORT_BLOBS")) {
        return vpux::envVarStrToBool("IE_KMB_TESTS_EXPORT_BLOBS", var);
    }

    return KmbTestBase::GENERATE_BLOBS && !KmbTestBase::DUMP_PATH.empty();
}();

const std::string KmbTestBase::LOG_LEVEL = []() -> std::string {
    if (const auto var = std::getenv("IE_KMB_TESTS_LOG_LEVEL")) {
        return var;
    }

    return std::string();
}();

const bool KmbTestBase::PRINT_PERF_COUNTERS = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_PRINT_PERF_COUNTERS")) {
        return vpux::envVarStrToBool("IE_KMB_TESTS_PRINT_PERF_COUNTERS", var);
    }

    return false;
}();

void KmbTestBase::SetUp() {
    ASSERT_NO_FATAL_FAILURE(CommonTestUtils::TestsCommon::SetUp());

    const auto testInfo = testing::UnitTest::GetInstance()->current_test_info();
    IE_ASSERT(testInfo != nullptr);

    rd.seed();

    core = PluginCache::get().ie();
    RUN_INFER = [this]() -> bool {
        if (const auto var = std::getenv("IE_KMB_TESTS_RUN_INFER")) {
            return vpux::envVarStrToBool("IE_KMB_TESTS_RUN_INFER", var);
        }

        const auto devices = core->GetAvailableDevices();
        const auto isVPUXDeviceAvailable =
                std::find_if(devices.cbegin(), devices.cend(), [](const std::string& device) {
                    return device.find("VPUX") != std::string::npos;
                }) != devices.cend();

        return isVPUXDeviceAvailable;
    }();

    BACKEND_NAME = [this]() -> std::string {
        const auto backendName = core->GetMetric("VPUX", VPUX_METRIC_KEY(BACKEND_NAME)).as<std::string>();
        return backendName;
    }();

    const std::string configDevice = DEVICE_NAME.substr(0, DEVICE_NAME.find("."));
    if (!LOG_LEVEL.empty()) {
        core->SetConfig({{CONFIG_KEY(LOG_LEVEL), LOG_LEVEL}}, configDevice);
    }
    if (PRINT_PERF_COUNTERS) {
        core->SetConfig({{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)}}, configDevice);
    }

    if ((RUN_REF_CODE && REF_DEVICE_NAME == "CPU") || ((RUN_COMPILER || RUN_INFER) && DEVICE_NAME == "CPU")) {
        core->SetConfig({{"LP_TRANSFORMS_MODE", CONFIG_VALUE(NO)}}, "CPU");
    }

    dumpBaseName = cleanName(vpux::printToString("{0}_{1}", testInfo->test_case_name(), testInfo->name()));

    if (const auto typeParam = testInfo->type_param()) {
        std::cout << "[ PARAMS   ] " << typeParam << std::endl;
    }
    if (const auto valueParam = testInfo->value_param()) {
        std::cout << "[ PARAMS   ] " << valueParam << std::endl;
    }

    std::cout << "[ MODE     ] " << DEVICE_NAME << " / " << (RUN_INFER ? "RUN INFER AND CHECK" : "NO INFER") << " / "
              << (RUN_COMPILER ? "COMPILE NETWORK" : "IMPORT BLOB") << " / "
              << (RUN_REF_CODE ? "CALC REF" : "IMPORT REF") << std::endl;
    if (!DUMP_PATH.empty()) {
        std::cout << "[ DUMP PATH] " << DUMP_PATH << std::endl;
    }
}

void KmbTestBase::TearDown() {
#ifdef __aarch64__
    if (RUN_INFER) {
        core.reset();
        // FIXME: reset cache every time to destroy VpualDispatcherResource
        // this workaround is required to free VPU device properly
        // Track number: H#18013110883
        PluginCache::get().reset();
    }
#endif

    ASSERT_NO_FATAL_FAILURE(TestsCommon::TearDown());
}

Blob::Ptr KmbTestBase::getBlobByName(const std::string& blobName) {
    const auto it = blobs.find(blobName);
    if (it != blobs.end()) {
        return it->second;
    }

    const auto blobDesc = blobGenerators.at(blobName).first;

    Blob::Ptr blob;

    if (GENERATE_BLOBS) {
        std::cout << "=== GENERATE BLOB " << blobName << std::endl;

        blob = blobGenerators.at(blobName).second(blobDesc);
        IE_ASSERT(blob->getTensorDesc() == blobDesc);

        if (EXPORT_BLOBS) {
            std::cout << "    === EXPORT BLOB " << blobName << std::endl;

            dumpBlob(blobName, blob);
        }
    } else if (RUN_INFER) {
        std::cout << "=== IMPORT BLOB " << blobName << std::endl;

        blob = importBlob(blobName, blobDesc);
    }

    blobs.insert({blobName, blob});

    return blob;
}

ExecutableNetwork KmbTestBase::getExecNetwork(const std::function<CNNNetwork()>& netCreator,
                                              const std::function<CompileConfig()>& configCreator,
                                              const bool forceCompilation) {
    ExecutableNetwork exeNet;

    if (RUN_COMPILER || forceCompilation) {
        std::cout << "=== COMPILE NETWORK" << std::endl;

        auto config = configCreator();
        config[VPUX_CONFIG_KEY(PLATFORM)] = PlatformEnvironment::PLATFORM;
        if (config.find(VPUX_CONFIG_KEY(COMPILER_TYPE)) == config.end())
            config[VPUX_CONFIG_KEY(COMPILER_TYPE)] = VPUX_CONFIG_VALUE(MLIR);

        std::ostringstream ostr;
        ostr << "LoadNetwork Config: ";
        for (const auto& item : config) {
            ostr << item.first << "=" << item.second << "; ";
        }
        std::cout << ostr.str() << std::endl;

        exeNet = core->LoadNetwork(netCreator(), DEVICE_NAME, config);

        if (EXPORT_NETWORK) {
            std::cout << "    === EXPORT NETWORK" << std::endl;

            exportNetwork(exeNet);
        }
    } else if (RUN_INFER) {
        std::cout << "=== IMPORT NETWORK" << std::endl;

        std::map<std::string, std::string> importConfig;
        importConfig[VPUX_CONFIG_KEY(COMPILER_TYPE)] = VPUX_CONFIG_VALUE(MLIR);
        exeNet = importNetwork(importConfig);
    }

    return exeNet;
}

void KmbTestBase::compareOutputs(const Blob::Ptr& refOutput, const Blob::Ptr& actualOutput, const float tolerance,
                                 const CompareMethod method) {
    const auto& refDesc = refOutput->getTensorDesc();
    const auto& actualDesc = actualOutput->getTensorDesc();

    ASSERT_EQ(refDesc.getDims(), actualDesc.getDims());

    const auto refFP32 = vpux::toFP32(vpux::toDefLayout(as<MemoryBlob>(refOutput)));
    const auto actualFP32 = vpux::toFP32(vpux::toDefLayout(as<MemoryBlob>(actualOutput)));

    {
        auto refMem = refFP32->cbuffer();
        auto actualMem = actualFP32->cbuffer();

        const auto refPtr = refMem.as<const float*>();
        const auto actualPtr = actualMem.as<const float*>();

        const auto printCount = std::min<size_t>(refOutput->size(), 10);

        for (size_t i = 0; i < printCount; ++i) {
            const auto refVal = refPtr[i];
            const auto actualVal = actualPtr[i];

            const auto absdiff = std::fabs(refVal - actualVal);

            std::cout << "        " << i << " :"
                      << " ref : " << std::setw(10) << refVal << " actual : " << std::setw(10) << actualVal
                      << " absdiff : " << std::setw(10) << absdiff << std::endl;
        }
    }

    EXPECT_NO_FATAL_FAILURE(compareBlobs(actualFP32, refFP32, tolerance, method));
}

void KmbTestBase::compareWithReference(const BlobMap& actualOutputs, const BlobMap& refOutputs, const float tolerance,
                                       const CompareMethod method) {
    if (refOutputs.size() == 1) {
        // HACK: It's necessary for back compatibility when blob names were lost after export
        // [Track number: S#35709]

        ASSERT_EQ(actualOutputs.size(), 1);

        const auto& refOutput = refOutputs.begin()->second;
        const auto& actualOutput = actualOutputs.begin()->second;

        std::cout << "    Blob : ref:" << refOutputs.begin()->first << " actual:" << actualOutputs.begin()->first
                  << std::endl;

        compareOutputs(refOutput, actualOutput, tolerance, method);
    } else {
        for (const auto& p : refOutputs) {
            const auto& refOutput = p.second;
            const auto& actualOutput = actualOutputs.at(p.first);

            std::cout << "    Blob : " << p.first << std::endl;

            compareOutputs(refOutput, actualOutput, tolerance, method);
        }
    }
}

void KmbTestBase::checkWithOutputsInfo(const BlobMap& actualOutputs, const std::vector<DataPtr>& outputsInfo) {
    for (const auto& info : outputsInfo) {
        auto it = actualOutputs.find(info->getName());
        ASSERT_TRUE(it != actualOutputs.end());

        const auto& actual_desc = it->second->getTensorDesc();
        ASSERT_EQ(info->getLayout(), actual_desc.getLayout());
        ASSERT_EQ(info->getPrecision(), actual_desc.getPrecision());
        ASSERT_EQ(info->getDims(), actual_desc.getDims());
    }
}

void KmbTestBase::exportNetwork(ExecutableNetwork& exeNet) {
    IE_ASSERT(!DUMP_PATH.empty());

    const auto fileName = vpux::printToString("{0}/{1}.net", DUMP_PATH, dumpBaseName);

    if (RAW_EXPORT) {
        exeNet.Export(fileName);
    } else {
        std::ofstream file(fileName, std::ios_base::out | std::ios_base::binary);
        if (!file.is_open())
            IE_THROW() << "exportNetwork() failed. Can't open file " << fileName;

        exeNet.Export(file);
    }
}

ExecutableNetwork KmbTestBase::importNetwork(const std::map<std::string, std::string>& importConfig) {
    IE_ASSERT(!DUMP_PATH.empty());

    const auto fileName = vpux::printToString("{0}/{1}.net", DUMP_PATH, dumpBaseName);
    std::ifstream file(fileName, std::ios_base::in | std::ios_base::binary);
    if (!file.is_open()) {
        std::stringstream str;
        str << "importNetwork() failed. Cannot open file " << fileName;
        throw import_error(str.str());
    }

    if (RAW_EXPORT) {
        return core->ImportNetwork(fileName, DEVICE_NAME, importConfig);
    } else {
        return core->ImportNetwork(file, DEVICE_NAME, importConfig);
    }
}

void KmbTestBase::dumpBlob(const std::string& blobName, const Blob::Ptr& blob) {
    IE_ASSERT(!DUMP_PATH.empty());

    const auto fileName = vpux::printToString("{0}/{1}_{2}.blob", DUMP_PATH, dumpBaseName, cleanName(blobName));

    std::ofstream file(fileName, std::ios_base::out | std::ios_base::binary);
    if (!file.is_open())
        IE_THROW() << "dumpBlob() failed. Can't open file " << fileName;

    file.write(blob->cbuffer().as<const char*>(), static_cast<std::streamsize>(blob->byteSize()));
}

void KmbTestBase::dumpBlobs(const BlobMap& blobs) {
    IE_ASSERT(!DUMP_PATH.empty());

    for (const auto& p : blobs) {
        dumpBlob(p.first, p.second);
    }
}

Blob::Ptr KmbTestBase::importBlob(const std::string& name, const TensorDesc& desc) {
    IE_ASSERT(!DUMP_PATH.empty());

    const auto blob = make_blob_with_precision(desc);
    blob->allocate();

    const auto fileName = vpux::printToString("{0}/{1}_{2}.blob", DUMP_PATH, dumpBaseName, cleanName(name));
    std::ifstream file(fileName, std::ios_base::in | std::ios_base::binary);
    if (!file.is_open()) {
        std::stringstream str;
        str << "importBlob() failed. Cannot open file " << fileName;
        throw import_error(str.str());
    }

    file.read(blob->buffer().as<char*>(), static_cast<std::streamsize>(blob->byteSize()));

    return blob;
}

BlobMap KmbTestBase::runInfer(ExecutableNetwork& exeNet, const BlobMap& inputs, bool printTime) {
    auto inferRequest = exeNet.CreateInferRequest();

    for (const auto& p : inputs) {
        inferRequest.SetBlob(p.first, p.second);
    }

    const auto start = std::chrono::high_resolution_clock::now();

    inferRequest.Infer();

    const auto end = std::chrono::high_resolution_clock::now();

    if (printTime) {
        const auto dur = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
        std::cout << "Total Inference time: " << dur.count() << " ms" << std::endl;

        if (PRINT_PERF_COUNTERS) {
            const auto perfMap = inferRequest.GetPerformanceCounts();

            using PerfVal = std::pair<std::string, InferenceEngineProfileInfo>;
            std::vector<PerfVal> perfVec(perfMap.begin(), perfMap.end());
            std::sort(perfVec.begin(), perfVec.end(), [=](const PerfVal& p1, const PerfVal& p2) {
                return p1.second.execution_index < p2.second.execution_index;
            });

            size_t maxLayerName = 0u, maxExecType = 0u;
            for (const auto& p : perfMap) {
                maxLayerName = std::max(maxLayerName, p.first.length());
                maxExecType = std::max(maxExecType, std::strlen(p.second.exec_type));
            }

            const int indexWidth = 7;
            const int nameWidth = static_cast<int>(maxLayerName) + 5;
            const int typeWidth = static_cast<int>(maxExecType) + 5;
            const int timeWidth = 10;
            const int totalWidth = indexWidth + nameWidth + typeWidth + timeWidth;

            std::cout << std::endl;
            std::cout << "Detailed Per Layer Profile:" << std::endl;

            for (int i = 0; i < totalWidth; i++) {
                std::cout << "=";
            }

            std::cout << std::endl;
            std::cout << std::setw(indexWidth) << std::left << "Index" << std::setw(nameWidth) << std::left << "Name"
                      << std::setw(typeWidth) << std::left << "Type" << std::setw(timeWidth) << std::right
                      << "Time (ms)" << std::endl;

            for (int i = 0; i < totalWidth; i++) {
                std::cout << "-";
            }
            std::cout << std::endl;

            for (const auto& p : perfVec) {
                const auto& stageName = p.first;
                const auto& info = p.second;

                if (info.status == InferenceEngineProfileInfo::EXECUTED) {
                    std::cout << std::setw(indexWidth) << std::left << info.execution_index << std::setw(nameWidth)
                              << std::left << stageName << std::setw(typeWidth) << std::left << info.exec_type
                              << std::setw(timeWidth) << std::right << info.realTime_uSec / 1000.0 << std::endl;
                }
            }

            for (int i = 0; i < totalWidth; i++) {
                std::cout << "-";
            }
            std::cout << std::endl;
        }
    }

    const auto outputsInfo = exeNet.GetOutputsInfo();

    BlobMap out;

    for (const auto& p : outputsInfo) {
        out.insert({p.first, inferRequest.GetBlob(p.first)});
    }

    return out;
}

BlobMap KmbTestBase::getInputs(const ExecutableNetwork& testNet) {
    BlobMap inputs;

    for (const auto& info : testNet.GetInputsInfo()) {
        const auto blob = getBlobByName(info.first);
        inputs.emplace(info.first, blob);
    }

    return inputs;
}

//
// KmbLayerTestBase
//

void KmbLayerTestBase::runTest(const NetworkBuilder& builder, const float tolerance, const CompareMethod method) {
    try {
        if (!RUN_COMPILER || !RUN_REF_CODE) {
            if (DUMP_PATH.empty()) {
                GTEST_SKIP() << "Compilation and/or REF_CODE were disabled and IE_KMB_TESTS_DUMP_PATH was not provided";
            }
        }
        TestNetwork testNet;
        builder(testNet);

        for (const auto& ext : testNet.getExtensions()) {
            core->AddExtension(ext);
        }
        auto exeNet = getExecNetwork(testNet);

        const auto inputs = getInputs(exeNet);

        const auto refOutputs = getRefOutputs(testNet, inputs);

        // TODO: layer inference for by-pass mode
        // [Track number: S#48139]
#ifdef __aarch64__
        if (RUN_INFER) {
            std::cout << "=== INFER" << std::endl;

            const auto actualOutputs = runInfer(exeNet, inputs, true);

            std::cout << "=== COMPARE WITH REFERENCE" << std::endl;

            checkWithOutputsInfo(actualOutputs, testNet.getOutputsInfo());
            compareWithReference(actualOutputs, refOutputs, tolerance, method);
        }
#endif
    } catch (const import_error& ex) {
        std::cerr << ex.what() << std::endl;
        GTEST_SKIP() << ex.what();
    }
}

ExecutableNetwork KmbLayerTestBase::getExecNetwork(TestNetwork& testNet) {
    return KmbTestBase::getExecNetwork(
            [&testNet]() {
                return testNet.getCNNNetwork();
            },
            [&testNet]() {
                return testNet.compileConfig();
            });
}

BlobMap KmbLayerTestBase::getRefOutputs(TestNetwork& testNet, const BlobMap& inputs) {
    BlobMap refOutputs;

    if (RUN_REF_CODE) {
        std::cout << "=== CALC REFERENCE" << std::endl;

        refOutputs = testNet.calcRef(inputs);

        if (EXPORT_BLOBS) {
            std::cout << "    === EXPORT REFERENCE" << std::endl;

            dumpBlobs(refOutputs);
        }
    } else if (RUN_INFER) {
        std::cout << "=== IMPORT REFERENCE" << std::endl;

        for (const auto& info : testNet.getOutputsInfo()) {
            const auto desc =
                    TensorDesc(Precision::FP32, info->getDims(), TensorDesc::getLayoutByDims(info->getDims()));
            const auto blob = importBlob(info->getName(), desc);
            refOutputs.insert({info->getName(), blob});
        }
    }

    return refOutputs;
}

//
// TestNetworkDesc
//

void TestNetworkDesc::fillUserInputInfo(InputsDataMap& info) const {
    if (info.size() == 1) {
        if (_inputPrecisions.size() == 1) {
            info.begin()->second->setPrecision(_inputPrecisions.begin()->second);
        } else if (_inputPrecisions.size() > 1) {
            IE_THROW() << "Input precision was set more than one time";
        }

        if (_inputLayouts.size() == 1) {
            info.begin()->second->setLayout(_inputLayouts.begin()->second);
        } else if (_inputLayouts.size() > 1) {
            IE_THROW() << "Input layout was set more than one time";
        }
    } else {
        for (const auto& p : info) {
            const auto precisionIt = _inputPrecisions.find(p.first);
            if (precisionIt != _inputPrecisions.end()) {
                p.second->setPrecision(precisionIt->second);
            }

            const auto layoutIt = _inputLayouts.find(p.first);
            if (layoutIt != _inputLayouts.end()) {
                p.second->setLayout(layoutIt->second);
            }
        }
    }
}

void TestNetworkDesc::fillUserOutputInfo(OutputsDataMap& info) const {
    if (info.size() == 1) {
        if (_outputPrecisions.size() == 1) {
            info.begin()->second->setPrecision(_outputPrecisions.begin()->second);
        }

        if (_outputLayouts.size() == 1) {
            info.begin()->second->setLayout(_outputLayouts.begin()->second);
        }
    } else {
        for (const auto& p : info) {
            const auto precisionIt = _outputPrecisions.find(p.first);
            if (precisionIt != _outputPrecisions.end()) {
                p.second->setPrecision(precisionIt->second);
            }

            const auto layoutIt = _outputLayouts.find(p.first);
            if (layoutIt != _outputLayouts.end()) {
                p.second->setLayout(layoutIt->second);
            }
        }
    }
}

//
// KmbNetworkTestBase
//

std::string KmbNetworkTestBase::getTestDataPath() {
    if (const auto envVar = std::getenv("DATA_PATH")) {
        return envVar;
    }

#ifdef DATA_PATH
    return DATA_PATH;
#else
    return {};
#endif
}

namespace {

std::string getTestModelsBasePath() {
    if (const auto envVar = std::getenv("MODELS_PATH")) {
        return envVar;
    }

#ifdef MODELS_PATH
    return MODELS_PATH;
#else
    return {};
#endif
}

std::string getExperimentalModelsPath() {
    if (const auto envVar = std::getenv("EXPERIMENTAL_MODELS_PATH"))
        return std::string(envVar);
    else
#ifdef EXPERIMENTAL_MODELS_PATH
        return EXPERIMENTAL_MODELS_PATH;
#else
        return {};
#endif
}

}  // namespace

std::string KmbNetworkTestBase::getTestModelsPath() {
    return getTestModelsBasePath() + "/src/models";
}

Blob::Ptr KmbNetworkTestBase::loadImage(const TestImageDesc& image, size_t channels, size_t height, size_t width) {
    std::ostringstream imageFilePath;
    imageFilePath << getTestDataPath() << "/" << image.imageFileName();

    FormatReader::ReaderPtr reader(imageFilePath.str().c_str());
    IE_ASSERT(reader.get() != nullptr);

    const size_t C = channels;
    const size_t H = height;
    const size_t W = width;

    const auto tensorDesc = TensorDesc(Precision::FP32, {1, C, H, W}, Layout::NHWC);

    const auto blob = make_blob_with_precision(tensorDesc);
    blob->allocate();

    const auto imagePtr = reader->getData(width, height).get();
    const auto blobPtr = blob->buffer().as<float*>();

    IE_ASSERT(imagePtr != nullptr);
    IE_ASSERT(blobPtr != nullptr);

    if (image.isBGR()) {
        std::copy_n(imagePtr, blob->size(), blobPtr);
    } else {
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                for (size_t c = 0; c < C; ++c) {
                    blobPtr[c + w * C + h * C * W] = imagePtr[(C - c - 1) + w * C + h * C * W];
                }
            }
        }
    }

    return blob;
}

Blob::Ptr KmbNetworkTestBase::loadBinFile(const TestBinFileDesc& binFile, size_t channels, size_t height,
                                          size_t width) {
    std::ostringstream filePath;
    const auto binFileShape = binFile.getShape();
    bool singleDimBinFile = (binFileShape.size() == 1);
    if (!singleDimBinFile) {
        IE_ASSERT(channels == binFileShape[1]);
        IE_ASSERT(height == binFileShape[2]);
        IE_ASSERT(width == binFileShape[3]);
    }

    filePath << getTestDataPath() << "/" << binFile.fileName();

    std::ifstream file(filePath.str().c_str(), std::ios_base::in | std::ios_base::binary | std::ios::ate);
    if (!file.is_open())
        IE_THROW() << "Load input file failed. Can't open file " << filePath.str();

    file.seekg(0, std::ios::end);
    int file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    IE_ASSERT(file_size == binFile.getSize());

    auto layout = singleDimBinFile ? Layout::C : Layout::NHWC;
    const TensorDesc tensorDesc = TensorDesc(binFile.getPrecision(), binFileShape, layout);

    const auto blob = make_blob_with_precision(tensorDesc);
    blob->allocate();

    const auto blobPtr = blob->buffer().as<char*>();
    IE_ASSERT(blobPtr != nullptr);

    file.read(blobPtr, static_cast<std::streamsize>(blob->byteSize()));
    file.close();

    return blob;
}

CNNNetwork KmbNetworkTestBase::readNetwork(const TestNetworkDesc& netDesc, bool fillUserInfo) {
    std::ostringstream modelPath;

    if (netDesc.isExperimental())
        modelPath << getExperimentalModelsPath() << "/" << netDesc.irFileName();
    else
        modelPath << getTestModelsPath() << "/" << netDesc.irFileName();

    auto net = core->ReadNetwork(modelPath.str());

    if (fillUserInfo) {
        auto inputsInfo = net.getInputsInfo();
        auto outputsInfo = net.getOutputsInfo();

        netDesc.fillUserInputInfo(inputsInfo);
        netDesc.fillUserOutputInfo(outputsInfo);
    }

    return net;
}

ExecutableNetwork KmbNetworkTestBase::getExecNetwork(const TestNetworkDesc& netDesc) {
    return KmbTestBase::getExecNetwork(
            [&netDesc, this]() {
                return readNetwork(netDesc, true);
            },
            [&netDesc]() {
                return netDesc.compileConfig();
            },
            netDesc.isCompilationForced());
}

BlobMap KmbNetworkTestBase::calcRefOutput(const TestNetworkDesc& netDesc, const BlobMap& inputs,
                                          const bool& enableLPTRef) {
    if (enableLPTRef) {
        core->SetConfig({{"LP_TRANSFORMS_MODE", CONFIG_VALUE(YES)}}, "CPU");
    }
    const auto refNet = readNetwork(netDesc, false);
    auto refExeNet = core->LoadNetwork(refNet, REF_DEVICE_NAME);

    const auto refInputsInfo = refNet.getInputsInfo();

    BlobMap refInputs;
    for (const auto& refInfo : refInputsInfo) {
        const auto& refInputName = refInfo.first;
        const auto& refInputInfo = refInfo.second;
        const auto& inputBlob = inputs.at(refInputName);
        const auto refInputBlob = vpux::toLayout(
                vpux::toPrecision(as<MemoryBlob>(inputBlob), refInputInfo->getTensorDesc().getPrecision()),
                refInputInfo->getTensorDesc().getLayout());
        refInputs.emplace(refInputName, refInputBlob);
    }

    auto refOutputs = runInfer(refExeNet, refInputs, false);
    return refOutputs;
}

void KmbNetworkTestBase::checkLayouts(const BlobMap& actualOutputs,
                                      const std::unordered_map<std::string, Layout>& layouts) const {
    // HACK: It's necessary for back compatibility when blob names were lost after export
    // [Track number: S#35709]
    if (layouts.size() == 1) {
        const auto& actual = *actualOutputs.begin();
        const auto& expected = *layouts.begin();
        ASSERT_EQ(expected.second, actual.second->getTensorDesc().getLayout());
    } else {
        for (const auto& layout : layouts) {
            auto blob_it = actualOutputs.find(layout.first);
            ASSERT_TRUE(blob_it != actualOutputs.end());
            ASSERT_EQ(layout.second, blob_it->second->getTensorDesc().getLayout());
        }
    }
}

void KmbNetworkTestBase::checkPrecisions(const BlobMap& actualOutputs,
                                         const std::unordered_map<std::string, Precision>& precisions) const {
    // HACK: It's necessary for back compatibility when blob names were lost after export
    // [Track number: S#35709]
    if (precisions.size() == 1) {
        const auto& actual = *actualOutputs.begin();
        const auto& expected = *precisions.begin();
        ASSERT_EQ(expected.second, actual.second->getTensorDesc().getPrecision());
    } else {
        for (const auto& precision : precisions) {
            auto blob_it = actualOutputs.find(precision.first);
            ASSERT_TRUE(blob_it != actualOutputs.end());
            ASSERT_EQ(precision.second, blob_it->second->getTensorDesc().getPrecision());
        }
    }
}

void KmbNetworkTestBase::runTest(const TestNetworkDesc& netDesc, const InitIntputCallback& inputCallback,
                                 const CheckCallback& checkCallback) {
    try {
        if (!RUN_COMPILER || !RUN_REF_CODE) {
            if (DUMP_PATH.empty()) {
                GTEST_SKIP() << "Compilation and/or REF_CODE were disabled and IE_KMB_TESTS_DUMP_PATH was not provided";
            }
        }

        if (netDesc.isExperimental() && getExperimentalModelsPath().empty()) {
            GTEST_SKIP() << "EXPERIMENTAL_MODELS_PATH is not set";
        }

        auto exeNet = getExecNetwork(netDesc);

        const auto inputsInfo = exeNet.GetInputsInfo();
        const auto outputsInfo = exeNet.GetOutputsInfo();

        inputCallback(inputsInfo);

        BlobMap inputs;
        for (const auto& inputInfo : inputsInfo) {
            const auto& inputName = inputInfo.first;
            // HACK: to overcome IE bug with incorrect TensorDesc::setLayout
            const auto& desc = inputInfo.second->getTensorDesc();
            const auto& inputBlob = vpux::toPrecision(
                    vpux::toLayout(as<MemoryBlob>(getBlobByName(inputName)), desc.getLayout()), desc.getPrecision());
            inputs.emplace(inputName, inputBlob);
        }

        BlobMap refOutputBlobs;

        if (RUN_REF_CODE) {
            std::cout << "=== CALC REFERENCE WITH " << REF_DEVICE_NAME << std::endl;
            refOutputBlobs = calcRefOutput(netDesc, inputs, netDesc.isLPTRefModeEnabled());

            if (EXPORT_BLOBS) {
                std::cout << "    === EXPORT REFERENCE" << std::endl;
                for (const auto& refOutput : refOutputBlobs) {
                    dumpBlob(refOutput.first,
                             vpux::toDefLayout(vpux::toDefPrecision(as<MemoryBlob>(refOutput.second))));
                }
            }
        } else if (RUN_INFER) {
            std::cout << "=== IMPORT REFERENCE" << std::endl;

            for (const auto& outputInfo : outputsInfo) {
                const auto& outputDims = outputInfo.second->getTensorDesc().getDims();

                const auto refOutputTensorDesc =
                        TensorDesc(Precision::FP32, outputDims, TensorDesc::getLayoutByDims(outputDims));

                refOutputBlobs.emplace(outputInfo.first, importBlob(outputInfo.first, refOutputTensorDesc));
            }
        }

        if (RUN_INFER) {
            if (skipInfer) {
                std::cout << skipMessage << std::endl;
                return;
            }
            std::cout << "=== INFER" << std::endl;

            const auto actualOutputs = runInfer(exeNet, inputs, true);

            std::cout << "=== COMPARE WITH REFERENCE" << std::endl;
            checkLayouts(actualOutputs, netDesc.outputLayouts());
            checkPrecisions(actualOutputs, netDesc.outputPrecisions());
            checkCallback(actualOutputs, refOutputBlobs, inputsInfo);
        }
    } catch (const import_error& ex) {
        std::cerr << ex.what() << std::endl;
        GTEST_SKIP() << ex.what();
    }
}

void KmbNetworkTestBase::registerSingleImage(const TestImageDesc& image, const std::string& inputName,
                                             const TensorDesc inputDesc) {
    registerBlobGenerator(inputName, inputDesc, [image](const TensorDesc& desc) {
        const auto blob = loadImage(image, desc.getDims()[1], desc.getDims()[2], desc.getDims()[3]);
        IE_ASSERT(blob->getTensorDesc().getDims() == desc.getDims());

        return vpux::toPrecision(vpux::toLayout(as<MemoryBlob>(blob), desc.getLayout()), desc.getPrecision());
    });
};

void KmbNetworkTestBase::registerSingleBinFile(const TestBinFileDesc& file, const std::string& inputName,
                                               const TensorDesc inputDesc) {
    registerBlobGenerator(inputName, inputDesc, [file](const TensorDesc& desc) {
        auto channel = desc.getDims()[0];
        auto height = desc.getDims()[0];
        auto width = desc.getDims()[0];
        if (desc.getDims().size() == 4) {
            channel = desc.getDims()[1];
            height = desc.getDims()[2];
            width = desc.getDims()[3];
        }
        const auto blob = loadBinFile(file, channel, height, width);
        IE_ASSERT(blob->getTensorDesc().getDims() == desc.getDims());

        return vpux::toPrecision(vpux::toLayout(as<MemoryBlob>(blob), desc.getLayout()), desc.getPrecision());
    });
};

//
// KmbClassifyNetworkTest
//
void KmbClassifyNetworkTest::checkCallbackHelper(const BlobMap& actualBlobs, const BlobMap& refBlobs, const size_t topK,
                                                 const float probTolerance) {
    IE_ASSERT(actualBlobs.size() == 1u && actualBlobs.size() == refBlobs.size());
    auto actualBlob = actualBlobs.begin()->second;
    auto refBlob = refBlobs.begin()->second;

    ASSERT_EQ(refBlob->getTensorDesc().getDims(), actualBlob->getTensorDesc().getDims());

    auto actualOutput = parseOutput(vpux::toFP32(as<MemoryBlob>(actualBlob)));
    auto refOutput = parseOutput(vpux::toFP32(as<MemoryBlob>(refBlob)));

    ASSERT_GE(actualOutput.size(), topK);
    actualOutput.resize(topK);

    ASSERT_GE(refOutput.size(), topK);
    refOutput.resize(topK);

    std::cout << "Ref Top:" << std::endl;
    for (size_t i = 0; i < topK; ++i) {
        std::cout << i << " : " << refOutput[i].first << " : " << refOutput[i].second << std::endl;
    }

    std::cout << "Actual top:" << std::endl;
    for (size_t i = 0; i < topK; ++i) {
        std::cout << i << " : " << actualOutput[i].first << " : " << actualOutput[i].second << std::endl;
    }

    for (const auto& refElem : refOutput) {
        const auto actualIt =
                std::find_if(actualOutput.cbegin(), actualOutput.cend(), [&refElem](const std::pair<int, float> arg) {
                    return refElem.first == arg.first;
                });
        ASSERT_NE(actualIt, actualOutput.end());

        const auto& actualElem = *actualIt;

        if (refElem.second > actualElem.second) {
            const auto probDiff = std::fabs(refElem.second - actualElem.second);
            EXPECT_LE(probDiff, probTolerance)
                    << refElem.first << " : " << refElem.second << " vs " << actualElem.second;
        }
    }
};

void KmbClassifyNetworkTest::runTest(const TestNetworkDesc& netDesc, const TestImageDesc& image, const size_t topK,
                                     const float probTolerance) {
    const auto check = [=](const BlobMap& actualBlobs, const BlobMap& refBlobs, const ConstInputsDataMap&) {
        checkCallbackHelper(actualBlobs, refBlobs, topK, probTolerance);
    };

    const auto init_input = [=](const ConstInputsDataMap& inputs) {
        IE_ASSERT(inputs.size() == 1);
        registerSingleImage(image, inputs.begin()->first, inputs.begin()->second->getTensorDesc());
    };

    KmbNetworkTestBase::runTest(netDesc, init_input, check);
}

void KmbClassifyNetworkTest::runTest(const TestNetworkDesc& netDesc, const TestBinFileDesc& file, const size_t topK,
                                     const float probTolerance) {
    const auto check = [=](const BlobMap& actualBlobs, const BlobMap& refBlobs, const ConstInputsDataMap&) {
        checkCallbackHelper(actualBlobs, refBlobs, topK, probTolerance);
    };

    const auto init_input = [=](const ConstInputsDataMap& inputs) {
        IE_ASSERT(inputs.size() == 1);
        registerSingleBinFile(file, inputs.begin()->first, inputs.begin()->second->getTensorDesc());
    };

    KmbNetworkTestBase::runTest(netDesc, init_input, check);
}

std::vector<std::pair<int, float>> KmbClassifyNetworkTest::parseOutput(const Blob::Ptr& blob) {
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

//
// KmbDetectionNetworkTest
//

void KmbDetectionNetworkTest::runTest(const TestNetworkDesc& netDesc, const TestImageDesc& image,
                                      const float confThresh, const float boxTolerance, const float probTolerance) {
    const auto check = [=](const BlobMap& actualBlobs, const BlobMap& refBlobs, const ConstInputsDataMap& inputsDesc) {
        IE_ASSERT(inputsDesc.size() == 1);
        IE_ASSERT(actualBlobs.size() == 1u && actualBlobs.size() == refBlobs.size());

        auto actualBlob = actualBlobs.begin()->second;
        auto refBlob = refBlobs.begin()->second;

        const auto& inputDesc = inputsDesc.begin()->second->getTensorDesc();

        const auto imgWidth = inputDesc.getDims().at(3);
        const auto imgHeight = inputDesc.getDims().at(2);

        auto actualOutput = parseOutput(vpux::toFP32(as<MemoryBlob>(actualBlob)), imgWidth, imgHeight, confThresh);
        auto refOutput = parseOutput(vpux::toFP32(as<MemoryBlob>(refBlob)), imgWidth, imgHeight, confThresh);

        checkBBoxOutputs(actualOutput, refOutput, imgWidth, imgHeight, boxTolerance, probTolerance);
    };

    const auto init_input = [=](const ConstInputsDataMap& inputs) {
        IE_ASSERT(inputs.size() == 1);
        registerSingleImage(image, inputs.begin()->first, inputs.begin()->second->getTensorDesc());
    };

    KmbNetworkTestBase::runTest(netDesc, init_input, check);
}

std::vector<utils::BoundingBox> KmbDetectionNetworkTest::parseOutput(const Blob::Ptr& blob, const size_t imgWidth,
                                                                     const size_t imgHeight, const float confThresh) {
    constexpr size_t ELEM_COUNT = 7;

    const auto count = blob->size() / ELEM_COUNT;

    std::vector<utils::BoundingBox> out;
    out.reserve(count);

    const auto ptr = blob->cbuffer().as<const float*>();
    IE_ASSERT(ptr != nullptr);

    for (size_t i = 0; i < count; ++i) {
        const int batch_id = static_cast<int>(ptr[i * ELEM_COUNT + 0]);
        if (batch_id < 0) {
            continue;
        }

        const int class_id = static_cast<int>(ptr[i * ELEM_COUNT + 1]);

        const float conf = ptr[i * ELEM_COUNT + 2];
        if (conf < confThresh) {
            continue;
        }

        const float xmin = ptr[i * ELEM_COUNT + 3];
        const float ymin = ptr[i * ELEM_COUNT + 4];
        const float xmax = ptr[i * ELEM_COUNT + 5];
        const float ymax = ptr[i * ELEM_COUNT + 6];

        utils::BoundingBox bb(class_id, imgWidth * xmin, imgHeight * ymin, imgWidth * xmax, imgHeight * ymax, conf);

        out.push_back(bb);
    }

    return out;
}

void KmbDetectionNetworkTest::checkBBoxOutputs(std::vector<utils::BoundingBox>& actualOutput,
                                               std::vector<utils::BoundingBox>& refOutput, const size_t imgWidth,
                                               const size_t imgHeight, const float boxTolerance,
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

        EXPECT_TRUE(found) << "maxBoxError=" << maxBoxError << " "
                           << "maxProbError=" << maxProbError;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// YOLOV2NetworkAdapter ////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void KmbYoloV2NetworkTest::runTest(const TestNetworkDesc& netDesc, const TestImageDesc& image, const float confThresh,
                                   const float boxTolerance, const float probTolerance, const bool isTiny) {
    const auto check = [=](const BlobMap& actualBlobs, const BlobMap& refBlobs, const ConstInputsDataMap& inputsDesc) {
        IE_ASSERT(inputsDesc.size() == 1);
        IE_ASSERT(actualBlobs.size() == 1u && actualBlobs.size() == refBlobs.size());
        auto actualBlob = actualBlobs.begin()->second;
        auto refBlob = refBlobs.begin()->second;

        const auto& inputDesc = inputsDesc.begin()->second->getTensorDesc();

        const auto imgWidth = inputDesc.getDims().at(3);
        const auto imgHeight = inputDesc.getDims().at(2);

        auto actualOutput = utils::parseYoloOutput(vpux::toFP32(as<MemoryBlob>(actualBlob)), imgWidth, imgHeight,
                                                   confThresh, isTiny);
        auto refOutput =
                utils::parseYoloOutput(vpux::toFP32(as<MemoryBlob>(refBlob)), imgWidth, imgHeight, confThresh, isTiny);

        checkBBoxOutputs(actualOutput, refOutput, imgWidth, imgHeight, boxTolerance, probTolerance);
    };

    const auto init_input = [=](const ConstInputsDataMap& inputs) {
        IE_ASSERT(inputs.size() == 1);
        registerSingleImage(image, inputs.begin()->first, inputs.begin()->second->getTensorDesc());
    };

    KmbNetworkTestBase::runTest(netDesc, init_input, check);
}
