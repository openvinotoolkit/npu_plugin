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

#include "kmb_test_base.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

#include <blob_factory.hpp>
#include "functional_test_utils/plugin_cache.hpp"
#include <test_model_path.hpp>
#include <single_layer_common.hpp>
#include <format_reader_ptr.h>
#include <vpu/utils/error.hpp>
#include <boost/filesystem.hpp>

//
// KmbTestBase
//

namespace {

bool strToBool(const char* varName, const char* varValue) {
    try {
        const auto intVal = std::stoi(varValue);
        if (intVal != 0 && intVal != 1) {
            throw std::invalid_argument("Only 0 and 1 values are supported");
        }
        return (intVal != 0);
    } catch (const std::exception& e) {
        THROW_IE_EXCEPTION << "Environment variable " << varName << " has wrong value : " << e.what();
    }
}

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

    return "KMB";
}();

const std::string KmbTestBase::REF_DEVICE_NAME = []() -> std::string {
    if (const auto var = std::getenv("IE_KMB_TESTS_REF_DEVICE_NAME")) {
        return var;
    }

    return "CPU";
}();

const bool KmbTestBase::RUN_COMPILER = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_RUN_COMPILER")) {
        return strToBool("IE_KMB_TESTS_RUN_COMPILER", var);
    }

    if (KmbTestBase::DEVICE_NAME == "CPU") {
        return true;
    }

#if defined(__aarch64__) || !defined(ENABLE_MCM_COMPILER)
    return false;
#else
    return true;
#endif
}();

const bool KmbTestBase::RUN_REF_CODE = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_RUN_REF_CODE")) {
        return strToBool("IE_KMB_TESTS_RUN_REF_CODE", var);
    }

#ifdef __aarch64__
    return false;
#else
    return true;
#endif
}();

const bool KmbTestBase::RUN_INFER = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_RUN_INFER")) {
        return strToBool("IE_KMB_TESTS_RUN_INFER", var);
    }

    if (KmbTestBase::DEVICE_NAME == "CPU") {
        return true;
    }

#ifdef __aarch64__
    return true;
#else
    return false;
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
        return strToBool("IE_KMB_TESTS_EXPORT_NETWORK", var);
    }

    if (KmbTestBase::DEVICE_NAME == "CPU") {
        return false;
    }

    return KmbTestBase::RUN_COMPILER && !KmbTestBase::DUMP_PATH.empty();
}();

const bool KmbTestBase::RAW_EXPORT = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_RAW_EXPORT")) {
        return strToBool("IE_KMB_TESTS_RAW_EXPORT", var);
    }

    if (KmbTestBase::DEVICE_NAME != "KMB" || !KmbTestBase::EXPORT_NETWORK) {
        return false;
    }

    return false;
}();

const bool KmbTestBase::GENERATE_BLOBS = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_GENERATE_BLOBS")) {
        return strToBool("IE_KMB_TESTS_GENERATE_BLOBS", var);
    }

    return KmbTestBase::RUN_REF_CODE;
}();

const bool KmbTestBase::EXPORT_BLOBS = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_EXPORT_BLOBS")) {
        return strToBool("IE_KMB_TESTS_EXPORT_BLOBS", var);
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
        return strToBool("IE_KMB_TESTS_PRINT_PERF_COUNTERS", var);
    }

    return false;
}();

void KmbTestBase::SetUp() {
    ASSERT_NO_FATAL_FAILURE(TestsCommon::SetUp());

    const auto testInfo = testing::UnitTest::GetInstance()->current_test_info();
    IE_ASSERT(testInfo != nullptr);

    rd.seed();

    core = PluginCache::get().ie();
    if (!LOG_LEVEL.empty()) {
        core->SetConfig({{CONFIG_KEY(LOG_LEVEL), LOG_LEVEL}}, DEVICE_NAME);
    }
    if (PRINT_PERF_COUNTERS) {
        core->SetConfig({{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)}}, DEVICE_NAME);
    }

    if ((RUN_REF_CODE               && REF_DEVICE_NAME == "CPU") ||
       ((RUN_COMPILER || RUN_INFER) && DEVICE_NAME     == "CPU" )) {
       core->SetConfig({{"LP_TRANSFORMS_MODE", CONFIG_VALUE(NO)}}, "CPU");
    }

    dumpBaseName = cleanName(vpu::formatString("%v_%v", testInfo->test_case_name(), testInfo->name()));

    if (const auto typeParam = testInfo->type_param()) {
        std::cout << "[ PARAMS   ] " << typeParam << std::endl;
    }
    if (const auto valueParam = testInfo->value_param()) {
        std::cout << "[ PARAMS   ] " << valueParam << std::endl;
    }

    std::cout << "[ MODE     ] "
              << DEVICE_NAME << " / "
              << (RUN_INFER ? "RUN INFER AND CHECK" : "NO INFER") << " / "
              << (RUN_COMPILER ? "COMPILE NETWORK" : "IMPORT BLOB") << " / "
              << (RUN_REF_CODE ? "CALC REF" : "IMPORT REF") << std::endl;
    if (!DUMP_PATH.empty()) {
        std::cout << "[ DUMP PATH] " << DUMP_PATH << std::endl;
    }
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

ExecutableNetwork KmbTestBase::getExecNetwork(
        const std::function<CNNNetwork()>& netCreator,
        const std::function<CompileConfig()>& configCreator) {
    ExecutableNetwork exeNet;

    if (RUN_COMPILER) {
        std::cout << "=== COMPILE NETWORK" << std::endl;

        exeNet = core->LoadNetwork(netCreator(), DEVICE_NAME, configCreator());

        if (EXPORT_NETWORK) {
            std::cout << "    === EXPORT NETWORK" << std::endl;

            exportNetwork(exeNet);
        }
    } else if (RUN_INFER) {
        std::cout << "=== IMPORT NETWORK" << std::endl;

        exeNet = importNetwork();
    }

    return exeNet;
}

namespace {

bool tensorIter(SizeVector& ind, const TensorDesc& desc) {
    const auto& dims = desc.getDims();

    for (auto i = static_cast<ptrdiff_t>(dims.size() - 1); i >= 0; --i) {
        const auto ui = static_cast<size_t>(i);

        if (++ind[ui] < dims[ui]) {
            return true;
        }

        ind[ui] = 0;
    }

    return false;
}

}

void KmbTestBase::compareOutputs(
        const Blob::Ptr& refOutput, const Blob::Ptr& actualOutput,
        float tolerance, CompareMethod method) {
    ASSERT_EQ(refOutput->size(), actualOutput->size());

    const auto& refDesc = refOutput->getTensorDesc();
    const auto& actualDesc = actualOutput->getTensorDesc();

    const auto& ref_dims = refDesc.getDims();
    const auto& actual_dims = actualDesc.getDims();
    ASSERT_TRUE(ref_dims == actual_dims);

    BufferWrapper refPtr(refOutput);
    BufferWrapper actualPtr(actualOutput);

    const auto printCount = std::min<size_t>(refOutput->size(), 10);

    SizeVector tensorInd(refDesc.getDims().size());

    for (size_t i = 0; (i < printCount) && tensorIter(tensorInd, refDesc); ++i) {
        const auto refOffset = refDesc.offset(tensorInd);
        const auto actualOffset = actualDesc.offset(tensorInd);

        const auto refVal = refPtr[refOffset];
        const auto actualVal = actualPtr[actualOffset];

        const auto absdiff = std::fabs(refVal - actualVal);

        std::cout << "        " << i << " :"
                  << " ref : " << std::setw(10) << refVal
                  << " actual : " << std::setw(10) << actualVal
                  << " absdiff : " << std::setw(10) << absdiff
                  << std::endl;
    }

    EXPECT_NO_FATAL_FAILURE(compareBlobs(actualOutput, refOutput, tolerance, method));
}

void KmbTestBase::compareWithReference(
        const BlobMap& actualOutputs,
        const BlobMap& refOutputs,
        float tolerance, CompareMethod method) {
    if (refOutputs.size() == 1) {
        // HACK: currently blob names are lost after Export, do not use them for single output case.

        ASSERT_EQ(actualOutputs.size(), 1);

        const auto& refOutput = refOutputs.begin()->second;
        const auto& actualOutput = actualOutputs.begin()->second;

        std::cout << "    Blob : ref:" << refOutputs.begin()->first << " actual:" << actualOutputs.begin()->first << std::endl;

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

void KmbTestBase::checkWithOutputsInfo(const BlobMap& actualOutputs,
                                       const std::vector<DataPtr>& outputsInfo) {
    for (const auto& info : outputsInfo) {
        auto it = actualOutputs.find(info->getName());
        ASSERT_TRUE(it != actualOutputs.end());

        const auto& actual_desc = it->second->getTensorDesc();
        ASSERT_EQ(info->getLayout(),    actual_desc.getLayout());
        ASSERT_EQ(info->getPrecision(), actual_desc.getPrecision());
        ASSERT_EQ(info->getDims(),      actual_desc.getDims());
    }
}

void KmbTestBase::exportNetwork(ExecutableNetwork& exeNet) {
    IE_ASSERT(!DUMP_PATH.empty());

    const auto fileName = vpu::formatString("%v/%v.net", DUMP_PATH, dumpBaseName);

    if (RAW_EXPORT) {
        exeNet.Export(fileName);
    } else {
        std::ofstream file(fileName, std::ios_base::out | std::ios_base::binary);
        IE_ASSERT(file.is_open());

        exeNet.Export(file);
    }
}

ExecutableNetwork KmbTestBase::importNetwork() {
    IE_ASSERT(!DUMP_PATH.empty());

    const auto fileName = vpu::formatString("%v/%v.net", DUMP_PATH, dumpBaseName);

    if (RAW_EXPORT) {
        return core->ImportNetwork(fileName, DEVICE_NAME);
    } else {
        std::ifstream file(fileName, std::ios_base::in | std::ios_base::binary);
        IE_ASSERT(file.is_open());

        return core->ImportNetwork(file, DEVICE_NAME);
    }
}

void KmbTestBase::dumpBlob(const std::string& blobName, const Blob::Ptr& blob) {
    IE_ASSERT(!DUMP_PATH.empty());

    namespace fs = boost::filesystem;
    if (!fs::exists(DUMP_PATH)) {
        fs::create_directories(DUMP_PATH);
    }

    const auto fileName = vpu::formatString("%v/%v_%v.blob", DUMP_PATH, dumpBaseName, cleanName(blobName));

    std::ofstream file(fileName, std::ios_base::out | std::ios_base::binary);
    IE_ASSERT(file.is_open());

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

    std::ifstream file(vpu::formatString("%v/%v_%v.blob", DUMP_PATH, dumpBaseName, cleanName(name)),
        std::ios_base::in | std::ios_base::binary);
    IE_ASSERT(file.is_open());

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
        std::cout << "Total Infererence time: " << dur.count() << " ms" << std::endl;

        if (PRINT_PERF_COUNTERS) {
            const auto perfMap = inferRequest.GetPerformanceCounts();

            using PerfVal = std::pair<std::string, InferenceEngineProfileInfo>;
            std::vector<PerfVal> perfVec(perfMap.begin(), perfMap.end());
            std::sort(perfVec.begin(), perfVec.end(),
                [=](const PerfVal& p1, const PerfVal& p2) {
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
            std::cout << std::setw(indexWidth) << std::left << "Index"
                      << std::setw(nameWidth) << std::left << "Name"
                      << std::setw(typeWidth) << std::left << "Type"
                      << std::setw(timeWidth) << std::right << "Time (ms)"
                      << std::endl;

            for (int i = 0; i < totalWidth; i++) {
                std::cout << "-";
            }
            std::cout << std::endl;

            for (const auto& p : perfVec) {
                const auto& stageName = p.first;
                const auto& info = p.second;

                if (info.status == InferenceEngineProfileInfo::EXECUTED) {
                    std::cout << std::setw(indexWidth) << std::left << info.execution_index
                              << std::setw(nameWidth) << std::left << stageName
                              << std::setw(typeWidth) << std::left << info.exec_type
                              << std::setw(timeWidth) << std::right << info.realTime_uSec / 1000.0
                              << std::endl;
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

//
// KmbLayerTestBase
//

void KmbLayerTestBase::runTest(
        const NetworkBuilder& builder,
        float tolerance, CompareMethod method) {
    if (!RUN_COMPILER || !RUN_REF_CODE) {
        if (DUMP_PATH.empty()) {
            SKIP() << "Compilation and/or REF_CODE were disabled, but IE_KMB_TESTS_DUMP_PATH were not provided";
        }
    }

    TestNetwork testNet;
    builder(testNet);

    const auto inputs = getInputs(testNet);

    auto exeNet = getExecNetwork(testNet);

    const auto refOutputs = getRefOutputs(testNet, inputs);

    if (RUN_INFER) {
        std::cout << "=== INFER" << std::endl;

        const auto actualOutputs = runInfer(exeNet, inputs, true);

        std::cout << "=== COMPARE WITH REFERENCE" << std::endl;

        checkWithOutputsInfo(actualOutputs, testNet.getOutputsInfo());
        compareWithReference(actualOutputs, refOutputs, tolerance, method);
    }
}

BlobMap KmbLayerTestBase::getInputs(TestNetwork& testNet) {
    BlobMap inputs;

    for (const auto& info : testNet.getInputsInfo()) {
        const auto blob = getBlobByName(info->getName());
        inputs.insert({info->getName(), blob});
    }

    return inputs;
}

ExecutableNetwork KmbLayerTestBase::getExecNetwork(
        TestNetwork& testNet) {
    return KmbTestBase::getExecNetwork(
        [&testNet]() {
            return testNet.getCNNNetwork();
        },
        [&testNet]() {
            return testNet.compileConfig();
        });
}

BlobMap KmbLayerTestBase::getRefOutputs(
        TestNetwork& testNet,
        const BlobMap& inputs) {
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
            const auto desc = TensorDesc(Precision::FP32, info->getDims(), TensorDesc::getLayoutByDims(info->getDims()));
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
        }

        if (_inputLayouts.size() == 1) {
            info.begin()->second->setLayout(_inputLayouts.begin()->second);
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

Blob::Ptr KmbNetworkTestBase::loadImage(const TestImageDesc& image, int channels, int height, int width) {
    std::ostringstream imageFilePath;
    imageFilePath << TestDataHelpers::get_data_path() << "/" << image.imageFileName();

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

CNNNetwork KmbNetworkTestBase::readNetwork(const TestNetworkDesc& netDesc, bool fillUserInfo) {
    ModelsPath modelPath;
    modelPath << "/" << netDesc.irFileName();

    auto net = core->ReadNetwork(modelPath);

    if (fillUserInfo) {
        auto inputsInfo = net.getInputsInfo();
        auto outputsInfo = net.getOutputsInfo();

        netDesc.fillUserInputInfo(inputsInfo);
        netDesc.fillUserOutputInfo(outputsInfo);
    }

    return net;
}

ExecutableNetwork KmbNetworkTestBase::getExecNetwork(
        const TestNetworkDesc& netDesc) {
    return KmbTestBase::getExecNetwork(
        [&netDesc, this]() {
            return readNetwork(netDesc, true);
        },
        [&netDesc]() {
            return netDesc.compileConfig();
        });
}

Blob::Ptr KmbNetworkTestBase::calcRefOutput(
        const TestNetworkDesc& netDesc,
        const Blob::Ptr& inputBlob) {
    const auto refNet = readNetwork(netDesc, false);
    auto refExeNet = core->LoadNetwork(refNet, REF_DEVICE_NAME);

    const auto refInputsInfo = refNet.getInputsInfo();
    const auto refOutputsInfo = refNet.getOutputsInfo();

    IE_ASSERT(refInputsInfo.size() == 1);
    IE_ASSERT(refOutputsInfo.size() == 1);

    const auto& refInputName = refInputsInfo.begin()->first;
    const auto& refInputInfo = refInputsInfo.begin()->second;

    const auto refInputBlob = toLayout(toPrecision(inputBlob, refInputInfo->getTensorDesc().getPrecision()), refInputInfo->getTensorDesc().getLayout());

    const auto refOutputs = runInfer(refExeNet, {{refInputName, refInputBlob}}, false);
    IE_ASSERT(refOutputs.size() == 1);

    return refOutputs.begin()->second;
}

void KmbNetworkTestBase::checkLayouts(const BlobMap& actualOutputs,
                                      const std::unordered_map<std::string, Layout>& layouts) const {
    // FIXME: Currently only one output is allowed
    // Network can not explicitly set output layout
    ASSERT_TRUE(layouts.size() <= 1u);
    ASSERT_EQ(1u, actualOutputs.size());

    if (layouts.size() == 1) {
        const auto& actual   = *actualOutputs.begin();
        const auto& expected = *layouts.begin();
        ASSERT_EQ(expected.second, actual.second->getTensorDesc().getLayout());
    }
}

void KmbNetworkTestBase::checkPrecisions(const BlobMap& actualOutputs,
                                         const std::unordered_map<std::string, Precision>& precisions) const {
    // FIXME: Currently only one output is allowed
    // Network can not explicitly set output precision
    ASSERT_TRUE(precisions.size() <= 1u);
    ASSERT_EQ(1u, actualOutputs.size());

    if (precisions.size() == 1) {
        const auto& actual   = *actualOutputs.begin();
        const auto& expected = *precisions.begin();
        ASSERT_EQ(expected.second, actual.second->getTensorDesc().getPrecision());
    }
}

void KmbNetworkTestBase::runTest(
        const TestNetworkDesc& netDesc,
        const TestImageDesc& image,
        const CheckCallback& checkCallback) {
    if (!RUN_COMPILER || !RUN_REF_CODE) {
        if (DUMP_PATH.empty()) {
            SKIP() << "Compilation and/or REF_CODE were disabled, but IE_KMB_TESTS_DUMP_PATH vere not provided";
        }
    }

    auto exeNet = getExecNetwork(netDesc);

    const auto inputsInfo = exeNet.GetInputsInfo();
    const auto outputsInfo = exeNet.GetOutputsInfo();

    IE_ASSERT(inputsInfo.size() == 1);
    IE_ASSERT(outputsInfo.size() == 1);

    // HACK: to overcome IE bug with incorrect TensorDesc::setLayout
    const auto& inputInfo = inputsInfo.begin()->second;
    const auto inputTensorDesc = TensorDesc(inputInfo->getTensorDesc().getPrecision(), inputInfo->getTensorDesc().getDims(), inputInfo->getTensorDesc().getLayout());

    registerBlobGenerator(
        "input",
        inputTensorDesc,
        [&image](const TensorDesc& desc) {
            const auto blob = loadImage(image, desc.getDims()[1], desc.getDims()[2], desc.getDims()[3]);
            IE_ASSERT(blob->getTensorDesc().getDims() == desc.getDims());

            return toPrecision(toLayout(blob, desc.getLayout()), desc.getPrecision());
        });

    const auto inputBlob = getBlobByName("input");

    Blob::Ptr refOutputBlob;

    if (RUN_REF_CODE) {
        std::cout << "=== CALC REFERENCE WITH " << REF_DEVICE_NAME << std::endl;

        refOutputBlob = toDefLayout(toDefPrecision(calcRefOutput(netDesc, inputBlob)));

        if (EXPORT_BLOBS) {
            std::cout << "    === EXPORT REFERENCE" << std::endl;

            dumpBlob("output", refOutputBlob);
        }
    } else if (RUN_INFER) {
        std::cout << "=== IMPORT REFERENCE" << std::endl;

        const auto& outputInfo = outputsInfo.begin()->second;
        const auto& outputDims = outputInfo->getTensorDesc().getDims();

        const auto refOutputTensorDesc = TensorDesc(Precision::FP32, outputDims, TensorDesc::getLayoutByDims(outputDims));

        refOutputBlob = importBlob("output", refOutputTensorDesc);
    }

    if (RUN_INFER) {
        std::cout << "=== INFER" << std::endl;

        const auto& inputName = inputsInfo.begin()->first;

        const auto actualOutputs = runInfer(exeNet, {{inputName, inputBlob}}, true);
        IE_ASSERT(actualOutputs.size() == 1);

        const auto actualOutputBlob = actualOutputs.begin()->second;

        std::cout << "=== COMPARE WITH REFERENCE" << std::endl;

        checkLayouts(actualOutputs,    netDesc.outputLayouts());
        checkPrecisions(actualOutputs, netDesc.outputPrecisions());

        checkCallback(actualOutputBlob, refOutputBlob, inputTensorDesc);
    }
}

//
// KmbClassifyNetworkTest
//

void KmbClassifyNetworkTest::runTest(
        const TestNetworkDesc& netDesc,
        const TestImageDesc& image,
        size_t topK, float probTolerance) {
    const auto check = [=](const Blob::Ptr& actualBlob, const Blob::Ptr& refBlob, const TensorDesc&) {
        ASSERT_EQ(refBlob->getTensorDesc().getDims(), actualBlob->getTensorDesc().getDims());

        auto actualOutput = parseOutput(toFP32(actualBlob));
        auto refOutput = parseOutput(toFP32(refBlob));

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
            const auto actualIt = std::find_if(
                actualOutput.cbegin(), actualOutput.cend(),
                [&refElem](const std::pair<int, float> arg) {
                    return refElem.first == arg.first;
                });
            ASSERT_NE(actualIt, actualOutput.end());

            const auto& actualElem = *actualIt;

            const auto probDiff = std::fabs(refElem.second - actualElem.second);
            EXPECT_LE(probDiff, probTolerance)
                << refElem.first << " : " << refElem.second << " vs " << actualElem.second;
        }
    };

    KmbNetworkTestBase::runTest(netDesc, image, check);
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

void KmbDetectionNetworkTest::runTest(
        const TestNetworkDesc& netDesc,
        const TestImageDesc& image,
        float confThresh,
        float boxTolerance, float probTolerance) {
    const auto check = [=](const Blob::Ptr& actualBlob, const Blob::Ptr& refBlob, const TensorDesc& inputDesc) {
        const auto imgWidth = inputDesc.getDims().at(3);
        const auto imgHeight = inputDesc.getDims().at(2);

        auto actualOutput = parseOutput(toFP32(actualBlob), imgWidth, imgHeight, confThresh);
        auto refOutput = parseOutput(toFP32(refBlob), imgWidth, imgHeight, confThresh);

        checkBBoxOutputs(actualOutput, refOutput, imgWidth, imgHeight, boxTolerance, probTolerance);
    };

    KmbNetworkTestBase::runTest(netDesc, image, check);
}

std::vector<utils::YoloBBox> KmbDetectionNetworkTest::parseOutput(
        const Blob::Ptr& blob,
        size_t imgWidth, size_t imgHeight,
        float confThresh) {
    constexpr size_t ELEM_COUNT = 7;

    const auto count = blob->size() / ELEM_COUNT;

    std::vector<utils::YoloBBox> out;
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

        utils::YoloBBox bb (class_id, imgWidth * xmin, imgWidth * xmax, imgHeight * ymin, imgHeight * ymax, conf);

        out.push_back(bb);
    }

    return out;
}

void KmbDetectionNetworkTest::checkBBoxOutputs(std::vector<utils::YoloBBox> &actualOutput,
        std::vector<utils::YoloBBox> &refOutput, int imgWidth, int imgHeight, float boxTolerance, float probTolerance) {
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

            const utils::YoloBox actualBox {
                    actualBB.left / imgWidth,
                    actualBB.top / imgHeight,
                    (actualBB.right - actualBB.left) / imgWidth,
                    (actualBB.bottom - actualBB.top) / imgHeight
            };
            const utils::YoloBox refBox {
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

        EXPECT_TRUE(found)
                            << "maxBoxError=" << maxBoxError << " "
                            << "maxProbError=" << maxProbError;
    }
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// YOLOV2NetworkAdapter ////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void KmbYoloV2NetworkTest::runTest(
        const TestNetworkDesc& netDesc,
        const TestImageDesc& image,
        float confThresh,
        float boxTolerance, float probTolerance,
        bool isTiny) {
    const auto check = [=](const Blob::Ptr& actualBlob, const Blob::Ptr& refBlob, const TensorDesc& inputDesc) {
        const auto imgWidth = inputDesc.getDims().at(3);
        const auto imgHeight = inputDesc.getDims().at(2);

        auto actualOutput = utils::parseYoloOutput(toFP32(actualBlob), imgWidth, imgHeight, confThresh, isTiny);
        auto refOutput = utils::parseYoloOutput(toFP32(refBlob), imgWidth, imgHeight, confThresh, isTiny);

        checkBBoxOutputs(actualOutput, refOutput, imgWidth, imgHeight, boxTolerance, probTolerance);
    };

    KmbNetworkTestBase::runTest(netDesc, image, check);
}
