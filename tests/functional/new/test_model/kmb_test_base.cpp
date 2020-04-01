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
#include <test_kmb_models_path.h>

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

    core = PluginCache::get().ie(DEVICE_NAME);
    if (!LOG_LEVEL.empty()) {
        core->SetConfig({{CONFIG_KEY(LOG_LEVEL), LOG_LEVEL}}, DEVICE_NAME);
    }
    if (PRINT_PERF_COUNTERS) {
        core->SetConfig({{CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(YES)}}, DEVICE_NAME);
    }
    if (RUN_COMPILER) {
        // MCM frontend supports only outputs with NHWC layout and U8 or FP16 precision
        // but many tests in this framework use classification, so NC output is required
        // also, FP32 precision is used to compare with references
        // TODO either update tests to use NHWC with FP16, or update MCM compiler to support NC with FP32
        core->SetConfig({{"VPU_COMPILER_ALLOW_NC_OUTPUT", CONFIG_VALUE(YES)},
            {"VPU_COMPILER_ALLOW_FP32_OUTPUT", CONFIG_VALUE(YES)}}, DEVICE_NAME);
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
    ASSERT_EQ(refDesc.getDims().size(), actualDesc.getDims().size());

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

Blob::Ptr KmbNetworkTestBase::loadImage(const TestImageDesc& image, int channels) {
    std::ostringstream imageFilePath;
    imageFilePath << get_data_path() << "/" << image.imageFileName();
    if (!exist(imageFilePath.str())) {
        imageFilePath.str("");
        imageFilePath << getTestDataPathNonFatal() << "/" << image.imageFileName();
    }

    FormatReader::ReaderPtr reader(imageFilePath.str().c_str());
    IE_ASSERT(reader.get() != nullptr);

    const size_t C = channels;
    const size_t H = reader->height();
    const size_t W = reader->width();

    const auto tensorDesc = TensorDesc(Precision::FP32, {1, C, H, W}, Layout::NHWC);

    const auto blob = make_blob_with_precision(tensorDesc);
    blob->allocate();

    const auto imagePtr = reader->getData().get();
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
    std::string modelPath = KmbModelsPath() + '/' + netDesc.irFileName();
    if (!exist(modelPath.c_str()))
        modelPath = ModelsPath() + "/" + netDesc.irFileName();

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
            const auto blob = loadImage(image, desc.getDims()[1]);
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

std::vector<KmbDetectionNetworkTest::BBox> KmbDetectionNetworkTest::parseOutput(
        const Blob::Ptr& blob,
        size_t imgWidth, size_t imgHeight,
        float confThresh) {
    constexpr size_t ELEM_COUNT = 7;

    const auto count = blob->size() / ELEM_COUNT;

    std::vector<KmbDetectionNetworkTest::BBox> out;
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

        BBox bb (class_id, imgWidth * xmin, imgWidth * xmax, imgHeight * ymin, imgHeight * ymax, conf);

        out.push_back(bb);
    }

    return out;
}

float KmbDetectionNetworkTest::overlap(float x1, float w1, float x2, float w2) {
    const float l1 = x1 - w1 / 2;
    const float l2 = x2 - w2 / 2;
    const float left = l1 > l2 ? l1 : l2;

    const float r1 = x1 + w1 / 2;
    const float r2 = x2 + w2 / 2;
    const float right = r1 < r2 ? r1 : r2;

    return right - left;
}

float KmbDetectionNetworkTest::boxIntersection(const Box& a, const Box& b) {
    const float w = overlap(a.x, a.w, b.x, b.w);
    const float h = overlap(a.y, a.h, b.y, b.h);

    if (w < 0 || h < 0) {
        return 0.0f;
    }

    return w * h;
}

float KmbDetectionNetworkTest::boxUnion(const Box& a, const Box& b) {
    const float i = boxIntersection(a, b);
    return a.w * a.h + b.w * b.h - i;
}

float KmbDetectionNetworkTest::boxIou(const Box& a, const Box& b) {
    return boxIntersection(a, b) / boxUnion(a, b);
}

void KmbDetectionNetworkTest::checkBBoxOutputs(std::vector<BBox> &actualOutput, std::vector<BBox> &refOutput,
        int imgWidth, int imgHeight, float boxTolerance, float probTolerance) {
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

            const KmbDetectionNetworkTest::Box actualBox {
                    actualBB.left / imgWidth,
                    actualBB.top / imgHeight,
                    (actualBB.right - actualBB.left) / imgWidth,
                    (actualBB.top - actualBB.bottom) / imgHeight
            };
            const KmbDetectionNetworkTest::Box refBox {
                    refBB.left / imgWidth,
                    refBB.top / imgHeight,
                    (refBB.right - refBB.left) / imgWidth,
                    (refBB.top - refBB.bottom) / imgHeight
            };

            const auto boxIntersection = boxIou(actualBox, refBox);
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

int KmbYoloV2NetworkTest::entryIndex(int lw, int lh, int lcoords, int lclasses, int lnum, int batch, int location, int entry) {
    int n = location / (lw * lh);
    int loc = location % (lw * lh);
    int loutputs = lh * lw * lnum * (lclasses + lcoords + 1);
    return batch * loutputs + n * lw * lh * (lcoords + lclasses + 1) + entry * lw * lh + loc;
}

KmbYoloV2NetworkTest::Box KmbYoloV2NetworkTest::getRegionBox(float *x, const std::vector<float> &biases, int n, int index, int i, int j, int w, int h, int stride) {
    Box b;
    b.x = (i + x[index + 0 * stride]) / w;
    b.y = (j + x[index + 1 * stride]) / h;
    b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;

    return b;
}

void KmbYoloV2NetworkTest::correctRegionBoxes(std::vector<Box> &boxes, int n, int w, int h, int netw, int neth, int relative) {
    int new_w = 0;
    int new_h = 0;
    if (((float) netw / w) < ((float) neth / h)) {
        new_w = netw;
        new_h = (h * netw) / w;
    } else {
        new_h = neth;
        new_w = (w * neth) / h;
    }

    IE_ASSERT(static_cast<int>(boxes.size()) >= n);
    for (int i = 0; i < n; ++i) {
        KmbDetectionNetworkTest::Box b = boxes[i];
        b.x = (b.x - (netw - new_w) / 2. / netw) / ((float) new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / ((float) new_h / neth);
        b.w *= (float) netw / new_w;
        b.h *= (float) neth / new_h;
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        boxes[i] = b;
    }
}

void KmbYoloV2NetworkTest::getRegionBoxes(std::vector<float> &predictions,
                      int lw,
                      int lh,
                      int lcoords,
                      int lclasses,
                      int lnum,
                      int w,
                      int h,
                      int netw,
                      int neth,
                      float thresh,
                      std::vector<std::vector<float>> &probs,
                      std::vector<Box> &boxes,
                      int relative,
                      const std::vector<float> &anchors) {
    for (int i = 0; i < lw * lh; ++i) {
        int row = i / lw;
        int col = i % lw;
        for (int n = 0; n < lnum; ++n) {
            int index = n * lw * lh + i;
            int obj_index = entryIndex(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, lcoords);
            int box_index = entryIndex(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, 0);
            float scale = predictions[obj_index];

            boxes[index] = getRegionBox(predictions.data(), anchors, n, box_index, col, row, lw, lh, lw * lh);

            float max = 0;
            for (int j = 0; j < lclasses; ++j) {
                int class_index = entryIndex(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, lcoords + 1 + j);
                float prob = scale * predictions[class_index];
                probs[index][j] = (prob > thresh) ? prob : 0;
                if (prob > max) max = prob;
            }
            probs[index][lclasses] = max;

        }
    }
    correctRegionBoxes(boxes, lw * lh * lnum, w, h, netw, neth, relative);
}

int KmbYoloV2NetworkTest::maxIndex(std::vector<float> &a, int n) {
    if (n <= 0) {
        return -1;
    }
    int max_i = 0;
    float max = a[0];
    for (int i = 1; i < n; ++i) {
        if (a[i] > max) {
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}
void KmbYoloV2NetworkTest::doNMSSort(std::vector<Box> &boxes, std::vector<std::vector<float>> &probs,
        int total, int classes, float thresh) {
    std::vector<sortableBBox> boxCandidates;

    for (int i = 0; i < total; ++i) {
        sortableBBox candidate(i, 0, probs);
        boxCandidates.push_back(candidate);
    }

    for (int k = 0; k < classes; ++k) {
        for (int i = 0; i < total; ++i) {
            boxCandidates[i].cclass = k;
        }
        std::sort(boxCandidates.begin(), boxCandidates.end(), [ ]( const sortableBBox& a, const sortableBBox& b)
        {
            float diff = a.probs[a.index][b.cclass] - b.probs[b.index][b.cclass];
            return diff > 0;
        });
        for (int i = 0; i < total; ++i) {
            if (probs[boxCandidates[i].index][k] == 0) continue;
            KmbDetectionNetworkTest::Box a = boxes[boxCandidates[i].index];
            for (int j = i + 1; j < total; ++j) {
                KmbDetectionNetworkTest::Box b = boxes[boxCandidates[j].index];
                if (KmbDetectionNetworkTest::boxIou(a, b) > thresh) {
                    probs[boxCandidates[j].index][k] = 0;
                }
            }
        }
    }
}

void KmbYoloV2NetworkTest::getDetections(int imw,
                    int imh,
                    int num,
                    float thresh,
                    KmbDetectionNetworkTest::Box *boxes,
                    std::vector<std::vector<float>> &probs,
                    int classes,
                    std::vector<KmbDetectionNetworkTest::BBox> &detect_result) {
    for (int i = 0; i < num; ++i) {
        int idxClass = maxIndex(probs[i], classes);
        float prob = probs[i][idxClass];

        if (prob > thresh) {
            KmbDetectionNetworkTest::Box b = boxes[i];

            float left = (b.x - b.w / 2.) * imw;
            float right = (b.x + b.w / 2.) * imw;
            float top = (b.y - b.h / 2.) * imh;
            float bot = (b.y + b.h / 2.) * imh;

            KmbDetectionNetworkTest::BBox bx(idxClass,
                              left,
                              top,
                              right,
                              bot,
                              prob);
            detect_result.push_back(bx);
        }
    }
}

std::vector<KmbDetectionNetworkTest::BBox> KmbYoloV2NetworkTest::yolov2BoxExtractor(
        float threshold,
        std::vector<float> &net_out,
        int imgWidth,
        int imgHeight,
        int class_num,
        bool isTiny
) {
    int classes = class_num;
    int coords = 4;
    int num = 5;
    std::vector<BBox> boxes_result;

    std::vector<float> TINY_YOLOV2_ANCHORS = {1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52};
    std::vector<float> YOLOV2_ANCHORS = {1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071};
    std::vector<float> YOLOV2_ANCHORS_80_CLASSES =
            {0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};

    int imw = 416;
    int imh = 416;

    int lw = 13;
    int lh = 13;
    float nms = 0.4;

    std::vector<Box> boxes(lw * lh * num);
    std::vector<std::vector<float>> probs(lw * lh * num, std::vector<float> (classes + 1, 0.0));

    std::vector<float> anchors;
    if (isTiny) {
        anchors = TINY_YOLOV2_ANCHORS;
    } else {
        anchors = YOLOV2_ANCHORS;
        if (class_num == 80) {
            anchors = YOLOV2_ANCHORS_80_CLASSES;
        }
    }

    getRegionBoxes(net_out,
                    lw,
                    lh,
                    coords,
                    classes,
                    num,
                    imgWidth,
                    imgHeight,
                    imw,
                    imh,
                    threshold,
                    probs,
                    boxes,
                    1,
                    anchors);

    doNMSSort(boxes, probs, lw * lh * num, classes, nms);
    getDetections(imw, imh, lw * lh * num, threshold, boxes.data(), probs, classes, boxes_result);

    return boxes_result;
}

std::vector<KmbDetectionNetworkTest::BBox> KmbYoloV2NetworkTest::parseOutput(
        const Blob::Ptr& blob,
        size_t imgWidth, size_t imgHeight,
        float confThresh, bool isTiny) {
    auto ptr = blob->cbuffer().as<float*>();
    IE_ASSERT(ptr != nullptr);

    std::vector<float> results (blob->size());
    for (size_t i = 0; i < blob->size(); i++) {
        results[i] = ptr[i];
    }

    std::vector<BBox> out;
    int classes = 20;
    out = yolov2BoxExtractor(confThresh, results, imgWidth, imgHeight, classes, isTiny);

    return out;
}

void KmbYoloV2NetworkTest::runTest(
        const TestNetworkDesc& netDesc,
        const TestImageDesc& image,
        float confThresh,
        float boxTolerance, float probTolerance,
        bool isTiny) {
    const auto check = [=](const Blob::Ptr& actualBlob, const Blob::Ptr& refBlob, const TensorDesc& inputDesc) {
        const auto imgWidth = inputDesc.getDims().at(3);
        const auto imgHeight = inputDesc.getDims().at(2);

        auto actualOutput = parseOutput(toFP32(actualBlob), imgWidth, imgHeight, confThresh, isTiny);
        auto refOutput = parseOutput(toFP32(refBlob), imgWidth, imgHeight, confThresh, isTiny);

        checkBBoxOutputs(actualOutput, refOutput, imgWidth, imgHeight, boxTolerance, probTolerance);
    };

    KmbNetworkTestBase::runTest(netDesc, image, check);
}
