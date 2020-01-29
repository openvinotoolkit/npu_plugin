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

#include <blob_factory.hpp>
#include <plugin_cache.hpp>

#include <test_model_path.hpp>
#include <single_layer_common.hpp>
#include <format_reader_ptr.h>
#include <vpu/utils/error.hpp>

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

const std::string DEVICE_NAME = []() -> std::string {
    if (const auto var = std::getenv("IE_KMB_TESTS_DEVICE_NAME")) {
        return var;
    }

    return "KMB";
}();

const std::string REF_DEVICE_NAME = []() -> std::string {
    if (const auto var = std::getenv("IE_KMB_TESTS_REF_DEVICE_NAME")) {
        return var;
    }

    return "CPU";
}();

const bool RUN_COMPILER = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_RUN_COMPILER")) {
        return strToBool("IE_KMB_TESTS_RUN_COMPILER", var);
    }

#if defined(PLATFORM_ARM) || !defined(ENABLE_MCM_COMPILER)
    return false;
#else
    return true;
#endif
}();

const bool RUN_REF_CODE = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_RUN_REF_CODE")) {
        return strToBool("IE_KMB_TESTS_RUN_REF_CODE", var);
    }

#ifdef PLATFORM_ARM
    return false;
#else
    return true;
#endif
}();

const bool RUN_INFER = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_RUN_INFER")) {
        return strToBool("IE_KMB_TESTS_RUN_INFER", var);
    }

#ifdef PLATFORM_ARM
    return true;
#else
    return false;
#endif
}();

const std::string DUMP_PATH = []() -> std::string {
    if (const auto var = std::getenv("IE_KMB_TESTS_DUMP_PATH")) {
        return var;
    }

    return std::string();
}();

const bool EXPORT_NETWORK = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_EXPORT_NETWORK")) {
        return strToBool("IE_KMB_TESTS_EXPORT_NETWORK", var);
    }

    return RUN_COMPILER && !DUMP_PATH.empty();
}();

const bool RAW_EXPORT = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_RAW_EXPORT")) {
        return strToBool("IE_KMB_TESTS_RAW_EXPORT", var);
    }

    return false;
}();

const bool GENERATE_BLOBS = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_GENERATE_BLOBS")) {
        return strToBool("IE_KMB_TESTS_GENERATE_BLOBS", var);
    }

    return RUN_REF_CODE;
}();

const bool EXPORT_BLOBS = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_EXPORT_BLOBS")) {
        return strToBool("IE_KMB_TESTS_EXPORT_BLOBS", var);
    }

    return GENERATE_BLOBS && !DUMP_PATH.empty();
}();

const std::string LOG_LEVEL = []() -> std::string {
    if (const auto var = std::getenv("IE_KMB_TESTS_LOG_LEVEL")) {
        return var;
    }

    return std::string();
}();

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

void KmbTestBase::SetUp() {
    ASSERT_NO_FATAL_FAILURE(TestsCommon::SetUp());

    const auto testInfo = testing::UnitTest::GetInstance()->current_test_info();
    IE_ASSERT(testInfo != nullptr);

    rd.seed();

    core = PluginCache::get().ie(DEVICE_NAME);
    if (!LOG_LEVEL.empty()) {
        core->SetConfig({{CONFIG_KEY(LOG_LEVEL), LOG_LEVEL}}, DEVICE_NAME);
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

#ifndef ENABLE_MCM_COMPILER
    if (RUN_COMPILER) {
        FAIL() << "KMB Plugin was built without mcmCompiler";
    }
#endif
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

BlobMap KmbTestBase::getInputs(TestNetwork& testNet) {
    BlobMap inputs;

    for (const auto& info : testNet.getInputsInfo()) {
        const auto blob = getBlobByName(info->getName());
        inputs.insert({info->getName(), blob});
    }

    return inputs;
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

ExecutableNetwork KmbTestBase::getExecNetwork(
        TestNetwork& testNet) {
    return getExecNetwork(
        [&testNet]() {
            return testNet.getCNNNetwork();
        },
        [&testNet]() {
            return testNet.compileConfig();
        });
}

BlobMap KmbTestBase::getRefOutputs(
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

void KmbTestBase::runTest(
        TestNetwork& testNet,
        const BlobMap& inputs,
        float tolerance, CompareMethod method) {
    auto exeNet = getExecNetwork(testNet);

    const auto refOutputs = getRefOutputs(testNet, inputs);

    if (RUN_INFER) {
        std::cout << "=== INFER" << std::endl;

        const auto actualOutputs = runInfer(exeNet, inputs);

        std::cout << "=== COMPARE WITH REFERENCE" << std::endl;

        compareWithReference(actualOutputs, refOutputs, tolerance, method);
    }
}

void KmbTestBase::runTest(
        TestNetwork& testNet,
        float tolerance, CompareMethod method) {
    const auto inputs = getInputs(testNet);
    runTest(testNet, inputs, tolerance, method);
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

BlobMap KmbTestBase::runInfer(ExecutableNetwork& exeNet, const BlobMap& inputs) {
    auto inferRequest = exeNet.CreateInferRequest();

    for (const auto& p : inputs) {
        inferRequest.SetBlob(p.first, p.second);
    }

    inferRequest.Infer();

    const auto outputsInfo = exeNet.GetOutputsInfo();

    BlobMap out;

    for (const auto& p : outputsInfo) {
        out.insert({p.first, inferRequest.GetBlob(p.first)});
    }

    return out;
}

//
// KmbNetworkTest
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

Blob::Ptr KmbNetworkTestBase::loadImage(const std::string& imageFilePath) {
    FormatReader::ReaderPtr reader(imageFilePath.c_str());
    IE_ASSERT(reader.get() != nullptr);

    const size_t C = 3;
    const size_t H = reader->height();
    const size_t W = reader->width();

    const auto tensorDesc = TensorDesc(Precision::FP32, {1, C, H, W}, Layout::NHWC);

    const auto blob = make_blob_with_precision(tensorDesc);
    blob->allocate();

    const auto imagePtr = reader->getData().get();
    const auto blobPtr = blob->buffer().as<float*>();

    IE_ASSERT(imagePtr != nullptr);
    IE_ASSERT(blobPtr != nullptr);

    std::copy_n(imagePtr, blob->size(), blobPtr);

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

    const auto refOutputs = runInfer(refExeNet, {{refInputName, refInputBlob}});
    IE_ASSERT(refOutputs.size() == 1);

    return refOutputs.begin()->second;
}

void KmbNetworkTestBase::runTest(
        const TestNetworkDesc& netDesc,
        const std::string& inputFileName,
        const CheckCallback& checkCallback) {
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
        [&inputFileName](const TensorDesc& desc) {
            std::ostringstream inputFilePath;
            inputFilePath << get_data_path() << "/" << inputFileName;

            const auto blob = loadImage(inputFilePath.str());
            IE_ASSERT(blob->getTensorDesc().getDims() == desc.getDims());

            return toPrecision(toLayout(blob, desc.getLayout()), desc.getPrecision());
        });

    const auto inputBlob = getBlobByName("input");

    Blob::Ptr refOutputBlob;

    if (RUN_REF_CODE) {
        std::cout << "=== CALC REFERENCE WITH " << REF_DEVICE_NAME << std::endl;

        refOutputBlob = toDefLayout(toFP32(calcRefOutput(netDesc, inputBlob)));

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

        const auto actualOutputs = runInfer(exeNet, {{inputName, inputBlob}});
        IE_ASSERT(actualOutputs.size() == 1);

        const auto actualOutputBlob = actualOutputs.begin()->second;

        std::cout << "=== COMPARE WITH REFERENCE" << std::endl;

        checkCallback(actualOutputBlob, refOutputBlob, inputTensorDesc);
    }
}

void KmbClassifyNetworkTest::runTest(
        const TestNetworkDesc& netDesc,
        const std::string& inputFileName,
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

    KmbNetworkTestBase::runTest(netDesc, inputFileName, check);
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

void KmbDetectionNetworkTest::runTest(
        const TestNetworkDesc& netDesc,
        const std::string& inputFileName,
        float confThresh,
        float boxTolerance, float probTolerance) {
    const auto check = [=](const Blob::Ptr& actualBlob, const Blob::Ptr& refBlob, const TensorDesc& inputDesc) {
        const auto imgWidth = inputDesc.getDims().at(3);
        const auto imgHeight = inputDesc.getDims().at(2);

        auto actualOutput = parseOutput(toFP32(actualBlob), imgWidth, imgHeight, confThresh);
        auto refOutput = parseOutput(toFP32(refBlob), imgWidth, imgHeight, confThresh);

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

        ASSERT_GE(actualOutput.size(), refOutput.size());

        for (const auto& refBB : refOutput) {
            bool found = false;

            float maxBoxError = 0.0f;
            float maxProbError = 0.0f;

            for (const auto& actualBB : actualOutput) {
                if (actualBB.idx != refBB.idx) {
                    continue;
                }

                const Box actualBox {
                    actualBB.left / imgWidth,
                    actualBB.top / imgHeight,
                    (actualBB.right - actualBB.left) / imgWidth,
                    (actualBB.bottom - actualBB.top) / imgHeight
                };
                const Box refBox {
                    refBB.left / imgWidth,
                    refBB.top / imgHeight,
                    (refBB.right - refBB.left) / imgWidth,
                    (refBB.bottom - refBB.top) / imgHeight
                };

                const auto boxError = box_iou(actualBox, refBox);
                maxBoxError = std::max(maxBoxError, boxError);

                const auto probError = std::fabs(actualBB.prob - refBB.prob);
                maxProbError = std::max(maxProbError, probError);

                if (boxError < boxTolerance) {
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
    };

    KmbNetworkTestBase::runTest(netDesc, inputFileName, check);
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

        BBox bb;
        bb.idx = class_id;
        bb.prob = conf;
        bb.left = imgWidth * xmin;
        bb.right = imgWidth * xmax;
        bb.top = imgHeight * ymin;
        bb.bottom = imgHeight * ymax;

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

float KmbDetectionNetworkTest::box_intersection(const Box& a, const Box& b) {
    const float w = overlap(a.x, a.w, b.x, b.w);
    const float h = overlap(a.y, a.h, b.y, b.h);

    if (w < 0 || h < 0) {
        return 0.0f;
    }

    return w * h;
}

float KmbDetectionNetworkTest::box_union(const Box& a, const Box& b) {
    const float i = box_intersection(a, b);
    return a.w * a.h + b.w * b.h - i;
}

float KmbDetectionNetworkTest::box_iou(const Box& a, const Box& b) {
    return box_intersection(a, b) / box_union(a, b);
}
