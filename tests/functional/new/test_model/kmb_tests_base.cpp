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

#include "kmb_tests_base.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>

#include <blob_factory.hpp>
#include <plugin_cache.hpp>

#include <test_model_path.hpp>
#include <single_layer_common.hpp>
#include <format_reader_ptr.h>

//
// KmbTestBase
//

namespace {

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
        return std::stoi(var);
    }

#ifdef PLATFORM_ARM
    return false;
#else
    return true;
#endif
}();

const bool RUN_REF_CODE = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_RUN_REF_CODE")) {
        return std::stoi(var);
    }

#ifdef PLATFORM_ARM
    return false;
#else
    return true;
#endif
}();

const bool RUN_INFER = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_RUN_INFER")) {
        return std::stoi(var);
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

const bool RAW_EXPORT = []() -> bool {
    if (const auto var = std::getenv("IE_KMB_TESTS_RAW_EXPORT")) {
        return std::stoi(var);
    }

    return false;
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
    core->SetConfig({{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)}}, DEVICE_NAME);

    dumpBaseName = cleanName(vpu::formatString("%v_%v", testInfo->test_case_name(), testInfo->name()));

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

#ifndef ENABLE_MCM_COMPILER
    if (RUN_COMPILER) {
        THROW_IE_EXCEPTION << "KMB Plugin was built without mcmCompiler";
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

    if (RUN_REF_CODE) {
        std::cout << "=== GENERATE BLOB " << blobName << std::endl;

        blob = blobGenerators.at(blobName).second(blobDesc);
        IE_ASSERT(blob->getTensorDesc() == blobDesc);

        if (!DUMP_PATH.empty()) {
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
        const CNNNetwork& net,
        const std::map<std::string, std::string>& config) {
    ExecutableNetwork exeNet;

    if (RUN_COMPILER) {
        std::cout << "=== COMPILE NETWORK" << std::endl;

        exeNet = core->LoadNetwork(net, DEVICE_NAME, config);

        if (!DUMP_PATH.empty()) {
            std::cout << "    === EXPORT NETWORK" << std::endl;

            exportNetwork(exeNet);
        }
    } else if (RUN_INFER) {
        std::cout << "=== IMPORT NETWORK" << std::endl;

        exeNet = importNetwork();
    }

    return exeNet;
}

BlobMap KmbTestBase::getRefOutputs(
        TestNetwork& testNet,
        const BlobMap& inputs) {
    BlobMap refOutputs;

    if (RUN_REF_CODE) {
        std::cout << "=== CALC REFERENCE" << std::endl;

        refOutputs = testNet.calcRef(inputs);

        if (!DUMP_PATH.empty()) {
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
    std::cout << "=== COMPARE WITH REFERENCE" << std::endl;

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
        const auto actualOutputs = runInfer(exeNet, inputs);

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
    std::cout << "=== INFER" << std::endl;

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

void KmbNetworkTest::runClassifyNetworkTest(
        const CNNNetwork& net,
        const Blob::Ptr& inputBlob,
        const std::vector<std::pair<int, float>>& refTopK, float probTolerance) {
    auto exeNet = getExecNetwork(net);

    if (RUN_INFER) {
        const auto inputsInfo = exeNet.GetInputsInfo();
        IE_ASSERT(inputsInfo.size() == 1);

        const auto actualOutputs = runInfer(exeNet, {{(*inputsInfo.begin()).first, inputBlob}});

        std::cout << "=== COMPARE WITH REFERENCE" << std::endl;

        const auto outputsInfo = exeNet.GetOutputsInfo();
        IE_ASSERT(outputsInfo.size() == 1);

        const auto& actualOutputBlob = actualOutputs.at((*outputsInfo.begin()).first);
        const auto actualOutput = parseClassifyOutput(toFP32(actualOutputBlob));

        ASSERT_GE(actualOutput.size(), refTopK.size());

        std::cout << "Ref Top:" << std::endl;
        for (size_t i = 0; i < refTopK.size(); ++i) {
            std::cout << i << " : " << refTopK[i].first << " : " << refTopK[i].second * 100 << "%" << std::endl;
        }

        std::cout << "Actual top:" << std::endl;
        for (size_t i = 0; i < refTopK.size(); ++i) {
            std::cout << i << " : " << actualOutput[i].first << " : " << actualOutput[i].second * 100 << "%" << std::endl;
        }

        for (const auto& refElem : refTopK) {
            const auto actualIt = std::find_if(actualOutput.cbegin(), actualOutput.cend(), [&refElem](const std::pair<int, float> arg) { return refElem.first == arg.first; });
            ASSERT_NE(actualIt, actualOutput.end());

            const auto& actualElem = *actualIt;

            const auto probDiff = std::fabs(refElem.second * 100 - actualElem.second * 100);
            EXPECT_LE(probDiff, probTolerance)
                << refElem.first << " : " << refElem.second << " vs " << actualElem.second;
        }
    }
}

void KmbNetworkTest::runClassifyNetworkTest(
        const std::string& modelFileName,
        const std::string& inputFileName,
        const std::vector<std::pair<int, float>>& refTopK, float probTolerance) {
    const auto net = loadNetwork(modelFileName);
    const auto inputBlob = loadImage(inputFileName);

    runClassifyNetworkTest(net, inputBlob, refTopK, probTolerance);
}

void KmbNetworkTest::runClassifyNetworkTest(
        const std::string& modelFileName,
        const std::string& inputFileName,
        size_t topK, float probTolerance) {
    std::vector<std::pair<int, float>> refTopK;

    const auto net = loadNetwork(modelFileName);
    const auto inputBlob = loadImage(inputFileName);

    const auto inputsInfo = net.getInputsInfo();
    IE_ASSERT(inputsInfo.size() == 1);
    const auto inputName = (*inputsInfo.begin()).first;

    const auto outputsInfo = net.getOutputsInfo();
    IE_ASSERT(outputsInfo.size() == 1);
    const auto outputName = (*outputsInfo.begin()).first;

    if (RUN_REF_CODE) {
        std::cout << "=== CALC REFERENCE WITH " << REF_DEVICE_NAME << std::endl;

        auto refExeNet = core->LoadNetwork(net, REF_DEVICE_NAME);
        const auto refOutputs = runInfer(refExeNet, {{inputName, inputBlob}});

        const auto& refOutputBlob = refOutputs.at(outputName);
        refTopK = parseClassifyOutput(toFP32(refOutputBlob));

        IE_ASSERT(refTopK.size() >= topK);
        refTopK.resize(topK);

        const auto refTopKDesc = TensorDesc(Precision::FP32, {topK, 2}, Layout::NC);
        const auto refTopKBlob = make_blob_with_precision(refTopKDesc, refTopK.data());

        if (!DUMP_PATH.empty()) {
            std::cout << "    === EXPORT REFERENCE" << std::endl;

            dumpBlob(outputName, refTopKBlob);
        }
    } else if (RUN_INFER) {
        std::cout << "=== IMPORT REFERENCE" << std::endl;

        const auto refTopKDesc = TensorDesc(Precision::FP32, {topK, 2}, Layout::NC);
        const auto refTopKBlob = importBlob(outputName, refTopKDesc);

        refTopK.resize(topK);
        std::copy_n(refTopKBlob->cbuffer().as<const std::pair<int, float>*>(), topK, refTopK.data());
    }

    runClassifyNetworkTest(net, inputBlob, refTopK, probTolerance);
}

CNNNetwork KmbNetworkTest::loadNetwork(const std::string& modelFileName) {
    std::ostringstream irFileName;
    irFileName << "/" << modelFileName << ".xml";

    ModelsPath modelPath;
    modelPath << irFileName.str();

    return core->ReadNetwork(modelPath);
}

Blob::Ptr KmbNetworkTest::loadImage(const std::string& imageFileName) {
    std::ostringstream imageFilePath;
    imageFilePath << get_data_path() << "/" << imageFileName;

    FormatReader::ReaderPtr reader(imageFilePath.str().c_str());
    IE_ASSERT(reader.get() != nullptr);

    const size_t C = 3;
    const size_t H = reader->height();
    const size_t W = reader->width();

    const auto tensorDesc = TensorDesc(Precision::FP32, {1, C, H, W}, Layout::NCHW);

    const auto blob = make_blob_with_precision(tensorDesc);
    blob->allocate();

    const auto imagePtr = reader->getData();
    const auto blobPtr = blob->buffer().as<float*>();

    IE_ASSERT(imagePtr != nullptr);
    IE_ASSERT(blobPtr != nullptr);

    for (size_t c = 0; c < C; ++c) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                const auto blobInd = tensorDesc.offset({1, c, h, w});
                blobPtr[blobInd] = imagePtr.get()[c + w * C + h * C * W];
            }
        }
    }

    return blob;
}

std::vector<std::pair<int, float>> KmbNetworkTest::parseClassifyOutput(const Blob::Ptr& blob) {
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
