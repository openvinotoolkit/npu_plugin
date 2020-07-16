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

#include <file_reader.h>
#include <gtest/gtest.h>
#include <fstream>

#include <allocators.hpp>
#include <test_model/kmb_test_utils.hpp>

#include "models/model_pooling.h"
#include "vpu_layers_tests.hpp"

using namespace InferenceEngine;
using namespace vpu;

using KmbPrivateConfig = std::map<std::string, std::string>;
using VPUAllocatorPtr = std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator>;

struct PrivateConfigTestParams final {
    std::string _testDescription;
    std::string _modelPath;
    std::string _inputPath;
    std::string _referencePath;
    bool _preProc;  // false -> create fake NHWC input
    bool _checkSIPP;
    KmbPrivateConfig _privateConfig;
    size_t _inputWidth;
    size_t _inputHeight;
    size_t _nClasses;

    PrivateConfigTestParams& testDescription(const std::string& test_description) {
        this->_testDescription = test_description;
        return *this;
    }

    PrivateConfigTestParams& modelPath(const std::string& model_path) {
        this->_modelPath = model_path;
        return *this;
    }

    PrivateConfigTestParams& inputPath(const std::string& input_path) {
        this->_inputPath = input_path;
        return *this;
    }

    PrivateConfigTestParams& referencePath(const std::string& reference_path) {
        this->_referencePath = reference_path;
        return *this;
    }

    PrivateConfigTestParams& preProc(const bool& pre_proc) {
        this->_preProc = pre_proc;
        return *this;
    }

    PrivateConfigTestParams& checkSIPP(const bool& check_SIPP) {
        this->_checkSIPP = check_SIPP;
        return *this;
    }

    PrivateConfigTestParams& privateConfig(const KmbPrivateConfig& private_config) {
        this->_privateConfig = private_config;
        return *this;
    }

    PrivateConfigTestParams& inputWidth(const size_t& input_width) {
        this->_inputWidth = input_width;
        return *this;
    }

    PrivateConfigTestParams& inputHeight(const size_t& input_height) {
        this->_inputHeight = input_height;
        return *this;
    }

    PrivateConfigTestParams& nClasses(const size_t& n_classes) {
        this->_nClasses = n_classes;
        return *this;
    }
};
std::ostream& operator<<(std::ostream& os, const PrivateConfigTestParams& p) {
    vpu::formatPrint(os,
        "[testDescription:%v, modelPath:%v, inputPath:%v, referencePath:%v, preProc:%v, checkSIPP:%v, "
        "privateConfig:%v, inputWidth:%v, inputHeight:%v, nClasses:%v]",
        p._testDescription, p._modelPath, p._inputPath, p._referencePath, p._preProc, p._checkSIPP, p._privateConfig,
        p._inputWidth, p._inputHeight, p._nClasses);
    return os;
}

class KmbPrivateConfigTests : public vpuLayersTests, public testing::WithParamInterface<PrivateConfigTestParams> {
protected:
    Blob::Ptr runInferWithConfig(const std::string& model_path, const std::string& input_path, bool pre_proc,
        const KmbPrivateConfig& config, size_t input_width, size_t input_height) const;

    Blob::Ptr readReference(const std::string& reference_path, const TensorDesc& tensor_desc) const;

    Blob::Ptr createFakeNHWCBlob(const Blob::Ptr& blob) const;

    bool isSIPP() const;
};

bool KmbPrivateConfigTests::isSIPP() const {
    std::string USE_SIPP = std::getenv("USE_SIPP") != nullptr ? std::getenv("USE_SIPP") : "";
    return !(USE_SIPP.find("0") != std::string::npos);
}

Blob::Ptr KmbPrivateConfigTests::createFakeNHWCBlob(const Blob::Ptr& blob) const {
    auto tensorDesc = blob->getTensorDesc();
    if (tensorDesc.getLayout() != Layout::NHWC) {
        THROW_IE_EXCEPTION << "fakeNHWCBlob works only with NHWC format";
    }

    if (tensorDesc.getDims()[1] != 3) {
        THROW_IE_EXCEPTION << "fakeNHWCBlob works only with channels == 3";
    }

    tensorDesc.setLayout(Layout::NHWC);
    Blob::Ptr fakeNHWC = make_shared_blob<uint8_t>(tensorDesc);
    fakeNHWC->allocate();

    const auto C = tensorDesc.getDims()[1];
    const auto H = tensorDesc.getDims()[2];
    const auto W = tensorDesc.getDims()[3];
    const auto multHW = H * W;
    const auto multWC = W * C;
    for (size_t c = 0; c < C; ++c) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
                static_cast<uint8_t*>(fakeNHWC->buffer())[c * multHW + W * h + w] =
                    static_cast<uint8_t*>(blob->buffer())[h * multWC + w * C + c];
            }
        }
    }

    return fakeNHWC;
}

// pre_proc = false -> create fake NHWC input
Blob::Ptr KmbPrivateConfigTests::runInferWithConfig(const std::string& model_path, const std::string& input_path,
    bool pre_proc, const KmbPrivateConfig& config, size_t input_width, size_t input_height) const {
    auto network = core->ImportNetwork(model_path, deviceName, config);

    auto request = network.CreateInferRequest();

    const auto inputName = network.GetInputsInfo().begin()->second->getInputData()->getName();
    Blob::Ptr inputBlob;
    VPUAllocatorPtr allocator;
    if (pre_proc) {
        allocator = std::make_shared<vpu::KmbPlugin::utils::VPUSMMAllocator>();
        inputBlob = vpu::KmbPlugin::utils::fromNV12File(input_path, input_width, input_height, allocator);
    } else {
        const auto inputTensorDesc = network.GetInputsInfo().begin()->second->getTensorDesc();
        inputBlob = vpu::KmbPlugin::utils::fromBinaryFile(input_path, inputTensorDesc);
    }

    PreProcessInfo preProcInfo;
    Blob::Ptr fakeNHWCInput;
    if (pre_proc) {
        preProcInfo.setColorFormat(ColorFormat::NV12);
        preProcInfo.setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
        request.SetBlob(inputName, inputBlob, preProcInfo);
    } else {
        fakeNHWCInput = createFakeNHWCBlob(inputBlob);
        request.SetBlob(inputName, fakeNHWCInput);
    }

    request.Infer();

    const auto outputName = network.GetOutputsInfo().begin()->second->getName();
    Blob::Ptr outputBlob = toFP32(request.GetBlob(outputName));

    return outputBlob;
}

Blob::Ptr KmbPrivateConfigTests::readReference(const std::string& reference_path, const TensorDesc& tensor_desc) const {
    Blob::Ptr referenceBlob = vpu::KmbPlugin::utils::fromBinaryFile(reference_path, tensor_desc);

    return referenceBlob;
}

TEST_P(KmbPrivateConfigTests, IE_VPU_KMB_PRIVATE_CONFIG_COMMON) {
#if !defined(__arm__) && !defined(__aarch64__)
    SKIP();
#endif

    const auto& p = GetParam();

    if (p._checkSIPP && !isSIPP()) SKIP() << "The test is intended to be run with SIPP enabled";

    Blob::Ptr outputBlob =
        runInferWithConfig(p._modelPath, p._inputPath, p._preProc, p._privateConfig, p._inputWidth, p._inputHeight);

    Blob::Ptr referenceBlob = readReference(p._referencePath, outputBlob->getTensorDesc());

    ASSERT_NO_THROW(compareTopClasses(outputBlob, referenceBlob, p._nClasses));
}

const std::vector<PrivateConfigTestParams> privateConfigParams {
    PrivateConfigTestParams()
        .testDescription("IE_VPU_KMB_SIPP_OUT_COLOR_FORMAT")
        .modelPath(ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob")
        .inputPath(ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/input-228x228-bgr-nv12.bin")
        .referencePath(ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/output-228x228-nv12.bin")
        .preProc(true)
        .checkSIPP(true)
        .privateConfig({{"VPU_KMB_SIPP_OUT_COLOR_FORMAT", "RGB"}})
        .inputWidth(228)
        .inputHeight(228)
        .nClasses(4),
    PrivateConfigTestParams()
        .testDescription("USE_SIPP")
        .modelPath(ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob")
        .inputPath(ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/input-228x228-nv12.bin")
        .referencePath(ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/output-228x228-nv12.bin")
        .preProc(true)
        .checkSIPP(false)
        .privateConfig({{"VPU_KMB_USE_SIPP", CONFIG_VALUE(YES)}})
        .inputWidth(228)
        .inputHeight(228)
        .nClasses(2),
    PrivateConfigTestParams()
        .testDescription("FORCE_NCHW_TO_NHWC")
        .modelPath(ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob")
        .inputPath(ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/input.bin")
        .referencePath(ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/output.bin")
        .preProc(false)
        .checkSIPP(false)
        .privateConfig({{"VPU_KMB_FORCE_NCHW_TO_NHWC", CONFIG_VALUE(YES)}})
        .inputWidth(228)
        .inputHeight(228)
        .nClasses(5)};

INSTANTIATE_TEST_CASE_P(DISABLED_SomeCase, KmbPrivateConfigTests, testing::ValuesIn(privateConfigParams));

TEST_F(KmbPrivateConfigTests, SERIALIZE_CNN_BEFORE_COMPILE_FILE) {
#if defined(__arm__) || defined(__aarch64__)
    SKIP();
#endif

    InferenceEngine::ExecutableNetwork network;
    ModelPooling_Helper modelPoolingHelper;
    InferenceEngine::CNNNetwork poolingNetwork = modelPoolingHelper.getNetwork();
    const std::string testFileName = "tmp_test.xml";
    std::remove(testFileName.c_str());
    for (auto&& input : poolingNetwork.getInputsInfo()) {
        input.second->setLayout(InferenceEngine::Layout::NHWC);
        input.second->setPrecision(InferenceEngine::Precision::U8);
    }
    for (auto&& output : poolingNetwork.getOutputsInfo()) {
        output.second->setLayout(InferenceEngine::Layout::NHWC);
        output.second->setPrecision(InferenceEngine::Precision::FP16);
    }
    network = core->LoadNetwork(poolingNetwork, deviceName);
    std::ifstream notExist(testFileName);
    ASSERT_FALSE(notExist.good());
    network = core->LoadNetwork(poolingNetwork, deviceName,
        {{"VPU_COMPILER_SERIALIZE_CNN_BEFORE_COMPILE_FILE", testFileName.c_str()}});
    std::ifstream exists(testFileName);
    ASSERT_TRUE(exists.good());
    std::remove(testFileName.c_str());
}

class KmbConfigTestsWithParams :
    public vpuLayersTests, public testing::WithParamInterface<std::string> {};

TEST_P(KmbConfigTestsWithParams, PERF_COUNT) {
#if !defined(__arm__) && !defined(__aarch64__)
    SKIP();
#endif
    const std::string perfCount = GetParam();
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";

    Core ie;
    InferenceEngine::ExecutableNetwork network;
    network = ie.ImportNetwork(modelFilePath, deviceName, {{"PERF_COUNT", perfCount}});

    InferenceEngine::InferRequest request;
    request = network.CreateInferRequest();

    std::string inputPath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/input.bin";
    const auto inputTensorDesc = network.GetInputsInfo().begin()->second->getTensorDesc();
    const auto inputBlob = vpu::KmbPlugin::utils::fromBinaryFile(inputPath, inputTensorDesc);

    const auto inputName = network.GetInputsInfo().begin()->second->getInputData()->getName();
    request.SetBlob(inputName, inputBlob);

    ASSERT_NO_THROW(request.Infer());

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap = {};
    // GetPerformanceCounts is still not implemented
    ASSERT_ANY_THROW(perfMap = request.GetPerformanceCounts());
}

const static std::vector<std::string> perfCountModes = {CONFIG_VALUE(YES), CONFIG_VALUE(NO)};

INSTANTIATE_TEST_CASE_P(perfCount, KmbConfigTestsWithParams, ::testing::ValuesIn(perfCountModes));
