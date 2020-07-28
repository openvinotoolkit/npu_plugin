//
// Copyright 2019-2020 Intel Corporation.
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

#pragma once

#include "kmb_test_utils.hpp"
#include "kmb_test_model.hpp"

#include "kmb_test_add_def.hpp"
#include "kmb_test_mul_def.hpp"
#include "kmb_test_scale_shift_def.hpp"
#include "kmb_test_convolution_def.hpp"
#include "kmb_test_deconvolution_def.hpp"
#include "kmb_test_normalize_def.hpp"
#include "kmb_test_fake_quantize_def.hpp"
#include "kmb_test_softmax_def.hpp"
#include "kmb_test_fully_connected_def.hpp"
#include "kmb_test_sigmoid_def.hpp"
#include "kmb_test_power_def.hpp"
#include "kmb_test_proposal_def.hpp"
#include "kmb_test_psroipooling_def.hpp"
#include "kmb_test_interp_def.hpp"
#include "kmb_test_topk_def.hpp"
#include "kmb_test_interp_def.hpp"
#include "kmb_test_reshape_def.hpp"
#include "kmb_test_permute_def.hpp"

#include <vpu/kmb_plugin_config.hpp>

#include <tests_common.hpp>

#include <gtest/gtest.h>
#include <yolo_helpers.hpp>

using namespace InferenceEngine;

// #define RUN_SKIPPED_TESTS

#ifdef RUN_SKIPPED_TESTS
#   define SKIP_ON(_device_, _reason_)
#   define SKIP_INFER_ON(_device_, _reason_)
#else

#   define SKIP_ON1(_device0_, _reason_)                                        \
        do {                                                                    \
            std::set<std::string> devices({_device0_});                         \
            if (devices.count(DEVICE_NAME) != 0) {                              \
                SKIP() << "Skip on " << DEVICE_NAME << " due to " << _reason_;  \
            }                                                                   \
        } while (false)

#   define SKIP_ON2(_device0_, _device1_, _reason_)                             \
        do {                                                                    \
            std::set<std::string> devices({_device0_, _device1_});              \
            if (devices.count(DEVICE_NAME) != 0) {                              \
                SKIP() << "Skip on " << DEVICE_NAME << " due to " << _reason_;  \
            }                                                                   \
        } while (false)

#   define SKIP_ON3(_device0_, _device1_, _device2_, _reason_)                  \
        do {                                                                    \
            std::set<std::string> devices({_device0_, _device1_, _device2_,});  \
            if (devices.count(DEVICE_NAME) != 0) {                              \
                SKIP() << "Skip on " << DEVICE_NAME << " due to " << _reason_;  \
            }                                                                   \
        } while (false)


#   define SKIP_INFER_ON1(_device0_, _reason_)                                          \
        do {                                                                            \
            std::set<std::string> devices({_device0_});                                 \
            if (KmbTestBase::RUN_INFER && devices.count(DEVICE_NAME) != 0) {            \
                SKIP() << "Skip infer on " << DEVICE_NAME << " due to " << _reason_;    \
            }                                                                           \
        } while (false)

#   define SKIP_INFER_ON2(_device0_, _device1_, _reason_)                               \
        do {                                                                            \
            std::set<std::string> devices({_device0_, _device1_});                      \
            if (KmbTestBase::RUN_INFER && devices.count(DEVICE_NAME) != 0) {            \
                SKIP() << "Skip infer on " << DEVICE_NAME << " due to " << _reason_;    \
            }                                                                           \
        } while (false)

#   define SKIP_INFER_ON3(_device0_, _device1_, _device2_, _reason_)                    \
        do {                                                                            \
            std::set<std::string> devices({_device0_, _device1_});                      \
            if (KmbTestBase::RUN_INFER && devices.count(DEVICE_NAME) != 0) {            \
                SKIP() << "Skip infer on " << DEVICE_NAME << " due to " << _reason_;    \
            }                                                                           \
        } while (false)
#endif

#define GET_MACRO(_1,_2,_3, _4, NAME,...) NAME
#define SKIP_INFER_ON(...) GET_MACRO(__VA_ARGS__, SKIP_INFER_ON3, SKIP_INFER_ON2, SKIP_INFER_ON1)(__VA_ARGS__)

#define SKIP_ON(...) GET_MACRO(__VA_ARGS__, SKIP_ON3, SKIP_ON2, SKIP_ON1)(__VA_ARGS__)

//
// KmbTestBase
//

class KmbTestBase : public TestsCommon {
public:
    using BlobGenerator = std::function<Blob::Ptr(const TensorDesc& desc)>;
    using CompileConfig = std::map<std::string, std::string>;

public:
    static const std::string DEVICE_NAME;
    static const std::string REF_DEVICE_NAME;
    static const bool RUN_COMPILER;
    static const bool RUN_REF_CODE;
    static const bool RUN_INFER;
    static const std::string DUMP_PATH;
    static const bool EXPORT_NETWORK;
    static const bool RAW_EXPORT;
    static const bool GENERATE_BLOBS;
    static const bool EXPORT_BLOBS;
    static const std::string LOG_LEVEL;
    static const bool PRINT_PERF_COUNTERS;

public:
    void registerBlobGenerator(
            const std::string& blobName,
            const TensorDesc& desc,
            const BlobGenerator& generator) {
        blobGenerators[blobName] = {desc, generator};
    }

    Blob::Ptr getBlobByName(const std::string& blobName);

    BlobMap getInputs(const ExecutableNetwork& testNet);

protected:
    void SetUp() override;

protected:
    ExecutableNetwork getExecNetwork(
            const std::function<CNNNetwork()>& netCreator,
            const std::function<CompileConfig()>& configCreator);

    void compareWithReference(
            const BlobMap& actualOutputs,
            const BlobMap& refOutputs,
            float tolerance, CompareMethod method = CompareMethod::Absolute);

    void compareOutputs(
            const Blob::Ptr& refOutput, const Blob::Ptr& actualOutput,
            float tolerance, CompareMethod method = CompareMethod::Absolute);

    void checkWithOutputsInfo(const BlobMap& actualOutputs, const std::vector<DataPtr>& outputsInfo);

protected:
    void exportNetwork(ExecutableNetwork& exeNet);

    ExecutableNetwork importNetwork();

    void dumpBlob(const std::string& blobName, const Blob::Ptr& blob);

    void dumpBlobs(const BlobMap& blobs);

    Blob::Ptr importBlob(const std::string& name, const TensorDesc& desc);

    BlobMap runInfer(ExecutableNetwork& exeNet, const BlobMap& inputs, bool printTime);

protected:
    std::default_random_engine rd;
    std::shared_ptr<Core> core;
    std::string dumpBaseName;
    std::unordered_map<std::string, Blob::Ptr> blobs;
    std::unordered_map<std::string, std::pair<TensorDesc, BlobGenerator>> blobGenerators;
};

//
// KmbLayerTestBase
//

class KmbLayerTestBase : public KmbTestBase {
    using NetworkBuilder = std::function<void(TestNetwork& testNet)>;

public:
    void runTest(
            const NetworkBuilder& builder,
            float tolerance, CompareMethod method = CompareMethod::Absolute);

protected:
    ExecutableNetwork getExecNetwork(
            TestNetwork& testNet);

    BlobMap getRefOutputs(
            TestNetwork& testNet,
            const BlobMap& inputs);
};

//
// TestNetworkDesc
//

class TestNetworkDesc final {
public:
    explicit TestNetworkDesc(std::string irFileName) : _irFileName(std::move(irFileName)) {}

    TestNetworkDesc& setUserInputPrecision(
            const std::string& name,
            const Precision& precision) {
        _inputPrecisions[name] = precision;
        return *this;
    }

    TestNetworkDesc& setUserInputLayout(
            const std::string& name,
            const Layout& layout) {
        _inputLayouts[name] = layout;
        return *this;
    }

    TestNetworkDesc& setUserOutputPrecision(
            const std::string& name,
            const Precision& precision) {
        _outputPrecisions[name] = precision;
        return *this;
    }

    TestNetworkDesc& setUserOutputLayout(
            const std::string& name,
            const Layout& layout) {
        _outputLayouts[name] = layout;
        return *this;
    }

    TestNetworkDesc& setCompileConfig(const std::map<std::string, std::string>& compileConfig) {
        _compileConfig = compileConfig;
        return *this;
    }

    const std::string& irFileName() const {
        return _irFileName;
    }

    void fillUserInputInfo(InputsDataMap& info) const;
    void fillUserOutputInfo(OutputsDataMap& info) const;

    const std::unordered_map<std::string, Precision>& outputPrecisions() const {
        return _outputPrecisions;
    }

    const std::unordered_map<std::string, Layout>& outputLayouts() const {
        return _outputLayouts;
    }

    const std::map<std::string, std::string>& compileConfig() const {
        return _compileConfig;
    }

private:
    std::string _irFileName;

    std::unordered_map<std::string, Precision> _inputPrecisions;
    std::unordered_map<std::string, Layout> _inputLayouts;

    std::unordered_map<std::string, Precision> _outputPrecisions;
    std::unordered_map<std::string, Layout> _outputLayouts;

    std::map<std::string, std::string> _compileConfig;
};

//
// TestImageDesc
//

class TestImageDesc final {
public:
    TestImageDesc(const char* imageFileName, bool isBGR = true) : _imageFileName(imageFileName), _isBGR(isBGR) {}
    TestImageDesc(std::string imageFileName, bool isBGR = true) : _imageFileName(std::move(imageFileName)), _isBGR(isBGR) {}

    const std::string& imageFileName() const {
        return _imageFileName;
    }

    bool isBGR() const {
        return _isBGR;
    }

private:
    std::string _imageFileName;
    bool _isBGR = true;
};

//
// KmbNetworkTestBase
//

class KmbNetworkTestBase : public KmbTestBase {
protected:
    using CheckCallback = std::function<void(const BlobMap& actualBlob,
                                             const BlobMap& refBlob,
                                             const ConstInputsDataMap& inputsDesc)>;

    using InitIntputCallback = std::function<void(const ConstInputsDataMap& inputs)>;

protected:
    static Blob::Ptr loadImage(const TestImageDesc& image, int channels, int height, int width);

    void registerSingleImage (const TestImageDesc& image,
                              const std::string& inputName,
                              const TensorDesc inputDesc);

    CNNNetwork readNetwork(
            const TestNetworkDesc& netDesc,
            bool fillUserInfo);

    ExecutableNetwork getExecNetwork(
            const TestNetworkDesc& netDesc);

    BlobMap calcRefOutput(
            const TestNetworkDesc& netDesc,
            const BlobMap& inputs);

    void runTest(
            const TestNetworkDesc& netDesc,
            const InitIntputCallback& image,
            const CheckCallback& checkCallback);

    void checkLayouts(const BlobMap& actualOutputs,
                      const std::unordered_map<std::string, Layout>& layouts) const;

    void checkPrecisions(const BlobMap& actualOutputs,
                         const std::unordered_map<std::string, Precision>&) const;
};

//
// KmbClassifyNetworkTest
//

class KmbClassifyNetworkTest : public KmbNetworkTestBase {
public:
    void runTest(
            const TestNetworkDesc& netDesc,
            const TestImageDesc& image,
            size_t topK, float probTolerance);

protected:
    static std::vector<std::pair<int, float>> parseOutput(const Blob::Ptr& blob);
};

//
// KmbDetectionNetworkTest
//

class KmbDetectionNetworkTest : public KmbNetworkTestBase {
public:
    void runTest(
            const TestNetworkDesc& netDesc,
            const TestImageDesc& image,
            float confThresh,
            float boxTolerance, float probTolerance);

    void runTest(
            const TestNetworkDesc& netDesc,
            float confThresh,
            float boxTolerance, float probTolerance);

protected:
    static std::vector<utils::YoloBBox> parseOutput(
            const Blob::Ptr& blob,
            size_t imgWidth, size_t imgHeight,
            float confThresh);

    void checkBBoxOutputs(std::vector<utils::YoloBBox> &actual, std::vector<utils::YoloBBox> &ref,
            int imgWidth, int imgHeight, float boxTolerance, float probTolerance);
};

class KmbRFCNNetworkTest : public KmbDetectionNetworkTest {
public:
    void runTest(
            const TestNetworkDesc& netDesc,
            const std::string& data_name,
            const TestImageDesc& image,
            const std::string& im_info_name,
            const std::vector<float>& im_info_values);
};

class KmbYoloV2NetworkTest : public KmbDetectionNetworkTest {
public:
    void runTest(
            const TestNetworkDesc& netDesc,
            const TestImageDesc& image,
            float confThresh,
            float boxTolerance, float probTolerance,
            bool isTiny);
};

using KmbYoloV1NetworkTest = KmbYoloV2NetworkTest;

//
// KmbSegmentationNetworkTest
//

class KmbSegmentationNetworkTest : public KmbNetworkTestBase {
public:
    void runTest(
            const TestNetworkDesc& netDesc,
            const TestImageDesc& image,
            const float meanIntersectionOverUnionTolerance);
};

class GazeEstimationNetworkTest : public KmbNetworkTestBase {
public:
    void runTest(
        const TestNetworkDesc& netDesc,
        const std::string& left_eye_input_name,
        const TestImageDesc& left_eye_image,
        const std::string right_eye_input_name,
        const TestImageDesc& right_eye_image,
        const std::string head_pos_input_name,
        std::vector<float> head_pos);
};

class SmokeNetworkTest : public KmbNetworkTestBase {
public:
    void runTest(const TestNetworkDesc& netDesc);

private:
    std::default_random_engine rd;
};
