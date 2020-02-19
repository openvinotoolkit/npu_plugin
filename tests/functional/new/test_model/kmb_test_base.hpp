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
#include "kmb_test_fake_quantize_def.hpp"
#include "kmb_test_softmax_def.hpp"

#include <vpu/kmb_plugin_config.hpp>

#include <tests_common.hpp>

#include <gtest/gtest.h>

using namespace InferenceEngine;

// #define RUN_SKIPPED_TESTS

#ifdef RUN_SKIPPED_TESTS
#   define SKIP_ON(_device_, _reason_)
#   define SKIP_INFER_ON(_device_, _reason_)
#else
#   define SKIP_ON(_device_, _reason_)                                          \
        do {                                                                    \
            if (DEVICE_NAME == _device_) {                                      \
                SKIP() << "Skip on " << DEVICE_NAME << " due to " << _reason_;  \
            }                                                                   \
        } while (false)

#   define SKIP_INFER_ON(_device_, _reason_)                                            \
        do {                                                                            \
            if (KmbTestBase::RUN_INFER && DEVICE_NAME == _device_) {                    \
                SKIP() << "Skip infer on " << DEVICE_NAME << " due to " << _reason_;    \
            }                                                                           \
        } while (false)
#endif

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
    BlobMap getInputs(TestNetwork& testNet);

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

    TestNetworkDesc& setUserInputPresision(
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

    TestNetworkDesc& setUserOutputPresision(
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
    using CheckCallback = std::function<void(const Blob::Ptr& actualBlob, const Blob::Ptr& refBlob, const TensorDesc& inputDesc)>;

protected:
    static Blob::Ptr loadImage(const TestImageDesc& image);

    CNNNetwork readNetwork(
            const TestNetworkDesc& netDesc,
            bool fillUserInfo);

    ExecutableNetwork getExecNetwork(
            const TestNetworkDesc& netDesc);

    Blob::Ptr calcRefOutput(
            const TestNetworkDesc& netDesc,
            const Blob::Ptr& inputBlob);

    void runTest(
            const TestNetworkDesc& netDesc,
            const TestImageDesc& image,
            const CheckCallback& checkCallback);
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

protected:
    struct Box final {
        float x, y, w, h;
    };

    struct BBox final {
        int idx;
        float left, right, top, bottom;
        float prob;
        BBox(int idx, float xmin, float ymin, float xmax, float ymax, float prob)
                : idx(idx), left(xmin), right(xmax), top(ymax), bottom(ymin), prob (prob) {};
    };

protected:
    static std::vector<BBox> parseOutput(
            const Blob::Ptr& blob,
            size_t imgWidth, size_t imgHeight,
            float confThresh);

    void checkBBoxOutputs(std::vector<BBox> &actual, std::vector<BBox> &ref,
            int imgWidth, int imgHeight, float boxTolerance, float probTolerance);

protected:
    static float overlap(float x1, float w1, float x2, float w2);
    static float boxIntersection(const Box& a, const Box& b);
    static float boxUnion(const Box& a, const Box& b);
    static float boxIou(const Box& a, const Box& b);
};


class KmbYoloV2NetworkTest : public KmbDetectionNetworkTest {
public:
    void runTest(
            const TestNetworkDesc& netDesc,
            const std::string& inputFileName,
            float confThresh,
            float boxTolerance, float probTolerance,
            bool isTiny);

    //TODO: It can be optimized
protected:
    struct sortableBBox {
        int index;
        int cclass;
        std::vector<std::vector<float>> probs;
        sortableBBox(int index, float cclass, std::vector<std::vector<float>> &probs)
                : index(index), cclass(cclass), probs(probs) {};
    };

protected:
    static std::vector<KmbDetectionNetworkTest::BBox> parseOutput(
            const Blob::Ptr& blob,
            size_t imgWidth, size_t imgHeight,
            float confThresh, bool isTiny);

    static std::vector<BBox> yolov2BoxExtractor(
            float threshold,
            std::vector<float> &net_out,
            int imgWidth,
            int imgHeight,
            int class_num,
            bool isTiny);

    static void getRegionBoxes(std::vector<float> &predictions,
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
                                                const std::vector<float> &anchors);

    static void correctRegionBoxes(std::vector<Box> &boxes, int n, int w, int h, int netw, int neth, int relative);
    static Box getRegionBox(float *x, const std::vector<float> &biases, int n, int index, int i, int j, int w, int h, int stride);
    static int entryIndex(int lw, int lh, int lcoords, int lclasses, int lnum, int batch, int location, int entry);
    static int maxIndex(std::vector<float> &a, int n);
    static void doNMSSort(std::vector<Box> &boxes, std::vector<std::vector<float>> &probs,
                            int total, int classes, float thresh);
    static void getDetections(int imw, int imh, int num, float thresh,
                               Box *boxes, std::vector<std::vector<float>> &probs,
                               int classes, std::vector<BBox> &detect_result);
};
