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

#include <gtest/gtest.h>

#include <tests_common.hpp>

#include "kmb_test_utils.hpp"
#include "kmb_test_model.hpp"

#include "kmb_test_add_def.hpp"
#include "kmb_test_mul_def.hpp"
#include "kmb_test_scale_shift_def.hpp"
#include "kmb_test_convolution_def.hpp"
#include "kmb_test_fake_quantize_def.hpp"

using namespace InferenceEngine;

//
// KmbTestBase
//

class KmbTestBase : public TestsCommon {
public:
    using BlobGenerator = std::function<Blob::Ptr(const TensorDesc& desc)>;
    using CompileConfig = std::map<std::string, std::string>;

public:
    void registerBlobGenerator(
            const std::string& blobName,
            const TensorDesc& desc,
            const BlobGenerator& generator) {
        blobGenerators[blobName] = {desc, generator};
    }

    Blob::Ptr getBlobByName(const std::string& blobName);

    void runTest(
            TestNetwork& testNet,
            float tolerance, CompareMethod method = CompareMethod::Absolute);

protected:
    void SetUp() override;

protected:
    void runTest(
            TestNetwork& testNet,
            const BlobMap& inputs,
            float tolerance, CompareMethod method = CompareMethod::Absolute);

protected:
    BlobMap getInputs(TestNetwork& testNet);

    ExecutableNetwork getExecNetwork(
            const std::function<CNNNetwork()>& netCreator,
            const std::function<CompileConfig()>& configCreator);

    ExecutableNetwork getExecNetwork(
            TestNetwork& testNet);

    BlobMap getRefOutputs(
            TestNetwork& testNet,
            const BlobMap& inputs);

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

    BlobMap runInfer(ExecutableNetwork& exeNet, const BlobMap& inputs);

protected:
    std::default_random_engine rd;
    std::shared_ptr<Core> core;
    std::string dumpBaseName;
    std::unordered_map<std::string, Blob::Ptr> blobs;
    std::unordered_map<std::string, std::pair<TensorDesc, BlobGenerator>> blobGenerators;
};

//
// KmbNetworkTest
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

class KmbNetworkTestBase : public KmbTestBase {
protected:
    using CheckCallback = std::function<void(const Blob::Ptr& actualBlob, const Blob::Ptr& refBlob, const TensorDesc& inputDesc)>;

protected:
    static Blob::Ptr loadImage(const std::string& imageFilePath);

    CNNNetwork readNetwork(
            const TestNetworkDesc& netDesc,
            bool fillUserInfo);

    ExecutableNetwork getExecNetwork(
            const TestNetworkDesc& netDesc);

    void runTest(
            const TestNetworkDesc& netDesc,
            const std::string& inputFileName,
            const CheckCallback& checkCallback);
};

class KmbClassifyNetworkTest : public KmbNetworkTestBase {
public:
    void runTest(
            const TestNetworkDesc& netDesc,
            const std::string& inputFileName,
            size_t topK, float probTolerance);

protected:
    static std::vector<std::pair<int, float>> parseOutput(const Blob::Ptr& blob);
};

class KmbDetectionNetworkTest : public KmbNetworkTestBase {
public:
    void runTest(
            const TestNetworkDesc& netDesc,
            const std::string& inputFileName,
            float confThresh,
            float boxTolerance, float probTolerance);

protected:
    struct Box final {
        float x, y, w, h;
    };

    struct BBox final {
        float left, right, top, bottom;
        float prob;
        int idx;
    };

protected:
    static std::vector<BBox> parseOutput(
            const Blob::Ptr& blob,
            size_t imgWidth, size_t imgHeight,
            float confThresh);

protected:
    static float overlap(float x1, float w1, float x2, float w2);
    static float box_intersection(const Box& a, const Box& b);
    static float box_union(const Box& a, const Box& b);
    static float box_iou(const Box& a, const Box& b);
};
