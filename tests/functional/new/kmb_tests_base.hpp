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

#pragma once

#include <gtest/gtest.h>

#include <tests_common.hpp>

#include "test_model/kmb_test_utils.hpp"
#include "test_model/kmb_test_model.hpp"
#include "test_model/kmb_test_convolution_def.hpp"
#include "test_model/kmb_test_fake_quantize_def.hpp"

using namespace InferenceEngine;

//
// KmbTestBase
//

extern const std::string DEVICE_NAME;
extern const std::string REF_DEVICE_NAME;
extern const bool RUN_COMPILER;
extern const bool RUN_REF_CODE;
extern const bool RUN_INFER;
extern const std::string BLOBS_PATH;
extern const std::string REFS_PATH;

class KmbTestBase : public TestsCommon {
public:
    using BlobMapGenerator = std::function<BlobMap()>;

    void SetUp() override;

    void runTest(
            TestNetwork& testNet,
            const BlobMap& inputs,
            float tolerance, CompareMethod method = CompareMethod::Absolute);
    void runTest(
            TestNetwork& testNet,
            const BlobMapGenerator& inputsGenerator,
            float tolerance, CompareMethod method = CompareMethod::Absolute);

    BlobMap getInputs(
            TestNetwork& testNet,
            const BlobMapGenerator& generator);

    ExecutableNetwork getExecNetwork(
            const CNNNetwork& net,
            const std::map<std::string, std::string>& config = {});
    ExecutableNetwork getExecNetwork(
            TestNetwork& testNet) {
        return getExecNetwork(testNet.toCNNNetwork(), testNet.compileConfig());
    }

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
    void dumpBlobs(const BlobMap& blobs);
    Blob::Ptr importBlob(const std::string& name, const TensorDesc& desc);
    BlobMap runInfer(ExecutableNetwork& exeNet, const BlobMap& inputs);

protected:
    std::default_random_engine rd;
    std::shared_ptr<Core> core;
    std::string dumpBaseName;
};

//
// KmbNetworkTest
//

class KmbNetworkTest : public KmbTestBase {
public:
    void runClassifyNetworkTest(
            const CNNNetwork& net,
            const Blob::Ptr& inputBlob,
            const std::vector<std::pair<int, float>>& refTopK, float probTolerance);
    void runClassifyNetworkTest(
            const std::string& modelFileName,
            const std::string& inputFileName,
            const std::vector<std::pair<int, float>>& refTopK, float probTolerance);
    void runClassifyNetworkTest(
            const std::string& modelFileName,
            const std::string& inputFileName,
            size_t topK, float probTolerance);

protected:
    CNNNetwork loadNetwork(const std::string& modelFileName);
    Blob::Ptr loadImage(const std::string& imageFileName);

    std::vector<std::pair<int, float>> parseClassifyOutput(const Blob::Ptr& blob);
};
