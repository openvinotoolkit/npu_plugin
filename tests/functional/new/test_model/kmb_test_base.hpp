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
            const CNNNetwork& net,
            const std::map<std::string, std::string>& config = {});

    ExecutableNetwork getExecNetwork(
            TestNetwork& testNet) {
        return getExecNetwork(testNet.getCNNNetwork(), testNet.compileConfig());
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

class KmbNetworkTest : public KmbTestBase {
public:
    void runClassifyNetworkTest(
            const std::string& modelFileName,
            const std::string& inputFileName,
            size_t topK, float probTolerance);

protected:
    void runClassifyNetworkTest(
            const CNNNetwork& net,
            const Blob::Ptr& inputBlob,
            const std::vector<std::pair<int, float>>& refTopK, float probTolerance);

    void runClassifyNetworkTest(
            const std::string& modelFileName,
            const std::string& inputFileName,
            const std::vector<std::pair<int, float>>& refTopK, float probTolerance);

    std::vector<std::pair<int, float>> parseClassifyOutput(const Blob::Ptr& blob);

protected:
    CNNNetwork loadNetwork(const std::string& modelFileName);
    Blob::Ptr loadImage(const std::string& imageFileName);
};
