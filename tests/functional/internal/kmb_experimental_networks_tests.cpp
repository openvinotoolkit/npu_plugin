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

#include "test_model/kmb_test_base.hpp"

TEST_F(KmbNetworkTestBase, split_conv_concat) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "Wrong results due to precision issues"); // TODO: create JIRA ticket
    const auto init_input = [=](const ConstInputsDataMap& inputs) {
        IE_ASSERT(inputs.size() == 1);
        registerSingleImage("28x28/image_1_28x28.bmp", inputs.begin()->first, inputs.begin()->second->getTensorDesc());
    };

    const auto check = [=](const BlobMap& actualBlobs, const BlobMap& refBlobs, const ConstInputsDataMap&) {
        compareWithReference(actualBlobs, refBlobs, 1e-2f, CompareMethod::Absolute);
    };
    runTest(
        TestNetworkDesc("KMB_models/INT8/customnets/split_conv_concat.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        init_input, check);
}

TEST_F(KmbNetworkTestBase, customnet_conv_strided_slice) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "Wrong results due to precision issues"); // TODO: create JIRA ticket
    const auto init_input = [=](const ConstInputsDataMap& inputs) {
        IE_ASSERT(inputs.size() == 1);
        registerSingleImage("28x28/image_1_28x28.bmp", inputs.begin()->first, inputs.begin()->second->getTensorDesc());
    };

    const auto check = [=](const BlobMap& actualBlobs, const BlobMap& refBlobs, const ConstInputsDataMap&) {
        compareWithReference(actualBlobs, refBlobs, 1e-2f, CompareMethod::Absolute);
    };
    runTest(
        TestNetworkDesc("KMB_models/INT8/customnets/conv_strided_slice.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        init_input, check);
}

TEST_F(KmbClassifyNetworkTest, precommit_customnet1_tf_int8_dense_grayscale_fashionmnist) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hangs on infer");  // [Track number: S#43799]
    runTest(
        TestNetworkDesc("KMB_models/INT8/customnets/customnet1_tf_int8_dense_grayscale_fashionmnist.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        "28x28/image_1_28x28.bmp",
        1, 0.5f);
}

TEST_F(KmbClassifyNetworkTest, precommit_customnet_sigmoid) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/customnets/customnet_sigmoid.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        "28x28/image_1_28x28.bmp",
        1, 0.5f);
}

TEST_F(KmbClassifyNetworkTest, customnet2_pytorch_int8_dense_cifar10) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hangs on infer");  // TODO: create JIRA ticket
    runTest(
        TestNetworkDesc("KMB_models/INT8/customnets/customnet2_pytorch_int8_dense_cifar10.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        "32x32/0_cat.bmp",
        1, 0.5f);
}

TEST_F(KmbClassifyNetworkTest, customnet3_mobilenet_v1_caffe_int8_dense) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/customnets/customnet3_mobilenet_v1.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16)
            .setUserOutputLayout("output", Layout::NHWC),
        "224x224/cat3.bmp",
        1, 0.5f);
}

TEST_F(KmbClassifyNetworkTest, customnet_tanh) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "Wrong results"); // TODO: create JIRA ticket
    runTest(
        TestNetworkDesc("KMB_models/INT8/customnets/customnet_tanh.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        "28x28/image_1_28x28.bmp",
        1, 0.5f);
}

namespace {
constexpr bool EXPERIMENTAL = true;
} // namespace

TEST_F(KmbClassifyNetworkTest, experimental_network_0000) {
    runTest(
        TestNetworkDesc("emotions-recognition-retail-0003/emotions-recognition-retail-0003_int8_from_fp16.xml", EXPERIMENTAL)
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputLayout("output", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        "vpu/emotions-recognition-retail-0003.png",
        2, 0.1f);
}

TEST_F(ModelAdk, precommit_ModelA) {
    runTest(
            TestNetworkDesc("ADK3/ModelA_INT8/ModelA_INT8.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::FP16)
                    .setUserOutputPrecision("output", Precision::FP16),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::BGR),
            0.0025f);
}

TEST_F(ModelAdk, ModelE) {
    runTest(
            TestNetworkDesc("ADK3/ModelE_INT8/ModelE_INT8.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::FP16)
                    .setUserOutputPrecision("PostProcess/stage0/x1/Sigmoid", Precision::FP32)
                    .setUserOutputPrecision("PostProcess/stage0/x4/Sigmoid", Precision::FP32)
                    .setUserOutputPrecision("PostProcess/stage1/x1/Sigmoid", Precision::FP32)
                    .setUserOutputPrecision("PostProcess/stage1/x4/Sigmoid", Precision::FP32),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::BGR),
            0.0025f);
}

// [Track number: S#47419]
TEST_F(SmokeNetworkTest, DISABLED_DeBlur) {
#ifdef _WIN32
    SKIP() << "SEH exception";
#endif
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "bad results");
    runTest(
            TestNetworkDesc("ADK3/DeBlur_INT8/DeBlur_INT8.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::FP16)
                    .setUserOutputPrecision("output", Precision::FP16));
}

// [Track number: S#47647]
TEST_F(SmokeNetworkTest, DISABLED_SuperResolution_ADK3) {
    runTest(
            TestNetworkDesc("ADK3/SuperRes_INT8/SuperRes_INT8.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::FP16)
                    .setUserOutputPrecision("output", Precision::FP16));
}
