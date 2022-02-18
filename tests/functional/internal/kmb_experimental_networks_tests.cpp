//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "test_model/kmb_test_base.hpp"

TEST_F(KmbNetworkTestBase, split_conv_concat) {
    SKIP_INFER("Wrong results - precision issues");  // TODO: create JIRA ticket
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

TEST_F(KmbNetworkTestBase, precommit_customnet_conv_strided_slice) {
    SKIP_INFER("Wrong results - precision issues");  // TODO: create JIRA ticket
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
    SKIP_INFER("hangs on infer");  // [Track number: S#43799]
    runTest(
        TestNetworkDesc("KMB_models/INT8/customnets/customnet1_tf_int8_dense_grayscale_fashionmnist.xml")
                .setUserInputPrecision("input", Precision::U8)
                .setUserInputLayout("input", Layout::NHWC)
                .setUserOutputPrecision("output", Precision::FP32),
        "28x28/image_1_28x28.bmp",
        1, 0.5f);
}

TEST_F(KmbClassifyNetworkTest, precommit_customnet_sigmoid) {
    SKIP_INFER("Leads to 'DDR MemMgr failed to free segment'");  // [Track number: E#27630]
    runTest(
        TestNetworkDesc("KMB_models/INT8/customnets/customnet_sigmoid.xml")
                .setUserInputPrecision("input", Precision::U8)
                .setUserInputLayout("input", Layout::NHWC)
                .setUserOutputPrecision("output", Precision::FP32),
        "28x28/image_1_28x28.bmp",
        1, 0.5f);
}

TEST_F(KmbClassifyNetworkTest, customnet2_pytorch_int8_dense_cifar10) {
    SKIP_INFER("hangs on infer");  // TODO: create JIRA ticket
    runTest(
        TestNetworkDesc("KMB_models/INT8/customnets/customnet2_pytorch_int8_dense_cifar10.xml")
                .setUserInputPrecision("input", Precision::U8)
                .setUserInputLayout("input", Layout::NHWC)
                .setUserOutputPrecision("output", Precision::FP32),
        "32x32/0_cat.bmp",
        1, 0.5f);
}

// [Track number: E#20726]
TEST_F(KmbClassifyNetworkTest, customnet3_mobilenet_v1_caffe_int8_dense) {
    SKIP_ON("LEVEL0", "Sporadic exception - throwOnFail: zeCommandQueueCreate result: 0x70000001");
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
    SKIP_INFER("Wrong results");  // TODO: create JIRA ticket
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
}  // namespace

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

TEST_F(KmbClassifyNetworkTest, efficient_b0_cars) {
    SKIP_ON("EMULATOR", "Wrong results due to missing implementation for eltwise UPATask");
    runTest(
            TestNetworkDesc("efficientnet-b0-stanford-cars/caffe2/FP16-INT8/efficientnet-b0-stanford-cars.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP32)
                    .enableLPTRefMode(),
            TestImageDesc("vpu/efficient/car_1.jpg", ImageFormat::RGB),
            1, 0.15f);
}

TEST_F(KmbClassifyNetworkTest, precommit_efficient_b0_dogs) {
    SKIP_ON("EMULATOR", "Wrong results due to missing implementation for eltwise UPATask");
    SKIP_INFER("Leads to 'DDR MemMgr failed to free segment'");  // [Track number: E#27630]
    runTest(
            TestNetworkDesc("efficientnet-b0-stanford-dogs/caffe2/FP16-INT8/efficientnet-b0-stanford-dogs.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP32)
                    .enableLPTRefMode(),
            TestImageDesc("vpu/efficient/dog_1.jpg", ImageFormat::RGB),
            1, 0.15f);
}

TEST_F(KmbClassifyNetworkTest, efficient_b0_aircrafts) {
    SKIP_ON("EMULATOR", "Wrong results due to missing implementation for eltwise UPATask");
    runTest(
            TestNetworkDesc("efficientnet-b0-aircrafts/caffe2/FP16-INT8/efficientnet-b0-aircrafts.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP32)
                    .enableLPTRefMode(),
            TestImageDesc("vpu/efficient/aircraft_1.jpg", ImageFormat::RGB),
            1, 0.15f);
}

TEST_F(KmbClassifyNetworkTest, mobilenet_v3_cars) {
    SKIP_ON("EMULATOR", "Wrong results due to missing implementation for eltwise UPATask");
    runTest(
            TestNetworkDesc("mobilenet-v3-small-stanford-cars/caffe2/FP16-INT8/mobilenet-v3-small-stanford-cars.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("vpu/efficient/car_1.jpg", ImageFormat::RGB),
            1, 0.15f);
}

// TODO: [Track number: E#9578]
TEST_F(KmbClassifyNetworkTest, precommit_mobilenet_v3_dogs) {
    SKIP_ON("HDDL2", "Bad accuracy");
    SKIP_ON("EMULATOR", "Wrong results due to missing implementation for eltwise UPATask");
    SKIP_INFER("Leads to 'DDR MemMgr failed to free segment'");  // [Track number: E#27630]
    runTest(
            TestNetworkDesc("mobilenet-v3-small-stanford-dogs/caffe2/FP16-INT8/mobilenet-v3-small-stanford-dogs.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("vpu/efficient/dog_1.jpg", ImageFormat::RGB),
            1, 0.15f);
}

// [Track number: E#15866]
TEST_F(KmbClassifyNetworkTest, mobilenet_v3_aircrafts) {
    SKIP_INFER("bad results");
    runTest(
            TestNetworkDesc("mobilenet-v3-small-aircrafts/caffe2/FP16-INT8/mobilenet-v3-small-aircrafts.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("vpu/efficient/aircraft_1.jpg", ImageFormat::RGB),
            1, 0.17f);
}

TEST_P(ModelAdk, precommit_ModelA_ADK3) {
    runTest(
            TestNetworkDesc("ADK3/ModelA_INT8/ModelA_INT8.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", GetParam())
                    .setUserOutputPrecision("output", Precision::FP16),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::BGR),
            0.0035f);
}

TEST_F(ModelAdk, ModelE_ADK3) {
    runTest(
            TestNetworkDesc("ADK3/ModelE_INT8/ModelE_INT8.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::FP16)
                    .setUserOutputPrecision("PostProcess/stage0/x1/Sigmoid", Precision::FP16)
                    .setUserOutputPrecision("PostProcess/stage0/x4/Sigmoid", Precision::FP16)
                    .setUserOutputPrecision("PostProcess/stage1/x1/Sigmoid", Precision::FP16)
                    .setUserOutputPrecision("PostProcess/stage1/x4/Sigmoid", Precision::FP16),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::BGR),
            0.0025f);
}

// [Track number: EISW-10831]
TEST_F(ModelAdk, DISABLED_DeBlur_AA_BDK2) {
    SKIP_INFER("bad results");
    runTest(
            TestNetworkDesc(
                    "../clientmodels/BDK2/Deblur/AccuracyAware/2020.1_INT8_Deblur_AccuracyAwareQuantization.xml", EXPERIMENTAL)
                    .setUserInputPrecision("img_placeholder", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP16)
                    .setCompileConfig({{"VPUX_THROUGHPUT_STREAMS", "1"}}),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::BGR),
            0.0025f);
}

// TODO: this is temporary added test to verify #E16291.
// The test should be replaced by a sub-graph test
// to verify different patterns which can be applied
// for detect_input_fq ngraph pass
// [Track number: #E19643]
TEST_F(KmbClassifyNetworkTest, googlenet_v3_BDK2) {
    runTest(
        TestNetworkDesc("googlenet_v3/FP16-INT8/googlenet-v3.xml", EXPERIMENTAL)
            .setUserInputPrecision("input", Precision::FP32),
        TestImageDesc("299x299/lassy_googlenet_big.bmp", ImageFormat::RGB),
        1, 0.03f);
}

const static std::vector<InferenceEngine::Precision> inputPrecision = {
                InferenceEngine::Precision::U8,
                InferenceEngine::Precision::FP16,
                InferenceEngine::Precision::FP32};

INSTANTIATE_TEST_SUITE_P(PrecisionCase, ModelAdk, ::testing::ValuesIn(inputPrecision));

TEST_F(KmbClassifyNetworkTest, precommit_MobilenetV2_ADK3) {
    SKIP_INFER("Leads to 'DDR MemMgr failed to free segment'");  // [Track number: E#27630]
    runTest(
            TestNetworkDesc("ADK3/mobilenet-v2/caffe2/FP16-INT8/mobilenet-v2.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP16),
            TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
            1, 2.0f);
}
