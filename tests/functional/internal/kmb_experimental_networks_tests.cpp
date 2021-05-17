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

TEST_F(KmbNetworkTestBase, precommit_customnet_conv_strided_slice) {
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

TEST_F(KmbClassifyNetworkTest, efficient_b0_cars) {
    runTest(
            TestNetworkDesc("efficientnet-b0-stanford-cars/caffe2/FP16-INT8/efficientnet-b0-stanford-cars.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP32)
                    .enableLPTRefMode(),
            TestImageDesc("vpu/efficient/car_1.jpg", ImageFormat::RGB),
            1, 0.15f);
}

TEST_F(KmbClassifyNetworkTest, precommit_efficient_b0_dogs) {
    runTest(
            TestNetworkDesc("efficientnet-b0-stanford-dogs/caffe2/FP16-INT8/efficientnet-b0-stanford-dogs.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP32)
                    .enableLPTRefMode(),
            TestImageDesc("vpu/efficient/dog_1.jpg", ImageFormat::RGB),
            1, 0.15f);
}

TEST_F(KmbClassifyNetworkTest, efficient_b0_aircrafts) {
    runTest(
            TestNetworkDesc("efficientnet-b0-aircrafts/caffe2/FP16-INT8/efficientnet-b0-aircrafts.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP32)
                    .enableLPTRefMode(),
            TestImageDesc("vpu/efficient/aircraft_1.jpg", ImageFormat::RGB),
            1, 0.15f);
}

TEST_F(KmbClassifyNetworkTest, mobilenet_v3_cars) {
    runTest(
            TestNetworkDesc("mobilenet-v3-small-stanford-cars/caffe2/FP16-INT8/mobilenet-v3-small-stanford-cars.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("vpu/efficient/car_1.jpg", ImageFormat::RGB),
            1, 0.15f);
}

// TODO: [Track number: E#9578]
TEST_F(KmbClassifyNetworkTest, precommit_mobilenet_v3_dogs) {
    if (isByPass()) {
        SKIP() << "Skip for by-pass mode due to bad accuracy";
    }
    runTest(
            TestNetworkDesc("mobilenet-v3-small-stanford-dogs/caffe2/FP16-INT8/mobilenet-v3-small-stanford-dogs.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("vpu/efficient/dog_1.jpg", ImageFormat::RGB),
            1, 0.15f);
}

TEST_F(KmbClassifyNetworkTest, mobilenet_v3_aircrafts) {
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

// [Track number: D#7790]
TEST_P(ModelAdk, precommit_StackedHourGlass_BDK1) {
    runTest(
            TestNetworkDesc("BDK1/StackedHourGlass_INT8/hg-s8-b1-mpii.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input.1", Precision::U8)
                    .setUserInputLayout("input.1", Layout::NCHW)
                    .setUserOutputPrecision("4702", Precision::FP16),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
            0.0025f);
}

// TODO: [Track number: E#9578]
TEST_F(SmokeNetworkTest, SuperResolution_AA_ADK3) {
    if (isByPass()) {
        SKIP() << "Skip inference for by-pass mode due to exception - dims and format are inconsistent.";
    }
    runTest(
            TestNetworkDesc("ADK3/SuperRes_INT8/SuperRes_INT8_AA.xml", EXPERIMENTAL)
                    .setUserInputPrecision("netInput", Precision::U8)
                    .setUserInputPrecision("t_param", Precision::U8)
                    .setUserInputPrecision("t_param1", Precision::U8)
                    .setUserOutputPrecision("tl_unet1x2x4x/out1x/add_2", Precision::FP16)
                    .setUserOutputPrecision("tl_unet1x2x4x/out2x/add_2", Precision::FP16)
                    .setUserOutputPrecision("tl_unet1x2x4x/out4x/add_1", Precision::FP16));
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

// [Track number: S#47419]
TEST_F(ModelAdk, precommit_DeBlur_ADK3) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "bad results");
    runTest(
            TestNetworkDesc("ADK3/DeBlur_INT8/DeBlur_deepImageDeblur.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP16)
                    .setCompileConfig({{"VPUX_THROUGHPUT_STREAMS", "1"}}),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::BGR),
            0.0025f);
}

// [Track number: EISW-10831]
TEST_F(ModelAdk, DISABLED_DeBlur_AA_BDK2) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "bad results");
    runTest(
            TestNetworkDesc("../clientmodels/BDK2/Deblur/AccuracyAware/2020.1_INT8_Deblur_AccuracyAwareQuantization.xml", EXPERIMENTAL)
                    .setUserInputPrecision("img_placeholder", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP16)
                    .setCompileConfig({{"VPUX_THROUGHPUT_STREAMS", "1"}}),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::BGR),
            0.0025f);
}

const static std::vector<InferenceEngine::Precision> inputPrecision = {
                InferenceEngine::Precision::U8,
                InferenceEngine::Precision::FP16,
                InferenceEngine::Precision::FP32};

INSTANTIATE_TEST_CASE_P(PrecisionCase, ModelAdk, ::testing::ValuesIn(inputPrecision));

TEST_F(UnetNetworkTest, UnetCamvidAva0001_ADK3) {
    runTest(
            TestNetworkDesc("ADK3/unet-camvid-onnx-0001/caffe2/FP16-INT8/unet-camvid-onnx-0001.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP16),
            TestImageDesc("480x360/0016E5_07959.png", ImageFormat::RGB),
            0.4f);  // mean intersection over union tolerance
}

TEST_F(KmbClassifyNetworkTest, precommit_MobilenetV2_ADK3) {
    runTest(
            TestNetworkDesc("ADK3/mobilenet-v2/caffe2/FP16-INT8/mobilenet-v2.xml", EXPERIMENTAL)
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP16),
            TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
            1, 2.0f);
}

// [Track number: S#47647]
TEST_F(KmbSuperResNetworkTest, precommit_SuperResolution_ADK3) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "bad results");
    const std::string imgName  = "netInput";
    const std::string paramName1 = "t_param";
    const std::string paramName2 = "t_param1";
    runTest(
            TestNetworkDesc("ADK3/SuperRes_INT8/SuperRes_INT8.xml", EXPERIMENTAL)
            .setUserInputPrecision(imgName, Precision::U8)
            .setUserInputLayout(imgName, Layout::NHWC)
            .setUserInputPrecision(paramName1, Precision::U8)
            .setUserInputLayout(paramName1, Layout::C)
            .setUserInputPrecision(paramName2, Precision::U8)
            .setUserInputLayout(paramName2, Layout::C)
            .setUserOutputPrecision("tl_unet1x2x4x/out1x/add_2", Precision::FP16)
            .setUserOutputPrecision("tl_unet1x2x4x/out2x/add_2", Precision::FP16)
            .setUserOutputPrecision("tl_unet1x2x4x/out4x/add_1", Precision::FP16),
            imgName,
            TestBinFileDesc("vpu/super_resolution_input/input-0-U8.bin", {1, 3, 192, 192}, Precision::U8),
            paramName1,
            TestBinFileDesc("vpu/super_resolution_input/input-1-U8.bin", {1}, Precision::U8),
            paramName2,
            TestBinFileDesc("vpu/super_resolution_input/input-2-U8.bin", {1}, Precision::U8));
}

class KmbSuperResNetworkTestParameterized : public KmbSuperResNetworkTest,
    public testing::WithParamInterface<std::tuple<InferenceEngine::Precision, InferenceEngine::Precision>> {};

// [Track number: S#47647]
TEST_P(KmbSuperResNetworkTestParameterized, PrecisionTest) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "bad results");
    auto inputPrecision = std::get<0>(GetParam());
    auto outputPrecision = std::get<1>(GetParam());
    const std::string imgName  = "netInput";
    const std::string paramName1 = "t_param";
    const std::string paramName2 = "t_param1";

    std::map<InferenceEngine::Precision, std::string> precisionToStr = {
        {InferenceEngine::Precision::U8, "U8"},
        {InferenceEngine::Precision::FP16, "FP16"},
        {InferenceEngine::Precision::FP32, "FP32"}
    };
    runTest(
            TestNetworkDesc("ADK3/SuperRes_INT8/SuperRes_INT8.xml", EXPERIMENTAL)
            .setUserInputPrecision(imgName, inputPrecision)
            .setUserInputLayout(imgName, Layout::NHWC)
            .setUserInputPrecision(paramName1, inputPrecision)
            .setUserInputLayout(paramName1, Layout::C)
            .setUserInputPrecision(paramName2, inputPrecision)
            .setUserInputLayout(paramName2, Layout::C)
            .setUserOutputPrecision("tl_unet1x2x4x/out1x/add_2", outputPrecision)
            .setUserOutputPrecision("tl_unet1x2x4x/out2x/add_2", outputPrecision)
            .setUserOutputPrecision("tl_unet1x2x4x/out4x/add_1", outputPrecision),
            imgName,
            TestBinFileDesc("vpu/super_resolution_input/input-0-"+precisionToStr[inputPrecision]+".bin", {1, 3, 192, 192}, inputPrecision),
            paramName1,
            TestBinFileDesc("vpu/super_resolution_input/input-1-"+precisionToStr[inputPrecision]+".bin", {1}, inputPrecision),
            paramName2,
            TestBinFileDesc("vpu/super_resolution_input/input-2-"+precisionToStr[inputPrecision]+".bin", {1}, inputPrecision));
    }

static const std::vector<InferenceEngine::Precision> SRInputPrecision = {
            InferenceEngine::Precision::U8,
            InferenceEngine::Precision::FP16,
            InferenceEngine::Precision::FP32
};

static const std::vector<InferenceEngine::Precision> SROutputPrecision = {
            InferenceEngine::Precision::FP16,
            InferenceEngine::Precision::FP32
};

static const auto allPrecisionCombine = ::testing::Combine(
        ::testing::ValuesIn(SRInputPrecision),
        ::testing::ValuesIn(SROutputPrecision));

INSTANTIATE_TEST_CASE_P(PrecisionCombine, KmbSuperResNetworkTestParameterized, allPrecisionCombine);
