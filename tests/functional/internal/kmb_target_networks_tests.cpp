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

#include <file_utils.h>

#include "test_model/kmb_test_base.hpp"

// TODO: [Track number: C#40310]
//       We need to remove or transform XML based tests before opening the source.

//
// precommit scope
//


//
// ResNet50 FP16 IRv10
//
// [Track number: S#48139]
TEST_F(KmbClassifyNetworkTest, precommit_resnet_50_pytorch_dense_fp16_IRv10) {
#ifndef __aarch64__
    SKIP_INFER_ON("VPUX", "bad results");
#endif
    runTest(
        TestNetworkDesc("KMB_models/FP16/resnet_50_pytorch/resnet-50-pytorch.xml")
            .setUserInputLayout("input", Layout::NHWC)
            .setUserInputPrecision("input", Precision::FP16)
            .setUserOutputPrecision("output", Precision::FP16),
        "224x224/cat3.bmp",
        3, 0.05);
}

// [Track number: S#48139]
TEST_F(KmbClassifyNetworkTest, precommit_resnet_50_pytorch_dense_fp16_IRv10_u8_input) {
#ifndef __aarch64__
    SKIP_INFER_ON("VPUX", "bad results");
#endif
    runTest(
            TestNetworkDesc("KMB_models/FP16/resnet_50_pytorch/resnet-50-pytorch.xml")
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP16)
                    .setCompileConfig({{"VPU_COMPILER_ALLOW_U8_INPUT_FOR_FP16_MODELS", "YES"}}),
            "224x224/cat3.bmp",
            3, 0.05);
}

// [Track number: S#48139]
// [Track number: E#7736]
TEST_F(KmbClassifyNetworkTest, precommit_mobilenet_v2_pytorch_dense_IRv10_fp16) {
    SKIP_INFER_ON("VPUX", "bad results");
    runTest(
            TestNetworkDesc("KMB_models/FP16/MobileNet_v2_pytorch/mobilenet-v2_pytorch_dense_fp16_ww34.xml")
                    .setUserInputPrecision("input", Precision::FP16)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP16),
            TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
            3, 2.5f);
}

TEST_F(KmbStereoNetworkTest, precommit_INT8_Stereo_720p) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/customnets/stereo/ngraph_stereo_720p.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NCHW)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestBinFileDesc("1280x720/stereo_1280x720.bin", {1, 1, 720, 1280}, Precision::U8), 0.0f);
}

#ifdef KMB_HAS_CUSTOM_OCL_KERNELS
TEST_F(KmbYoloV2NetworkTest, precommit_yolo_tiny_v2_ava_0001_tf_dense_int8_IRv10_from_fp32_custom) {
    const auto customLayers = std::make_pair(VPU_COMPILER_CONFIG_KEY(CUSTOM_LAYERS),
                                             getIELibraryPath() + "/kmb_custom_ocl_kernels/yolov2.xml");
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/yolo-tiny-v2-ava-0001/yolo_tiny_v2_ava_0001_tf_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32)
                    .setCompileConfig({customLayers}),
            TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
            0.6, 0.4, 0.4, false);
}
#endif  // KMB_HAS_CUSTOM_OCL_KERNELS

// KMB : Bad inference results. Possible bug in test system.
// [Track number: S#28790]
TEST_F(KmbYoloV2NetworkTest, precommit_yolo_tiny_v2_ava_0001_tf_dense_int8_IRv10_from_fp32) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/yolo-tiny-v2-ava-0001/yolo_tiny_v2_ava_0001_tf_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
            0.6, 0.4, 0.4, false);
}

#ifdef KMB_HAS_CUSTOM_OCL_KERNELS
TEST_F(KmbYoloV2NetworkTest, precommit_yolo_v2_ava_0001_tf_dense_int8_IRv10_from_fp32_custom) {
    const auto customLayers = std::make_pair(VPU_COMPILER_CONFIG_KEY(CUSTOM_LAYERS),
                                             getIELibraryPath() + "/kmb_custom_ocl_kernels/yolov2.xml");
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/yolo-v2-ava-0001/yolo_v2_ava_0001_tf_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32)
                    .setCompileConfig({customLayers}),
            TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
            0.6, 0.4, 0.4, false);
}
#endif  // KMB_HAS_CUSTOM_OCL_KERNELS

TEST_F(KmbYoloV2NetworkTest, precommit_yolo_v2_ava_0001_tf_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/yolo-v2-ava-0001/yolo_v2_ava_0001_tf_dense_int8_IRv10_fp16_to_int8.xml")

                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
            0.6, 0.4, 0.4, false);
}

TEST_F(KmbYoloV1NetworkTest, precommit_INT8_Dense_TF_DarkNet_TinyYoloV1) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/tiny_yolo_v1/tiny_yolo_v1_tf_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("512x512/dog_croped512.bmp", ImageFormat::RGB),
            0.6, 0.4, 0.4, true);
}

TEST_F(KmbClassifyNetworkTest, precommit_mobilenet_v2_pytorch_caffe2_dense_int8_IRv10_from_fp32_no_align) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/MobileNet_V2/mobilenet_v2_pytorch_caffe2_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32)
                    .setCompileConfig({{"VPU_COMPILER_ELTWISE_SCALES_ALIGNMENT", "NO"}}),
            TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
            1, 7.0f);
}

TEST_F(KmbClassifyNetworkTest, precommit_facenet_20180408_102900_tf_dense_int8_IRv10_from_fp32) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/facenet-20180408-102900/facenet_20180408_102900_tf_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
            1, 0.05f);
}

// Required w/a is disabling SplitOverH clustering strategy in compilation descriptor
// [Track number: H#18012385770]
// [Track number: H#18013202155]
TEST_F(KmbRetinaFaceNetworkTest, precommit_retinaface_mobilenetv2_0_25_modified) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/private/retinaface-mobilenetv2-0.25-modified/retinaface-mobilenetv2-0.25-modified.xml")
                    .setUserInputPrecision("input", Precision::U8),
            "data",
            TestImageDesc("300x300/20_Family_Group_Family_Group_20_1003.jpg", ImageFormat::RGB));
}

TEST_F(AgeGenderNetworkTest, precommit_age_gender_retail_0013) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/age-gender-recognition-retail-0013/caffe/FP16-INT8/age-gender-recognition-retail-0013_ww22.xml")
                    .setUserInputPrecision("input", Precision::U8),
            TestImageDesc("62x62/face62.bmp", ImageFormat::RGB),
            0.1f);
}

TEST_F(KmbVasFDStage1Test, precommit_vasfd_stage1) {
// [Track number: #7733]
#ifdef __aarch64__
    SKIP_INFER_ON("VPUX", "Wrong results");
#endif
    const std::string inputName = "data";
    const std::vector<std::string> layerNames = {
            "b12", "b16", "b24", "b32", "b48",
            "b64", "b96", "b128", "b192"};
    const std::vector<int> anchorSizes = {4, 3, 2, 3, 2, 3, 2, 3, 2};
    const std::vector<int> windowScales = {8, 8, 8, 16, 16, 32, 32, 64, 64};
    const std::vector<int> windowLengths = {12, 16, 24, 32, 48, 64, 96, 128, 192};

    runTest(
            TestNetworkDesc("KMB_models/FP16/face_detection_stage1/vasfd_stage1.xml")
                    .setUserInputLayout(inputName, Layout::NHWC)
                    .setUserInputPrecision(inputName, Precision::U8),
            TestImageDesc("320x240/Alma_Powell_0_0.1133.jpg", ImageFormat::BGR),
            0.35f, 0.1f, 0.3f, layerNames, anchorSizes, windowScales, windowLengths);
}

TEST_F(KmbVasFRTest, precommit_vasfr_feature) {
    const std::string inputName = "input_data";
    runTest(
            TestNetworkDesc("KMB_models/FP16-INT8/private/face_recognition/vasfr_feature.xml")
                    .setUserInputPrecision(inputName, Precision::U8),
            TestImageDesc("112x112/Charlize_Theron_0001.jpg", ImageFormat::BGR),
            0.6f);
}

// MTL target compilation test
// [Track number: C#46795]
TEST_F(KmbClassifyNetworkTest, precommit_resnet_50_pytorch_dense_int8_IRv10_fp16_to_int8_MTL) {
    SKIP() << "LpScheduler - RuntimeError: Precondition violation";
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/ResNet-50/resnet_50_pytorch_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::U8)  // currently FP16 is not supported by runtime
                    .setCompileConfig({{"VPU_COMPILER_COMPILATION_DESCRIPTOR", "release_mtl-sc"},
                                       {"VPU_COMPILER_TARGET_DESCRIPTOR", "release_mtl"},
                                       {"VPU_COMPILER_ALLOW_U8_INPUT_FOR_FP16_MODELS", "NO"}}),


            "224x224/cat3.bmp",
            3, 0.05);
}

TEST_F(KmbClassifyNetworkTest, precommit_squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8_MTL) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "Wrong detection results");  // At the moment no EVM is setup so cannot run
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1/squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::U8)  // currently FP16 is not supported by runtime
                    .setUserOutputLayout("output", Layout::NHWC)
                    .setCompileConfig({{"VPU_COMPILER_COMPILATION_DESCRIPTOR", "release_mtl-sc"},
                                       {"VPU_COMPILER_TARGET_DESCRIPTOR", "release_mtl"},
                                       {"VPU_COMPILER_ALLOW_U8_INPUT_FOR_FP16_MODELS", "NO"}}),
            TestImageDesc("227x227/watch.bmp", ImageFormat::RGB),
            1, 0.5f);
}

TEST_F(PersonAttrRecNetworkTest, precommit_person_attribute_recognitnion_crossroad_0238) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/person-attributes-recognition-crossroad/person-attributes-recognition-crossroad-0238.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP16),
            TestImageDesc("vpu/person-attributes-recognition-crossroad.jpg", ImageFormat::BGR), 0.07f);
}

// This test checks correctness of handling FP16 input in case of quantized model
// for which inner network precision will be U8
// [Track number: S#48139]
TEST_F(KmbClassifyNetworkTest, precommit_mobilenet_v1_025_128_FP16) {
#ifndef __aarch64__
    SKIP_INFER_ON("VPUX", "bad results");
#endif
    runTest(
            TestNetworkDesc("KMB_models/FP16-INT8/public/mobilenet-v1-0.25-128/mobilenet-v1-0.25-128.xml")
                    .setUserInputPrecision("input", Precision::FP16),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::BGR),
            1,
            0.3f);
}

// [Track number: S#48139]
TEST_F(KmbClassifyNetworkTest, precommit_mobilenet_v1_025_128_FP32) {
#ifndef __aarch64__
    SKIP_INFER_ON("VPUX", "bad results");
#endif
    runTest(
            TestNetworkDesc("KMB_models/FP16-INT8/public/mobilenet-v1-0.25-128/mobilenet-v1-0.25-128.xml")
                    .setUserInputPrecision("input", Precision::FP32),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::BGR),
            1,
            0.3f);
}

// [Track number: S#48139]
TEST_F(KmbClassifyNetworkTest, precommit_aclnet_des_53_vpu) {
#ifndef __aarch64__
    SKIP_INFER_ON("VPUX", "exception - load graph to device");
#endif
    runTest(
            TestNetworkDesc("KMB_models/FP16-INT8/public/aclnet-des-53-vpu/aclnet-des-53-vpu.xml")
                    .setUserInputPrecision("input", Precision::FP16),
            TestBinFileDesc("vpu/audio_16k/airplane_3_17-FP16.bin", {1, 1, 1, 16000}, Precision::FP16),
            1,
            0.3f);
}

// TODO: [Track number: E#9578]
TEST_F(SmokeNetworkTest, precommit_yolo_v4_tf_full) {
    if (isByPass()) {
        SKIP() << "Skip for by-pass mode due to exception - couldn't load the graph into the device";
    }
#ifdef _WIN32
    SKIP() << "SEH exception";
#endif
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/yolo_v4/yolo_v4_tf.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP32));
}

TEST_F(KmbClassifyNetworkTest, precommit_shufflenet_v2_x1_0_pytorch) {
    runTest(
            TestNetworkDesc("KMB_models/FP16-INT8/public/shufflenet-v2-x1_0-pytorch/shufflenet-v2-x1_0-pytorch.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NCHW)
                    .setUserOutputPrecision("output", Precision::FP32)
                    .setUserOutputLayout("output", Layout::NC),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
            3, 0.5f);
}

TEST_F(KmbDetectionNetworkTest, precommit_peleenet) {
    runTest(
            TestNetworkDesc("KMB_models/FP16-INT8/public/peleenet/peleenet.xml")
                    .setUserInputPrecision("input", Precision::U8),
            TestImageDesc("300x300/dog.bmp", ImageFormat::BGR),
            0.3f,
            0.1f, 0.3f);
}

// TODO: [Track number: E#9578]
TEST_F(KmbDetectionNetworkTest, precommit_vehicle_license_plate_detection_barrier_0106_tf_dense_int8_IRv10_from_fp32) {
    if (isByPass()) {
        SKIP() << "Skip for by-pass mode due to bad accuracy";
    }
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/vehicle-license-plate-detection-barrier-0106/vehicle_license_plate_detection_barrier_0106_tf_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8),
            TestImageDesc("736x416/dss_val_05.png", ImageFormat::BGR),
            0.3f,
            0.25f, 0.3f);
}


//
// General scope
//

TEST_F(SmokeNetworkTest, yolo_v4_subgraph_ddr_output_test) {
#ifdef _WIN32
    SKIP() << "SEH exception";
#endif
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "bad results");
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/yolo_v4_subgraph/FP16-INT8/yolo_v4_subgraph.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP16));
}

//
// Stereo360p
//
// [Track number: S#11812]
TEST_F(KmbStereoNetworkTest, INT8_Stereo_360p) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "bad results");
    runTest(
        TestNetworkDesc("KMB_models/INT8/customnets/stereo/ngraph_stereo_360p.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NCHW)
            .setUserOutputPrecision("output", Precision::FP32),
        TestBinFileDesc("640x360/stereo_640x360.bin", {1, 1, 360, 640}, Precision::U8), 0.0f);
}

//
// TinyYolo V2
//

// [Track number: S#40783]
TEST_F(KmbYoloV1NetworkTest, INT8_Dense_Cntk_TinyYoloV2) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "bad results");
    runTest(
        TestNetworkDesc("KMB_models/INT8/private/tiny_yolo_v2/tiny_yolo_v2_cntk_dense_int8_IRv10.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("512x512/dog_croped512.bmp", ImageFormat::BGR),
        0.6, 0.4, 0.4, true);
}

//
// Yolo V3
//

const static std::vector<InferenceEngine::Layout> inputLayout = {
        InferenceEngine::Layout::NCHW,
        InferenceEngine::Layout::NHWC};

const static std::vector<InferenceEngine::Layout> outputLayout = {
        InferenceEngine::Layout::NCHW,
        InferenceEngine::Layout::NHWC};

class KmbYoloV3NetworkTestWithSpecificLayout : public KmbYoloV3NetworkTest,
    public testing::WithParamInterface<std::tuple<InferenceEngine::Layout, InferenceEngine::Layout>> {};

// [Track number: S#48139]
TEST_P(KmbYoloV3NetworkTestWithSpecificLayout, INT8_Dense_TF_YoloV3) {
#ifndef __aarch64__
    SKIP_INFER_ON("VPUX", "exception - load graph to device");
#endif
    std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0,
                        45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/yolo_v3/yolo_v3_tf_dense_int8_IRv10.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", std::get<0>(GetParam()))
            .setUserOutputPrecision("conv2d_58/Conv2D/YoloRegion", Precision::FP32)
            .setUserOutputPrecision("conv2d_66/Conv2D/YoloRegion", Precision::FP32)
            .setUserOutputPrecision("conv2d_74/Conv2D/YoloRegion", Precision::FP32)
            .setUserOutputLayout("conv2d_58/Conv2D/YoloRegion", std::get<1>(GetParam()))
            .setUserOutputLayout("conv2d_66/Conv2D/YoloRegion", std::get<1>(GetParam()))
            .setUserOutputLayout("conv2d_74/Conv2D/YoloRegion", std::get<1>(GetParam())),
        TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
        0.6, 0.4, 0.4, 80, 4, 3,
        anchors);
}

INSTANTIATE_TEST_CASE_P(all_layouts, KmbYoloV3NetworkTestWithSpecificLayout,
    ::testing::Combine(::testing::ValuesIn(inputLayout), ::testing::ValuesIn(outputLayout)));

class KmbClassifyNetworkTestWithSpecificLayout : public KmbClassifyNetworkTest, public testing::WithParamInterface<InferenceEngine::Layout> {};

INSTANTIATE_TEST_CASE_P(precommit, KmbClassifyNetworkTestWithSpecificLayout, ::testing::ValuesIn(inputLayout));

TEST_F(KmbClassifyNetworkTest, squeezenet1_1_caffe2_force_compilation) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1/squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32)
            .setUserOutputLayout("output", Layout::NHWC)
            .enableForcedCompilation(),
        TestImageDesc("227x227/cat3.bmp", ImageFormat::RGB),
        1, 2.0f);
}

class KmbDetectionNetworkTestWithSpecificLayout : public KmbDetectionNetworkTest, public testing::WithParamInterface<InferenceEngine::Layout> {};

// [Track number: E#11501]
TEST_P(KmbDetectionNetworkTestWithSpecificLayout, DISABLED_face_detection_retail_caffe_IRV10_fp16_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/face-detection-retail-0004/caffe/FP16-INT8/face-detection-retail-0004-ww22.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", GetParam()),
            TestImageDesc("300x300/20_Family_Group_Family_Group_20_1003.jpg", ImageFormat::RGB),
            0.3f,
            1.f, 0.3f);
}


INSTANTIATE_TEST_CASE_P(precommit, KmbDetectionNetworkTestWithSpecificLayout, ::testing::ValuesIn(inputLayout));

// TODO 4 similar tests face_detection_retail_caffe_IRV10_fp16_int8_*
TEST_F(KmbDetectionNetworkTest, face_detection_retail_caffe_IRV10_fp16_int8_nchw_fuse_scale_input_accuracy_drop) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/face-detection-retail-0004/caffe/FP16-INT8/face-detection-retail-0004-ww22.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NCHW),
            TestImageDesc("300x300/0_Parade_marchingband_1_1004.jpg", ImageFormat::RGB),
            0.3f,
            1.f, 0.3f);
}

// [Track number: S#41097]
TEST_F(KmbDetectionNetworkTest, face_detection_retail_caffe_IRV10_fp16_int8_nhwc_fuse_scale_input_accuracy_drop) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/face-detection-retail-0004/caffe/FP16-INT8/face-detection-retail-0004-ww22.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC),
            TestImageDesc("300x300/0_Parade_marchingband_1_1004.jpg", ImageFormat::RGB),
            0.3f,
            1.f, 0.3f);
}

// Compilation time is about 10 minutes
// Required > 13 GB DDR
// [Track number: S#42011]
// [Track number: S#39223]
TEST_F(KmbClassifyNetworkTest, DISABLED_vgg16_caffe_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/vgg16/vgg16_caffe_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
            1, 0.05f);
}

class SmokeNetworkTestWithSpecificLayout : public SmokeNetworkTest, public testing::WithParamInterface<InferenceEngine::Layout> {};
TEST_P(SmokeNetworkTestWithSpecificLayout, openpose_pose_cf) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/OpenPose/FP16-INT8/openpose-pose_cf_ww22.xml")
            .setUserInputPrecision("image", Precision::U8)
            .setUserInputLayout("image", GetParam())
            .setUserOutputPrecision("output", Precision::FP32));
}
INSTANTIATE_TEST_CASE_P(precommit, SmokeNetworkTestWithSpecificLayout, ::testing::ValuesIn(inputLayout));

// [Track number: E#12913]
TEST_F(KmbDetectionNetworkTest, face_detection_adas_0001) {
    SKIP() << "Invalid overwrite state of inplace output";
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/face-detection-adas-0001/face-detection-adas-0001.xml")
	    .setUserInputPrecision("input", Precision::U8)
	    .setUserInputLayout("input", Layout::NHWC),
        TestImageDesc("300x300/20_Family_Group_Family_Group_20_1003.jpg", ImageFormat::BGR),
        0.3f,
        1.f, 0.3f);
}

// TODO Create ticket
TEST_F(HeadPoseEstimationNetworkTest, head_pose_estimation_adas_0001) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hang on infer");
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/head_pose_estimation_adas_0001/head-pose-estimation-adas-0001.xml")
	    .setUserInputPrecision("input", Precision::U8),
        TestImageDesc("60x60/head-pose-estimation-adas-0001.png", ImageFormat::BGR),
        0.1f);
}

TEST_F(PersonAttrRecNetworkTest, person_attribute_recognitnion_crossroad_0234) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/person-attributes-recognition-crossroad/person-attributes-recognition-crossroad-0234.xml")
                .setUserInputPrecision("input", Precision::U8)
                .setUserInputLayout("input", Layout::NHWC)
                .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("vpu/person-attributes-recognition-crossroad.jpg", ImageFormat::BGR), 0.02f);
}

TEST_F(KmbDetectionNetworkTest, person_detection_retail_0013) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hang on infer");
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/person-detection-retail-0013/person-detection-retail-0013.xml")
	    .setUserInputPrecision("input", Precision::U8),
        TestImageDesc("544x320/pedestrian.jpg", ImageFormat::BGR),
        0.3f,
        0.1f, 0.3f);
}

// C++ exception with description "Cannot convert layer "efficientnet-b0/model/stem/swish_f32"
// due to unsupported layer type "Swish"
// [Track number: D#3769]
TEST_F(KmbClassifyNetworkTest, DISABLED_efficientnet_b0) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/efficientnet-b0/efficientnet-b0.xml")
            .setUserInputPrecision("input", Precision::U8),
	TestImageDesc("224x224/husky.bmp", ImageFormat::BGR),
        1,
        0.3f);
}

// C++ exception with description "Cannot convert layer "MobilenetV3/Conv/hard_swish/mul_1"
// due to unsupported layer type "HSwish"
// [Track number: D#3775]
// Ngraph pass to align concat scales is under development
// Once that is merged, re enable this networks compilation
TEST_F(KmbClassifyNetworkTest, DISABLED_mobilenet_v3_small) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hang on infer");
    runTest(
        TestNetworkDesc("KMB_models/FP16-INT8/private/mobilenet-v3-small-1.0-224/mobilenet-v3-small-1.0-224.xml")
            .setUserInputPrecision("input", Precision::U8),
	TestImageDesc("224x224/husky.bmp", ImageFormat::BGR),
	1,
	0.3f);
}
