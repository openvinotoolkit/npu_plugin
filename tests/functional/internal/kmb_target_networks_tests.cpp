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

#include <file_utils.h>

#include "test_model/kmb_test_base.hpp"

// TODO: [Track number: C#40310]
//       We need to remove or transform XML based tests before opening the source.

//
// ResNet50 FP16 IRv10
//
// [Track number: S#48139]
TEST_F(KmbClassifyNetworkTest, precommit_resnet_50_pytorch_dense_fp16_IRv10) {
    SKIP_INFER_BYPASS_ON("VPUX", "bad results");
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
    SKIP_INFER_BYPASS_ON("VPUX", "bad results");
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
TEST_F(KmbClassifyNetworkTest, precommit_mobilenet_v2_pytorch_dense_IRv10_fp16) {
    SKIP_INFER_BYPASS_ON("VPUX", "bad results");
    runTest(
            TestNetworkDesc("KMB_models/FP16/MobileNet_v2_pytorch/mobilenet-v2_pytorch_dense_fp16_ww34.xml")
                    .setUserInputPrecision("input", Precision::FP16)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP16),
            TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
            3, 2.5f);
}

TEST_F(KmbClassifyNetworkTest, INT8_Dense_PyTorch_IRv10_ResNet_50) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/ResNet-50/resnet-50-pytorch-from-icv-bench-cache.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
        1, 2.5f);
}

TEST_F(KmbClassifyNetworkTest, precommit_INT8_Dense_Caffe2_IRv10_ResNet_50_v1) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/private/ResNet-50/resnet50_v1_caffe2_dense_int8_IRv10.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("224x224/watch.bmp", ImageFormat::BGR),
        1, 2.5f);
}

TEST_F(KmbClassifyNetworkTest, precommit_INT8_Dense_Caffe2_IRv10_ResNet_50_v2) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/private/ResNet-50/resnet50_v2_caffe2_dense_int8_IRv10.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("224x224/watch.bmp", ImageFormat::BGR),
        1, 2.5f);
}

//
// MobileNetV2
//

TEST_F(KmbClassifyNetworkTest, INT8_Dense_Caffe_IRv10_MobileNet_V2) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/MobileNet_V2/mobilenet-v2-caffe-IRv10.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16)
            .setUserOutputLayout("output", Layout::NHWC),
        TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
        2, 0.7f);
}

// CPU : Supported primitive descriptors list is empty for node: Add1_/Fused_Add_
TEST_F(KmbClassifyNetworkTest, INT8_Dense_PyTorch_IRv10_MobileNet_V2) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/MobileNet_V2/mobilenet-v2-pytorch-from-icv-bench-cache.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
        3, 2.15f);
}

//
// InceptionV1
//

TEST_F(KmbClassifyNetworkTest, INT8_Dense_Caffe_IRv10_Inception_V1) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v1_caffe/googlenet-v1-caffe-from-icv-bench-cache.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
        3, 0.05f);
}

//
// InceptionV3
//

TEST_F(KmbClassifyNetworkTest, INT8_Dense_PyTorch_IRv10_Inception_V3) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "bad results");
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v3_tf/googlenet-v3-pytorch-from-icv-bench-cache.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("299x299/n01537544_28.bmp", ImageFormat::RGB),
        1, 1e-1f);
}

TEST_F(KmbClassifyNetworkTest, INT8_Dense_TF_IRv10_Inception_V3) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v3_tf/googlenet-v3-tf-frozen-from-icv-bench-cache.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("299x299/n01537544_28.bmp", ImageFormat::RGB),
        1, 1e-1f);
}

//
// SqueezeNet 1.1
//

// FIXME: Missing IR in models-ir repository
// ????
TEST_F(KmbClassifyNetworkTest, DISABLED_INT8_Dense_Caffe2_IRv10_SqueezeNet_1_1) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1_caffe/squeezenet1.1-caffe2-uint8-int8-weights-perchannel-IRv10.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("227x227/cat3.bmp", ImageFormat::RGB),
        3, 1e-1f);
}

//
// TinyYolo V1
//

TEST_F(KmbYoloV1NetworkTest, INT8_Dense_TF_DarkNet_TinyYoloV1) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/tiny_yolo_v1/tiny_yolo_v1_tf_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("512x512/dog_croped512.bmp", ImageFormat::RGB),
        0.6, 0.4, 0.4, true);
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
    SKIP_INFER_BYPASS_ON("VPUX", "exception - load graph to device");
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

//////////////////////////////////////////
// Start of test-set for KMB-alpha IRv10
//////////////////////////////////////////

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


// Compilation fails on windows
// [Track number: D#44765]
TEST_F(KmbYoloV2NetworkTest, precommit_yolo_v2_ava_0001_tf_dense_int8_IRv10_from_fp32) {
#ifdef _WIN32
    SKIP() << "LpScheduler - RuntimeError: input is not a DAG";
#endif
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/yolo-v2-ava-0001/yolo_v2_ava_0001_tf_dense_int8_IRv10_from_fp32.xml")
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

// Wrong detection results
// [Track number: S#41494]
TEST_F(KmbYoloV2NetworkTest, precommit_yolo_v2_pytorch_dense_int8_IRv10_fp16_to_int8) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "Wrong detection results");
    runTest(
            TestNetworkDesc("KMB_models/INT8/private/yolo_v2_pytorch/yolov2_pytorch_dense_int8_IRv10.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
            0.6, 0.4, 0.4, false);
}

class KmbClassifyNetworkTestWithSpecificLayout : public KmbClassifyNetworkTest, public testing::WithParamInterface<InferenceEngine::Layout> {};

TEST_P(KmbClassifyNetworkTestWithSpecificLayout, precommit_resnet_50_pytorch_dense_int8_IRv10_from_fp32) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/ResNet-50/resnet_50_pytorch_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", GetParam())
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("224x224/husky.bmp", ImageFormat::RGB),
        1, 0.7f);
}

INSTANTIATE_TEST_CASE_P(precommit, KmbClassifyNetworkTestWithSpecificLayout, ::testing::ValuesIn(inputLayout));

TEST_F(KmbClassifyNetworkTest, precommit_mobilenet_v2_pytorch_caffe2_dense_int8_IRv10_from_fp32) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/MobileNet_V2/mobilenet_v2_pytorch_caffe2_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
        1, 7.0f);
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

TEST_F(KmbClassifyNetworkTest, precommit_googlenet_v1_tf_dense_int8_IRv10_from_fp32) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/googlenet-v1/googlenet_v1_tf_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
        1, 0.05f);
}

// Bad accuracy
// [Track number: S#39435]
TEST_F(KmbClassifyNetworkTest, precommit_googlenet_v3_tf_dense_int8_IRv10_from_fp32) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/googlenet-v3/googlenet_v3_tf_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("299x299/n01537544_28.bmp", ImageFormat::RGB),
        1, 0.05f);
}

TEST_F(KmbClassifyNetworkTest, precommit_squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_from_fp32) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1/squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32)
            .setUserOutputLayout("output", Layout::NHWC),
        TestImageDesc("227x227/cat3.bmp", ImageFormat::RGB),
        1, 2.0f);
}

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

//////////////////////////////////////////
// End of test-set for KMB-alpha IRv10
//////////////////////////////////////////

//////////////////////////////////////////
// Start of test-set for KMB-beta IRv10
//////////////////////////////////////////

class KmbDetectionNetworkTestWithSpecificLayout : public KmbDetectionNetworkTest, public testing::WithParamInterface<InferenceEngine::Layout> {};

TEST_P(KmbDetectionNetworkTestWithSpecificLayout, face_detection_retail_caffe_IRV10_fp16_int8) {
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

// Sporadic accuracy fail
// [Track number: S#41921]
TEST_F(KmbSSDNetworkTest, DISABLED_precommit_ssd512_caffe_dense_int8_IRv10_from_fp32) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/ssd512/ssd512_caffe_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8),
            TestImageDesc("512x512/dog_croped512.bmp", ImageFormat::RGB),
            0.3f,
            0.1f, 0.3f);
}

// C++ exception with description "Only single input is supported currently
// kmb-plugin/src/frontend_mcm/src/frontend_mcm.cpp:785
// [Track number: D#2723]
// Convertor for operation McmFC_7719 failed due to runtime error
// Op:McmFC_7719 - OpError: Invalid input data (0) - Inconsistent total size of input tensor (input 0) 204800
// and 1st dimension of weights tensor (input 1) 2048
// [Track number: D#40919]
TEST_F(KmbDetectionNetworkTest, DISABLED_precommit_faster_rcnn_resnet101_coco_tf_dense_int8_IRv10_from_fp32) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/faster_rcnn_resnet101_coco/faster_rcnn_resnet101_coco_tf_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP16),
            TestImageDesc("1024x600/frankfurt_001016.jpg", ImageFormat::RGB),
            0.3f,
            0.1f, 0.3f);
}

TEST_F(KmbClassifyNetworkTest, precommit_googlenet_v4_tf_dense_int8_IRv10_from_fp32) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/googlenet-v4/googlenet_v4_tf_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("300x300/dog.bmp", ImageFormat::RGB),
            1, 0.06f);
}

TEST_F(KmbSSDNetworkTest, precommit_ssd_mobilenet_v1_coco_tf_dense_int8_IRv10_from_fp32) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco_tf_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8),
            TestImageDesc("300x300/dog.bmp", ImageFormat::RGB),
            0.3f,
            0.1f, 0.3f);
}

// Interrupted by signal 6: SIGABRT
// KmbFunctionalTests: kmb-plugin/src/frontend_mcm/src/frontend_mcm.cpp:1635:
// void vpu::FrontEndMcm::parseNormalize(const CNNLayerPtr&, const McmNodeVector&):
// Assertion `(dims[1] == weightsSize)' failed.
// TODO Check 2918
// [Track number: D#2918]
TEST_F(KmbClassifyNetworkTest, precommit_facenet_20180408_102900_tf_dense_int8_IRv10_from_fp32) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/facenet-20180408-102900/facenet_20180408_102900_tf_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
            1, 0.05f);
}

// [Track number: D#3675]
// ngraph and legacy parsers:
// Op:bottleneck4_2/dim_inc/bn/variance/Fused_Add_ - ArgumentError: attribute identifer splitStrategy - Undefined identifier
TEST_F(KmbDetectionNetworkTest, DISABLED_precommit_person_vehicle_bike_detection_crossroad_0078_caffe_dense_int8_IRv10_from_fp32) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/person-vehicle-bike-detection-crossroad-0078/person_vehicle_bike_detection_crossroad_0078_caffe_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP16),
            TestImageDesc("1024x1024/frankfurt_001016.png", ImageFormat::RGB),
            0.3f,
            0.1f, 0.3f);
}

TEST_F(KmbDetectionNetworkTest, precommit_vehicle_license_plate_detection_barrier_0106_tf_dense_int8_IRv10_from_fp32) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/vehicle-license-plate-detection-barrier-0106/vehicle_license_plate_detection_barrier_0106_tf_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8),
            TestImageDesc("736x416/dss_val_05.png", ImageFormat::BGR),
            0.3f,
            0.1f, 0.3f);
}

// TODO Update to YoloV3
// FIXME change adapter to Yolo V3 when available
// [Track number: H#1801262299]
// "output strides are set to represent NHWC"
TEST_F(KmbYoloV2NetworkTest, person_vehicle_bike_detection_crossroad_yolov3_1020) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "Bad results");
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/"
                    "person-vehicle-bike-detection-crossroad-yolov3-1020/"
                    "person-vehicle-bike-detection-crossroad-yolov3-1020.xml")
                    .setUserInputPrecision("input", Precision::U8),
            TestImageDesc("500x500/car_fcn8.bmp", ImageFormat::RGB),
            0.6, 0.4, 0.4, false);
}
// Model file KMB_models/INT8/icv/face-detection-retail-0004/face_detection_retail_0004_caffe_dense_int8_IRv10_from_fp32.xml
// cannot be opened!
TEST_F(KmbDetectionNetworkTest, DISABLED_precommit_face_detection_retail_0004_caffe_dense_int8_IRv10_from_fp32) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/face-detection-retail-0004/face_detection_retail_0004_caffe_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP16),
            TestImageDesc("300x300/dog.bmp", ImageFormat::RGB),
            0.3f,
            0.1f, 0.3f);
}

TEST_F(KmbClassifyNetworkTest, precommit_resnet_101_caffe_dense_int8_IRv10_from_fp32) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/resnet-101/resnet_101_caffe_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
            1, 0.05f);
}

TEST_F(KmbClassifyNetworkTest, precommit_resnet_152_caffe_dense_int8_IRv10_from_fp32) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/resnet-152/resnet_152_caffe_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
            1, 0.065f);
}

// C++ exception with description "Op:conv2 - OpError: Invalid input weights (1) -
// Does not match the channel dimension of input 96
// [Track number: D#2799]
// "OpModel:AlexNet - ArgumentError: op:name slice0 - Duplicated op name
TEST_F(KmbClassifyNetworkTest, DISABLED_precommit_alexnet_caffe_dense_int8_IRv10_from_fp32) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/alexnet/alexnet_caffe_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("227x227/cat3.bmp", ImageFormat::RGB),
            1, 0.05f);
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

//////////////////////////////////////////
// End of test-set for KMB-beta IRv10
//////////////////////////////////////////

////////////////////////////////////////////////////////////
// Start of test-set for IRv10 FP16 to INT8 quantization
////////////////////////////////////////////////////////////

// C++ exception with description "Layer Power_123537 supports only power = 1
// kmb-plugin/src/frontend_mcm/src/frontend_mcm.cpp:1464
// [Track number: D#2809]
// Convertor for operation image_preprocess/sub failed due to runtime error Op:image_preprocess/sub - OpError:
// Invalid input inputs (0) - All the inputs of eltwise ops have to share the same size or the other inputs
// must have size 1 and be populated
// Track ?
TEST_F(KmbYoloV2NetworkTest, DISABLED_yolo_tiny_v2_ava_0001_tf_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/yolo-tiny-v2-ava-0001/yolo_tiny_v2_ava_0001_tf_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
            0.6, 0.4, 0.4, false);
}

TEST_F(KmbYoloV2NetworkTest, yolo_v2_ava_0001_tf_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/yolo-v2-ava-0001/yolo_v2_ava_0001_tf_dense_int8_IRv10_fp16_to_int8.xml")

                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
            0.6, 0.4, 0.4, false);
}

TEST_F(KmbClassifyNetworkTest, resnet_50_pytorch_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/ResNet-50/resnet_50_pytorch_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
            1, 2.f);
}

TEST_F(KmbClassifyNetworkTest, mobilenet_v2_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/MobileNet_V2/mobilenet_v2_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
            1, 2.62f);
}

TEST_F(KmbClassifyNetworkTest, googlenet_v1_tf_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/googlenet-v1/googlenet_v1_tf_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
            1, 0.05f);
}

TEST_F(KmbClassifyNetworkTest, googlenet_v3_tf_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/googlenet-v3/googlenet_v3_tf_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("299x299/n01537544_28.bmp", ImageFormat::RGB),
            1, 0.6f);
}

TEST_F(KmbClassifyNetworkTest, squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1/squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32)
                    .setUserOutputLayout("output", Layout::NHWC),
            TestImageDesc("227x227/watch.bmp", ImageFormat::RGB),
            1, 0.5f);
}

TEST_F(KmbClassifyNetworkTest, googlenet_v4_tf_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/googlenet-v4/googlenet_v4_tf_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("300x300/dog.bmp", ImageFormat::RGB),
            1, 0.06f);
}

TEST_F(KmbClassifyNetworkTest, resnet_101_caffe_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/resnet-101/resnet_101_caffe_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
            1, 0.5f);
}

TEST_F(KmbClassifyNetworkTest, resnet_152_caffe_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/resnet-152/resnet_152_caffe_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/watch.bmp", ImageFormat::RGB),
            1, 0.5f);
}

// [Track number: D#3453]
// GraphOptimizer-StrategyManager - LogicError: No strategies created for layer data:0_implicit. Layer possibly unsupported.
// [Track number: S#3331]
// Hang on execution
// [Track number: D#41406]
// Unsupported operation: psroipooled_cls_rois with name PSROIPooling_3288 with C++ type ngraph::op::v0::PSROIPooling
TEST_F(KmbRFCNNetworkTest, DISABLED_rfcn_resnet50_caffe_IRV10_fp16_int8) {
    const std::string data_name    = "data";
    const std::string im_info_name = "im_info";
    runTest(
        TestNetworkDesc("KMB_models/INT8/private/rfcn-resnet50/caffe/FP16-INT8/rfcn-resnet50_ww22.xml")
            .setUserInputPrecision(data_name, Precision::U8)
            .setUserInputLayout(data_name, Layout::NCHW)
            .setUserInputPrecision(im_info_name, Precision::FP32)
            .setUserInputLayout(im_info_name, Layout::NC)
            .setUserOutputPrecision("cls_prob_reshape",  Precision::FP32)
            .setUserOutputPrecision("bbox_pred_reshape", Precision::FP32),
        "data",
        TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
        "im_info",
        {224.f, 224.f, 1.f});
}

////////////////////////////////////////////////////////////
// End of test-set for IRv10 FP16 to INT8 quantization
////////////////////////////////////////////////////////////

TEST_F(KmbClassifyNetworkTest, emotion_recognition_retail_0003) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/emotions-recognition-retail-0003/emotions-recognition-retail-0003_int8_from_fp16.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputLayout("output", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        "vpu/emotions-recognition-retail-0003.png",
        2, 0.1f);
}

TEST_F(KmbSegmentationNetworkTest, icnet_camvid_ava_0001) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/icnet-camvid-ava-tf-0001/icnet_camvid_ava_tf_0001_tf_dense_int8_IRv10.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputLayout("output", Layout::CHW)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("1024x1024/frankfurt_001016.png", ImageFormat::RGB),
        0.3f);  // mean intersection over union tolerance
}

// 10Gb Memory allocation failed
// [Track number: S#42880]
class UnetNetworkTestWithSpecificLayout : public UnetNetworkTest, public testing::WithParamInterface<InferenceEngine::Layout> {};
TEST_P(UnetNetworkTestWithSpecificLayout, DISABLED_unet_camvid_ava_0001) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/unet-camvid-onnx-0001/caffe2/FP16-INT8/unet_camvid_onnx_0001_WW34.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", GetParam())
            .setUserOutputLayout("output", Layout::NCHW),
        TestImageDesc("480x360/0016E5_07959.png", ImageFormat::RGB),
        0.3f);  // mean intersection over union tolerance
}
INSTANTIATE_TEST_CASE_P(precommit, UnetNetworkTestWithSpecificLayout, ::testing::ValuesIn(inputLayout));

// Compilation fails with exception:
// "Caught exception during unit run: QuantizationParams: quantParams -
// ArgumentError: attribute identifer mult - Undefine identifier"
// QuantizationParams: quantParams - ArgumentError: attribute identifer mult - Undefined identifier
// [Track number: D#3707]
TEST_F(GazeEstimationNetworkTest, DISABLED_gaze_estimation_adas_0002) {
    const auto left_eye_input_name = "left_eye_image";
    const auto right_eye_input_name = "right_eye_image";
    const auto head_pos_input_name = "head_pose_angles";
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/gaze-estimation-adas-0002/gaze_estimation_adas_0002_int8_from_fp16_ww22.xml")
            .setUserInputPrecision(left_eye_input_name, Precision::U8)
            .setUserInputLayout(left_eye_input_name, Layout::NHWC)
            .setUserInputPrecision(right_eye_input_name, Precision::U8)
            .setUserInputLayout(right_eye_input_name, Layout::NHWC)
            .setUserInputPrecision(head_pos_input_name, Precision::FP32)
            .setUserInputLayout(head_pos_input_name, Layout::NC)
            .setUserOutputPrecision("output", Precision::FP32),
        left_eye_input_name,
        "vpu/gm_0000_left.png",
        right_eye_input_name,
        "vpu/gm_0000_right.png",
        head_pos_input_name,
        std::vector<float>{-2.076815605163574, -2.1021695137023926, 0.13159990310668945});
}

class SmokeNetworkTestWithSpecificLayout : public SmokeNetworkTest, public testing::WithParamInterface<InferenceEngine::Layout> {};
TEST_P(SmokeNetworkTestWithSpecificLayout, openpose_pose_cf) {
#ifdef _WIN32
    SKIP() << "Skip openpose_pose_cf test on windows due to unexpected error during test execution";
#endif
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/OpenPose/FP16-INT8/openpose-pose_cf_ww22.xml")
            .setUserInputPrecision("image", Precision::U8)
            .setUserInputLayout("image", GetParam())
            .setUserOutputPrecision("output", Precision::FP32));
}
INSTANTIATE_TEST_CASE_P(precommit, SmokeNetworkTestWithSpecificLayout, ::testing::ValuesIn(inputLayout));

TEST_F(AgeGenderNetworkTest, precommit_age_gender_retail_0013) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/age-gender-recognition-retail-0013/caffe/FP16-INT8/age-gender-recognition-retail-0013_ww22.xml")
            .setUserInputPrecision("input", Precision::U8),
        TestImageDesc("62x62/face62.bmp", ImageFormat::RGB),
        0.1f);
}

// [Track number: D#3604]
TEST_F(KmbSSDNetworkTest, ssdlite_mobilenet_v2) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/ssdlite_mobilenet_v2/ssdlite_mobilenet_v2.xml")
            .setUserInputPrecision("image_tensor", Precision::U8),
        TestImageDesc("300x300/dog.bmp", ImageFormat::BGR),
        0.3f,
        0.1f, 0.3f);
}

TEST_F(VehicleAttrRecNetworkTest, vehicle_attributes_recognition_barrier_0042) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/"
                        "vehicle-attributes-recognition-barrier-0042/"
                        "vehicle-attributes-recognition-barrier-0042.xml")
            .setUserInputPrecision("input", Precision::U8),
        TestImageDesc("500x500/test.bmp", ImageFormat::BGR),
        0.25f);
}

// C++ exception with description "Op:L0067_AddBackward1 - OpError: Invalid input inputs (0) -
// All the inputs of eltwise ops have to share the same size
// or the other inputs must have size 1 and be populated
// [Track number: D#3627]
// Convertor for operation L0067_AddBackward1 failed due to runtime error
// TODO Create ticket
TEST_F(KmbSegmentationNetworkTest, DISABLED_road_segmentation_adas_0001) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/road_segmentation_adas_0001/road-segmentation-adas-0001.xml"),
        TestImageDesc("512x896/road-segmentation-adas-0001.png", ImageFormat::BGR),
        0.3f);
}

TEST_F(KmbDetectionNetworkTest, face_detection_adas_0001) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/face-detection-adas-0001/face-detection-adas-0001.xml")
	    .setUserInputPrecision("input", Precision::U8)
	    .setUserInputLayout("input", Layout::NHWC),
        TestImageDesc("300x300/20_Family_Group_Family_Group_20_1003.jpg", ImageFormat::BGR),
        0.3f,
        1.f, 0.3f);
}

// TODO Create ticket
// Disabled for now due to hw incompatible dtype combination
// U8 input and FP16 weights
// Future PR will provide a mitigation and renable this test case
// Issue to track: CVS-39964
TEST_F(HeadPoseEstimationNetworkTest, DISABLED_head_pose_estimation_adas_0001) {
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

TEST_F(PersonAttrRecNetworkTest, precommit_person_attribute_recognitnion_crossroad_0238) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/person-attributes-recognition-crossroad/person-attributes-recognition-crossroad-0238.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("vpu/person-attributes-recognition-crossroad.jpg", ImageFormat::BGR), 0.07f);
}

// C++ exception with description "Tile layer is not supported by kmbPlugin
// Unsupported operation: Tile with name TileIE_1896 with C++ type ngraph::op::TileIE (MR940)
// [Track number: D#3657]
TEST_F(KmbClassifyNetworkTest, DISABLED_license_plate_recognition_barrier_0007) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/license-plate-recognition-barrier-0007/license-plate-recognition-barrier-0007.xml")
	    .setUserInputPrecision("input", Precision::U8),
        TestImageDesc("24x94/000000.bmp", ImageFormat::BGR),
        7,
        0.3f);
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


TEST_F(KmbClassifyNetworkTest, densenet_121_caffe_dense_int8_IRv10) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/densenet-121/caffe/densenet_121_caffe_dense_int8_IRv10-ww42.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("224x224/watch.bmp", ImageFormat::BGR),
        1,
        0.3f);
}


TEST_F(KmbClassifyNetworkTest, densenet_121_tf_dense_int8_IRv10) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/densenet-121/tf/densenet_121_tf_dense_int8_IRv10-ww42.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("224x224/watch.bmp", ImageFormat::BGR),
        1,
        0.3f);
}

TEST_F(KmbClassifyNetworkTest, densenet_169_caffe_dense_int8_IRv10) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/densenet-169/caffe/densenet_169_caffe_dense_int8_IRv10-ww42.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("224x224/rattlesnake.bmp", ImageFormat::BGR),
        1,
        0.3f);
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
// Disabled for now due to hw incompatible dtype combination
// U8 input and FP16 weights
// Future PR will provide a mitigation and renable this test case
// Issue to track: CVS-39964
TEST_F(KmbClassifyNetworkTest, DISABLED_mobilenet_v3_small) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hang on infer");
    runTest(
        TestNetworkDesc("KMB_models/FP16-INT8/private/mobilenet-v3-small-1.0-224/mobilenet-v3-small-1.0-224.xml")
            .setUserInputPrecision("input", Precision::U8),
	TestImageDesc("224x224/husky.bmp", ImageFormat::BGR),
	1,
	0.3f);
}

TEST_F(KmbClassifyNetworkTest, precommit_mobilenet_v1_025_128_U8) {
    runTest(
	TestNetworkDesc("KMB_models/FP16-INT8/public/mobilenet-v1-0.25-128/mobilenet-v1-0.25-128.xml")
	    .setUserInputPrecision("input", Precision::U8),
	TestImageDesc("224x224/cat3.bmp", ImageFormat::BGR),
        1,
        0.3f);
}

// This test checks correctness of handling FP16 input in case of quantized model
// for which inner network precision will be U8
// [Track number: S#48139]
TEST_F(KmbClassifyNetworkTest, precommit_mobilenet_v1_025_128_FP16) {
    SKIP_INFER_BYPASS_ON("VPUX", "bad results");
    runTest(
	TestNetworkDesc("KMB_models/FP16-INT8/public/mobilenet-v1-0.25-128/mobilenet-v1-0.25-128.xml")
	    .setUserInputPrecision("input", Precision::FP16),
	TestImageDesc("224x224/cat3.bmp", ImageFormat::BGR),
        1,
        0.3f);
}

// [Track number: S#48139]
TEST_F(KmbClassifyNetworkTest, precommit_mobilenet_v1_025_128_FP32) {
    SKIP_INFER_BYPASS_ON("VPUX", "bad results");
    runTest(
	TestNetworkDesc("KMB_models/FP16-INT8/public/mobilenet-v1-0.25-128/mobilenet-v1-0.25-128.xml")
	    .setUserInputPrecision("input", Precision::FP32),
	TestImageDesc("224x224/cat3.bmp", ImageFormat::BGR),
        1,
        0.3f);
}

// [Track number: S#48139]
TEST_F(KmbClassifyNetworkTest, precommit_aclnet_des_53_vpu) {
    SKIP_INFER_BYPASS_ON("VPUX", "exception - load graph to device");
    runTest(
    TestNetworkDesc("KMB_models/FP16-INT8/public/aclnet-des-53-vpu/aclnet-des-53-vpu.xml")
        .setUserInputPrecision("input", Precision::FP16),
    TestBinFileDesc("vpu/audio_16k/airplane_3_17-FP16.bin", {1, 1, 1, 16000}, Precision::FP16),
        1,
        0.3f);
}

// Compilation time is about 18 minutes
TEST_F(KmbSSDNetworkTest, ssd_mobilenet_v2_coco) {
    runTest(
        TestNetworkDesc("KMB_models/FP16-INT8/public/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco.xml")
            .setUserInputPrecision("input", Precision::U8),
        TestImageDesc("300x300/dog.bmp", ImageFormat::BGR),
        0.3f,
        0.1f, 0.35f);
}

// [Track number: D#45024]
TEST_F(SmokeNetworkTest, precommit_text_detection_0004_tf_dense_int8_IRv10_from_fp32) {
#ifdef _WIN32
    SKIP() << "SEH exception";
#endif
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/text-detection-0004/tf/FP16-INT8/text-detection-0004-ww48.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP32));
}

// [Track number: D#45024]
TEST_F(SmokeNetworkTest, text_detection_0003_tf_dense_int8_IRv10_from_fp32) {
#ifdef _WIN32
    SKIP() << "SEH exception";
#endif
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/text-detection-0003/tf/FP16-INT8/text-detection-0003-ww48.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP32));
}

// Prevent DDR2DDR DMA Test
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


// Regression on compilation due to latest rebase
TEST_F(KmbVasFDStage1Test, DISABLED_precommit_vasfd_stage1) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hang on infer");
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
            .setUserInputPrecision(inputName, Precision::FP16),
    TestImageDesc("320x240/Alma_Powell_0_0.1133.jpg", ImageFormat::BGR),
    0.35f, 0.1f, 0.3f, layerNames, anchorSizes, windowScales, windowLengths);
}

// MemoryAllocator:ProgrammableOutput - ArgumentError:
// ImplicitOutput_2_conversion:0::Order NCHW - Does not match the order NHWC of
// the tensor ImplicitOutput_2 already allocated in the given buffer
// [Track number: D#47570]
TEST_F(KmbVasFDStage2Test, precommit_vasfd_stage2) {
#ifdef _WIN32
    SKIP() << "Order NCHW - Does not match the order NHWC of the tensor";
#endif
    const std::string inputName = "data";
    const KmbVasFDStage2Test::Candidate candidate = {118.36408299, 50.26568365, 158.98897427, 125.54895544};
    runTest(
        TestNetworkDesc("KMB_models/FP16-INT8/private/face_detection_stage2/vasfd_stage2.xml")
            .setUserInputPrecision(inputName, Precision::U8),
        TestImageDesc("48x48/Alma_Powell_0_0.1133.jpg", ImageFormat::BGR),
        0.5f, 1, 0.3f, candidate);
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
                    .setUserOutputPrecision("output", Precision::U8) //currently FP16 is not supported by runtime
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

TEST_F(KmbClassifyNetworkTest, shufflenet_v2_x1_0_pytorch) {
    runTest(
            TestNetworkDesc("KMB_models/FP16-INT8/public/shufflenet-v2-x1_0-pytorch/shufflenet-v2-x1_0-pytorch.xml")
                .setUserInputPrecision("input", Precision::U8)
                .setUserInputLayout("input", Layout::NCHW)
                .setUserOutputPrecision("output", Precision::FP32)
                .setUserOutputLayout("output", Layout::NC),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
            3, 0.5f);
}

TEST_F(KmbDetectionNetworkTest, peleenet) {
    runTest(
            TestNetworkDesc("KMB_models/FP16-INT8/public/peleenet/peleenet.xml")
                .setUserInputPrecision("input", Precision::U8),
            TestImageDesc("300x300/dog.bmp", ImageFormat::BGR),
            0.3f,
            0.1f, 0.3f);
}

// [Track number: D#45024]
TEST_F(SmokeNetworkTest, text_detection_0004_tf_dense_int8_IRv10_from_fp32) {
#ifdef _WIN32
    SKIP() << "SEH exception";
#endif
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/text-detection-0004/tf/FP16-INT8/text-detection-0004-ww48.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserOutputPrecision("output", Precision::FP32));
}
