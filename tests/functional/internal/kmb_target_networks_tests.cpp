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
TEST_F(KmbClassifyNetworkTest, precommit_resnet_50_pytorch_dense_fp16_IRv10) {
    runTest(
        TestNetworkDesc("KMB_models/FP16/resnet_50_pytorch/resnet-50-pytorch.xml")
            .setUserInputLayout("input", Layout::NHWC)
            .setUserInputPrecision("input", Precision::FP16)
            .setUserOutputPrecision("output", Precision::FP16),
        "224x224/cat3.bmp",
        3, 0.05);
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

TEST_F(KmbClassifyNetworkTest, INT8_Dense_Caffe2_IRv10_ResNet_50_v1) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/private/ResNet-50/resnet50_v1_caffe2_dense_int8_IRv10.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("224x224/watch.bmp", ImageFormat::BGR),
        1, 2.5f);
}

// [Track number: S#40780]
TEST_F(KmbClassifyNetworkTest, INT8_Dense_Caffe2_IRv10_ResNet_50_v2) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "bad results");
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

// KMB : Op:pool5/7x7_s1 - OpError: Invalid input data (0) - Filter kernel width (7) exceeds the padded input width (6)
// Unsupported operation: pool1/norm16960 with name LRN_IE_3845 with C++ type ngraph::op::LRN_IE
// [Track number: D#40918]
TEST_F(KmbClassifyNetworkTest, DISABLED_INT8_Dense_Caffe_IRv10_Inception_V1) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v1_caffe/googlenet-v1-caffe-from-icv-bench-cache.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
        3, 1e-2f);
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

TEST_P(KmbYoloV3NetworkTestWithSpecificLayout, INT8_Dense_TF_YoloV3) {
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

TEST_F(KmbYoloV2NetworkTest, precommit_yolo_tiny_v2_ava_0001_tf_dense_int8_IRv10_legacy_parser) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/yolo-tiny-v2-ava-0001/yolo_tiny_v2_ava_0001_tf_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32)
            .setCompileConfig({{"VPU_COMPILER_USE_NGRAPH_PARSER", CONFIG_VALUE(NO)}}),
        TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
        0.6, 0.4, 0.4, false);
}

TEST_F(KmbYoloV2NetworkTest, precommit_yolo_tiny_v2_ava_0001_tf_dense_int8_IRv10_from_fp32_custom) {
    const auto customLayers = std::make_pair(VPU_COMPILER_CONFIG_KEY(CUSTOM_LAYERS),
        getIELibraryPath() + "/kmb_custom_kernels/yolov2.xml");
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/yolo-tiny-v2-ava-0001/yolo_tiny_v2_ava_0001_tf_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32)
            .setCompileConfig({customLayers}),
        TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
        0.6, 0.4, 0.4, false);
}

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

TEST_F(KmbYoloV2NetworkTest, precommit_yolo_v2_ava_0001_tf_dense_int8_IRv10_from_fp32) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/yolo-v2-ava-0001/yolo_v2_ava_0001_tf_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
        TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
        0.6, 0.4, 0.4, false);
}

TEST_F(KmbYoloV2NetworkTest, precommit_yolo_v2_ava_0001_tf_dense_int8_IRv10_from_fp32_custom) {
    const auto customLayers = std::make_pair(VPU_COMPILER_CONFIG_KEY(CUSTOM_LAYERS),
        getIELibraryPath() + "/kmb_custom_kernels/yolov2.xml");
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/yolo-v2-ava-0001/yolo_v2_ava_0001_tf_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32)
            .setCompileConfig({customLayers}),
        TestImageDesc("416x416/person.bmp", ImageFormat::RGB),
        0.6, 0.4, 0.4, false);
}

TEST_F(KmbYoloV2NetworkTest, yolo_v2_ava_0001_tf_dense_int8_IRv10_legacy_parser) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/yolo-v2-ava-0001/yolo_v2_ava_0001_tf_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32)
            .setCompileConfig({{"VPU_COMPILER_USE_NGRAPH_PARSER", CONFIG_VALUE(NO)}}),
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

TEST_F(KmbClassifyNetworkTest, precommit_googlenet_v3_tf_dense_int8_IRv10_legacy_parser) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/googlenet-v3/googlenet_v3_tf_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32)
            .setCompileConfig({{"VPU_COMPILER_USE_NGRAPH_PARSER", CONFIG_VALUE(NO)}}),
        TestImageDesc("299x299/n01537544_28.bmp", ImageFormat::RGB),
        1, 0.1f);
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

INSTANTIATE_TEST_CASE_P(precommit, KmbDetectionNetworkTestWithSpecificLayout, ::testing::ValuesIn({Layout::NHWC}));
// [Track number: S#41097]
// Wrong strategy generated: tensor fire3/squeeze1x1:0_crop:0_align:0 needs sparsity but it can't be sparsified
// INSTANTIATE_TEST_CASE_P(precommit, KmbDetectionNetworkTestWithSpecificLayout, ::testing::ValuesIn(inputLayout));

// TODO Check [Track number: D#3467]
// TODO 4 similar tests face_detection_retail_caffe_IRV10_fp16_int8_*
TEST_F(KmbDetectionNetworkTest, face_detection_retail_caffe_IRV10_fp16_int8_nchw_fuse_scale_input_accuracy_drop) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "Bad results");
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/face-detection-retail-0004/caffe/FP16-INT8/face-detection-retail-0004-ww22.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC),
            TestImageDesc("300x300/0_Parade_marchingband_1_1004.jpg", ImageFormat::RGB),
            0.3f,
            1.f, 0.3f);
}

// [Track number: S#41097]
// [Track number: S#40935]
TEST_F(KmbDetectionNetworkTest, face_detection_retail_caffe_IRV10_fp16_int8_nhwc_fuse_scale_input_accuracy_drop) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "Accuracy regression");
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/face-detection-retail-0004/caffe/FP16-INT8/face-detection-retail-0004-ww22.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            // [Track number: D#3634]
            .setCompileConfig({{"VPU_COMPILER_SCALE_FUSE_INPUT", CONFIG_VALUE(NO)}}),
            TestImageDesc("300x300/0_Parade_marchingband_1_1004.jpg", ImageFormat::RGB),
            0.3f,
            1.f, 0.3f);
}

// Disabled due to accuracy regression with nGraph frontend
// [Track number: S#40575]
TEST_F(KmbSSDNetworkTest, precommit_ssd512_caffe_dense_int8_IRv10_from_fp32) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX","ngraph_parser_enabled bad infer results");
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

// C++ exception with description "Caught exception during unit run:
// propagateParameters ERROR: inputs of the Eltwise/Concat do not have the same QuantParams
// [Track number: D#3675]
// Unsupported operation: bottleneck1_1/dim_red/fn with name Elu_4549 with C++ type ngraph::op::v0::Elu (MR1176)
// Caught exception during unit run: propagateParameters ERROR: inputs of the Eltwise/Concat do not have the same QuantParams
// TODO Check after 1176 and 1377
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
            1, 0.05f);
}

// C++ exception with description "Op:conv2 - OpError: Invalid input weights (1) -
// Does not match the channel dimension of input 96
// [Track number: D#2799]
// Unsupported operation: norm11198 with name LRN_IE_1388 with C++ type ngraph::op::LRN_IE
// [Track number: D#40918]
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
// [Track number: D#3640]
TEST_F(KmbClassifyNetworkTest, vgg16_caffe_dense_int8_IRv10_from_fp32) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hang on infer");  // TODO Create Ticket
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/vgg16/vgg16_caffe_dense_int8_IRv10_from_fp32.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
            1, 0.05f);
}


// Compilation time is about 10 minutes
// [Track number: D#3640]
TEST_F(KmbClassifyNetworkTest, vgg16_caffe_dense_int8_IRv10_fp16_to_int8) {
    // [Track number: S#39223]
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hang on infer");
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/vgg16/vgg16_caffe_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB),
            1, 0.05f);
}

// [Track number: S#41097]
// segfault on compile time
TEST_F(KmbRetinaFaceNetworkTest, DISABLED_precommit_retinaface_mobilenetv2_0_25_modified) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/private/retinaface-mobilenetv2-0.25-modified/retinaface-mobilenetv2-0.25-modified.xml")
                    .setUserInputPrecision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setCompileConfig({{"VPU_COMPILER_COMPILATION_DESCRIPTOR", "release_kmb_retinaface"}}),
            "data",
            TestImageDesc("224x224/cat3.bmp", ImageFormat::RGB));
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
            1, 0.5f);
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

// Compilation fails with exception:
// "Caught exception during unit run: propagateParameters ERROR:
// inputs of the Eltwise/Concat do not have the same QuantParams"
// [Track number: D#3453]
// TODO CHECK [Track number: S#3331]
// C++ exception with description "Proposal layer doesn't support framework: " thrown in the test body.
// TOO Create ticket
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

// Bad accuracy
// [Track number: S#39421]
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

// [Track number: S#25636]
// Segmentation fault in MCM replacement_passes.cpp:303
// [Track number: S#39621]
// [Track number: D#40921]
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
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/OpenPose/FP16-INT8/openpose-pose_cf_ww22.xml")
            .setUserInputPrecision("image", Precision::U8)
            .setUserInputLayout("image", GetParam())
            .setUserOutputPrecision("output", Precision::FP32));
}
INSTANTIATE_TEST_CASE_P(precommit, SmokeNetworkTestWithSpecificLayout, ::testing::ValuesIn(inputLayout));

// Disabled due to unstable CPU results
// [Track number: S#40572]
TEST_F(AgeGenderNetworkTest, precommit_age_gender_retail_0013) {
    const std::string input_name = "input";
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "CPU ref generate different results time to time");
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/age-gender-recognition-retail-0013/caffe/FP16-INT8/age-gender-recognition-retail-0013_ww22.xml")
            .setUserInputPrecision(input_name, Precision::U8),
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

// C++ exception with description "Caught exception during unit run: MemoryAllocator:VPU_CMX_NN - ArgumentError:
// conv4_3_0_norm_mbox_locNeutral_copy0conv4_3_0_norm_mbox_locNeutral_copyDMAconv5_5/sep/bn/variance/Fused_Add_:0:0:0::paddedShape[2]
// 192 - Does not match the dimension 184 of the tensor conv4_3_0_norm_mbox_locNeutral:0 already allocated in the given buffer
// TODO Check ticket
// [Track number: D#3656]
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
TEST_F(HeadPoseEstimationNetworkTest, head_pose_estimation_adas_0001) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hang on infer");
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/head_pose_estimation_adas_0001/head-pose-estimation-adas-0001.xml")
	    .setUserInputPrecision("input", Precision::U8),
        TestImageDesc("60x60/head-pose-estimation-adas-0001.png", ImageFormat::BGR),
        0.1f);
}

// TODO: Need to fix bad check in gather layer parser in runtime
TEST_F(PersonAttrRecNetworkTest, person_attribute_recognitnion_crossroad_0234) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hang on infer");
    const std::string input_name = "input";
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/person-attributes-recognition-crossroad/person-attributes-recognition-crossroad-0234.xml")
                .setUserInputPrecision("input", Precision::U8)
                .setUserInputLayout("input", Layout::NHWC)
                .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("vpu/person-attributes-recognition-crossroad.jpg", ImageFormat::BGR), 0.2f);
}

// TODO: Need to fix bad check in gather layer parser in runtime
TEST_F(PersonAttrRecNetworkTest, person_attribute_recognitnion_crossroad_0238) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "hang on infer");
    const std::string input_name = "input";
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/person-attributes-recognition-crossroad/person-attributes-recognition-crossroad-0238.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("vpu/person-attributes-recognition-crossroad.jpg", ImageFormat::BGR), 0.35f);
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

// C++ exception with description "Caught exception during unit run:
// propagateParameters ERROR: inputs of the Eltwise/Concat do not have the same QuantParams
// [Track number: S#40387]
TEST_F(KmbClassifyNetworkTest, DISABLED_densenet_121) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/densenet-121/densenet-121.xml")
            .setUserInputPrecision("input", Precision::U8),
        TestImageDesc("224x224/cat3.bmp", ImageFormat::BGR),
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
