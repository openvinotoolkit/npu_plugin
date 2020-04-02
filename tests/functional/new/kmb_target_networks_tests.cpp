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

//
// ResNet50
//

// Fails on parsing stage:
// "C++ exception with description "AssertionFailed: weightableLayer->insData.size() > 1"
// or "C++ exception with description "Tensor::0 - ArgumentError: data vector type double - Unable to populate, data type is not doubleDType of tensor is Float16 but populating with double data" thrown in the test body."
//    if isWeightableLayerQuantized insData.size() assertion excluded
// [Track number: S#28568]
TEST_F(KmbClassifyNetworkTest, DISABLED_resnet_50_pytorch_dense_fp16_IRv10) {
    runTest(
        TestNetworkDesc("KMB_models/FP16/resnet_50_pytorch/resnet-50-pytorch.xml")
            .setUserInputPresision("input", Precision::FP16)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        "224x224/cat3.bmp",
        3, 1e-5f);
}

TEST_F(KmbClassifyNetworkTest, INT8_Dense_PyTorch_IRv10_ResNet_50) {
    SKIP_INFER_ON("KMB", "bad results");  // TODO: create JIRA ticket
    SKIP_INFER_ON("CPU", "segfault on infer");  // TODO: create JIRA ticket

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/ResNet-50/resnet-50-pytorch-from-icv-bench-cache.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        "224x224/cat3.bmp",
        3, 1e-5f);
}

// [Track number: S#27965]
TEST_F(KmbClassifyNetworkTest, INT8_Dense_PyTorch_IRv7_ResNet_50) {
    SKIP_INFER_ON("KMB", "bad result");
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/ResNet-50/resnet50_uint8_int8_weights_pertensor.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        TestImageDesc("224x224/cat3.bmp", false),
        1, 0.7f);
}

// KMB : Hangs on infer stage [Track number: D#2245]
TEST_F(KmbClassifyNetworkTest, INT8_SparseV1_TF_IRv7_ResNet_50) {  // 28.4% sparsity
    SKIP_INFER_ON("KMB", "hang on infer");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/ResNet-50-tf/resnetv1-int8-sparse-v1-tf-0001.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        "224x224/cat3.bmp",
        1, 0.05f);
}

// KMB : Hangs on infer stage 
// [Track number: D#2245]
TEST_F(KmbClassifyNetworkTest, INT8_SparseV2_TF_IRv7_ResNet_50) {  // 60.4% sparsity
    SKIP_INFER_ON("KMB", "hang on infer");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/ResNet-50-tf/resnetv1-int8-sparse-v2-tf-0001.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        "224x224/cat3.bmp",
        1, 0.05f);
}

// KMB : Bad inference results.
// Track number: D#2245]
TEST_F(KmbClassifyNetworkTest, INT8_SparseV2_ONNX_IRv7_ResNet_50) {
    SKIP_ON("KMB", "Compiler Error: min > max");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/ResNet-50-onnx/resnet50-int8-sparse-v2.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        "224x224/cat3.bmp",
        1, 0.05f);
}

//
// MobileNetV2
//

TEST_F(KmbClassifyNetworkTest, INT8_Dense_Caffe_IRv10_MobileNet_V2) {
    SKIP_ON("CPU", "compile error");  // TODO: create JIRA ticket
    SKIP_INFER_ON("KMB", "bad results");  // TODO: create JIRA ticket

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/MobileNet_V2/mobilenet-v2-caffe-IRv10.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16)
            .setUserOutputLayout("output", Layout::NHWC),
        "224x224/cat3.bmp",
        3, 1e-5f);
}

// CPU : Supported primitive descriptors list is empty for node: Add1_/Fused_Add_
TEST_F(KmbClassifyNetworkTest, INT8_Dense_PyTorch_IRv10_MobileNet_V2) {
    SKIP_ON("CPU", "compile error");  // TODO: create JIRA ticket
    SKIP_INFER_ON("KMB", "bad results");  // TODO: create JIRA ticket

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/MobileNet_V2/mobilenet-v2-pytorch-from-icv-bench-cache.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        "224x224/cat3.bmp",
        3, 1e-5f);
}

// KMB : Bad inference results. 
// [Track number: D#2473]
TEST_F(KmbClassifyNetworkTest, INT8_Dense_PyTorch_IRv7_MobileNet_V2) {
    SKIP_INFER_ON("KMB", "bad results");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/MobileNet_V2/mobilenet_v2_uint8_int8_weights_perchannel.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        TestImageDesc("224x224/cat3.bmp", false),
        1, 0.05f);
}

// KMB : Bad inference results. 
// [Track number: D#2246]
TEST_F(KmbClassifyNetworkTest, INT8_SparseV1_TF_IRv7_MobileNet_V2) {  // 30.8% sparsity
    SKIP_INFER_ON("KMB", "bad results");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/MoblieNet-v2-tf/mobilenetv2-int8-sparse-v1-tf-0001.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        "224x224/cat3.bmp",
        1, 0.05f);
}

// KMB : Bad inference results. 
// [Track number: D#2246 D#2691]
TEST_F(KmbClassifyNetworkTest, INT8_SparseV2_TF_IRv7_MobileNet_V2) {  // 59.3% sparsity
    SKIP_INFER_ON("KMB", "bad results, mixed up top2 classes");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/MoblieNet-v2-tf/mobilenetv2-int8-sparse-v2-tf-0001.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        "224x224/cat3.bmp",
        1, 0.05f);
}

// KMB : Bad inference results. 
// [Track number: D#2246]
TEST_F(KmbClassifyNetworkTest, INT8_SparseV2_ONNX_IRv7_MobileNet_V2) {
    SKIP_INFER_ON("KMB", "bad results");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/MoblieNet-v2-onnx/mobilenetv2-int8-sparse-v2.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        "224x224/cat3.bmp",
        1, 0.05f);
}

//
// InceptionV1
//

// KMB : Op:pool5/7x7_s1 - OpError: Invalid input data (0) - Filter kernel width (7) exceeds the padded input width (6)
TEST_F(KmbClassifyNetworkTest, INT8_Dense_Caffe_IRv10_Inception_V1) {
    SKIP_ON("KMB", "compile error");  // TODO: create JIRA ticket

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v1_caffe/googlenet-v1-caffe-from-icv-bench-cache.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        "224x224/cat3.bmp",
        3, 1e-2f);
}

// KMB : Test on caffe based inception_v1 fails on IE to mcmCompiler parsing stage
// KMB : C++ exception with description "Op:pool5/7x7_s1 - OpError: Invalid input data (0) -
// KMB : Filter kernel width (7) exceeds the padded input width (6)
// [Track number: S#25483/D#2374]
TEST_F(KmbClassifyNetworkTest, INT8_Dense_Caffe_IRv7_Inception_V1) {
    SKIP_ON("KMB", "compile error");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v1_caffe/googlenet-v1.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        "224x224/cat3.bmp",
        1, 0.05f);
}

// KMB : Bad inference results. 
// [Track number: D#2475]
TEST_F(KmbClassifyNetworkTest, INT8_Dense_TF_IRv7_Inception_V1) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v1_tf/inception-v1_tf_uint8_int8_weights_pertensor.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        TestImageDesc("224x224/cat3.bmp", false),
        1, 0.05f);
}

// KMB : Hangs on infer stage 
// [Track number: D#2293]
TEST_F(KmbClassifyNetworkTest, INT8_Sparse_TF_IRv7_Inception_V1) {
    SKIP_INFER_ON("KMB", "hang on infer");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/GoogLeNet-v1-tf/inceptionv1-int8-sparse-tf-0001.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        "224x224/cat3.bmp",
        1, 0.05f);
}

//
// InceptionV3
//

// KMB : Power layer is not supported by kmbPlugin
TEST_F(KmbClassifyNetworkTest, INT8_Dense_PyTorch_IRv10_Inception_V3) {
    SKIP_ON("KMB", "compile error");  // TODO: create JIRA ticket

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v3_tf/googlenet-v3-pytorch-from-icv-bench-cache.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        "299x299/n01537544_28.bmp",
        1, 1e-1f);
}

// KMB : Power layer is not supported by kmbPlugin
TEST_F(KmbClassifyNetworkTest, INT8_Dense_TF_IRv10_Inception_V3) {
    SKIP_ON("KMB", "compile error");  // TODO: create JIRA ticket

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v3_tf/googlenet-v3-tf-frozen-from-icv-bench-cache.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        "299x299/n01537544_28.bmp",
        1, 1e-1f);
}

// KMB : Bad inference results. 
// [Track number: D#2474]
TEST_F(KmbClassifyNetworkTest, INT8_Dense_TF_IRv7_Inception_V3) {
    SKIP_INFER_ON("KMB", "bad results");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v3_tf/inception-v3_tf_uint8_int8_weights_pertensor.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        TestImageDesc("299x299/n01537544_28.bmp", false),
        1, 0.05f);
}

//
// SqueezeNet 1.1
//

// FIXME: Missing IR in models-ir repository
TEST_F(KmbClassifyNetworkTest, DISABLED_INT8_Dense_Caffe2_IRv10_SqueezeNet_1_1) {
    SKIP_INFER_ON("KMB", "bad results");  // TODO: create JIRA ticket

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1_caffe/squeezenet1.1-caffe2-uint8-int8-weights-perchannel-IRv10.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        "227x227/cat3.bmp",
        3, 1e-1f);
}

// KMB : Following test on caffe based squeezenet1_1 fails on IE to mcmCompiler parsing stage
// KMB : with message
// KMB : C++ exception with description "Op:pool10 - OpError: Invalid input data (0) -
// KMB : Filter kernel width (14) exceeds the padded input width (13)
// [Track number: S#25483/D#2374]
TEST_F(KmbClassifyNetworkTest, INT8_Dense_Caffe_IRv7_SqueezeNet_1_1) {
    SKIP_INFER_ON("KMB", "bad results");  // TODO: create JIRA ticket

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1_caffe/squeezenet1.1.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16)
            .setUserOutputLayout("input", Layout::NHWC),
        TestImageDesc("227x227/cat3.bmp", false),
        1, 0.05f);
}

TEST_F(KmbClassifyNetworkTest, INT8_Dense_PyTorch_IRv7_SqueezeNet_1_1) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1_pytorch/squeezenet1_1_pytorch_uint8_int8_weights_pertensor.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        TestImageDesc("224x224/cat3.bmp", false),
        1, 2.f);
}

// KMB : Inference hangs.
// [Track number: D#2476]
TEST_F(KmbClassifyNetworkTest, INT8_Sparse_ONNX_IRv7_SqueezeNet_1_1) {
    SKIP_INFER_ON("KMB", "hang on infer");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/squeezenetv1.1-int8-onnx/squeezenetv1.1-int8-sparse-v2.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        "224x224/cat3.bmp",
        1, 0.05f);
}

//
// SSD 512
//

// KMB : Unsupported case, we expect only one child
TEST_F(KmbDetectionNetworkTest, INT8_Dense_Caffe_IRv10_SSD_512) {
    SKIP_ON("KMB", "compile error");  // TODO: create JIRA ticket

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/ssd512/ssd512_caffe_uint8_int8_weights_pertensor.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        "512x512/dog_croped512.bmp",
        0.3f,
        0.1f, 0.3f);
}

//
// TinyYolo V2
//

// KMB : Bad inference results.
// [Track number: S#28195]
TEST_F(KmbYoloV2NetworkTest, INT8_Dense_TF_DarkNet_TinyYoloV2) {
    SKIP_INFER_ON("KMB", "bad results");

    runTest(
        TestNetworkDesc("KMB_models/INT8/ava/TinyYolo_V2/tiny_yolo_v2_uint8_int8_weights_pertensor.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
            TestImageDesc("416x416/person.bmp", false),
        0.6, 0.4, 0.4, false);
}

//
// Yolo V2
//

TEST_F(KmbYoloV2NetworkTest, INT8_Dense_TF_DarkNet_YoloV2) {
    SKIP_INFER_ON("KMB", "bad results");  // TODO: create JIRA ticket

    runTest(
        TestNetworkDesc("KMB_models/INT8/ava/Yolo_V2/yolo_v2_uint8_int8_weights_pertensor.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
            TestImageDesc("416x416/person.bmp", false),
        0.6, 0.4, 0.4, false);
}


//////////////////////////////////////////
// Start of test-set for KMB-alpha IRv10
//////////////////////////////////////////

// KMB : Bad inference results.
// [Track number: S#28790]
TEST_F(KmbYoloV2NetworkTest, yolo_tiny_v2_ava_0001_dense_int8_IRv10) {
    SKIP_INFER_ON("KMB", "bad results");

    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/yolo-tiny-v2-ava-0001/yolo_tiny_v2_ava_0001_tf_dense_int8_IRv10.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        TestImageDesc("416x416/person.bmp", false),
        0.6, 0.4, 0.4, false);
}

TEST_F(KmbYoloV2NetworkTest, yolo_v2_ava_0001_dense_int8_IRv10) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/icv/yolo-v2-ava-0001/yolo_v2_ava_0001_tf_dense_int8_IRv10.xml")

            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        TestImageDesc("416x416/person.bmp", false),
        0.6, 0.4, 0.4, false);
}

// KMB : Bad inference results.
// [Track number: S#28791]
TEST_F(KmbClassifyNetworkTest, resnet_50_pytorch_dense_int8_IRv10) {
    SKIP_INFER_ON("KMB", "bad results");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/ResNet-50/resnet_50_pytorch_dense_int8_IRv10.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        "224x224/husky.bmp",
        1, 0.7f);
}

TEST_F(KmbClassifyNetworkTest, mobilenet_v2_caffe2_dense_int8_IRv10) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/MobileNet_V2/mobilenet_v2_pytorch_caffe2_dense_int8_IRv10.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        "224x224/watch.bmp",
        1, 7.0f);
}

// KMB : Bad inference results.
// [Track number: S#28792]
TEST_F(KmbClassifyNetworkTest, googlenet_v1_tf_dense_int8_IRv10) {
    SKIP_INFER_ON("KMB", "bad results");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/googlenet-v1/googlenet_v1_tf_dense_int8_IRv10.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        "224x224/cat3.bmp",
        1, 0.05f);
}

TEST_F(KmbClassifyNetworkTest, googlenet_v3_tf_dense_int8_IRv10) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/googlenet-v3/googlenet_v3_tf_dense_int8_IRv10.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        "299x299/n01537544_28.bmp",
        1, 0.05f);
}

// KMB : Bad inference results.
// [Track number: S#28793]
TEST_F(KmbClassifyNetworkTest, squeezenet_v1_1_caffe2_dense_int8_IRv10) {
    SKIP_INFER_ON("KMB", "bad results");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1/squeezenet1_1_pytorch_caffe2_dense_int8_IRv10.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32)
            .setUserOutputLayout("output", Layout::NHWC),
        "227x227/cat3.bmp",
        1, 2.0f);
}
//////////////////////////////////////////
// End of test-set for KMB-alpha IRv10
//////////////////////////////////////////


//////////////////////////////////////////
// Start of test-set for KMB-beta IRv10
//////////////////////////////////////////

// C++ exception with description "Unsupported case, we expect only one child
// kmb-plugin/src/frontend_mcm/src/quantization_helpers.cpp:170
// [Track number: D#2722]
TEST_F(KmbDetectionNetworkTest, ssd512_caffe_dense_int8_IRv10) {
    SKIP_ON("KMB", "compile error");

    runTest(
            TestNetworkDesc("KMB_models/INT8/public/ssd512/ssd512_caffe_dense_int8_IRv10.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP16),
            "512x512/dog_croped512.bmp",
            0.3f,
            0.1f, 0.3f);
}

// C++ exception with description "Only single input is supported currently
// kmb-plugin/src/frontend_mcm/src/frontend_mcm.cpp:824
// [Track number: D#2723]
TEST_F(KmbDetectionNetworkTest, faster_rcnn_resnet101_coco_tf_dense_int8_IRv10) {
    SKIP_ON("KMB", "compile error");

    runTest(
            TestNetworkDesc("KMB_models/INT8/public/faster_rcnn_resnet101_coco/faster_rcnn_resnet101_coco_tf_dense_int8_IRv10.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP16),
            "1024x600/frankfurt_001016.jpg",
            0.3f,
            0.1f, 0.3f);
}

// KMB : Bad inference results.
// [Track number: S#28624]
TEST_F(KmbClassifyNetworkTest, googlenet_v4_tf_dense_int8_IRv10) {
    SKIP_INFER_ON("KMB", "bad results");

    runTest(
            TestNetworkDesc("KMB_models/INT8/public/googlenet-v4/googlenet_v4_tf_dense_int8_IRv10.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP32),
            "299x299/sofa.bmp",
            1, 0.05f);
}

// C++ exception with description "Unsupported case, we expect only one child
// kmb-plugin/src/frontend_mcm/src/quantization_helpers.cpp:170
// [Track number: D#2724]
TEST_F(KmbDetectionNetworkTest, ssd_mobilenet_v1_coco_tf_dense_int8_IRv10) {
    SKIP_ON("KMB", "compile error");

    runTest(
            TestNetworkDesc("KMB_models/INT8/public/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco_tf_dense_int8_IRv10.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP16),
            "300x300/dog.bmp",
            0.3f,
            0.1f, 0.3f);
}

// C++ exception with description "Op:Add1_36140/Fused_Add_ - OpError: Invalid input weights (1)
// - Invalid shape of scales tensor (input 1) - the dimension has to equal to the last dimension
// of the input tensor which is 1
// Exception is raised in kmb-plugin/thirdparty/movidius/mcmCompiler/src/computation/op/def/scale.cpp:29
// [Track number: S#28499]
TEST_F(KmbClassifyNetworkTest, facenet_20180408_102900_tf_dense_int8_IRv10) {
    SKIP_ON("KMB", "compile error");

    runTest(
            TestNetworkDesc("KMB_models/INT8/public/facenet-20180408-102900/facenet_20180408_102900_tf_dense_int8_IRv10.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP32),
            "160x160/cat3.bmp",
            1, 0.05f);
}

// C++ exception with description "ELU layer is not supported by kmbPlugin
// kmb-plugin/src/frontend_mcm/src/frontend_mcm.cpp:1787
// [Track number: D#2725]
TEST_F(KmbDetectionNetworkTest, person_vehicle_bike_detection_crossroad_0078_caffe_dense_int8_IRv10) {
    SKIP_ON("KMB", "compile error");

    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/person-vehicle-bike-detection-crossroad-0078/person_vehicle_bike_detection_crossroad_0078_caffe_dense_int8_IRv10.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP16),
            "1024x1024/frankfurt_001016.png",
            0.3f,
            0.1f, 0.3f);
}

// C++ exception with description "DetectionOutput layer is not supported by kmbPlugin
// kmb-plugin/src/frontend_mcm/src/frontend_mcm.cpp:1744
// [Track number: D#2726]
TEST_F(KmbDetectionNetworkTest, vehicle_license_plate_detection_barrier_0106_tf_dense_int8_IRv10) {
    SKIP_ON("KMB", "compile error");

    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/vehicle-license-plate-detection-barrier-0106/vehicle_license_plate_detection_barrier_0106_tf_dense_int8_IRv10.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP16),
            "300x300/dog.bmp",
            0.3f,
            0.1f, 0.3f);
}

// C++ exception with description "PriorBoxClustered layer is not supported by kmbPlugin
// kmb-plugin/src/frontend_mcm/src/frontend_mcm.cpp:1877
// [Track number: D#2727]
TEST_F(KmbDetectionNetworkTest, face_detection_retail_0004_caffe_dense_int8_IRv10) {
    SKIP_ON("KMB", "compile error");

    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/face-detection-retail-0004/face_detection_retail_0004_caffe_dense_int8_IRv10.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP16),
            "300x300/dog.bmp",
            0.3f,
            0.1f, 0.3f);
}

// KMB : Bad inference results.
// [Track number: S#28623
TEST_F(KmbClassifyNetworkTest, resnet_101_caffe_dense_int8_IRv10) {
    SKIP_INFER_ON("KMB", "bad results");

    runTest(
            TestNetworkDesc("KMB_models/INT8/public/resnet-101/resnet_101_caffe_dense_int8_IRv10.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP32),
            "224x224/cat3.bmp",
            1, 0.05f);
}

// KMB : Bad inference results.
// [Track number: S#28622]
TEST_F(KmbClassifyNetworkTest, resnet_152_caffe_dense_int8_IRv10) {
    SKIP_INFER_ON("KMB", "bad results");

    runTest(
            TestNetworkDesc("KMB_models/INT8/public/resnet-152/resnet_152_caffe_dense_int8_IRv10.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP32),
            "224x224/cat3.bmp",
            1, 0.05f);
}

// C++ exception with description "Caught exception during unit run: Tensor:norm11194:0 -
// ArgumentError: attribute identifer splitStrategy - Undefined identifier
// kmb-plugin/src/frontend_mcm/src/mcm_adapter.cpp:157
// [Track number: D#2728]
TEST_F(KmbClassifyNetworkTest, alexnet_caffe_dense_int8_IRv10) {
    SKIP_ON("KMB", "compile error");

    runTest(
            TestNetworkDesc("KMB_models/INT8/public/alexnet/alexnet_caffe_dense_int8_IRv10.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP32),
            "227x227/cat3.bmp",
            1, 0.05f);
}

// Compilation time is very long - more than 40 minutes.
// [Track number: S#28620]
TEST_F(KmbClassifyNetworkTest, vgg16_caffe_dense_int8_IRv10) {
    SKIP_ON("KMB", "very long compile time");

    runTest(
            TestNetworkDesc("KMB_models/INT8/public/vgg16/vgg16_caffe_dense_int8_IRv10.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP32),
            "224x224/cat3.bmp",
            1, 0.05f);
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
TEST_F(KmbYoloV2NetworkTest, yolo_tiny_v2_ava_0001_tf_dense_int8_IRv10_fp16_to_int8) {
    SKIP_ON("KMB", "compile error");

    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/yolo-tiny-v2-ava-0001/yolo_tiny_v2_ava_0001_tf_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP32),
            TestImageDesc("416x416/person.bmp", false),
            0.6, 0.4, 0.4, false);
}

TEST_F(KmbYoloV2NetworkTest, yolo_v2_ava_0001_tf_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/icv/yolo-v2-ava-0001/yolo_v2_ava_0001_tf_dense_int8_IRv10_fp16_to_int8.xml")

                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP32),
            TestImageDesc("416x416/person.bmp", false),
            0.6, 0.4, 0.4, false);
}

TEST_F(KmbClassifyNetworkTest, resnet_50_pytorch_dense_int8_IRv10_fp16_to_int8) {
    SKIP_INFER_ON("KMB", "bad results");
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/ResNet-50/resnet_50_pytorch_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP32),
            "224x224/husky.bmp",
            1, 0.7f);
}

TEST_F(KmbClassifyNetworkTest, mobilenet_v2_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/MobileNet_V2/mobilenet_v2_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP32),
            "224x224/watch.bmp",
            1, 7.0f);
}

TEST_F(KmbClassifyNetworkTest, googlenet_v1_tf_dense_int8_IRv10_fp16_to_int8) {
    SKIP_INFER_ON("KMB", "bad results");
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/googlenet-v1/googlenet_v1_tf_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP32),
            "224x224/cat3.bmp",
            1, 0.05f);
}

TEST_F(KmbClassifyNetworkTest, googlenet_v3_tf_dense_int8_IRv10_fp16_to_int8) {
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/googlenet-v3/googlenet_v3_tf_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP32),
            "299x299/n01537544_28.bmp",
            1, 0.05f);
}

TEST_F(KmbClassifyNetworkTest, squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8) {
    SKIP_INFER_ON("KMB", "bad results");
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1/squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP32)
                    .setUserOutputLayout("output", Layout::NHWC),
            "227x227/cat3.bmp",
            1, 2.0f);
}

TEST_F(KmbClassifyNetworkTest, googlenet_v4_tf_dense_int8_IRv10_fp16_to_int8) {
    SKIP_INFER_ON("KMB", "bad results");
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/googlenet-v4/googlenet_v4_tf_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP32),
            "299x299/sofa.bmp",
            1, 0.05f);
}

TEST_F(KmbClassifyNetworkTest, resnet_101_caffe_dense_int8_IRv10_fp16_to_int8) {
    SKIP_INFER_ON("KMB", "bad results");
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/resnet-101/resnet_101_caffe_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP32),
            "224x224/cat3.bmp",
            1, 0.05f);
}

TEST_F(KmbClassifyNetworkTest, resnet_152_caffe_dense_int8_IRv10_fp16_to_int8) {
    SKIP_INFER_ON("KMB", "bad results");
    runTest(
            TestNetworkDesc("KMB_models/INT8/public/resnet-152/resnet_152_caffe_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP32),
            "224x224/cat3.bmp",
            1, 0.05f);
}

// Compilation time is very long - more than 40 minutes.
// The same situation as for vgg16_caffe_dense_int8_IRv10
// [Track number: S#28620]
TEST_F(KmbClassifyNetworkTest, vgg16_caffe_dense_int8_IRv10_fp16_to_int8) {
    SKIP_ON("KMB", "very long compile time");

    runTest(
            TestNetworkDesc("KMB_models/INT8/public/vgg16/vgg16_caffe_dense_int8_IRv10_fp16_to_int8.xml")
                    .setUserInputPresision("input", Precision::U8)
                    .setUserInputLayout("input", Layout::NHWC)
                    .setUserOutputPresision("output", Precision::FP32),
            "224x224/cat3.bmp",
            1, 0.05f);
}
////////////////////////////////////////////////////////////
// End of test-set for IRv10 FP16 to INT8 quantization
////////////////////////////////////////////////////////////
