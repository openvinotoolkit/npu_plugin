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

TEST_F(KmbClassifyNetworkTest, INT8_Dense_PyTorch_IRv7_ResNet_50) {
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

// KMB : Hangs on infer stage [Track number: D#2245]
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

// KMB : Bad inference results. [Track number: D#2245]
TEST_F(KmbClassifyNetworkTest, INT8_SparseV2_ONNX_IRv7_ResNet_50) {
    SKIP_INFER_ON("KMB", "bad results");

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
            .setUserOutputPresision("output", Precision::FP16),
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

// KMB : Bad inference results. [Track number: D#2473]
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

// KMB : Bad inference results. [Track number: D#2246]
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

// KMB : Bad inference results. [Track number: D#2246]
TEST_F(KmbClassifyNetworkTest, INT8_SparseV2_TF_IRv7_MobileNet_V2) {  // 59.3% sparsity
    SKIP_INFER_ON("KMB", "bad results");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/MoblieNet-v2-tf/mobilenetv2-int8-sparse-v2-tf-0001.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        "224x224/cat3.bmp",
        1, 0.05f);
}

// KMB : Bad inference results. [Track number: D#2246]
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
// KMB : [Track number: S#25483, D#2374]
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

TEST_F(KmbClassifyNetworkTest, INT8_Dense_TF_IRv7_Inception_V1) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v1_tf/inception-v1_tf_uint8_int8_weights_pertensor.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        TestImageDesc("224x224/cat3.bmp", false),
        1, 0.05f);
}

// KMB : Hangs on infer stage [Track number: D#2293]
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
        "299x299/lassy_googlenet_big.bmp",
        3, 1e-1f);
}

// KMB : Power layer is not supported by kmbPlugin
TEST_F(KmbClassifyNetworkTest, INT8_Dense_TF_IRv10_Inception_V3) {
    SKIP_ON("KMB", "compile error");  // TODO: create JIRA ticket

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v3_tf/googlenet-v3-tf-frozen-from-icv-bench-cache.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        "299x299/lassy_googlenet_big.bmp",
        3, 1e-1f);
}

// KMB : Bad inference results. [Track number: D#2474]
TEST_F(KmbClassifyNetworkTest, INT8_Dense_TF_IRv7_Inception_V3) {
    SKIP_INFER_ON("KMB", "bad results");

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v3_tf/inception-v3_tf_uint8_int8_weights_pertensor.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
        TestImageDesc("299x299/lassy_googlenet_big.bmp", false),
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
// KMB : [Track number: S#25483, D#2374]
TEST_F(KmbClassifyNetworkTest, INT8_Dense_Caffe_IRv7_SqueezeNet_1_1) {
    SKIP_INFER_ON("KMB", "bad results");  // TODO: create JIRA ticket

    runTest(
        TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1_caffe/squeezenet1.1.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP16),
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
        1, 1.2f);
}

// KMB : Inference hangs.  [Track number: D#2476]
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

TEST_F(KmbYoloV2NetworkTest, INT8_Dense_TF_DarkNet_TinyYoloV2) {
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
