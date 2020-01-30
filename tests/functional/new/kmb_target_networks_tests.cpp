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

TEST_F(KmbClassifyNetworkTest, INT8_Dense_PyTorch_ResNet_50) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/ResNet-50/resnet-50-pytorch-uint8-int8-weights-perchannel-IRv10.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NCHW)
            .setUserOutputPresision("output", Precision::FP32),
        "224x224/cat3.bmp",
        3, 1e-5f);
}

// CPU : Supported primitive descriptors list is empty for node: Add1_/Fused_Add_
TEST_F(KmbClassifyNetworkTest, INT8_Dense_PyTorch_MobileNet_V2) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/MobileNet_V2/mobilenet-v2-pytorch-uint8-int8-weights-perchannel-IRv10.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NCHW)
            .setUserOutputPresision("output", Precision::FP32),
        "224x224/cat3.bmp",
        3, 1e-5f);
}

// KMB : Op:pool5/7x7_s1 - OpError: Invalid input data (0) - Filter kernel width (7) exceeds the padded input width (6)
TEST_F(KmbClassifyNetworkTest, DISABLED_INT8_Dense_Caffe_Inception_V1) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v1_caffe/googlenet-v1-uint8-int8-weights-perchannel-IRv10.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NCHW)
            .setUserOutputPresision("output", Precision::FP32),
        "224x224/cat3.bmp",
        3, 1e-2f);
}

// KMB : Power layer is not supported by kmbPlugin
TEST_F(KmbClassifyNetworkTest, DISABLED_INT8_Dense_PyTorch_Inception_V3) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v3_tf/googlenet-v3-pytorch-uint8-int8-weights-perchannel-IRv10.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NCHW)
            .setUserOutputPresision("output", Precision::FP32),
        "299x299/lassy_googlenet_big.bmp",
        3, 1e-1f);
}

TEST_F(KmbClassifyNetworkTest, INT8_Dense_Caffe_SqueezeNet_1_1) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1_caffe/squeezenet1.1-caffe2-uint8-int8-weights-perchannel-IRv10.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NCHW)
            .setUserOutputPresision("output", Precision::FP32),
        "227x227/cat3.bmp",
        3, 1e-1f);
}

// KMB : Unsupported case, we expect only one child
TEST_F(KmbDetectionNetworkTest, DISABLED_INT8_Dense_Caffe_SSD_512) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/ssd512/ssd512_caffe_uint8_int8_weights_pertensor.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NCHW)
            .setUserOutputPresision("output", Precision::FP32),
        "512x512/dog_croped512.bmp",
        0.3f,
        0.1f, 0.3f);
}

// TODO: use correct network type
TEST_F(KmbClassifyNetworkTest, INT8_Dense_TF_TinyYolo_V2) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/ava/TinyYolo_V2/tiny_yolo_v2_uint8_int8_weights_pertensor.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NCHW)
            .setUserOutputPresision("output", Precision::FP32),
        "416x416/person.bmp",
        3, 1e-5f);
}

// TODO: use correct network type
// KMB : too long compilation time
TEST_F(KmbClassifyNetworkTest, DISABLED_INT8_Dense_TF_Yolo_V2) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/ava/Yolo_V2/yolo_v2_uint8_int8_weights_pertensor.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NCHW)
            .setUserOutputPresision("output", Precision::FP32),
        "416x416/person.bmp",
        3, 1e-5f);
}

// Hangs on infer stage [Track number: D#2245]
TEST_F(KmbClassifyNetworkTest, DISABLED_ResNet_50_v1_tf_int8_sparse_v2) {  // 60.4% sparsity
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/ResNet-50-tf/resnetv1-int8-sparse-v2-tf-0001.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        "224x224/cat3.bmp",
        1, 0.05f);
}

// Hangs on infer stage [Track number: D#2245]
TEST_F(KmbClassifyNetworkTest, DISABLED_ResNet_50_v1_tf_int8_sparse_v1) {  // 28.4% sparsity
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/ResNet-50-tf/resnetv1-int8-sparse-v1-tf-0001.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        "224x224/cat3.bmp",
        1, 0.05f);
}

// Bad inference results. [Track number: D#2245]
TEST_F(KmbClassifyNetworkTest, ResNet_50_v1_onnx_int8_sparse_v2) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/ResNet-50-onnx/resnet50-int8-sparse-v2.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        "224x224/cat3.bmp",
        1, 0.05f);
}

// Bad inference results. [Track number: D#2246]
TEST_F(KmbClassifyNetworkTest, MobileNet_v2_onnx_int8_sparse_v2) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/MoblieNet-v2-onnx/mobilenetv2-int8-sparse-v2.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        "224x224/cat3.bmp",
        1, 0.05f);
}

// Bad inference results. [Track number: D#2473]
TEST_F(KmbClassifyNetworkTest, mobilenet_v2_uint8_int8_weights_perchannel) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/MobileNet_V2/mobilenet_v2_uint8_int8_weights_perchannel.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        "224x224/cat3.bmp",
        1, 0.05f);
}
// Bad inference results. [Track number: D#2474]
TEST_F(KmbClassifyNetworkTest, inception_v3_tf_uint8_int8_weights_pertensor) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v3_tf/inception-v3_tf_uint8_int8_weights_pertensor.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        "299x299/lassy_googlenet_big.bmp",
        1, 0.05f);
}

// Bad inference results. [Track number: D#2475]
TEST_F(KmbClassifyNetworkTest, inception_v1_tf_uint8_int8_weights_pertensor) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v1_tf/inception-v1_tf_uint8_int8_weights_pertensor.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        "224x224/cat3.bmp",
        1, 0.05f);
}
// Test on caffe based inception_v1 fails on IE to mcmCompiler parsing stage
// C++ exception with description "Op:pool5/7x7_s1 - OpError: Invalid input data (0) -
// Filter kernel width (7) exceeds the padded input width (6)
// [Track number: S#25483, D#2374]
TEST_F(KmbClassifyNetworkTest, DISABLED_inception_v1_caffe_benchmark) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/inception-v1_caffe/googlenet-v1.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        "224x224/cat3.bmp",
        1, 0.05f);
}
// Following test on caffe based squeezenet1_1 fails on IE to mcmCompiler parsing stage
// with message
// C++ exception with description "Op:pool10 - OpError: Invalid input data (0) -
// Filter kernel width (14) exceeds the padded input width (13)
// [Track number: S#25483, D#2374]
TEST_F(KmbClassifyNetworkTest, DISABLED_squeezenet1_1_caffe_benchmark) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1_caffe/squeezenet1.1.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        "227x227/cat3.bmp",
        1, 0.05f);
}
TEST_F(KmbClassifyNetworkTest, resnet50_uint8_int8_weights_pertensor) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/ResNet-50/resnet50_uint8_int8_weights_pertensor.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        "224x224/cat3.bmp",
        1, 0.7f);
}
// Hangs on infer stage [Track number: D#2293]
TEST_F(KmbClassifyNetworkTest, DISABLED_GoogLeNet_v1_tf_int8_sparse) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/GoogLeNet-v1-tf/inceptionv1-int8-sparse-tf-0001.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        "224x224/cat3.bmp",
        1, 0.05f);
}
// Bad inference results. [Track number: D#2246] 
TEST_F(KmbClassifyNetworkTest, DISABLED_MobileNet_v2_tf_int8_sparse_v2) {  // 59.3% sparsity
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/MoblieNet-v2-tf/mobilenetv2-int8-sparse-v2-tf-0001.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        "224x224/cat3.bmp",
        1, 0.05f);
}
// Bad inference results. [Track number: D#2246] 
TEST_F(KmbClassifyNetworkTest, MobileNet_v2_tf_int8_sparse_v1) {  // 30.8% sparsity
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/MoblieNet-v2-tf/mobilenetv2-int8-sparse-v1-tf-0001.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        "224x224/cat3.bmp",
        1, 0.05f);
}
// Inference hangs.  [Track number: D#2476]
TEST_F(KmbClassifyNetworkTest, DISABLED_squeezenet1_1_pytorch_uint8_int8_weights_pertensor) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1_pytorch/squeezenet1_1_pytorch_uint8_int8_weights_pertensor.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        "224x224/cat3.bmp",
        1, 0.05f);
}
// Inference hangs.  [Track number: D#2476]
TEST_F(KmbClassifyNetworkTest, DISABLED_SqueezeNetv1_1_onnx_int8_sparse) {
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/sparse/squeezenetv1.1-int8-onnx/squeezenetv1.1-int8-sparse-v2.xml")
            .setUserInputPresision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPresision("output", Precision::FP32),
        "224x224/cat3.bmp",
        1, 0.05f);
}
