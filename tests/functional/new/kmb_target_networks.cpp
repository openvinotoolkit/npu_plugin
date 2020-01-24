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

#include "test_model/kmb_tests_base.hpp"

#ifdef ENABLE_MCM_COMPILER

// Fails on IE to mcmCompiler parsing stage with message
// C++ exception with description "quant_model/resnet_v1_50/block1/unit_3/bottleneck_v1/addQuantize Eltwise
// should has FakeQuantize on inputs
TEST_F(KmbNetworkTest, DISABLED_ResNet_50_v1_tf_int8_sparse_v2) {  // 60.4% sparsity
    runClassifyNetworkTest(
        "KMB_models/INT8/public/sparse/ResNet-50-tf/resnetv1-int8-sparse-v2-tf-0001",
        "224x224/cat3.bmp",
        1, 5.0f);
}
// Fails on IE to mcmCompiler parsing stage with message
// C++ exception with description "quant_model/resnet_v1_50/block1/unit_3/bottleneck_v1/addQuantize Eltwise
// should has FakeQuantize on inputs
TEST_F(KmbNetworkTest, DISABLED_ResNet_50_v1_tf_int8_sparse_v1) {  // 28.4% sparsity
    runClassifyNetworkTest(
        "KMB_models/INT8/public/sparse/ResNet-50-tf/resnetv1-int8-sparse-v1-tf-0001",
        "224x224/cat3.bmp",
        1, 5.0f);
}
// Fails on mcmCompiler compilation stage with message
// C++ exception with description "Caught std::runtime_error during unit run:
// Populated tensor with DType Int32 with out of bound value -9223372036854775808
TEST_F(KmbNetworkTest, DISABLED_ResNet_50_v1_onnx_int8_sparse_v2) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/sparse/ResNet-50-onnx/resnet50-int8-sparse-v2",
        "224x224/cat3.bmp",
        1, 5.0f);
}
// Fails on mcmCompiler compilation stage with message
// C++ exception with description "Caught std::runtime_error during unit run:
// Populated tensor with DType Int32 with out of bound value -4315556704
TEST_F(KmbNetworkTest, DISABLED_MobileNet_v2_onnx_int8_sparse_v2) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/sparse/MoblieNet-v2-onnx/mobilenetv2-int8-sparse-v2",
        "224x224/cat3.bmp",
        1, 5.0f);
}
// post training models
// To learn where the post training IRs from and how to update them (if necessary) see
// scripts/post_training_quantization/README.md and
// scripts/post_training_quantization/<corresponding network dir>/run.txt files

// Fails on mcmCompiler compilation stage with message
// C++ exception with description "Caught std::runtime_error during unit run:
// quantParams - ArgumentError: channel 24 - Invalid index: channel is greater than zeroPoint vector
TEST_F(KmbNetworkTest, DISABLED_mobilenet_v2_uint8_int8_weights_perchannel) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/MobileNet_V2/mobilenet_v2_uint8_int8_weights_perchannel",
        "224x224/cat3.bmp",
        1, 5.0f);
}
// post training models
// Test on inception_v1 fails on mcmCompiler compilation stage with message.
// C++ exception with description "Caught std::runtime_error during unit run:
// Populated tensor with DType Int32 with out of bound value -9223372036854775808
TEST_F(KmbNetworkTest, DISABLED_inception_v1_tf_uint8_int8_weights_pertensor) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/inception-v1_tf/inception-v1_tf_uint8_int8_weights_pertensor",
        "224x224/cat3.bmp",
        1, 5.0f);
}
// post training models
// Following test on caffe based squeezenet1_1 fails on IE to mcmCompiler parsing stage
// with message
// C++ exception with description "Op:pool10 - OpError: Invalid input data (0) -
// Filter kernel width (14) exceeds the padded input width (13)
TEST_F(KmbNetworkTest, DISABLED_squeezenet1_1_caffe_benchmark) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/squeezenet1_1_caffe/squeezenet1.1",
        "227x227/cat3.bmp",
        1, 5.0f);
}

TEST_F(KmbNetworkTest, resnet50_uint8_int8_weights_pertensor) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/ResNet-50/resnet50_uint8_int8_weights_pertensor",
        "224x224/cat3.bmp",
        1, 5.0f);
}
TEST_F(KmbNetworkTest, GoogLeNet_v1_tf_int8_sparse) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/sparse/GoogLeNet-v1-tf/inceptionv1-int8-sparse-tf-0001",
        "224x224/cat3.bmp",
        1, 5.0f);
}
TEST_F(KmbNetworkTest, DISABLED_MobileNet_v2_tf_int8_sparse_v2) {  // 59.3% sparsity
    runClassifyNetworkTest(
        "KMB_models/INT8/public/sparse/MoblieNet-v2-tf/mobilenetv2-int8-sparse-v2-tf-0001",
        "224x224/cat3.bmp",
        1, 5.0f);
}
TEST_F(KmbNetworkTest, MobileNet_v2_tf_int8_sparse_v1) {  // 30.8% sparsity
    runClassifyNetworkTest(
        "KMB_models/INT8/public/sparse/MoblieNet-v2-tf/mobilenetv2-int8-sparse-v1-tf-0001",
        "224x224/cat3.bmp",
        1, 5.0f);
}
TEST_F(KmbNetworkTest, squeezenet1_1_pytorch_uint8_int8_weights_pertensor) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/squeezenet1_1_pytorch/squeezenet1_1_pytorch_uint8_int8_weights_pertensor",
        "224x224/cat3.bmp",
        1, 5.0f);
}
TEST_F(KmbNetworkTest, SqueezeNetv1_1_onnx_int8_sparse) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/sparse/squeezenetv1.1-int8-onnx/squeezenetv1.1-int8-sparse-v2",
        "224x224/cat3.bmp",
        1, 5.0f);
}

#endif
