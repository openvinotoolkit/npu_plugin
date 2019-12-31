//
// Copyright 2019 Intel Corporation.
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

#include "kmb_tests_base.hpp"

// Fails on IE to mcmCompiler parsing stage with message
// C++ exception with description "quant_model/resnet_v1_50/block1/unit_3/bottleneck_v1/addQuantize Eltwise
// should has FakeQuantize on inputs
TEST_F(KmbNetworkTest, DISABLED_ResNet_50_v1_tf_int8_sparse_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/sparse/ResNet-50-tf/resnetv1-int8-sparse-v2-tf-0001",
        "224x224/cat3.bmp",
        1, 5.0f);
}
// Fails on mcmCompiler compilation stage with message
// C++ exception with description "Caught std::runtime_error during unit run:
// Populated tensor with DType Int32 with out of bound value -9223372036854775808
TEST_F(KmbNetworkTest, DISABLED_ResNet_50_v1_onnx_int8_sparse_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/sparse/ResNet-50-onnx/resnet50-int8-sparse-v2",
        "224x224/cat3.bmp",
        1, 5.0f);
}
// Fails on mcmCompiler compilation stage with message
// C++ exception with description "Caught std::runtime_error during unit run:
// Populated tensor with DType Int32 with out of bound value -4315556704
TEST_F(KmbNetworkTest, DISABLED_MobileNet_v2_onnx_int8_sparse_new) {
    runClassifyNetworkTest(
        "/KMB_models/INT8/public/sparse/MoblieNet-v2-onnx/mobilenetv2-int8-sparse-v2",
        "224x224/cat3.bmp",
        1, 5.0f);
}
// IRs from DL_benchmarking_models
// Fails on IE to mcmCompiler parsing stage with message
// C++ exception with description "Only single input is supported currently
TEST_F(KmbNetworkTest, DISABLED_FasterRcnnResnet101_tf_fp16_new) {
    runClassifyNetworkTest(
        "KMB_models/FP16/faster_rcnn_resnet101_coco/tf/tf_frozen/FP16/1/dldt/faster_rcnn_resnet101_coco",
        "512x512/dog_croped512.bmp",
        1, 5.0f);
}
// Fails on IE to mcmCompiler parsing stage with message
// C++ exception with description "Unexpected biases precision
TEST_F(KmbNetworkTest, DISABLED_ICNet_caffe_fp16_new) {
    runClassifyNetworkTest(
        "/KMB_models/FP16/icnet/caffe/caffe/FP16/1/dldt/icnet",
        "1024x2048/frankfurt_001016.bmp",
        1, 5.0f);
}
// post training models
// To learn where the post trainig IRs from and how to update them (if necessary) see
// scripts/post_training_quantization/README.md and
// scripts/post_training_quantization/<corresponding network dir>/run.txt files

// Fails on mcmCompiler compilation stage with message
// C++ exception with description "Caught std::runtime_error during unit run:
// quantParams - ArgumentError: channel 24 - Invalid index: channel is greater than zeroPoint vector
TEST_F(KmbNetworkTest, DISABLED_mobilenet_v2_uint8_int8_weights_perchannel_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/MobileNet_V2/mobilenet_v2_uint8_int8_weights_perchannel",
        "224x224/cat3.bmp",
        1, 5.0f);
}
// post training models
// Following test on yolo_v3 fails on IE to mcmCompiler parsing stage with message.
// C++ exception with description "Resample layer is not supported by kmbPlugin
TEST_F(KmbNetworkTest, DISABLED_yolo_v3_uint8_int8_weights_pertensor_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/MobileNet_V2/mobilenet_v2_uint8_int8_weights_perchannel",
        "416x416/person.bmp",
        1, 5.0f);
}
// Test on ssd_mobilenet_v1_coco fails on IE to mcmCompiler parsing stage with message:
// C++ exception with description "Power layer is not supported by kmbPlugin
TEST_F(KmbNetworkTest, DISABLED_ssd_mobilenet_v1_coco_uint8_int8_weights_pertensor_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco_uint8_int8_weights_pertensor",
        "300x300/dog.bmp",
        1, 5.0f);
}
// post trainig models
// Test on inception_v1 fails on mcmCompiler compilation stage with message.
// C++ exception with description "Caught std::runtime_error during unit run:
// Populated tensor with DType Int32 with out of bound value -9223372036854775808
TEST_F(KmbNetworkTest, DISABLED_inception_v1_tf_uint8_int8_weights_pertensor_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/inception-v1_tf/inception-v1_tf_uint8_int8_weights_pertensor",
        "224x224/cat3.bmp",
        1, 5.0f);
}
// post training models
// Test on inception_v3 fails on mcmCompiler compilation stage with message
// C++ exception with description "Caught std::runtime_error during unit run:
// Populated tensor with DType Int32 with out of bound value -9223372036854775808
TEST_F(KmbNetworkTest, DISABLED_inception_v3_tf_uint8_int8_weights_pertensor_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/inception-v3_tf/inception-v3_tf_uint8_int8_weights_pertensor",
        "224x224/cat3.bmp",
        1, 5.0f);
}
// post training models
// Following test on road-segmentation-adas-0001 fails on IE to mcmCompiler parsing stage with following message
// C++ exception with description "OpEntry:Eltwise - IndexError: index 1 -
// Passed input index exceeds inputs count registered for the op type Eltwise
TEST_F(KmbNetworkTest, DISABLED_road_segmentation_adas_0001_uint8_int8_weights_pertensor_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/icv/road-segmentation-adas-0001/"
        "road-segmentation-adas-0001-uint8-int8-weights-pertensor",
        "512x512/dog_croped512.bmp",
        1, 5.0f);
}
// post training models
// Following test on person-vehicle-bike-detection-crossroad-0078 fails on IE to mcmCompiler parsing stage with
// message C++ exception with description "ELU layer is not supported by kmbPlugin
TEST_F(KmbNetworkTest, DISABLED_person_vehicle_bike_detection_crossroad_0078_uint8_int8_weights_pertensor_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/icv/person-vehicle-bike-detection-crossroad-0078/"
        "person-vehicle-bike-detection-crossroad-0078-uint8-int8-weights-pertensor",
        "1024x1024/frankfurt_001016.png",
        1, 5.0f);
}
// post training models
// Test on vehicle-license-plate-detection-barrier-0106 fails on IE to mcmCompiler parsing stage with
// message C++ exception with description "Tensor:SSD/ssd_head/layer_14/output_mbox_loc/Conv2D/Transpose:0
// - ArgumentError: attribute identifer quantParams - Undefined identifier or C++ exception with description
// "DetectionOutput layer is not supported by kmbPlugin
// TODO : use more relevant for network image when it will be added in validation set
// and when inference part of the test will be implemented
TEST_F(KmbNetworkTest, DISABLED_vehicle_license_plate_detection_barrier_0106_uint8_int8_weights_pertensor_new) {
    runClassifyNetworkTest(
            "/KMB_models/INT8/icv/vehicle-license-plate-detection-barrier-0106/"
            "vehicle-license-plate-detection-barrier-0106-uint8_int8_weights_pertensor.xml",
        "1024x1024/frankfurt_001016.png",
        1, 5.0f);
}
// post training models
// Test on face-detection-retail-0004 fails on IE to mcmCompiler parsing stage with message
// C++ exception with description "PriorBoxClustered layer is not supported by kmbPlugin
// TODO : use more relevant for network image when it will be added in validation set
// and when inference part of the test will be implemented
TEST_F(KmbNetworkTest, DISABLED_face_detection_retail_0004_uint8_int8_weights_pertensor_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/icv/face-detection-retail-0004/"
        "face-detection-retail-0004-uint8_int8_weights_pertensor",
        "300x300/dog.bmp",
        1, 5.0f);
}
// Test on ssd512 fail on IE to mcmCompiler parsing stage with message
// C++ exception with description "Unsupported case, we expect only one child"
// Also there are unsupported layers PriorBox and DetectionOutput
TEST_F(KmbNetworkTest, DISABLED_ssd512_caffe_uint8_int8_weights_pertensor_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/ssd512/ssd512_caffe_uint8_int8_weights_pertensor",
        "512x512/dog_croped512.bmp",
        1, 5.0f);
}
// post training models
// Test on caffe based inception_v1 fails on IE to mcmCompiler parsing stage
// C++ exception with description "Op:pool5/7x7_s1 - OpError: Invalid input data (0) -
// Filter kernel width (7) exceeds the padded input width (6)
TEST_F(KmbNetworkTest, DISABLED_inception_v1_caffe_benchmark_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/inception-v1_caffe/googlenet-v1",
        "24x224/cat3.bmp",
        1, 5.0f);
}
// post training models
// Following test on caffe based squeezenet1_1 fails on IE to mcmCompiler parsing stage
// with message
// C++ exception with description "Op:pool10 - OpError: Invalid input data (0) -
// Filter kernel width (14) exceeds the padded input width (13)
TEST_F(KmbNetworkTest, DISABLED_squeezenet1_1_caffe_benchmark_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/squeezenet1_1_caffe/squeezenet1.1",
        "227x227/cat3.bmp",
        1, 5.0f);
}

TEST_F(KmbNetworkTest, resnet50_uint8_int8_weights_pertensor_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/ResNet-50/resnet50_uint8_int8_weights_pertensor",
        "224x224/cat3.bmp",
        1, 5.0f);
}
TEST_F(KmbNetworkTest, GoogLeNet_v1_tf_int8_sparse_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/sparse/GoogLeNet-v1-tf/inceptionv1-int8-sparse-tf-0001",
        "224x224/cat3.bmp",
        1, 5.0f);
}
TEST_F(KmbNetworkTest, MobileNet_v2_tf_int8_sparse_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/sparse/MoblieNet-v2-tf/mobilenetv2-int8-sparse-v2-tf-0001",
        "224x224/cat3.bmp",
        1, 5.0f);
}
TEST_F(KmbNetworkTest, tiny_yolo_v2_uint8_int8_weights_pertensor_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/ava/TinyYolo_V2/tiny_yolo_v2_uint8_int8_weights_pertensor",
        "416x416/person.bmp",
        1, 5.0f);
}
TEST_F(KmbNetworkTest, yolo_v2_uint8_int8_weights_pertensor_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/ava/Yolo_V2/yolo_v2_uint8_int8_weights_pertensor",
        "416x416/person.bmp",
        1, 5.0f);
}
TEST_F(KmbNetworkTest, tiny_yolo_v1_uint8_int8_weights_pertensor_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/YoloTiny-v1-caffe/tiny_yolo_v1_caffe_uint8_int8_weights_per_tensor",
        "448x448/cat3.bmp",
        1, 5.0f);
}
TEST_F(KmbNetworkTest, squeezenet1_1_pytorch_uint8_int8_weights_pertensor_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/squeezenet1_1_pytorch/squeezenet1_1_pytorch_uint8_int8_weights_pertensor",
        "224x224/cat3.bmp",
        1, 5.0f);
}
TEST_F(KmbNetworkTest, SqueezeNetv1_1_onnx_int8_sparse_new) {
    runClassifyNetworkTest(
        "KMB_models/INT8/public/sparse/squeezenetv1.1-int8-onnx/squeezenetv1.1-int8-sparse-v2",
        "224x224/cat3.bmp",
        1, 5.0f);
}
