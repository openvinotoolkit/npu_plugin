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

#include <test_kmb_models_path.h>
#include "test_model/kmb_test_base.hpp"

TEST_F(KmbClassifyNetworkTest, INT8_Dense_PyTorch_IRv7_ResNet_50_Alpha) {
    if (std::getenv("KMB_ALPHA_TESTS_DATA_PATH") == nullptr)
        SKIP() << " KMB_ALPHA_TESTS_DATA_PATH is not set";

    runTest(
        TestNetworkDesc("/alpha/resnet50_uint8_int8_weights_pertensor.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("/test_pictures/224x224/husky.bmp", ImageFormat::RGB),
        1, 0.7f);
}

TEST_F(KmbClassifyNetworkTest, INT8_Dense_PyTorch_IRv7_MobileNet_V2_Alpha) {
    if (std::getenv("KMB_ALPHA_TESTS_DATA_PATH") == nullptr)
        SKIP() << " KMB_ALPHA_TESTS_DATA_PATH is not set";

    runTest(
        TestNetworkDesc("/alpha/mobilenet_v2_uint8_int8_weights_perchannel.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("/test_pictures/224x224/watch.bmp", ImageFormat::RGB),
        // TODO: threshold is 7
        1, 7.05f);
}

TEST_F(KmbClassifyNetworkTest, INT8_Dense_TF_IRv7_Inception_V1_Alpha) {
    if (std::getenv("KMB_ALPHA_TESTS_DATA_PATH") == nullptr)
        SKIP() << " KMB_ALPHA_TESTS_DATA_PATH is not set";

    runTest(
        TestNetworkDesc("/alpha/inception-v1_tf_uint8_int8_weights_pertensor.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("/test_pictures/224x224/cat3.bmp", ImageFormat::RGB),
        1, 0.05f);
}

TEST_F(KmbClassifyNetworkTest, INT8_Dense_TF_IRv7_Inception_V3_Alpha) {
    if (std::getenv("KMB_ALPHA_TESTS_DATA_PATH") == nullptr)
        SKIP() << " KMB_ALPHA_TESTS_DATA_PATH is not set";

    runTest(
        TestNetworkDesc("/alpha/inception-v3_tf_uint8_int8_weights_pertensor.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("/test_pictures/299x299/n01537544_28.bmp", ImageFormat::RGB),
        1, 0.05f);
}

TEST_F(KmbClassifyNetworkTest, INT8_Dense_PyTorch_IRv7_SqueezeNet_1_1_Alpha) {
    if (std::getenv("KMB_ALPHA_TESTS_DATA_PATH") == nullptr)
        SKIP() << " KMB_ALPHA_TESTS_DATA_PATH is not set";

    runTest(
        TestNetworkDesc("/alpha/squeezenet1_1_pytorch_uint8_int8_weights_pertensor.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP16),
        TestImageDesc("/test_pictures/224x224/cat3.bmp", ImageFormat::RGB),
        1, 2.f);
}

//
// TinyYolo V2
//

TEST_F(KmbYoloV2NetworkTest, INT8_Dense_TF_DarkNet_TinyYoloV2_Alpha) {
    if (std::getenv("KMB_ALPHA_TESTS_DATA_PATH") == nullptr)
        SKIP() << " KMB_ALPHA_TESTS_DATA_PATH is not set";

    runTest(
        TestNetworkDesc("/alpha/tiny_yolo_v2_uint8_int8_weights_pertensor.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("/test_pictures/416x416/person.bmp", ImageFormat::RGB),
        0.6, 0.4, 0.4, false);
}

//
// Yolo V2
//

TEST_F(KmbYoloV2NetworkTest, INT8_Dense_TF_DarkNet_YoloV2_Alpha) {
    if (std::getenv("KMB_ALPHA_TESTS_DATA_PATH") == nullptr)
        SKIP() << " KMB_ALPHA_TESTS_DATA_PATH is not set";

    runTest(
        TestNetworkDesc("/alpha/yolo_v2_uint8_int8_weights_pertensor.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32),
            TestImageDesc("/test_pictures/416x416/person.bmp", ImageFormat::RGB),
        0.6, 0.4, 0.4, false);
}
