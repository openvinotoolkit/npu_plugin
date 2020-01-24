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

#include "test_model/kmb_test_base.hpp"

TEST_F(KmbClassifyNetworkTest, DISABLED_ResNet50) {
    TestNetworkDesc net("ResNet-50/ResNet-50_fp32.xml");

    net
        .setUserInputPresision("input", Precision::U8)
        .setUserInputLayout("input", Layout::NHWC)
        .setUserOutputPresision("output", Precision::FP32);

    runTest(
        net,
        "224x224/cat3.bmp",
        1, 0.05f);
}

TEST_F(KmbDetectionNetworkTest, DISABLED_SSD300) {
    TestNetworkDesc net("SSD_300/ssd_300_fp16.xml");

    net
        .setUserInputPresision("input", Precision::U8)
        .setUserInputLayout("input", Layout::NHWC)
        .setUserOutputPresision("output", Precision::FP32);

    runTest(
        net,
        "300x300/dog.bmp",
        0.3f,
        0.85f, 0.1f);
}
