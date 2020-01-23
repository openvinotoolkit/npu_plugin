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

TEST_F(KmbNetworkTest, DISABLED_ResNet50) {
    runClassifyNetworkTest(
        "resnet50_tf_int8/0d/resnet-50-int8-tf-0001",
        "224x224/cat3.bmp",
        1, 5.0f);
}
