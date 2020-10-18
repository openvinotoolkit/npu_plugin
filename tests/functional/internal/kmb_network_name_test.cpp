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

class KmbNetworkNameTest : public KmbNetworkTestBase {
public:
    void runTest(const TestNetworkDesc& netDesc);
};

void KmbNetworkNameTest::runTest(const TestNetworkDesc& netDesc) {
    if (RUN_COMPILER) {
        CNNNetwork cnnNet = KmbNetworkTestBase::readNetwork(netDesc, true);
        ExecutableNetwork exeNet = core->LoadNetwork(cnnNet, DEVICE_NAME, netDesc.compileConfig());
        KmbTestBase::exportNetwork(exeNet);
    }
    ExecutableNetwork importedNet = KmbTestBase::importNetwork();
    const std::string netName = importedNet.GetMetric(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME));
    ASSERT_EQ(netName, "squeezenet1.1");
}

// [Track number: S#40932]
TEST_F(KmbNetworkNameTest, fetchNetworkName) {
    SKIP_ON("KMB", "HDDL2", "VPUX", "netName 'net1' is not equal to 'squeezenet1.1'");
    runTest(
        TestNetworkDesc("KMB_models/INT8/public/squeezenet1_1/squeezenet1_1_pytorch_caffe2_dense_int8_IRv10_from_fp32.xml")
            .setUserInputPrecision("input", Precision::U8)
            .setUserInputLayout("input", Layout::NHWC)
            .setUserOutputPrecision("output", Precision::FP32)
            .setUserOutputLayout("output", Layout::NHWC));
}

