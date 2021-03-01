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

class KmbNetworkNameTest : public KmbLayerTestBase {
public:
    TestNetwork buildPowerLayer(const std::string& netName);
    std::string runTest(const TestNetwork& netDesc, const std::string& netName);
};

TestNetwork KmbNetworkNameTest::buildPowerLayer(const std::string& netName) {
    const Precision precision = Precision::FP16;
    const std::vector<size_t> dims = {1, 3, 224, 224};
    const Layout layout = Layout::NCHW;
    const auto userInDesc = TensorDesc(precision, dims, layout);
    registerBlobGenerator("input", userInDesc, [&](const TensorDesc& desc) {
        return vpux::makeSplatBlob(desc, 1.0f);
    });

    const auto powerTensorDesc = TensorDesc(Precision::FP32, {1, 1, 1, 1}, Layout::NCHW);
    registerBlobGenerator("scale", powerTensorDesc, [&](const TensorDesc& desc) {
        return vpux::makeSplatBlob(desc, 1.0f);
    });

    TestNetwork testNet;
    testNet
        .setUserInput("input", precision, layout)
        .addNetInput("input", dims, Precision::FP32)
        .addLayer<PowerLayerDef>("power")
            .input1("input")
            .input2(getBlobByName("scale"))
            .build()
        .setUserOutput(PortInfo("power"), precision, layout)
        .addNetOutput(PortInfo("power"))
        .finalize(netName);

    return testNet;
}

std::string KmbNetworkNameTest::runTest(const TestNetwork& netDesc, const std::string& netFileName) {
    const auto blobFileName = vpu::formatString("%v/%v.net", DUMP_PATH, netFileName);
    if (RUN_COMPILER) {
        CNNNetwork cnnNet = netDesc.getCNNNetwork();
        ExecutableNetwork exeNet = core->LoadNetwork(cnnNet, DEVICE_NAME, netDesc.compileConfig());
        exeNet.Export(blobFileName);
    }
    ExecutableNetwork importedNet = core->ImportNetwork(blobFileName, DEVICE_NAME, {});
    const std::string netName = importedNet.GetMetric(EXEC_NETWORK_METRIC_KEY(NETWORK_NAME));

    return netName;
}

TEST_F(KmbNetworkNameTest, fetchNetworkName) {
    const std::string expectedNetName = "singleLayerNet";
    TestNetwork testNet = buildPowerLayer(expectedNetName);
    std::string netName = runTest(testNet, expectedNetName);

    ASSERT_EQ(netName, expectedNetName);
}

TEST_F(KmbNetworkNameTest, checkUniqueId) {
    const std::string firstExpectedNetName = "firstNet";
    const std::string secondExpectedNetName = "secondNet";
    TestNetwork firstNetDesc = buildPowerLayer(firstExpectedNetName);
    TestNetwork secondNetDesc = buildPowerLayer(secondExpectedNetName);
    std::string squeezeNetName = runTest(firstNetDesc, firstExpectedNetName);
    std::string googleNetName = runTest(secondNetDesc, secondExpectedNetName);

    ASSERT_NE(squeezeNetName, googleNetName);
}
