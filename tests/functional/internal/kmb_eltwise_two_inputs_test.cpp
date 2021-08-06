//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "test_model/kmb_test_base.hpp"

using EltwiseTwoInputsTestParams = std::tuple<
    SizeVector,  // inDims
    Precision,   // inPrecision
    Precision    // outPrecision
>;

std::ostream& operator<<(std::ostream& os, const EltwiseTwoInputsTestParams& p) {
    vpu::formatPrint(os, "[inDims:%v, inPrecision:%v, outPrecision:%v]", std::get<0>(p), std::get<1>(p), std::get<2>(p));
    return os;
}

class KmbEltwiseTwoInputsTest : public KmbLayerTestBase, public testing::WithParamInterface<EltwiseTwoInputsTestParams> {};

TEST_P(KmbEltwiseTwoInputsTest, eltwiseAdd) {
    const auto &p = GetParam();
    const auto& inDims = std::get<0>(p);
    const auto& inPrecision  = std::get<1>(p);
    const auto& outPrecision = std::get<2>(p);

    const auto userInDesc = TensorDesc(inPrecision, inDims, Layout::NHWC);
    const auto userOutDesc = TensorDesc(outPrecision, Layout::NHWC);

    const auto inputRange = std::make_pair(0.0f, 10.0f);

    const auto tolerance = 5e-2f;

    registerBlobGenerator(
        "input1", userInDesc,
        [&](const TensorDesc& desc) {
            return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
        }
    );

    registerBlobGenerator(
        "input2", userInDesc,
        [&](const TensorDesc& desc) {
            return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
        }
    );

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input1", userInDesc.getPrecision(), userInDesc.getLayout())
            .setUserInput("input2", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input1", userInDesc.getDims(), Precision::FP32)
            .addNetInput("input2", userInDesc.getDims(), Precision::FP32)
            .addLayer<AddLayerDef>("eltwiseAdd")
                .input1("input1")
                .input2("input2")
                .build()
            .addNetOutput(PortInfo("eltwiseAdd"))
            .setUserOutput(PortInfo("eltwiseAdd"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::set<Precision> inPrecisions = {
    Precision::U8,
    Precision::FP16,
    Precision::FP32
};

INSTANTIATE_TEST_SUITE_P(precommit, KmbEltwiseTwoInputsTest,
    testing::Combine(
        testing::Values(SizeVector({1, 3, 32, 32})),
        testing::ValuesIn(inPrecisions),
        testing::Values(Precision::FP16)
    )
);
