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

struct SigmoidTestParams final {
    SizeVector _inDims;

    SigmoidTestParams& inDims(const SizeVector& inDims) {
        this->_inDims = inDims;
        return *this;
    }
};

std::ostream& operator<<(std::ostream& os, const SigmoidTestParams& p) {
    vpu::formatPrint(os, "[inDims:%v]", p._inDims);
    return os;
}

class KmbSigmoidLayerTests : public KmbLayerTestBase, public testing::WithParamInterface<SigmoidTestParams> {};

TEST_P(KmbSigmoidLayerTests, Single_FP32) {
    // TODO: Remove following line after mcm compiler fix.
    GTEST_SKIP() << "MCM compiler is unable to parse a standalone sigmoid layer.";
    const auto &p = GetParam();

    const auto userInDesc = TensorDesc(Precision::U8, p._inDims, Layout::NCHW);
    const auto userOutDesc = TensorDesc(Precision::FP32, Layout::NCHW);

    const auto inputRange = std::make_pair(0.0f, 10.0f);

    const auto tolerance = 1e-2f;

    registerBlobGenerator(
        "input", userInDesc,
        [&](const TensorDesc& desc) {
            return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
        }
    );

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), Precision::FP32)
            .addLayer<SigmoidLayerDef>("sigmoid")
                .input("input")
                .build()
            .addNetOutput(PortInfo("sigmoid"))
            .setUserOutput(PortInfo("sigmoid"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<SigmoidTestParams> sigmoidParams {
        SigmoidTestParams()
            .inDims({1, 1000, 1, 1})
};

INSTANTIATE_TEST_SUITE_P(precommit, KmbSigmoidLayerTests, testing::ValuesIn(sigmoidParams));
