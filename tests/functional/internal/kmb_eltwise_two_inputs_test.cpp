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

struct EltwiseTwoInputsTestParams final {
    SizeVector _inDims;
    Precision _outPrecision;

    EltwiseTwoInputsTestParams& inDims(const SizeVector& inDims) {
        this->_inDims = inDims;
        return *this;
    }

    EltwiseTwoInputsTestParams& outPrecision(const Precision& outPrecision) {
        this->_outPrecision = outPrecision;
        return *this;
    }
};

std::ostream& operator<<(std::ostream& os, const EltwiseTwoInputsTestParams& p) {
    vpu::formatPrint(os, "[inDims:%v, outPrecision:%v]", p._inDims, p._outPrecision);
    return os;
}

class KmbEltwiseTwoInputsTest : public KmbLayerTestBase, public testing::WithParamInterface<EltwiseTwoInputsTestParams> {};

TEST_P(KmbEltwiseTwoInputsTest, eltwiseAdd) {
    const auto &p = GetParam();

    const auto userInDesc = TensorDesc(Precision::U8, p._inDims, Layout::NHWC);
    const auto userOutDesc = TensorDesc(p._outPrecision, Layout::NHWC);

    const auto inputRange = std::make_pair(0.0f, 10.0f);

    const auto tolerance = 1e-2f;

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

const std::vector<EltwiseTwoInputsTestParams> eltwiseParams {
        EltwiseTwoInputsTestParams()
            .inDims({1, 3, 32, 32})
            .outPrecision(Precision::FP16),
};

INSTANTIATE_TEST_SUITE_P(precommit, KmbEltwiseTwoInputsTest, testing::ValuesIn(eltwiseParams));
