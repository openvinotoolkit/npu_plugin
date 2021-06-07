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

struct EltwisePopulatedInputTestParams final {
    SizeVector _inDims;
    Precision _outPrecision;

    EltwisePopulatedInputTestParams& inDims(const SizeVector& inDims) {
        this->_inDims = inDims;
        return *this;
    }

    EltwisePopulatedInputTestParams& outPrecision(const Precision& outPrecision) {
        this->_outPrecision = outPrecision;
        return *this;
    }
};

std::ostream& operator<<(std::ostream& os, const EltwisePopulatedInputTestParams& p) {
    vpu::formatPrint(os, "[inDims:%v, outPrecision:%v]", p._inDims, p._outPrecision);
    return os;
}

class KmbEltwisePopulatedInputTest : public KmbLayerTestBase, public testing::WithParamInterface<EltwisePopulatedInputTestParams> {};

TEST_P(KmbEltwisePopulatedInputTest, eltwiseAdd) {
    const auto &p = GetParam();

    const auto userInDesc = TensorDesc(Precision::U8, p._inDims, Layout::NHWC);
    const auto ConstDesc = TensorDesc(Precision::U8, {1, 1, 1, 1}, Layout::NHWC);
    const auto userOutDesc = TensorDesc(p._outPrecision, Layout::NHWC);

    const auto inputRange = std::make_pair(0.0f, 10.0f);

    const auto tolerance = 1e-2f;

    registerBlobGenerator(
        "constant", ConstDesc,
        [&](const TensorDesc& desc) {
            return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
        }
    );

    registerBlobGenerator(
        "input", userInDesc,
        [&](const TensorDesc& desc) {
            return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
        }
    );

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), Precision::U8)
            .addLayer<AddLayerDef>("eltwiseAdd")
                .input1(getBlobByName("constant"))
                .input2("input")
                .build()
            .addNetOutput(PortInfo("eltwiseAdd"))
            .setUserOutput(PortInfo("eltwiseAdd"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<EltwisePopulatedInputTestParams> eltwiseParams {
        EltwisePopulatedInputTestParams()
            .inDims({1, 3, 32, 32})
            .outPrecision(Precision::FP16),
};

INSTANTIATE_TEST_CASE_P(precommit, KmbEltwisePopulatedInputTest, testing::ValuesIn(eltwiseParams));
