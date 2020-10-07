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

INSTANTIATE_TEST_CASE_P(precommit, KmbEltwiseTwoInputsTest, testing::ValuesIn(eltwiseParams));
