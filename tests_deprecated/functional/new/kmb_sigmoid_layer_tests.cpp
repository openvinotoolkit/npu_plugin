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
    SKIP() << "MCM compiler is unable to parse a standalone sigmoid layer.";
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

INSTANTIATE_TEST_CASE_P(precommit, KmbSigmoidLayerTests, testing::ValuesIn(sigmoidParams));
