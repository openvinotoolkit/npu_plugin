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

struct SoftmaxTestParams final {
    SizeVector _inDims;
    size_t _axisSet = 0;

    SoftmaxTestParams& inDims(const SizeVector& inDims) {
        this->_inDims = inDims;
        return *this;
    }

    SoftmaxTestParams& axisSet(const size_t& axis_set) {
        this->_axisSet = axis_set;
        return *this;
    }
};
std::ostream& operator<<(std::ostream& os, const SoftmaxTestParams& p) {
    vpu::formatPrint(os, "[inDims:%v, axisSet:%v]", p._inDims, p._axisSet);
    return os;
}

class KmbSoftmaxLayerTests : public KmbLayerTestBase, public testing::WithParamInterface<SoftmaxTestParams> {};

TEST_P(KmbSoftmaxLayerTests, Single_FP32) {
    const auto &p = GetParam();

    const auto userInDesc = TensorDesc(Precision::FP16, p._inDims, Layout::NHWC);
    const auto userOutDesc = TensorDesc(Precision::FP32, Layout::NHWC);

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
            .addLayer<SoftmaxLayerDef>("softmax", p._axisSet)
                .input("input")
                .build()
            .addNetOutput(PortInfo("softmax"))
            .setUserOutput(PortInfo("softmax"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<SoftmaxTestParams> softmaxParams {
        SoftmaxTestParams()
            .inDims({1, 1000, 2, 2})
            .axisSet({1}),
        SoftmaxTestParams()
            .inDims({1, 1001, 2, 2})
            .axisSet({1}),
        SoftmaxTestParams()
            .inDims({1, 1000, 1, 1})
            .axisSet({1}),
        SoftmaxTestParams()
            .inDims({1, 1001, 1, 1})
            .axisSet({1}),
};

INSTANTIATE_TEST_CASE_P(precommit, KmbSoftmaxLayerTests, testing::ValuesIn(softmaxParams));
