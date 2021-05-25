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

struct SoftmaxTestParams final {
    SizeVector _inDims;
    size_t _axisSet = 0;
    Precision _inPrecision;
    Precision _outPrecision;

    SoftmaxTestParams& inDims(const SizeVector& inDims) {
        this->_inDims = inDims;
        return *this;
    }

    SoftmaxTestParams& axisSet(const size_t& axis_set) {
        this->_axisSet = axis_set;
        return *this;
    }

    SoftmaxTestParams& inPrecision(const Precision& inPrecision) {
        this->_inPrecision = inPrecision;
        return *this;
    }

    SoftmaxTestParams& outPrecision(const Precision& outPrecision) {
        this->_outPrecision = outPrecision;
        return *this;
    }
};
std::ostream& operator<<(std::ostream& os, const SoftmaxTestParams& p) {
    vpu::formatPrint(os, "[inDims:%v, axisSet:%v, inPrecision:%v, outPrecision:%v]", p._inDims, p._axisSet, p._inPrecision, p._outPrecision);
    return os;
}

class KmbSoftmaxLayerTests : public KmbLayerTestBase, public testing::WithParamInterface<SoftmaxTestParams> {};

TEST_P(KmbSoftmaxLayerTests, Single_FP32) {
    const auto &p = GetParam();

    const auto userInDesc = TensorDesc(p._inPrecision, p._inDims, Layout::NHWC);
    const auto userOutDesc = TensorDesc(p._outPrecision, Layout::NHWC);

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
        /* FIXME: "Softmax fails with following params"
        * [Track number: CVS-37052 */
        // SoftmaxTestParams()
        //     .inDims({1, 1000, 2, 2})
        //     .axisSet({1})
        //     .inPrecision(Precision::FP16)
        //     .outPrecision(Precision::FP32),
        // SoftmaxTestParams()
        //     .inDims({1, 1001, 2, 2})
        //     .axisSet({1})
        //     .inPrecision(Precision::FP16)
        //     .outPrecision(Precision::FP32),
        SoftmaxTestParams()
            .inDims({1, 3, 32, 32})
            .axisSet({1})
            .inPrecision(Precision::U8)
            .outPrecision(Precision::FP16),
        SoftmaxTestParams()
            .inDims({1, 3, 32, 32})
            .axisSet({1})
            .inPrecision(Precision::U8)
            .outPrecision(Precision::FP32),
        SoftmaxTestParams()
            .inDims({1, 3, 16, 16})
            .axisSet({1})
            .inPrecision(Precision::U8)
            .outPrecision(Precision::FP16),
        SoftmaxTestParams()
            .inDims({1, 3, 16, 16})
            .axisSet({1})
            .inPrecision(Precision::U8)
            .outPrecision(Precision::FP32)
        // SoftmaxTestParams()
        //     .inDims({1, 1000, 2, 2})
        //     .axisSet({1})
        //     .inPrecision(Precision::U8)
        //     .outPrecision(Precision::FP32),
        // SoftmaxTestParams()
        //     .inDims({1, 1001, 2, 2})
        //     .axisSet({1})
        //     .inPrecision(Precision::U8)
        //     .outPrecision(Precision::FP32)
};


INSTANTIATE_TEST_CASE_P(precommit, KmbSoftmaxLayerTests, testing::ValuesIn(softmaxParams));
