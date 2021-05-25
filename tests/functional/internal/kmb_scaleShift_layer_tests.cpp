//
// Copyright 2019 Intel Corporation.
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

struct ScaleShiftTestParams final {
    SizeVector _inDims;
    float _scale;
    float _shift;

    ScaleShiftTestParams& inDims(const SizeVector& inDims) {
        this->_inDims = inDims;
        return *this;
    }

    ScaleShiftTestParams& scale(float scale) {
        this->_scale = scale;
        return *this;
    }

    ScaleShiftTestParams& shift(float shift) {
        this->_shift = shift;
        return *this;
    }
};
std::ostream& operator<<(std::ostream& os, const ScaleShiftTestParams& p) {
    vpu::formatPrint(os, "[inDims:%v, scale:%v, shift:%v]", p._inDims, p._scale, p._shift);
    return os;
}

class KmbScaleShiftLayerTests : public KmbLayerTestBase, public testing::WithParamInterface<ScaleShiftTestParams> {};

// [Track number: S#27239]
TEST_P(KmbScaleShiftLayerTests, DISABLED_SimpleScaleShift) {
    const auto& p = GetParam();

    const auto netPresicion = Precision::FP32;

    const auto userInDesc = TensorDesc(Precision::U8, p._inDims, Layout::NHWC);
    const auto userOutDesc = TensorDesc(Precision::FP16, Layout::NHWC);

    const auto inputRange = std::make_pair(1.0f, 1.0f);

    const auto inputScale = p._scale;
    const auto inputShift = p._shift;

    const auto tolerance = 1e-3f;  // obtained based on CPU plugin

    registerBlobGenerator(
            "input", userInDesc,
            [&](const TensorDesc& desc) {
                return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
            }
    );

    registerBlobGenerator(
            "scales", TensorDesc(netPresicion, {1, p._inDims.at(1), 1, 1}, Layout::NCHW),
            [&](const TensorDesc& desc) {
                return vpux::makeSplatBlob(desc, inputScale);
            }
    );
    registerBlobGenerator(
            "shift", TensorDesc(netPresicion, {1, p._inDims.at(1), 1, 1}, Layout::NCHW),
            [&](const TensorDesc& desc) {
                return vpux::makeSplatBlob(desc, inputShift);
            }
    );

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), netPresicion)
            .addLayer<ScaleShiftLayerDef>("scaleShift")
                .input("input")
                .scale(getBlobByName("scales"))
                .shift(getBlobByName("shift"))
                .build()
            .addNetOutput(PortInfo("scaleShift"))
            .setUserOutput(PortInfo("scaleShift"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<ScaleShiftTestParams> scaleShiftParams {
        {ScaleShiftTestParams()
                 .inDims({1, 3, 8, 8})
                 .scale(2.0f)
                 .shift(0.0f)},
        {ScaleShiftTestParams()
                 .inDims({1, 3, 8, 8})
                 .scale(2.0f)
                 .shift(1.0f)},
        {ScaleShiftTestParams()
                 .inDims({1, 3, 224, 224})
                 .scale(2.0f)
                 .shift(0.0f)},
        {ScaleShiftTestParams()
                 .inDims({1, 3, 224, 224})
                 .scale(2.5f)
                 .shift(0.3f)}
};

INSTANTIATE_TEST_CASE_P(precommit, KmbScaleShiftLayerTests, testing::ValuesIn(scaleShiftParams));
