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

TEST_P(KmbScaleShiftLayerTests, SimpleScaleShift) {
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
                return makeSingleValueBlob(desc, inputScale);
            }
    );
    registerBlobGenerator(
            "shift", TensorDesc(netPresicion, {1, p._inDims.at(1), 1, 1}, Layout::NCHW),
            [&](const TensorDesc& desc) {
                return makeSingleValueBlob(desc, inputShift);
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

INSTANTIATE_TEST_CASE_P(SomeCase, KmbScaleShiftLayerTests, testing::ValuesIn(scaleShiftParams));
