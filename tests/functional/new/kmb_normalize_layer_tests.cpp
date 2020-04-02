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
#include <blob_factory.hpp>

struct NormTestParams final {
    NormTestParams(const NormalizeParams& param) : _normParams(param) {}
    SizeVector _inDims;
    Layout _inLayout;
    Layout _outLayout;
    NormalizeParams _normParams;

    NormTestParams& inDims(const SizeVector& inDims) {
        this->_inDims = inDims;
        return *this;
    }

    NormTestParams& inLayout(const Layout& inLayout) {
        this->_inLayout = inLayout;
        return *this;
    }

    NormTestParams& outLayout(const Layout& outLayout) {
        this->_outLayout = outLayout;
        return *this;
    }

    NormTestParams& normParams(const NormalizeParams& normParams) {
        this->_normParams = normParams;
        return *this;
    }
};
std::ostream& operator<<(std::ostream& os, const NormTestParams& p) {
    vpu::formatPrint(os, "[inDims:%v, normParams:%v]", p._inDims, p._normParams);
    return os;
}

class KmbNormalizeLayerTests : public KmbLayerTestBase, public testing::WithParamInterface<NormTestParams> {};

TEST_P(KmbNormalizeLayerTests, EqualWithCPU) {

    const auto& p = GetParam();

    const auto netPresicion = Precision::FP32;

    const auto userInDesc = TensorDesc(Precision::U8, p._inDims, p._inLayout);
    const auto userOutDesc = TensorDesc(Precision::FP16, p._outLayout);

    const auto tolerance = 1e-3f;

    registerBlobGenerator("input", userInDesc, [&](const TensorDesc& desc) {
        return genBlobUniform(desc, rd, 0, 10);
    });
    registerBlobGenerator("axes", TensorDesc(Precision::I64, {1}, Layout::C),[&](const TensorDesc& desc) {
            return makeSingleValueBlob(desc, 1l);
        }
    );
    registerBlobGenerator("scales", TensorDesc(Precision::FP32, {1, p._inDims[1], 1, 1}, Layout::NCHW),[&](const TensorDesc& desc) {
            return makeSingleValueBlob(desc, 1.f);
        }
    );

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet.setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), netPresicion)
            .addLayer<NormalizeLayerDef>("normalize_layer", p._normParams)
                .input("input")
                .axes(getBlobByName("axes"))
                .build()
            .addLayer<MultiplyLayerDef>("multiply_layer")
                .input1("normalize_layer")
                .input2(getBlobByName("scales"))
                .build()
            .addNetOutput(PortInfo("multiply_layer"))
            .setUserOutput(PortInfo("multiply_layer"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<NormTestParams> normParams {
   NormTestParams(NormalizeParams(1e-10f, ngraph::op::EpsMode::ADD))
        .inDims({1, 4, 16, 16})
        .inLayout(Layout::NHWC)
        .outLayout(Layout::NHWC),
    NormTestParams(NormalizeParams(1e-10f, ngraph::op::EpsMode::ADD))
        .inDims({1, 512, 64, 64})
        .inLayout(Layout::NHWC)
        .outLayout(Layout::NHWC)
};

INSTANTIATE_TEST_CASE_P(Normalize, KmbNormalizeLayerTests, testing::ValuesIn(normParams));

