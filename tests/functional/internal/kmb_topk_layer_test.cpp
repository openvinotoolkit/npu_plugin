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
#include <blob_factory.hpp>

struct TopKTestParams final {
    TopKTestParams(const TopKParams& param) : _topkParams(param) {}
    SizeVector _inDims;
    Layout _inLayout;
    Layout _outLayout;
    TopKParams _topkParams;

    TopKTestParams& inDims(const SizeVector& inDims) {
        this->_inDims = inDims;
        return *this;
    }

    TopKTestParams& inLayout(const Layout& inLayout) {
        this->_inLayout = inLayout;
        return *this;
    }

    TopKTestParams& outLayout(const Layout& outLayout) {
        this->_outLayout = outLayout;
        return *this;
    }

    TopKTestParams& topkParams(const TopKParams& topkParams) {
        this->_topkParams = topkParams;
        return *this;
    }
};
std::ostream& operator<<(std::ostream& os, const TopKTestParams& p) {
    vpu::formatPrint(os, "[inDims:%v, topkParams:%v ]", p._inDims, p._topkParams);
    return os;
}

class KmbTopKLayerTests : public KmbLayerTestBase, public testing::WithParamInterface<TopKTestParams> {};

TEST_P(KmbTopKLayerTests, Top_EqualWithCPU) {
    const auto& p = GetParam();

    const auto netPresicion = Precision::FP32;

    const auto userInDesc = TensorDesc(Precision::U8, p._inDims, p._inLayout);
    const auto userOutDesc = TensorDesc(Precision::FP16, p._outLayout);

    const auto tolerance = 1e-3f;

    registerBlobGenerator("input", userInDesc, [&](const TensorDesc& desc) {
        return genBlobUniform(desc, rd, 0, 10);
    });

    auto scalarKTensorDesc = TensorDesc(Precision::I64, {}, Layout::SCALAR);
    registerBlobGenerator("scalarK", scalarKTensorDesc, [&](const TensorDesc& desc) {
        return vpux::makeSplatBlob(desc, 1);
    });

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet.setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), netPresicion)
            .addLayer<TopKLayerDef>("topk", p._topkParams)
                .input("input")
                .scalarK(getBlobByName("scalarK"))
                .build()
            .addNetOutput(PortInfo("topk", 1))
            .setUserOutput(PortInfo("topk", 1), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

// Params from ICNet network
const std::vector<TopKTestParams> topkParams {
   TopKTestParams(TopKParams(1, ngraph::op::v1::TopK::Mode::MAX,ngraph::op::v1::TopK::SortType::SORT_INDICES))
        .inDims({1, 12, 720, 960})
        .inLayout(Layout::NHWC)
        .outLayout(Layout::NHWC)
};

INSTANTIATE_TEST_SUITE_P(precommit_TopK, KmbTopKLayerTests, testing::ValuesIn(topkParams));
