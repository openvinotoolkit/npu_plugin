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

// Sсalar blob is unsupported
// Layers with multiple outputs is unsupported
// [Track number: S#31485/S#31238]
TEST_P(KmbTopKLayerTests, DISABLED_Top_EqualWithCPU) {

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
        auto blob = make_plain_blob(desc.getPrecision(), {});
        blob->allocate();
        blob->buffer().as<uint64_t*>()[0] = 1l;
        return blob;
    });

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet.setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), netPresicion)
            .addLayer<TopKLayerDef>("topk", p._topkParams)
                .input("input")
                .scalarK(getBlobByName("scalarK"))
                .build()
            .addNetOutput(PortInfo("topk"))
            .setUserOutput(PortInfo("topk"), userOutDesc.getPrecision(), userOutDesc.getLayout())
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

INSTANTIATE_TEST_CASE_P(TopK, KmbTopKLayerTests, testing::ValuesIn(topkParams));

