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

#include <common/include/vpu/utils/ie_helpers.hpp>

#include "test_model/kmb_test_base.hpp"

struct ConvertTestParams final {
    ConvertParams params;

    LAYER_PARAMETER(ngraph::element::Type, destination_type);
    PARAMETER(ngraph::element::Type, source_type);
    PARAMETER(SizeVector, dims);
};

std::ostream& operator<<(std::ostream& os, const ConvertTestParams &p) {
    vpu::formatPrint(
        os, "dims: %v, source_type: %t, destination_type: %t", p.dims(), p.source_type(), p.destination_type());
    return os;
}

class KmbConvertLayerTests : public KmbLayerTestBase, public testing::WithParamInterface<ConvertTestParams> {};

TEST_P(KmbConvertLayerTests, accuracy) {
    const auto &p = GetParam();

    const auto netPrecision = Precision::FP32;

    const auto dims = p.dims();
    const auto userInDesc = TensorDesc(typeToPrecision(p.source_type()), dims, Layout::NCHW);
    const auto userOutDesc = TensorDesc(typeToPrecision(p.destination_type()), dims, Layout::NCHW);

    const auto inputRange = std::make_pair(0.0f, 10.0f);

    const auto tolerance = 1e-2f;

    registerBlobGenerator("input", userInDesc, [&](const TensorDesc& desc) {
        return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
    });

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), netPrecision)
            .addLayer<ConvertLayerDef>("convert", p.params)
                .input("input")
                .build()
            .addNetOutput(PortInfo("convert"))
            .setUserOutput(PortInfo("convert"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .useCustomLayers()
            .disableMcmPasses({{"kmb_adapt", "KMBQuantizeConversion"}})
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<ConvertTestParams> convertParams = {
        ConvertTestParams()
            .dims({1, 3, 360, 480})
            .source_type(ngraph::element::Type_t::u8)
            .destination_type(ngraph::element::Type_t::f16)
};

INSTANTIATE_TEST_CASE_P(SomeCase, KmbConvertLayerTests, testing::ValuesIn(convertParams));
