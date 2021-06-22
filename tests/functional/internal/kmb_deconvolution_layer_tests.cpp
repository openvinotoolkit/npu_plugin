//
// Copyright 2019-2020 Intel Corporation.
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
#include <precision_utils.h>

struct DeconvTestParams final {
    SizeVector _inDims;
    DeconvolutionParams _deconvParams;

    DeconvTestParams& inDims(const SizeVector& inDims) {
        this->_inDims = inDims;
        return *this;
    }

    DeconvTestParams& deconvParams(const DeconvolutionParams& deconvParams) {
        this->_deconvParams = deconvParams;
        return *this;
    }
};
std::ostream& operator<<(std::ostream& os, const DeconvTestParams& p) {
    vpu::formatPrint(os, "[inDims:%v, deconvParams:%v]",
        p._inDims, p._deconvParams);
    return os;
}

class KmbDeconvLayerTests : public KmbLayerTestBase, public testing::WithParamInterface<DeconvTestParams> {};

// out_spatial_shape == infered_out_spatial_shape
// [Track number: E#13189]
// Bad results
// [Track number: S#39622]
TEST_P(KmbDeconvLayerTests, DepthWiseFP16) {
#ifdef _WIN32
    GTEST_SKIP() << "out_spatial_shape == infered_out_spatial_shape";
#endif
    SKIP_ON("KMB", "HDDL2", "VPUX", "Bad results");
    const auto& p = GetParam();

    const auto netPresicion = Precision::FP32;

    // TODO change input to FP16 when supported
    const auto userInDesc = TensorDesc(Precision::U8, p._inDims, Layout::NHWC);
    const auto userOutDesc = TensorDesc(Precision::FP16, Layout::NHWC);

    const auto tolerance = 1e+3f;

    registerBlobGenerator(
        "input", userInDesc,
        [&](const TensorDesc& desc) {
            return vpux::makeSplatBlob(desc, 1.0f);
        }
    );

    registerBlobGenerator(
        "weights", getDeconvDwWeightsDesc(p._deconvParams, netPresicion),
        [&](const TensorDesc& desc) {
            return vpux::makeSplatBlob(desc, 1.0f);
        }
    );

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), netPresicion)
            .addLayer<DeconvolutionLayerDef>("deconv", p._deconvParams)
                .input("input")
                .weights(getBlobByName("weights"))
                .build()
            .addNetOutput(PortInfo("deconv"))
            .setUserOutput(PortInfo("deconv"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .finalize();
    };


    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<DeconvTestParams> deconvParams {
    DeconvTestParams()
        .inDims({1, 16, 20, 20})
        .deconvParams(DeconvolutionParams().outChannels(16).kernel({2, 2}).strides({2, 2}).pad({0, 0, 0, 0}).group(16)),
};

INSTANTIATE_TEST_SUITE_P(precommit, KmbDeconvLayerTests, testing::ValuesIn(deconvParams));
