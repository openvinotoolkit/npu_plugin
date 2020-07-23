//
// Copyright 2019-2020 Intel Corporation.
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

TEST_P(KmbDeconvLayerTests, DepthWiseFP16) {

    const auto& p = GetParam();

    const auto netPresicion = Precision::FP32;

    // TODO change input to FP16 when supported
    const auto userInDesc = TensorDesc(Precision::U8, p._inDims, Layout::NHWC);  
    const auto userOutDesc = TensorDesc(Precision::FP16, Layout::NHWC);

    const auto tolerance = 1e+3f;

    registerBlobGenerator(
        "input", userInDesc,
        [&](const TensorDesc& desc) {
            return makeSingleValueBlob(desc, 1.0f);
        }
    );

    registerBlobGenerator(
        "weights", getDeconvDwWeightsDesc(p._deconvParams, netPresicion),
        [&](const TensorDesc& desc) {
            return makeSingleValueBlob(desc, 1.0f);
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

INSTANTIATE_TEST_CASE_P(precommit, KmbDeconvLayerTests, testing::ValuesIn(deconvParams));
