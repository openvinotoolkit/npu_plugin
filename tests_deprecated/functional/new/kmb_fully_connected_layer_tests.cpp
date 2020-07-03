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

struct FCTestParams final {
    SizeVector _inDims;

    FullyConnectedParams _fcParams;

    FCTestParams& inDims(const SizeVector& inDims) {
        this->_inDims = inDims;
        return *this;
    }

    FCTestParams& fcParams(const FullyConnectedParams& fcParams) {
        this->_fcParams = fcParams;
        return *this;
    }
};
std::ostream& operator<<(std::ostream& os, const FCTestParams& p) {
    vpu::formatPrint(os, "[inDims:%v, fcParams:%v]", p._inDims, p._fcParams);
    return os;
}

class KmbFullyConnectedLayerTests : public KmbLayerTestBase, public testing::WithParamInterface<FCTestParams> {};

TEST_P(KmbFullyConnectedLayerTests, FakeQuantize_FC) {
    SKIP_ON("KMB", "Check 'arg0_shape compatible arg1_shape' failed");
    const auto& p = GetParam();

    const auto netPresicion = Precision::FP32;

    const auto userInDesc = TensorDesc(Precision::U8, p._inDims, Layout::NC);
    const auto userOutDesc = TensorDesc(Precision::FP32, Layout::NC);

    const auto tolerance = 1e-3f;

    registerBlobGenerator(
        "input", userInDesc,
        [&](const TensorDesc& desc) {
            return makeSingleValueBlob(desc, 1.0f);
        }
    );
    registerBlobGenerator(
        "weights", getFCWeightsDesc(p._fcParams, p._inDims.at(1), netPresicion),
        [&](const TensorDesc& desc) {
            return makeSingleValueBlob(desc, 1.0f);
        }
    );
    registerBlobGenerator(
        "biases", getFCBiasesDesc(p._fcParams, netPresicion),
        [&](const TensorDesc& desc) {
            return makeSingleValueBlob(desc, 1.0f);
        }
    );

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet.setCompileConfig({{"VPU_KMB_FORCE_2D_TO_NC", CONFIG_VALUE(YES)}});
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), netPresicion)
            .addLayer<FakeQuantizeLayerDef>("input_quantize", 256)
                .input("input")
                .low(0.0f, netPresicion)
                .high(255.0f, netPresicion)
                .build()
            .addConst("weights", getBlobByName("weights"))
            .addLayer<FakeQuantizeLayerDef>("weights_quantize", 256)
                .input("weights")
                .low(0.0f, netPresicion)
                .high(255.0, netPresicion)
                .build()
            .addLayer<FullyConnectedLayerDef>("fc", p._fcParams)
                .input("input_quantize")
                .weights("weights_quantize")
                .biases(getBlobByName("biases"))
                .build()
            .addNetOutput(PortInfo("fc"))
            .setUserOutput(PortInfo("fc"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<FCTestParams> fcParams {
    FCTestParams()
        .inDims({1, 1024})
        .fcParams(FullyConnectedParams().outChannels(1000))
};

INSTANTIATE_TEST_CASE_P(precommit, KmbFullyConnectedLayerTests, testing::ValuesIn(fcParams));
