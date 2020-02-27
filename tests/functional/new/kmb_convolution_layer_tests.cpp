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

struct ConvTestParams final {
    SizeVector _inDims;
    ConvolutionParams _convParams;

    ConvTestParams& inDims(const SizeVector& inDims) {
        this->_inDims = inDims;
        return *this;
    }

    ConvTestParams& convParams(const ConvolutionParams& convParams) {
        this->_convParams = convParams;
        return *this;
    }
};
std::ostream& operator<<(std::ostream& os, const ConvTestParams& p) {
    vpu::formatPrint(os, "[inDims:%v, convParams:%v]",
        p._inDims, p._convParams);
    return os;
}

class KmbConvolutionLayerTests : public KmbLayerTestBase, public testing::WithParamInterface<ConvTestParams> {};

// [Track number: S#26002]
TEST_P(KmbConvolutionLayerTests, ScaleShift_FQ) {
    SKIP_INFER_ON("KMB", "bad results");

    const auto& p = GetParam();

    const auto netPresicion = Precision::FP32;

    const auto userInDesc = TensorDesc(Precision::U8, p._inDims, Layout::NHWC);
    const auto userOutDesc = TensorDesc(Precision::FP16, Layout::NHWC);

    const auto tolerance = 1e-3f;

    registerBlobGenerator(
        "input", userInDesc,
        [&](const TensorDesc& desc) {
            return makeSingleValueBlob(desc, 1.0f);
        }
    );
    registerBlobGenerator(
        "scales", TensorDesc(netPresicion, {1, p._inDims.at(1), 1, 1}, Layout::NCHW),
        [&](const TensorDesc& desc) {
            return makeSingleValueBlob(desc, 1.0f);
        }
    );
    registerBlobGenerator(
        "shift", TensorDesc(netPresicion, {1, p._inDims.at(1), 1, 1}, Layout::NCHW),
        [&](const TensorDesc& desc) {
            return makeSingleValueBlob(desc, 0.0f);
        }
    );
    registerBlobGenerator(
        "weights", getConvWeightsDesc(p._convParams, p._inDims.at(1), netPresicion),
        [&](const TensorDesc& desc) {
            return makeSingleValueBlob(desc, 1.0f);
        }
    );
    registerBlobGenerator(
        "biases", getConvBiasesDesc(p._convParams, netPresicion),
        [&](const TensorDesc& desc) {
            return makeSingleValueBlob(desc, 1.0f);
        }
    );

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), netPresicion)
            .addLayer<ScaleShiftLayerDef>("input_scale_shift")
                .input("input")
                .scale(getBlobByName("scales"))
                .shift(getBlobByName("shift"))
                .build()
            .addLayer<FakeQuantizeLayerDef>("input_quantize", 256)
                .input("input_scale_shift")
                .low(0.0f, netPresicion)
                .high(255.0f, netPresicion)
                .build()
            .addLayer<FakeQuantizeLayerDef>("weights_quantize", 256)
                .input(getBlobByName("weights"))
                .low(0.0f, netPresicion)
                .high(4.0f, netPresicion)
                .build()
            .addLayer<ConvolutionLayerDef>("conv", p._convParams)
                .input("input_quantize")
                .weights("weights_quantize")
                .biases(getBlobByName("biases"))
                .build()
            .addNetOutput(PortInfo("conv"))
            .setUserOutput(PortInfo("conv"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .finalize();

        testNet.setCompileConfig({{VPU_COMPILER_CONFIG_KEY(USE_NGRAPH_PARSER), CONFIG_VALUE(YES)}});
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<ConvTestParams> convParams {
    ConvTestParams()
        .inDims({1, 3, 64, 64})
        .convParams(ConvolutionParams().outChannels(64).kernel({3, 3}).strides({2, 2}).pad({0, 0, 0, 0}))
};

INSTANTIATE_TEST_CASE_P(SomeCase, KmbConvolutionLayerTests, testing::ValuesIn(convParams));
