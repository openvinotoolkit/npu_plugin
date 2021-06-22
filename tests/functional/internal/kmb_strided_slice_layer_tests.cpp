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

struct StridedSliceTestParams final {
    StridedSliceParams params;

    LAYER_PARAMETER(std::vector<int64_t>, begins);
    LAYER_PARAMETER(std::vector<int64_t>, ends);
    LAYER_PARAMETER(std::vector<int64_t>, strides);
    LAYER_PARAMETER(std::vector<int64_t>, beginMask);
    LAYER_PARAMETER(std::vector<int64_t>, endMask);
    LAYER_PARAMETER(std::vector<int64_t>, newAxisMask);
    LAYER_PARAMETER(std::vector<int64_t>, shrinkAxisMask);
    LAYER_PARAMETER(std::vector<int64_t>, ellipsisAxisMask);
    PARAMETER(SizeVector, inputShape);
};
std::ostream& operator<<(std::ostream& os, const StridedSliceTestParams& p) {
    vpu::formatPrint(os, "[inputShape:%v, begin:%v, end:%v, strides:%v, beginMask:%v, endMask:%v, newAxisMask:%v, shrinkAxisMask:%v, ellipsisAxisMask:%v]",
        p.inputShape(), p.begins(), p.ends(), p.strides(), p.beginMask(), p.endMask(), p.newAxisMask(), p.shrinkAxisMask(), p.ellipsisAxisMask());
    return os;
}

class KmbStridedSliceLayerTests : public KmbLayerTestBase, public testing::WithParamInterface<std::tuple<StridedSliceTestParams, Layout>> {};

TEST_P(KmbStridedSliceLayerTests, Single_FP32) {
    const auto &p = std::get<0>(GetParam());
    const auto &layout = std::get<1>(GetParam());

    const auto userInDesc = TensorDesc(Precision::FP16, p.inputShape(), layout);
    const auto userOutDesc = TensorDesc(Precision::FP16, layout);

    const auto inputRange = std::make_pair(0.0f, 10.0f);

    const auto tolerance = 0.f;

    registerBlobGenerator(
        "input", userInDesc,
        [&](const TensorDesc& desc) {
            return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
        }
    );
    registerBlobGenerator(
        "begins", TensorDesc(Precision::I64, {p.begins().size()}, Layout::C),
        [&](const TensorDesc& desc) {
            return genBlobFromData(desc, p.begins());
        }
    );
    registerBlobGenerator(
        "ends", TensorDesc(Precision::I64, {p.ends().size()}, Layout::C),
        [&](const TensorDesc& desc) {
            return genBlobFromData(desc, p.ends());
        }
    );
    registerBlobGenerator(
        "strides", TensorDesc(Precision::I64, {p.strides().size()}, Layout::C),
        [&](const TensorDesc& desc) {
            return genBlobFromData(desc, p.strides());
        }
    );

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), Precision::FP32)
            .addLayer<StridedSliceLayerDef>("stridedSlice", p.params)
                .input("input")
                .begins(getBlobByName("begins"))
                .ends(getBlobByName("ends"))
                .strides(getBlobByName("strides"))
                .build()
            .addNetOutput(PortInfo("stridedSlice"))
            .setUserOutput(PortInfo("stridedSlice"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .allowNCHWLayoutForMcmModelInput(true)
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<StridedSliceTestParams> stridedSliceParams {
        StridedSliceTestParams()
            .inputShape({1, 2, 3, 4})
            .begins({0, 0, 0, 0})
            .ends({0, 0, 0, 0})
            .strides({1, 1, 1, 1})
            .beginMask({1, 1, 1, 1})
            .endMask({1, 1, 1, 1})
            .newAxisMask({0, 0, 0, 0})
            .shrinkAxisMask({0, 0, 0, 0})
            .ellipsisAxisMask({0, 0, 0, 0}),
        StridedSliceTestParams()
            .inputShape({1, 2, 3, 4})
            .begins({0, 0, 0, 0})
            .ends({1, 2, 3, 4})
            .strides({1, 1, 1, 2})
            .beginMask({0, 1, 1, 1})
            .endMask({1, 1, 1, 1})
            .newAxisMask({0, 0, 0, 0})
            .shrinkAxisMask({0, 0, 0, 0})
            .ellipsisAxisMask({0, 0, 0, 0}),
    StridedSliceTestParams()
            .inputShape({1, 4, 8, 32})
            .begins({0, 0, 0, 0})
            .ends({1, 100, 8, 200})
            .strides({1, 2, 1, 4})
            .beginMask({0, 1, 1, 1})
            .endMask({0, 1, 0, 1})
            .newAxisMask({0, 0, 0, 0})
            .shrinkAxisMask({0, 0, 0, 0})
            .ellipsisAxisMask({0, 0, 0, 0})
            .ellipsisAxisMask({0, 0, 0, 0})
};

INSTANTIATE_TEST_SUITE_P(precommit, KmbStridedSliceLayerTests,
    testing::Combine(
        testing::ValuesIn(stridedSliceParams),
        testing::Values<Layout>(NCHW, NHWC)
    )
);
