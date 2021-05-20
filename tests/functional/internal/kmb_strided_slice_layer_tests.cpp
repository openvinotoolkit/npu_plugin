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

INSTANTIATE_TEST_CASE_P(precommit, KmbStridedSliceLayerTests,
    testing::Combine(
        testing::ValuesIn(stridedSliceParams),
        testing::Values<Layout>(NCHW, NHWC)
    )
);
