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

class KmbLayoutTests : public KmbLayerTestBase,
    public testing::WithParamInterface<std::tuple<Precision, Precision, Layout, bool>> {};

static const std::set<Precision> supportedInPrecisions = { Precision::U8 };
static const std::set<Precision> supportedOutPrecisions = { Precision::UNSPECIFIED, Precision::U8, Precision::FP16, Precision::FP32 };
static const std::set<Layout> supportedInLayouts = { Layout::NHWC };
static const std::set<Layout> supportedOutLayouts = { Layout::NHWC, Layout::NC };

static bool is_supported(const Precision& inPrecision, const Layout& inLayout, const Precision& outPrecision, const Layout& outLayout) {
    bool inPrecSupported = (supportedInPrecisions.find(inPrecision) != supportedInPrecisions.end());
    bool outPrecSupported = (supportedOutPrecisions.find(outPrecision) != supportedOutPrecisions.end());
    bool inLayoutSupported = (supportedInLayouts.find(inLayout) != supportedInLayouts.end());
    bool outLayoutSupported = (supportedOutLayouts.find(outLayout) != supportedOutLayouts.end());
    bool compareWithReferenceSupported = outPrecision == Precision::FP16 || outPrecision == Precision::FP32 ||
                                         outPrecision == Precision::UNSPECIFIED;
    bool compareWithReferenceRequired = KmbTestBase::RUN_INFER;
    bool compareSupported = compareWithReferenceSupported || !compareWithReferenceRequired;

    return (inPrecSupported && outPrecSupported && inLayoutSupported && outLayoutSupported && compareSupported);
}

static std::vector<size_t> composeDimsByLayout(const Layout& layout) {
    std::vector<size_t> resultDims;
    switch(layout) {
    case Layout::NCHW:
    case Layout::NHWC:
    case Layout::OIHW:
        resultDims = {1, 3, 16, 16};
        break;
    case Layout::NCDHW:
    case Layout::NDHWC:
    case Layout::GOIHW:
    case Layout::OIDHW:
        resultDims = {1, 3, 16, 16, 1};
        break;
    case Layout::GOIDHW:
        resultDims = {1, 3, 16, 16, 1, 1};
        break;
    case Layout::C:
        resultDims = {3 * 16 * 16};
        break;
    case Layout::CHW:
        resultDims = {3, 16, 16};
        break;
    case Layout::HW:
        resultDims = {16 * 3, 16 * 3};
        break;
    case Layout::NC:
        resultDims = {1, 16 * 16 * 3};
        break;
    case Layout::CN:
        resultDims = {16 * 16 * 3, 1};
        break;
    default:
        resultDims = {1};
    }

    return resultDims;
}

TEST_P(KmbLayoutTests, SetUnsupportedLayout) {
    const auto& p = GetParam();
    Precision in_precision = std::get<0>(p);
    Precision out_precision = std::get<1>(p);
    Layout layout = std::get<2>(p);
    bool forceFP16ToFP32 = std::get<3>(p);
    std::vector<size_t> dims = composeDimsByLayout(layout);

    const auto netPrecision = Precision::FP32;

    const auto userInDesc = TensorDesc(in_precision, dims, layout);
    const auto userOutDesc = TensorDesc(out_precision, dims, layout);

    const auto inputRange = std::make_pair(0.0f, 1.0f);

    const auto tolerance = 1e-3f;  // obtained based on CPU plugin

    registerBlobGenerator(
            "input", userInDesc,
            [&](const TensorDesc& desc) {
                return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
            }
    );

    if (!is_supported(userInDesc.getPrecision(), userInDesc.getLayout(), userOutDesc.getPrecision(), userOutDesc.getLayout())) {
        SKIP_INFER_ON("KMB", "Parameters are not supported, no graph to infer");
    }

    const auto netBuidler = [&](TestNetwork& testNet) {
        if (forceFP16ToFP32) {
            testNet.setCompileConfig({{"VPU_KMB_FORCE_FP16_TO_FP32", CONFIG_VALUE(YES)}});
        } else {
            testNet.setCompileConfig({{"VPU_KMB_FORCE_FP16_TO_FP32", CONFIG_VALUE(NO)}});
        }
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), netPrecision)
            .addLayer<SoftmaxLayerDef>("softmax", dims.size() > 1 ? 1 : 0)
                .input("input", 0)
                .build()
            .setUserOutput(PortInfo("softmax"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .addNetOutput(PortInfo("softmax"))
            .finalize();
    };

    if (is_supported(userInDesc.getPrecision(), userInDesc.getLayout(), userOutDesc.getPrecision(), userOutDesc.getLayout())) {
        ASSERT_NO_THROW(runTest(netBuidler, tolerance, CompareMethod::Absolute));
    } else {
        ASSERT_THROW(runTest(netBuidler, tolerance, CompareMethod::Absolute), details::InferenceEngineException);
    }
}

static const std::vector<Layout> all_layouts = { Layout::NCHW, Layout::NHWC, Layout::NCDHW, Layout::NDHWC,
    Layout::OIHW, Layout::GOIHW, Layout::OIDHW, Layout::GOIDHW, Layout::SCALAR, Layout::C, Layout::CHW,
    Layout::HW, Layout::NC, Layout::CN };

static const std::vector<Precision> all_precisions = {Precision::UNSPECIFIED, Precision::MIXED,
    Precision::FP32, Precision::FP16, Precision::Q78, Precision::I16, Precision::I8, Precision::U8, Precision::U16,
    Precision::I32, Precision::I64, Precision::U64, Precision::BIN, Precision::BOOL, Precision::CUSTOM};

static auto checkInputPrecisions = ::testing::Combine(::testing::ValuesIn(all_precisions),
    ::testing::Values(Precision::FP16), ::testing::Values(Layout::NHWC), ::testing::Values(true));

static auto checkOutputPrecisions = ::testing::Combine(::testing::Values(Precision::U8),
    ::testing::ValuesIn(all_precisions), ::testing::Values(Layout::NHWC), ::testing::Values(true));

static auto checkLayouts = ::testing::Combine(::testing::Values(Precision::U8),
    ::testing::Values(Precision::FP16), ::testing::ValuesIn(all_layouts), ::testing::Values(true));

static auto checkOutputPrecisionsForceFP16 = ::testing::Combine(::testing::Values(Precision::U8),
    ::testing::ValuesIn(all_precisions), ::testing::Values(Layout::NHWC), ::testing::Values(false));

INSTANTIATE_TEST_CASE_P(InPrecisions, KmbLayoutTests, checkInputPrecisions);
INSTANTIATE_TEST_CASE_P(OutPrecisions, KmbLayoutTests, checkOutputPrecisions);
INSTANTIATE_TEST_CASE_P(Layouts, KmbLayoutTests, checkLayouts);
INSTANTIATE_TEST_CASE_P(OutPrecisionsForceFP16, KmbLayoutTests, checkOutputPrecisionsForceFP16);
