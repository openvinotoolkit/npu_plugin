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

class KmbLayoutTests : public KmbLayerTestBase,
    public testing::WithParamInterface<std::tuple<Precision, Precision, Layout, bool>> {};

static const std::set<Precision> supportedInPrecisions = { Precision::U8, Precision::FP16, Precision::FP32 };
static const std::set<Precision> supportedOutPrecisions = { Precision::U8, Precision::FP16, Precision::FP32, Precision::I32 };
static const std::set<Layout> supportedInLayouts = { Layout::NHWC, Layout::NCHW, Layout::CHW, Layout::NC, Layout::C };
static const std::set<Layout> supportedOutLayouts = { Layout::NHWC, Layout::NCHW, Layout::CHW, Layout::NC, Layout::C };

static bool is_supported(const Precision& inPrecision,
                         const Layout& inLayout,
                         const Precision& outPrecision,
                         const Layout& outLayout,
                         bool compareWithReferenceRequired) {
    bool inPrecSupported = (supportedInPrecisions.find(inPrecision) != supportedInPrecisions.end());
    bool outPrecSupported = (supportedOutPrecisions.find(outPrecision) != supportedOutPrecisions.end());
    bool inLayoutSupported = (supportedInLayouts.find(inLayout) != supportedInLayouts.end());
    bool outLayoutSupported = (supportedOutLayouts.find(outLayout) != supportedOutLayouts.end());
    bool compareWithReferenceSupported = outPrecision == Precision::FP16 || outPrecision == Precision::FP32;
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

// [Track number: D#49269, E#8151]
TEST_P(KmbLayoutTests, DISABLED_SetUnsupportedLayout) {
#ifdef _WIN32
    GTEST_SKIP() << "SEH exception";
#endif
    const auto& p = GetParam();
    Precision in_precision = std::get<0>(p);
    Precision out_precision = std::get<1>(p);
    Layout layout = std::get<2>(p);
    bool useIncorrectInputLayout = std::get<3>(p);
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

    const std::vector<size_t> powerTensorDims(dims.size(), 1);
    Layout powerTensorLayout = layout;
    if (powerTensorDims.size() == 4) {
        powerTensorLayout = useIncorrectInputLayout ? Layout::NCHW : Layout::NHWC;
    }
    const auto powerTensorDesc = TensorDesc(Precision::FP32, powerTensorDims, powerTensorLayout);
    registerBlobGenerator(
            "scale", powerTensorDesc,
            [&](const TensorDesc& desc) {
                return vpux::makeSplatBlob(desc, 1.f);
            }
    );

    if (!is_supported(userInDesc.getPrecision(),
                      userInDesc.getLayout(),
                      userOutDesc.getPrecision(),
                      userOutDesc.getLayout(),
                      KmbTestBase::RUN_INFER)) {
        SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "Parameters are not supported, no graph to infer");
    }

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
            .addNetInput("input", userInDesc.getDims(), netPrecision)
            .addLayer<PowerLayerDef>("power")
                .input1("input")
                .input2(getBlobByName("scale"))
                .build()
            .setUserOutput(PortInfo("power"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .addNetOutput(PortInfo("power"))
            .finalize();
    };

    if (is_supported(userInDesc.getPrecision(),
                     userInDesc.getLayout(),
                     userOutDesc.getPrecision(),
                     userOutDesc.getLayout(),
                     KmbTestBase::RUN_INFER)) {
        // FIXME: Power doesn't work with FP precision (hangs in runtime)
        // [Track number: S#31382]
        if (userInDesc.getPrecision() != Precision::FP16 && userInDesc.getPrecision() != Precision::FP32) {
            ASSERT_NO_THROW(runTest(netBuidler, tolerance, CompareMethod::Absolute));
        }
    } else {
        ASSERT_THROW(runTest(netBuidler, tolerance, CompareMethod::Absolute), Exception);
    }
}

static const std::vector<Layout> all_layouts = { Layout::NCHW, Layout::NHWC, Layout::NCDHW, Layout::NDHWC,
    Layout::OIHW, Layout::GOIHW, Layout::OIDHW, Layout::GOIDHW, Layout::SCALAR, Layout::C, Layout::CHW,
    Layout::HW, Layout::NC, Layout::CN };

static const std::vector<Precision> all_precisions = {Precision::UNSPECIFIED, Precision::MIXED,
    Precision::FP32, Precision::FP16, Precision::Q78, Precision::I16, Precision::I8, Precision::U8, Precision::U16,
    Precision::I32, Precision::I64, Precision::U64, Precision::BIN, Precision::BOOL, Precision::CUSTOM};

static auto checkInputPrecisions = ::testing::Combine(::testing::ValuesIn(all_precisions),
    ::testing::Values(Precision::FP16), ::testing::Values(Layout::NHWC), ::testing::Values(false));

static auto checkOutputPrecisions = ::testing::Combine(::testing::Values(Precision::U8),
    ::testing::ValuesIn(all_precisions), ::testing::Values(Layout::NHWC), ::testing::Values(false));

static auto checkLayouts = ::testing::Combine(::testing::Values(Precision::U8),
    ::testing::Values(Precision::FP16), ::testing::ValuesIn(all_layouts), ::testing::Values(false));

static auto checkIncorrectLayouts = ::testing::Combine(::testing::Values(Precision::U8),
    ::testing::Values(Precision::FP16), ::testing::ValuesIn(all_layouts), ::testing::Values(true));

static auto checkOutputPrecisionsForceFP16 = ::testing::Combine(::testing::Values(Precision::U8),
    ::testing::ValuesIn(all_precisions), ::testing::Values(Layout::NHWC), ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(precommit_InPrecisions, KmbLayoutTests, checkInputPrecisions);
INSTANTIATE_TEST_SUITE_P(precommit_OutPrecisions, KmbLayoutTests, checkOutputPrecisions);
INSTANTIATE_TEST_SUITE_P(precommit_Layouts, KmbLayoutTests, checkLayouts);
INSTANTIATE_TEST_SUITE_P(precommit_IncorrectLayouts, KmbLayoutTests, checkIncorrectLayouts);
INSTANTIATE_TEST_SUITE_P(precommit_OutPrecisionsForceFP16, KmbLayoutTests, checkOutputPrecisionsForceFP16);
