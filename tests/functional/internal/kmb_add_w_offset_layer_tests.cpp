//
// Copyright 2021 Intel Corporation.
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

#include <vpu/utils/ie_helpers.hpp>
#include <file_utils.h>
#include "test_model/kmb_test_base.hpp"
#include "test_model/kmb_test_add_w_offset_def.hpp"

struct AddWOffsetTestParams final {
    AddWOffsetParams params;
    LAYER_PARAMETER(float, offset);
    PARAMETER(SizeVector, dims);
};

std::ostream& operator<<(std::ostream& os, const AddWOffsetTestParams& p) {
    vpu::formatPrint(
        os, "offset: %l", p.offset());
    return os;
}

class KmbAddWOffsetLayerTests : public KmbLayerTestBase, public testing::WithParamInterface<AddWOffsetTestParams> {};

TEST_P(KmbAddWOffsetLayerTests, accuracy) {
    const auto &p = GetParam();

    const auto& dims = p.dims();
    const auto userIn1Desc = TensorDesc(Precision::FP16, dims, Layout::NCHW);
    const auto userIn2Desc = TensorDesc(Precision::FP16, dims, Layout::NCHW);
    const auto userOutDesc = TensorDesc(Precision::FP32, dims, Layout::NCHW);

    const auto inputRange = std::make_pair(0.0f, 10.0f);

    const auto tolerance = 1e-2f;

    registerBlobGenerator("input1", userIn1Desc, [&](const TensorDesc& desc) {
        return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
    });
    registerBlobGenerator("input2", userIn2Desc, [&](const TensorDesc& desc) {
        return genBlobUniform(desc, rd, inputRange.first, inputRange.second);
    });

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input1", userIn1Desc.getPrecision(), userIn1Desc.getLayout())
            .setUserInput("input2", userIn2Desc.getPrecision(), userIn2Desc.getLayout())
            .addNetInput("input1", userIn1Desc.getDims(), Precision::FP32)
            .addNetInput("input2", userIn2Desc.getDims(), Precision::FP32)
            .addLayer<AddWOffsetLayerDef>("add_with_offset", p.params)
                .input("input1", "input2")
                .build()
            .addNetOutput(PortInfo("add_with_offset"))
            .setUserOutput(PortInfo("add_with_offset"), userOutDesc.getPrecision(), userOutDesc.getLayout())
            .useCustomLayers(getIELibraryPath() + "/kmb_custom_extension_library/sampleExtensionOclLayerBindings.xml")
            .useExtension(getIELibraryPath() + "/libcustom_extension_library.so")
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<AddWOffsetTestParams> convertParams = {
        AddWOffsetTestParams()
            .dims({1, 3, 2, 3})
            .offset(1.2)
};
// Test disabled on Windows because OCL kernels compilation is not unblocked yet
#if !defined(_WIN32)
INSTANTIATE_TEST_CASE_P(precommit, KmbAddWOffsetLayerTests, testing::ValuesIn(convertParams));
#endif
