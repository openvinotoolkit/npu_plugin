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

#include <random>

#include "test_model/kmb_test_base.hpp"

#include <blob_factory.hpp>

struct ROIPoolingTestParams final {
    SizeVector input_dims_;
    SizeVector coords_dims_;
    ROIPoolingParams params_;

    ROIPoolingTestParams& input_dims(SizeVector input_dims) {
        input_dims_ = std::move(input_dims);
        return *this;
    }

    ROIPoolingTestParams& coords_dims(SizeVector coords_dims) {
        coords_dims_ = std::move(coords_dims);
        return *this;
    }

    ROIPoolingTestParams& params(ROIPoolingParams params) {
        params_ = std::move(params);
        return *this;
    }
};

std::ostream& operator<<(std::ostream& os, const ROIPoolingTestParams& p) {
        vpu::formatPrint(os, "[input_dims:%v, coords_dims:%v, params:%v]",
                p.input_dims_, p.coords_dims_, p.params_);
	return os;
}

namespace {

Blob::Ptr generateCoords(const TensorDesc& desc, size_t width, size_t height) {
    auto blob = make_blob_with_precision(desc);
    blob->allocate();
    auto* output = blob->buffer().as<float*>();
    const size_t num_rois = desc.getDims()[0];

    const int max_range_width  = width  * 4 / 5;
    const int max_range_height = height * 4 / 5;

    std::mt19937 gen(0u);
    std::uniform_int_distribution<int> dist(0, std::max(max_range_width, max_range_height));

    for (size_t i = 0; i < num_rois; ++i)
    {
        int x0 = dist(gen) % max_range_width;
        int x1 = x0 + (dist(gen) % (width - x0 - 1)) + 1;
        int y0 = dist(gen) % max_range_height;
        int y1 = y0 + (dist(gen) % (height - y0 - 1)) + 1;

        output[i * 5 + 0] = 0;
        output[i * 5 + 1] = x0;
        output[i * 5 + 2] = y0;
        output[i * 5 + 3] = x1;
        output[i * 5 + 4] = y1;
    }

    return blob;
}

}

class KmbROIPoolingLayerTests : public KmbLayerTestBase,
                                public testing::WithParamInterface<ROIPoolingTestParams> {};

// [Track number: D#36225]
TEST_P(KmbROIPoolingLayerTests, AccuracyTest) {
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "bad results");

    const auto& p = GetParam();

    const auto precision = Precision::FP32;

    const auto in_desc     = TensorDesc(Precision::FP32, p.input_dims_,  Layout::NHWC);
    const auto coords_desc = TensorDesc(Precision::FP32, p.coords_dims_, Layout::NC);
    const auto out_desc    = TensorDesc(Precision::FP32, Layout::NHWC);

    const auto tolerance = 1e-3f;

    registerBlobGenerator(
            "input", in_desc,
            [&](const TensorDesc& desc) {
                const auto range = std::make_pair(0.0f, 3.0f);
                return genBlobUniform(desc, rd, range.first, range.second);
            }
            );

    registerBlobGenerator(
            "coords", coords_desc,
            [&](const TensorDesc& desc) {
                const size_t image_height = p.input_dims_[2] / p.params_.spatial_scale_;
                const size_t image_width  = p.input_dims_[3] / p.params_.spatial_scale_;
                return generateCoords(desc, image_height, image_width);
            }
            );

    const auto builder = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", in_desc.getPrecision(), in_desc.getLayout())
            .addNetInput("input", in_desc.getDims(), precision)
            .setUserInput("coords", coords_desc.getPrecision(), coords_desc.getLayout())
            .addNetInput("coords", coords_desc.getDims(), precision)
            .addLayer<ROIPoolingLayerDef>("roi", p.params_)
                .input("input")
                .coords("coords")
                .build()
            .addNetOutput(PortInfo("roi"))
            .setUserOutput(PortInfo("roi"), out_desc.getPrecision(), out_desc.getLayout())
            .finalize();
    };

    runTest(builder, tolerance, CompareMethod::Absolute);
}

const std::vector<ROIPoolingTestParams> roiPoolingParams {
    ROIPoolingTestParams()
        .input_dims({1, 392, 14, 14})
        .coords_dims({300, 5})
        .params(ROIPoolingParams().pooled_w(7).pooled_h(7).spatial_scale(0.0625f).mode("max")),
    ROIPoolingTestParams()
        .input_dims({1, 1029, 14, 14})
        .coords_dims({300, 5})
        .params(ROIPoolingParams().pooled_w(7).pooled_h(7).spatial_scale(0.0625f).mode("bilinear")),
    ROIPoolingTestParams()
        .input_dims({1, 32, 14, 14})
        .coords_dims({5, 5})
        .params(ROIPoolingParams().pooled_w(7).pooled_h(7).spatial_scale(0.0625f).mode("bilinear"))
};

INSTANTIATE_TEST_SUITE_P(precommit, KmbROIPoolingLayerTests, testing::ValuesIn(roiPoolingParams));
