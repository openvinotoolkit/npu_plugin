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

#include <blob_factory.hpp>

struct ReshapeTestParams final {
    SizeVector in_dims_;
    SizeVector shape_;

    ReshapeTestParams& in_dims(SizeVector in_dims) {
        in_dims_ = std::move(in_dims);
        return *this;
    }

    ReshapeTestParams& shape(SizeVector shape) {
        shape_ = std::move(shape);
        return *this;
    }
};

std::ostream& operator<<(std::ostream& os, const ReshapeTestParams& p) {
    vpu::formatPrint(os, "[in_dims:%v, shape:%v]", p.in_dims_, p.shape_);
    return os;
}

class KmbReshapeLayerTests : public KmbLayerTestBase,
                             public testing::WithParamInterface<ReshapeTestParams> {};

TEST_P(KmbReshapeLayerTests, AccuracyTest) {
    const auto& p = GetParam();
    const auto precision = Precision::FP32;

    const auto input_desc  = TensorDesc(Precision::FP16, p.in_dims_,
                                        TensorDesc::getLayoutByDims(p.in_dims_));
    const auto shape_desc  = TensorDesc(Precision::I64, {p.shape_.size()}, Layout::C);
    const auto output_desc = TensorDesc(Precision::FP16, p.shape_,
                                        TensorDesc::getLayoutByDims(p.shape_));

    const auto range = std::make_pair(0.0f, 1.0f);
    const auto tolerance = 1e-3f;

    registerBlobGenerator(
        "input", input_desc,
        [&](const TensorDesc& desc) {
            return genBlobUniform(desc, rd, range.first, range.second);
        }
    );

    registerBlobGenerator(
        "shape", shape_desc,
        [&](const TensorDesc& desc) {
            auto blob = make_blob_with_precision(desc);
            blob->allocate();
            CopyVectorToBlob(blob, p.shape_);
            return blob;
        }
    );

    const auto builder = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", input_desc.getPrecision(), input_desc.getLayout())
            .addNetInput("input", input_desc.getDims(), precision)
            .addConst("shape", getBlobByName("shape"))
            .addLayer<ReshapeLayerDef>("reshape")
                .input("input")
                .shape("shape")
                .build()
            .addNetOutput(PortInfo("reshape"))
            .setUserOutput(PortInfo("reshape"), output_desc.getPrecision(), output_desc.getLayout())
            .finalize();
    };

    runTest(builder, tolerance, CompareMethod::Absolute);
}

const std::vector<ReshapeTestParams> supportedReshapeParams {
    ReshapeTestParams()
        .in_dims({2048})
        .shape({1, 32, 32, 2}),

    ReshapeTestParams()
          .in_dims({4})
          .shape({1, 1, 2, 2}),

    ReshapeTestParams()
          .in_dims({1, 16})
          .shape({1, 4, 2, 2}),

    ReshapeTestParams()
         .in_dims({2, 2, 16})
         .shape({1, 4, 8, 2}),

    ReshapeTestParams()
          .in_dims({1, 2, 16})
          .shape({1, 4, 4, 2}),

    ReshapeTestParams()
          .in_dims({1, 4, 2, 2})
          .shape({1, 2, 4, 2}),

    ReshapeTestParams()
        .in_dims({1, 4, 2, 2})
        .shape({1, 16}),

    ReshapeTestParams()
        .in_dims({1, 4, 2, 4})
        .shape({32}),

    ReshapeTestParams()
        .in_dims({1, 2, 2})
        .shape({4}),

    ReshapeTestParams()
        .in_dims({1, 8, 2, 4})
        .shape({1, 8, 8}),

    ReshapeTestParams()
        .in_dims({1, 8, 2, 4})
        .shape({2, 4, 8}),

    ReshapeTestParams()
        .in_dims({8, 2, 4})
        .shape({1, 8, 8}),

    ReshapeTestParams()
        .in_dims({8, 2, 4})
        .shape({2, 4, 8}),

    ReshapeTestParams()
        .in_dims({1, 32})
        .shape({2, 2, 8}),

    ReshapeTestParams()
        .in_dims({1, 32})
        .shape({1, 16, 2}),

    ReshapeTestParams()
        .in_dims({1, 32})
        .shape({8, 2, 2}),
};

const std::vector<ReshapeTestParams> unsupportedReshapeParams {
    /* FIXME: "Flic NN doesn't support batch not equal to one"
     * [Track number: H#18011923106] */
    ReshapeTestParams()
        .in_dims({8, 2, 4})
        .shape({4, 4, 4}),

    ReshapeTestParams()
        .in_dims({2048})
        .shape({32, 64}),

    ReshapeTestParams()
        .in_dims({2048})
        .shape({32, 32, 2}),

    ReshapeTestParams()
        .in_dims({32, 64})
        .shape({8, 4, 64}),

    ReshapeTestParams()
        .in_dims({2, 4})
        .shape({1, 8}),

    ReshapeTestParams()
        .in_dims({2048})
        .shape({8, 4, 32, 2}),

    /* FIXME: Hangs when input has the same dimensions as the output
     * Even if they have different orders */
    ReshapeTestParams()
        .in_dims({1, 1, 1, 4})
        .shape({1, 1, 1, 4}),

    ReshapeTestParams()
        .in_dims({1, 1, 1, 4})
        .shape({1, 1, 4, 1}),

    ReshapeTestParams()
        .in_dims({1, 4})
        .shape({1, 4}),

    ReshapeTestParams()
        .in_dims({4})
        .shape({4}),
};

INSTANTIATE_TEST_CASE_P(precommit, KmbReshapeLayerTests, testing::ValuesIn(supportedReshapeParams));
INSTANTIATE_TEST_CASE_P(DISABLED_precommit, KmbReshapeLayerTests, testing::ValuesIn(unsupportedReshapeParams));
