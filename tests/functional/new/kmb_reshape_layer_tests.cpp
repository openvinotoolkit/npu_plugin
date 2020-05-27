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

TEST_P(KmbReshapeLayerTests, DISABLED_AccuracyTest) {
    const auto& p = GetParam();
    const auto precision = Precision::FP32;

    const auto input_desc  = TensorDesc(Precision::FP32, p.in_dims_,
                                        TensorDesc::getLayoutByDims(p.in_dims_));
    const auto shape_desc  = TensorDesc(Precision::I64, {p.shape_.size()}, Layout::C);
    const auto output_desc = TensorDesc(Precision::FP32, p.shape_,
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

const std::vector<ReshapeTestParams> reshapeParams {
    // FIXME: Input data type is not supported: FP32 [kmb plugin]
    ReshapeTestParams()
        .in_dims({1, 4, 2, 2})
        .shape({1, 2, 4, 2}),
    // FIXME: Output layout is not supported: CHW [kmb plugin]
    ReshapeTestParams()
        .in_dims({1, 4, 2, 2})
        .shape({1, 8, 2}),
    // FIXME: Input data type is not supported: FP32 [kmb plugin]
    ReshapeTestParams()
        .in_dims({1, 4, 2, 2})
        .shape({1, 16}),
    // FIXME: Output layout is not supported: C [kmb plugin]
    ReshapeTestParams()
        .in_dims({2, 4, 2, 2})
        .shape({32}),
    // FIXME: Input layout is not supported: CHW [kmb plugin]
    ReshapeTestParams()
        .in_dims({8, 2, 4})
        .shape({4, 4, 4}),
    // FIXME: Input layout is not supported: NC [kmb plugin]
    ReshapeTestParams()
        .in_dims({2, 4})
        .shape({1, 8}),
    // FIXME: Input layout is not supported: C [kmb plugin]
    ReshapeTestParams()
        .in_dims({2048})
        .shape({32, 64}),
    // FIXME: Input layout is not supported: C [kmb plugin]
    ReshapeTestParams()
        .in_dims({2048})
        .shape({32, 32, 2}),
    // FIXME: Input layout is not supported: C [kmb plugin]
    ReshapeTestParams()
        .in_dims({2048})
        .shape({8, 4, 32, 2}),
    // FIXME: Input layout is not supported: NC [kmb plugin]
    ReshapeTestParams()
        .in_dims({32, 64})
        .shape({8, 4, 64}),
    // FIXME: Input layout is not supported: CHW [kmb plugin]
    ReshapeTestParams()
        .in_dims({16, 2, 32})
        .shape({8, 4, 2, 16}),
};

INSTANTIATE_TEST_CASE_P(SomeCase, KmbReshapeLayerTests, testing::ValuesIn(reshapeParams));
