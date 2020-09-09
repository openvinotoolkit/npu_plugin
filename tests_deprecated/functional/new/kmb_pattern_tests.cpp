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

struct BatchNotEqualOneParams final {
    SizeVector shape0_;
    SizeVector shape1_;
    Precision prec_;
    SizeVector in_dims_;

    BatchNotEqualOneParams& in_dims(SizeVector in_dims) {
        in_dims_ = std::move(in_dims);
        return *this;
    }

    BatchNotEqualOneParams& shape0(SizeVector shape0) {
        shape0_ = std::move(shape0);
        return *this;
    }

    BatchNotEqualOneParams& shape1(SizeVector shape1) {
        shape1_ = std::move(shape1);
        return *this;
    }

    BatchNotEqualOneParams& precision(Precision prec) {
        prec_ = prec;
        return *this;
    }
};

class KmbPatternTests : public KmbLayerTestBase,
                        public testing::WithParamInterface<BatchNotEqualOneParams> {};

TEST_P(KmbPatternTests, BatchNotEqualToOneInTheMiddle) {
    /* FIXME: "Flic NN doesn't support batch not equal to one"
     * [Track number: H#18011923106] */
    SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "bad results");

    const auto& p = GetParam();

    const auto netPresicion = Precision::FP32;

    const auto input_desc  = TensorDesc(p.prec_, p.in_dims_, TensorDesc::getLayoutByDims(p.in_dims_));  
    const auto output_desc = TensorDesc(p.prec_, p.shape1_,  TensorDesc::getLayoutByDims(p.shape1_));
    const auto shape_desc0 = TensorDesc(Precision::I64,  {p.shape0_.size()},  Layout::C);
    const auto shape_desc1 = TensorDesc(Precision::I64,  {p.shape1_.size()},  Layout::C);

    const auto tolerance = 0.f;

    registerBlobGenerator(
        "input", input_desc,
        [&](const TensorDesc& desc) {
            return makeSingleValueBlob(desc, 1.0f);
        }
    );

    registerBlobGenerator(
        "shape0", shape_desc0,
        [&](const TensorDesc& desc) {
            auto blob = make_blob_with_precision(desc);
            blob->allocate();
            CopyVectorToBlob(blob, p.shape0_);
            return blob;
        }
    );

    registerBlobGenerator(
        "shape1", shape_desc1,
        [&](const TensorDesc& desc) {
            auto blob = make_blob_with_precision(desc);
            blob->allocate();
            CopyVectorToBlob(blob, p.shape1_);
            return blob;
        }
    );

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", input_desc.getPrecision(), input_desc.getLayout())
            .addNetInput("input", input_desc.getDims(), netPresicion)
            .addConst("shape0", getBlobByName("shape0"))
            .addConst("shape1", getBlobByName("shape1"))
            .addLayer<ReshapeLayerDef>("reshape0") 
                .input("input")
                .shape("shape0")
                .build()
            .addLayer<ReshapeLayerDef>("reshape1") 
                .input("reshape0")
                .shape("shape1")
                .build()
            .addNetOutput(PortInfo("reshape1"))
            .setUserOutput(PortInfo("reshape1"), output_desc.getPrecision(), output_desc.getLayout())
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute); 
}

const std::vector<BatchNotEqualOneParams> params = {
    BatchNotEqualOneParams().in_dims({1, 2, 4, 4})
                            .shape0({2, 4, 4, 1})
                            .shape1({1, 4, 2, 4})
                            .precision(Precision::FP16)
};

INSTANTIATE_TEST_CASE_P(BatchNotEqualOneTest, KmbPatternTests, testing::ValuesIn(params));
