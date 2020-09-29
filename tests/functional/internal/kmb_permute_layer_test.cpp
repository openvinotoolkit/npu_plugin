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

struct PermuteTestParams final {
    TensorDesc in_desc_;
    std::vector<int64_t> order_;

    PermuteTestParams& in_desc(TensorDesc in_desc) {
        this->in_desc_ = std::move(in_desc);
        return *this;
    }

    PermuteTestParams& order(std::vector<int64_t> order) {
        this->order_= std::move(order);
        return *this;
    }
};

std::ostream& operator<<(std::ostream& os, const PermuteTestParams& p) {
    vpu::formatPrint(os, "[in_dims:%v, in_precision:%v, in_layout:%v, order:%v]",
            p.in_desc_.getDims(), p.in_desc_.getPrecision(), p.in_desc_.getLayout(), p.order_);
    return os;
}

class KmbPermuteLayerTests : public KmbLayerTestBase, public testing::WithParamInterface<PermuteTestParams> {};

TEST_P(KmbPermuteLayerTests, Accuracy) {
    const auto &p = GetParam();

    const size_t num_dims = p.in_desc_.getDims().size();
    IE_ASSERT(num_dims == 4 && "Only 4D tensor is available for permute");
    IE_ASSERT(p.order_.size() == num_dims && "Order size must match the size of the input dimensions");

    const auto output_desc = TensorDesc(p.in_desc_.getPrecision(), p.in_desc_.getLayout());
    const auto order_desc = TensorDesc(Precision::I64, {p.order_.size()}, Layout::C);

    const auto range = std::make_pair(0.0f, 10.0f);
    const auto tolerance = 0.f;

    registerBlobGenerator(
        "input", p.in_desc_,
        [&](const TensorDesc& desc) {
            return genBlobUniform(desc, rd, range.first, range.second);
        }
    );

    registerBlobGenerator(
        "order", order_desc,
        [&](const TensorDesc& desc) {
            auto blob = make_blob_with_precision(desc);
            blob->allocate();
            CopyVectorToBlob(blob, p.order_);
            return blob;
        }
    );

    const auto netBuidler = [&](TestNetwork& testNet) {
        testNet
            .setUserInput("input", p.in_desc_.getPrecision(), p.in_desc_.getLayout())
            .addNetInput("input", p.in_desc_.getDims(), Precision::FP32)
            .addConst("order", getBlobByName("order"))
            .addLayer<PermuteLayerDef>("permute")
                .input("input")
                .order("order")
                .build()
            .addNetOutput(PortInfo("permute"))
            .setUserOutput(PortInfo("permute"), output_desc.getPrecision(), output_desc.getLayout())
            .finalize();
    };

    runTest(netBuidler, tolerance, CompareMethod::Absolute);
}

const std::vector<PermuteTestParams> FaceDetectionRetailCases {
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 36, 19, 19}, Layout::NCHW})
        .order({0, 2, 3, 1}),
};

const std::vector<PermuteTestParams> supportedCases {
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 3, 10, 5}, Layout::NCHW})
        .order({0, 2, 3, 1}),
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 3, 10, 5}, Layout::NCHW})
        .order({0, 3, 2, 1}),
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 3, 10, 5}, Layout::NCHW})
        .order({0, 3, 1, 2}),
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 3, 10, 5}, Layout::NCHW})
        .order({0, 1, 3, 2}),
};

// NB: Please note that these test cases doesn't include cases when batch not equal to one
const std::vector<PermuteTestParams> unsupportedCases {
    // FIXME: Doesn't match with CPU
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 3, 10, 5}, Layout::NCHW})
        .order({0, 2, 1, 3}),
    // FIXME: It works fine for NCHW layout
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 18, 19, 19}, Layout::NHWC})
        .order({0, 2, 3, 1}),
    // FIXME: Disabled suppportedCase [Track number: D#3455]
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 3, 10, 5}, Layout::NCHW})
        .order({0, 1, 2, 3}),
    // FIXME: Disabled FaceDetectionRetailCase [Track number: D#3455]
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 18, 19, 19}, Layout::NCHW})
        .order({0, 2, 3, 1}),
};

INSTANTIATE_TEST_CASE_P(SupportedCases,            KmbPermuteLayerTests, testing::ValuesIn(supportedCases));
// [Track number: S-37612]
INSTANTIATE_TEST_CASE_P(DISABLED_FaceDetectionRetail, KmbPermuteLayerTests, testing::ValuesIn(FaceDetectionRetailCases));
INSTANTIATE_TEST_CASE_P(DISABLED_UnsupportedCases, KmbPermuteLayerTests, testing::ValuesIn(unsupportedCases));

static std::vector<std::vector<int64_t>> genPermutations(std::vector<int64_t> seq) {
    std::vector<std::vector<int64_t>> permutations;
    do {
        permutations.push_back(seq);
    } while (std::next_permutation(seq.begin(), seq.end()));

    return permutations;
}

static std::vector<PermuteTestParams> getTestCases(TensorDesc desc, std::vector<std::vector<int64_t>> order_vec) {
    std::vector<PermuteTestParams> cases;
    cases.reserve(order_vec.size());
    for (auto&& ord : order_vec) {
        cases.push_back(PermuteTestParams()
                .in_desc(desc)
                .order(std::move(ord)));
    }
    return cases;
}

// FIXME: Currently it's impossible to use tensor with batch not equal to one due:
// [Track number: H#18011923106]
// Enable it when this problem & unsupportedCases are fixed
INSTANTIATE_TEST_CASE_P(DISABLED_StressTest,
                        KmbPermuteLayerTests,
                        testing::ValuesIn(getTestCases(TensorDesc{Precision::FP16, {1, 3, 10, 5}, Layout::NCHW},
                                                       genPermutations({0, 1, 2, 3}))));


/* FIXME: What about testing on different types and layouts ?
 * It can be represented like this:
 INSTANTIATE_TEST_CASE_P(StressTest, KmbPermuteLayerTests,
                         Combine(Values({1,3,10,5}),
                                 Values(Layout::NCHW, Layout::NHWC),
                                 Values(Precision::FP32, Precision::FP16),
                                 Values(genPermutations({0,1,2,3})))
 */
