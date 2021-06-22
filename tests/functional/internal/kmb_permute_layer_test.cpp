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

#include <blob_factory.hpp>

struct PermuteTestParams final {
    TensorDesc in_desc_;
    std::vector<int64_t> order_;
    std::string permute_nd_ = InferenceEngine::PluginConfigParams::NO;

    PermuteTestParams& in_desc(TensorDesc in_desc) {
        this->in_desc_ = std::move(in_desc);
        return *this;
    }

    PermuteTestParams& order(std::vector<int64_t> order) {
        this->order_= std::move(order);
        return *this;
    }

    PermuteTestParams& permute_nd(const std::string& permute_nd) {
        this->permute_nd_= permute_nd;
        return *this;
    }
};

std::ostream& operator<<(std::ostream& os, const PermuteTestParams& p) {
    vpu::formatPrint(os, "[in_dims:%v, in_precision:%v, in_layout:%v, order:%v, permute_nd:%s]",
            p.in_desc_.getDims(), p.in_desc_.getPrecision(), p.in_desc_.getLayout(), p.order_,
            p.permute_nd_);
    return os;
}

class KmbPermuteLayerTests : public KmbLayerTestBase, public testing::WithParamInterface<PermuteTestParams> {};

TEST_P(KmbPermuteLayerTests, Accuracy) {
    const auto &p = GetParam();

    const size_t num_dims = p.in_desc_.getDims().size();
    IE_ASSERT(p.order_.size() == num_dims && "Order size must match the size of the input dimensions");
    if (num_dims >= 5) {
        SKIP_INFER_ON("KMB", "HDDL2", "VPUX", "Inference does not yet support 5-d inputs and outputs");
    }

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

    const auto keepNoopPermute = CompileConfig{
        { VPU_COMPILER_CONFIG_KEY(REMOVE_PERMUTE_NOOP), InferenceEngine::PluginConfigParams::NO},
        { "VPU_COMPILER_ALLOW_PERMUTE_ND", p.permute_nd_},
    };

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
            .setCompileConfig(keepNoopPermute)
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
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 3, 10, 5}, Layout::NCHW})
        .order({0, 1, 2, 3}),
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 3, 10, 5}, Layout::NHWC})
        .order({0, 1, 2, 3}),
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 3, 10, 5}, Layout::NCHW})
        .order({0, 2, 1, 3}),
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 18, 19, 19}, Layout::NHWC})
        .order({0, 2, 3, 1}),
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 18, 19, 19}, Layout::NCHW})
        .order({0, 2, 3, 1}),

    ///// use permuteND
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 3, 10, 5}, Layout::NCHW})
        .order({0, 1, 2, 3})
        .permute_nd(InferenceEngine::PluginConfigParams::YES),
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 13, 13, 3, 85}, Layout::NCDHW})
        .order({0, 1, 2, 4, 3})
        .permute_nd(InferenceEngine::PluginConfigParams::YES),
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 3, 10, 5}, Layout::NCHW})
        .order({0, 2, 3, 1})
        .permute_nd(InferenceEngine::PluginConfigParams::YES),
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 3, 10, 5}, Layout::NCHW})
        .order({0, 3, 2, 1})
        .permute_nd(InferenceEngine::PluginConfigParams::YES),
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 3, 10, 5}, Layout::NCHW})
        .order({0, 3, 1, 2})
        .permute_nd(InferenceEngine::PluginConfigParams::YES),
    PermuteTestParams()
        .in_desc(TensorDesc{Precision::FP16, {1, 3, 10, 5}, Layout::NCHW})
        .order({0, 1, 3, 2})
        .permute_nd(InferenceEngine::PluginConfigParams::YES),
};

INSTANTIATE_TEST_SUITE_P(precommit_SupportedCases, KmbPermuteLayerTests, testing::ValuesIn(supportedCases));
INSTANTIATE_TEST_SUITE_P(precommit_FaceDetectionRetail, KmbPermuteLayerTests, testing::ValuesIn(FaceDetectionRetailCases));

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
INSTANTIATE_TEST_SUITE_P(DISABLED_precommit_StressTest,
                        KmbPermuteLayerTests,
                        testing::ValuesIn(getTestCases(TensorDesc{Precision::FP16, {1, 3, 10, 5}, Layout::NCHW},
                                                       genPermutations({0, 1, 2, 3}))));


/* FIXME: What about testing on different types and layouts ?
 * It can be represented like this:
 INSTANTIATE_TEST_SUITE_P(StressTest, KmbPermuteLayerTests,
                         Combine(Values({1,3,10,5}),
                                 Values(Layout::NCHW, Layout::NHWC),
                                 Values(Precision::FP32, Precision::FP16),
                                 Values(genPermutations({0,1,2,3})))
 */
