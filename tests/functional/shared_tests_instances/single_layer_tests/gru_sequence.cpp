//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/gru_sequence.hpp"
#include <vector>
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "vpu_ov1_layer_test.hpp"

using namespace ngraph::helpers;

namespace LayerTestsDefinitions {

class VPUXGRUSequenceLayerTest : public GRUSequenceTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void GenerateInputs() override {
        SequenceTestsMode mode;
        size_t seqLengths;
        size_t batchSize;
        size_t hiddenSize;
        std::vector<std::string> activations;
        float clip;
        bool linear_before_reset;
        ngraph::op::RecurrentSequenceDirection direction;
        InferenceEngine::Precision netPrecision;
        std::tie(mode, seqLengths, batchSize, hiddenSize, activations, clip, linear_before_reset, direction,
                 netPrecision, targetDevice) = this->GetParam();
        inputs.clear();
        for (const auto& input : executableNetwork.GetInputsInfo()) {
            const auto& info = input.second;
            auto blob = GenerateInput(*info);
            // Avoid default initialization in OV. The data value is too large for GRU, and it does not validate
            // anything.
            blob = FuncTestUtils::createAndFillBlobFloatNormalDistribution(info->getTensorDesc(), 0, 1);
            if (input.first == "seq_lengths") {
                blob = FuncTestUtils::createAndFillBlob(info->getTensorDesc(), m_max_seq_len, 0);
            }
            inputs.push_back(blob);
        }
    }

    void SetUp() override {
        inPrc = InferenceEngine::Precision::FP16;
        outPrc = InferenceEngine::Precision::FP16;
        size_t seq_lengths;
        size_t batch;
        size_t hidden_size;

        std::vector<std::string> activations;
        std::vector<float> activations_alpha;
        std::vector<float> activations_beta;
        size_t input_size = 10;
        float clip;
        bool linear_before_reset;
        ngraph::op::RecurrentSequenceDirection direction;
        InferenceEngine::Precision netPrecision;
        std::tie(m_mode, seq_lengths, batch, hidden_size, activations, clip, linear_before_reset, direction,
                 netPrecision, targetDevice) = this->GetParam();
        size_t num_directions = direction == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;
        std::vector<std::vector<size_t>> inputShapes = {
                {{batch, seq_lengths, input_size},
                 {batch, num_directions, hidden_size},
                 {batch},
                 {num_directions, 3 * hidden_size, input_size},
                 {num_directions, 3 * hidden_size, hidden_size},
                 {num_directions, (linear_before_reset ? 4 : 3) * hidden_size}},
        };
        m_max_seq_len = seq_lengths;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShapes[0], inputShapes[1]});
        if (m_mode == SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM ||
            m_mode == SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM ||
            m_mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM) {
            auto seq_lengths_param = ngraph::builder::makeParams(ngraph::element::i64, {inputShapes[2]}).at(0);
            seq_lengths_param->set_friendly_name("seq_lengths");
            params.push_back(seq_lengths_param);
        }

        std::vector<ngraph::Shape> constants = {inputShapes[3], inputShapes[4], inputShapes[5], inputShapes[2]};
        auto in = convert2OutputVector(castOps2Nodes(params));
        std::vector<float> empty;
        std::shared_ptr<ov::Node> gru_sequence;
        // Avoid default initialization in OV. The data value is too large for GRU, and it does not validate anything.
        auto W = ngraph::builder::makeConstant(in[0].get_element_type(), constants[0], empty, true, 1.0f, 0.0f);
        auto R = ngraph::builder::makeConstant(in[0].get_element_type(), constants[1], empty, true, 1.0f, 0.0f);
        auto B = ngraph::builder::makeConstant(in[0].get_element_type(), constants[2], empty, true, 1.0f, 0.0f);

        if (in.size() > 2 && in[2].get_partial_shape().is_dynamic()) {
            gru_sequence = std::make_shared<ov::op::v5::GRUSequence>(in[0], in[1], in[2], W, R, B, hidden_size,
                                                                     direction, activations, activations_alpha,
                                                                     activations_beta, clip, linear_before_reset);
        } else {
            std::shared_ptr<ov::Node> seq_lengths_node;
            switch (m_mode) {
            case SequenceTestsMode::PURE_SEQ:
            case SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_CONST: {
                std::vector<float> lengths(in[0].get_partial_shape()[0].get_min_length(),
                                           in[0].get_partial_shape()[1].get_min_length());
                seq_lengths_node = ngraph::builder::makeConstant(ngraph::element::i64, constants[3], lengths, false);
                break;
            }
            case SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_CONST:
            case SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST: {
                for (size_t i = 0; i <= in[0].get_shape().at(0); ++i) {
                    std::vector<float> lengths;
                    seq_lengths_node = ngraph::builder::makeConstant(ngraph::element::i64, constants[3], lengths, true,
                                                                     static_cast<float>(in[0].get_shape()[1]), 0.f);
                }
                break;
            }
            case SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM:
            case SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM:
            case SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM: {
                // Seq_lengths should be as a Parameter node for these two modes
                seq_lengths_node = in.at(2).get_node_shared_ptr();
                break;
            }
            default:
                throw std::runtime_error("Incorrect mode for creation of Sequence operation");
            }
            gru_sequence = std::make_shared<ov::op::v5::GRUSequence>(
                    in[0], in[1], seq_lengths_node, W, R, B, hidden_size, direction, activations, activations_alpha,
                    activations_beta, clip, linear_before_reset);
        }

        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gru_sequence->output(0)),
                                     std::make_shared<ngraph::opset1::Result>(gru_sequence->output(1))};
        function = std::make_shared<ngraph::Function>(results, params, "gru_sequence");
        bool is_pure_sequence =
                (m_mode == SequenceTestsMode::PURE_SEQ || m_mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM ||
                 m_mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST);
        if (!is_pure_sequence) {
            ngraph::pass::Manager manager;
            if (direction == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL)
                manager.register_pass<ov::pass::BidirectionalGRUSequenceDecomposition>();
            manager.register_pass<ov::pass::ConvertGRUSequenceToTensorIterator>();
            manager.run_passes(function);
            bool ti_found = is_tensor_iterator_exist(function);
            EXPECT_EQ(ti_found, true);
        } else {
            bool ti_found = is_tensor_iterator_exist(function);
            EXPECT_EQ(ti_found, false);
        }
    }
};

class VPUXGRUSequenceLayerTest_VPU3720 : public VPUXGRUSequenceLayerTest {};

TEST_P(VPUXGRUSequenceLayerTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;
using GRUDirection = ngraph::op::RecurrentSequenceDirection;

namespace {

const auto testMode = SequenceTestsMode::PURE_SEQ;
const std::vector<size_t> seqLength{5};
const std::vector<size_t> seqLengthTiling{1000};
const std::vector<size_t> seqLengthSplit{1};
const size_t batchSize = 2;
const size_t batchSizeTiling = 100;
const size_t batchSizeSplit = 1;
const size_t hiddenSize = 4;
const size_t hiddenSizeTiling = 1;
const size_t hiddenSizeSplit = 569;
const size_t hiddenSizeSplit1 = 200;
const std::vector<std::string> activations = {"sigmoid", "tanh"};
const float clip = 0.0f;
const std::vector<bool> shouldLinearBeforeReset{true, false};
const std::vector<GRUDirection> directionMode{GRUDirection::FORWARD, GRUDirection::REVERSE,
                                              GRUDirection::BIDIRECTIONAL};
const InferenceEngine::Precision netPrecisions = InferenceEngine::Precision::FP16;

const auto gruSequenceParam0 = testing::Combine(
        ::testing::Values(testMode), ::testing::ValuesIn(seqLength), ::testing::Values(batchSize),
        ::testing::Values(hiddenSize), ::testing::Values(activations), ::testing::Values(clip),
        ::testing::ValuesIn(shouldLinearBeforeReset), ::testing::ValuesIn(directionMode),
        ::testing::Values(netPrecisions), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto gruSequenceParam1 = testing::Combine(
        ::testing::Values(testMode), ::testing::ValuesIn(seqLengthTiling), ::testing::Values(batchSizeTiling),
        ::testing::Values(hiddenSizeTiling), ::testing::Values(activations), ::testing::Values(clip),
        ::testing::ValuesIn(shouldLinearBeforeReset), ::testing::ValuesIn(directionMode),
        ::testing::Values(netPrecisions), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto gruSequenceParam2 = testing::Combine(
        ::testing::Values(testMode), ::testing::ValuesIn(seqLengthSplit), ::testing::Values(batchSizeSplit),
        ::testing::Values(hiddenSizeSplit), ::testing::Values(activations), ::testing::Values(clip),
        ::testing::ValuesIn(shouldLinearBeforeReset), ::testing::ValuesIn(directionMode),
        ::testing::Values(netPrecisions), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto gruSequenceParam3 = testing::Combine(
        ::testing::Values(testMode), ::testing::ValuesIn(seqLengthSplit), ::testing::Values(batchSizeSplit),
        ::testing::Values(hiddenSizeSplit1), ::testing::Values(activations), ::testing::Values(clip),
        ::testing::ValuesIn(shouldLinearBeforeReset), ::testing::ValuesIn(directionMode),
        ::testing::Values(netPrecisions), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

//    VPU3720
INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRUSequence, VPUXGRUSequenceLayerTest_VPU3720, gruSequenceParam0,
                         GRUSequenceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRUSequence_Tiling, VPUXGRUSequenceLayerTest_VPU3720, gruSequenceParam1,
                         GRUSequenceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRUSequence_Split, VPUXGRUSequenceLayerTest_VPU3720, gruSequenceParam2,
                         GRUSequenceTest::getTestCaseName);

}  // namespace
