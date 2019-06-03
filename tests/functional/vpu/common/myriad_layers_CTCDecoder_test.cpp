// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

PRETTY_PARAM(hwAcceleration, std::string);

typedef myriadLayerTestBaseWithParam<hwAcceleration> myriadCTCDecoderLayerTests_nightly;

void refCTCDecoder(const Blob::Ptr src, const Blob::Ptr seq_ind, Blob::Ptr dst) {
    ie_fp16 *src_data = static_cast<ie_fp16*>(src->buffer());
    ie_fp16 *src_seq_inp = static_cast<ie_fp16*>(seq_ind->buffer());
    ie_fp16 *output_sequences = static_cast<ie_fp16*>(dst->buffer());
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(src_seq_inp, nullptr);
    ASSERT_NE(output_sequences, nullptr);

    size_t in_width      = src->dims()[0];
    size_t in_height     = src->dims()[1];
    size_t in_channels   = src->dims()[2];

    size_t T_ = in_channels;
    size_t N_ = in_height;
    size_t C_ = in_width;

    std::vector<ie_fp16> probabilities(in_width * in_height * in_channels) ;

    std::vector<int> seq_ind_data(88) ;
    seq_ind_data[0] = 0;
    for(int i = 1; i < 88; i++) {
        seq_ind_data[i] = (int)(PrecisionUtils::f16tof32(src_seq_inp[i]));
    }

    for(size_t k = 0; k < in_channels; ++k) {
        for(size_t j = 0; j < in_height; ++j) {
            for(size_t i = 0; i < in_width; ++i) {
                int dst_index = i + in_width * j + in_width * in_height * k;
                int src_index = k + in_channels * i + in_channels * in_width * j;
                probabilities[dst_index] = src_data[src_index];
            }
        }
    }
    // Fill output_sequences with -1
    for (size_t ii = 0; ii < T_; ii++) {
        output_sequences[ii] = PrecisionUtils::f32tof16(-1.0);
    }
    size_t output_index = 0;

    // Caffe impl
    for(size_t n = 0; n < N_; ++n) {
        int prev_class_idx = -1;

        for (size_t t = 0; /* check at end */; ++t) {
            // get maximum probability and its index
            int max_class_idx = 0;
            ie_fp16* probs;
            ie_fp16 max_prob;

            probs = probabilities.data() + t*C_;
            max_prob = probs[0];
            ++probs;

            for (size_t c = 1; c < C_; ++c, ++probs) {
                if (*probs > max_prob) {
                    max_class_idx = c;
                    max_prob = *probs;
                }
            }

            //if (max_class_idx != blank_index_
            //        && !(merge_repeated_&& max_class_idx == prev_class_idx))
            if (max_class_idx < (int)C_-1 && !(1 && max_class_idx == prev_class_idx)) {
                output_sequences[output_index] =  PrecisionUtils::f32tof16((float)max_class_idx);
                output_index++;
            }

            prev_class_idx = max_class_idx;

            // Assume sequence_indicators is always 1
//             if (t + 1 == T_)
            if (t + 1 == T_ || seq_ind_data[t + 1] == 0) {
                break;
            }
        }
    }
}


TEST_P(myriadCTCDecoderLayerTests_nightly, CTCDecoder) {
    std::string model = R"V0G0N(
        <net name="testCTC" version="2" batch="1">
            <layers>
                <layer name="seq_ind" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>88</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data" type="Input" precision="FP16" id="0">
                    <output>
                        <port id="0">
                            <dim>1</dim>
                            <dim>88</dim>
                            <dim>1</dim>
                            <dim>71</dim>
                        </port>
                    </output>
                </layer>
                <layer name="decode" type="CTCGreedyDecoder" precision="FP16" id="2">
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>88</dim>
                            <dim>1</dim>
                            <dim>71</dim>
                        </port>
                        <port id="3">
                            <dim>88</dim>
                            <dim>1</dim>
                        </port>
                    </input>
                    <output>
                        <port id="4">
                            <dim>1</dim>
                            <dim>88</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="2" to-port="2"/>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="3"/>
            </edges>
        </net>
    )V0G0N";
    SetSeed(DEFAULT_SEED_VALUE + 6);

    std::string HWConfigValue = GetParam();

    StatusCode st;

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    CNNNetwork network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["data"]->setInputPrecision(Precision::FP16);
    _inputsInfo["data"]->setLayout(NHWC);
    _inputsInfo["seq_ind"]->setInputPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["decode"]->setPrecision(Precision::FP16);

    ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(_exeNetwork, network,
                                                      {{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), HWConfigValue}}, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

    ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr data;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("data", data, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr seq_ind;
    ASSERT_NO_THROW(st = _inferRequest->GetBlob("seq_ind", seq_ind, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    uint16_t *blobRawSeqFp16 = seq_ind->buffer().as<uint16_t *>();
    size_t count = seq_ind->size();
    blobRawSeqFp16[0] = PrecisionUtils::f32tof16(0.0);
    for (size_t indx = 1; indx < count; ++indx) {
        blobRawSeqFp16[indx] = PrecisionUtils::f32tof16(1.0);
    }

    uint16_t *blobRawDataFp16 = data->buffer().as<uint16_t *>();
    size_t count1 = data->size();

    std::string inputTensorBinary = get_data_path() + "/vpu/InputGreedyDecoderMyriad.bin";

    ASSERT_TRUE(fromBinaryFile(inputTensorBinary, data));

    ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    Blob::Ptr outputBlob;
    ASSERT_NO_THROW(_inferRequest->GetBlob("decode", outputBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    _refBlob = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, ANY, outputBlob->dims());
    _refBlob->allocate();

    refCTCDecoder(data, seq_ind, _refBlob);

    ie_fp16 *output_sequences = static_cast<ie_fp16*>(_refBlob->buffer());
    ie_fp16 *out_myriad = static_cast<ie_fp16*>(outputBlob->buffer());

    Compare(outputBlob, _refBlob, 0.0);
}

INSTANTIATE_TEST_CASE_P(myriad, myriadCTCDecoderLayerTests_nightly,
                        ::testing::Values(CONFIG_VALUE(YES), CONFIG_VALUE(NO))
);
