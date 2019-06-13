// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "ie_layouts.h"
#include "myriad_layers_tests.hpp"
#include <vpu/private_plugin_config.hpp>
#include "myriad_layers_reference_functions.hpp"

using namespace InferenceEngine;

typedef std::vector<int32_t> IndicesVector;

static void ref_squeeze(const InferenceEngine::Blob::Ptr src,
                        InferenceEngine::Blob::Ptr dst) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    int32_t OW = 1;
    int32_t OH = 1;
    int32_t OC = 1;
    int32_t ON = 1;
    int32_t IW = 1;
    int32_t IH = 1;
    int32_t IC = 1;
    int32_t I_N = 1;

    get_dims(src, IW, IH, IC, I_N);
    get_dims(dst, OW, OH, OC, ON);

    ASSERT_EQ(IW * IH * IC * I_N, OW * OH * OC * ON);

    const ie_fp16 *src_data = src->buffer();
    ie_fp16 *dst_data = dst->buffer();
    size_t src_size = src->size();
    size_t dst_size = dst->size();

    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(dst_data, nullptr);
    ASSERT_EQ(src_size, dst_size);

    std::memcpy(dst_data, src_data, src_size * sizeof(ie_fp16));
}

PRETTY_PARAM(layout, const char*)

class myriadLayersTestsSqueezeBase : public
        myriadLayerTestBaseWithParam<std::tuple<InferenceEngine::SizeVector, IndicesVector, int32_t, layout>>
{
protected:
    virtual void InitBody()
    {
        auto input = std::get<0>(GetParam());
        auto indices = std::get<1>(GetParam());
        auto keep_at_least_1d = std::get<2>(GetParam());
        auto layout = std::get<3>(GetParam());

        _config[VPU_CONFIG_KEY(COMPUTE_LAYOUT)] = layout;

        std::string in_dims{};
        std::string out_dims{};

        InferenceEngine::SizeVector output{};

        for (auto &index : indices) {
            if (index < 0)
                index += input.size();
            ASSERT_LT(abs(index), input.size());
            ASSERT_EQ(input[index], 1);
        }

        std::sort(indices.begin(), indices.end());

        for (size_t k = 0; k < input.size(); k++) {
            if (std::find(indices.begin(), indices.end(), k) == indices.end()) {
                output.push_back(input[k]);
            }
        }

        if (output.size() == 0) {
            if (keep_at_least_1d) {
                output.push_back({ 1 });
            } else {
                output.push_back({ 0 });
            }
        }

        for (auto in_dim : input) {
            in_dims += R"V0G0N(
                        <dim>
)V0G0N"
                            + std::to_string(in_dim) +
R"V0G0N(
                        </dim>
)V0G0N";
        }

        for (auto out_dim : output) {
            out_dims += R"V0G0N(
                        <dim>
)V0G0N"
                            + std::to_string(out_dim) +
R"V0G0N(
                        </dim>
)V0G0N";
        }

        std::string SQUEEZE_MODEL_FORMATTED = R"V0G0N(
        <net name="SQUEEZE_MODEL" version="2" batch="1">
            <layers>
                <layer id="0" name="input" precision="FP16" type="Input">
                <output>
                    <port id="0">
)V0G0N"
                    + in_dims +
R"V0G0N(
                    </port>
                </output>
                </layer>
                <layer id="1" name="indices" precision="FP16" type="Const">
                    <output>
                        <port id="1">
                            <dim>
)V0G0N"
                                + std::to_string(indices.size()) +
R"V0G0N(
                            </dim>
                        </port>
                    </output>
                    <blobs>
                        <custom offset="0" size=")V0G0N"
                                              + std::to_string(indices.size() * sizeof(ie_fp16)) +
                                              R"V0G0N("/>
                    </blobs>
                </layer>
                <layer id="2" name="squeeze" precision="FP16" type="Squeeze">
                    <data keep_at_least_1d=")V0G0N"
                                            + std::to_string(keep_at_least_1d) +
                                            R"V0G0N("/>
                    <input>
                        <port id="0">
)V0G0N"
                        + in_dims +
R"V0G0N(
                        </port>
                        <port id="1">
                            <dim>
)V0G0N"
                                + std::to_string(indices.size()) +
R"V0G0N(
                            </dim>
                        </port>
                    </input>
                    <output>
                        <port id="1">
)V0G0N"
                        + out_dims +
R"V0G0N(
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
            </edges>
        </net>
)V0G0N";

        InferenceEngine::StatusCode st = InferenceEngine::OK;
        InferenceEngine::ResponseDesc resp;

        InferenceEngine::TBlob<uint8_t> *weights_raw = new InferenceEngine::TBlob<uint8_t>(
                InferenceEngine::Precision::U8,
                InferenceEngine::C,
                {indices.size() * sizeof(ie_fp16)});
        weights_raw->allocate();
        ie_fp16 *inputBlobRawDataFp16 = weights_raw->data().as<ie_fp16 *>();

        for (size_t index = 0; index < indices.size(); ++index) {
            inputBlobRawDataFp16[index] = InferenceEngine::PrecisionUtils::f32tof16(indices[index]);
        }

        TBlob<uint8_t>::Ptr weights(weights_raw);

        _net_reader.ReadNetwork(SQUEEZE_MODEL_FORMATTED.data(), SQUEEZE_MODEL_FORMATTED.length());
        ASSERT_EQ(_net_reader.isParseSuccess(), true);
        if (weights != nullptr)
            ASSERT_NO_THROW(_net_reader.SetWeights(weights));
        setup(InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP16, true);
        ASSERT_TRUE(Infer());
        ref_squeeze(_inputMap.begin()->second, _refBlob);
        auto outBlob = _outputMap.begin()->second;
        ASSERT_EQ(outBlob->dims().size(), _refBlob->dims().size());
        for (size_t i = 0; i < outBlob->dims().size(); i++) {
            ASSERT_EQ(outBlob->dims()[i], _refBlob->dims()[i]);
        }

        const ie_fp16 *out_data = outBlob->buffer();
        const ie_fp16 *ref_data = _refBlob->buffer();
        size_t out_size = outBlob->size();
        size_t ref_size = _refBlob->size();
        ASSERT_EQ(out_size, ref_size);
        for (size_t i = 0; i < out_size; i++) {
            ASSERT_EQ(out_data[i], ref_data[i]);
        }
    }
};

class myriadLayersTestsSqueezeTC1 : public myriadLayersTestsSqueezeBase
{
};

class myriadLayersTestsSqueezeTC2 : public myriadLayersTestsSqueezeBase
{
};

class myriadLayersTestsSqueezeTC3 : public myriadLayersTestsSqueezeBase
{
};

class myriadLayersTestsSqueezeTC4 : public myriadLayersTestsSqueezeBase
{
};

TEST_P(myriadLayersTestsSqueezeTC1, Squeeze) {
    InitBody();
}

TEST_P(myriadLayersTestsSqueezeTC2, Squeeze) {
    InitBody();
}

TEST_P(myriadLayersTestsSqueezeTC3, Squeeze) {
    InitBody();
}

TEST_P(myriadLayersTestsSqueezeTC4, Squeeze) {
    InitBody();
}

static std::vector<InferenceEngine::SizeVector> s_squeezeTensorsTC1 = {
    {{1, 3, 1}, {1, 1, 1}}
};

static std::vector<IndicesVector> s_squeezeIndicesTC1 = {
    {{0, 2}, {0}, {2}, {-3}, {-1}, {-3, -1}}
};

static std::vector<InferenceEngine::SizeVector> s_squeezeTensorsTC2 = {
    {{3, 1, 2}}
};

static std::vector<IndicesVector> s_squeezeIndicesTC2 = {
    {{1, -2}, {-2, 1}}
};

static std::vector<InferenceEngine::SizeVector> s_squeezeTensorsTC3 = {
        {{3, 1, 2, 3}}
};

static std::vector<IndicesVector> s_squeezeIndicesTC3 = {
        {{1, -3}, {-3, 1}}
};

static std::vector<InferenceEngine::SizeVector> s_squeezeTensorsTC4 = {
        {{3, 1, 2, 1}}
};

static std::vector<IndicesVector> s_squeezeIndicesTC4 = {
        {{1}, {3}, {1, 3}, {3, 1}, {-1}, {-3}, {-1, -3}}
};

static std::vector<int32_t> s_squeezeKeepAtLeast1D = {
    0, 1
};

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsSqueezeTC1,
    ::testing::Combine(
        ::testing::ValuesIn(s_squeezeTensorsTC1),
        ::testing::ValuesIn(s_squeezeIndicesTC1),
        ::testing::ValuesIn(s_squeezeKeepAtLeast1D),
        ::testing::Values<layout>(VPU_CONFIG_VALUE(AUTO))
    )
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsSqueezeTC2,
    ::testing::Combine(
        ::testing::ValuesIn(s_squeezeTensorsTC2),
        ::testing::ValuesIn(s_squeezeIndicesTC2),
        ::testing::ValuesIn(s_squeezeKeepAtLeast1D),
        ::testing::Values<layout>(VPU_CONFIG_VALUE(NCHW))
    )
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsSqueezeTC3,
    ::testing::Combine(
        ::testing::ValuesIn(s_squeezeTensorsTC3),
        ::testing::ValuesIn(s_squeezeIndicesTC3),
        ::testing::ValuesIn(s_squeezeKeepAtLeast1D),
        ::testing::Values<layout>(VPU_CONFIG_VALUE(AUTO))
    )
);

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsSqueezeTC4,
        ::testing::Combine(
        ::testing::ValuesIn(s_squeezeTensorsTC4),
        ::testing::ValuesIn(s_squeezeIndicesTC4),
        ::testing::ValuesIn(s_squeezeKeepAtLeast1D),
        ::testing::Values<layout>(VPU_CONFIG_VALUE(AUTO))
    )
);