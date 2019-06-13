// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

struct PriorBoxParams {
    tensor_test_params in1 = {1, 512, 38, 38};
    tensor_test_params in2 = {1, 3, 300, 300};

    std::vector<float> min_size = {21.0};
    std::vector<float> max_size = {45.0};
    std::vector<float> aspect_ratio = {2.0};
    int flip = 1;
    int clip = 0;
    std::vector<float> variance = {0.1f, 0.1f, 0.2f, 0.2f};
    int img_size = 0;
    int img_h = 0;
    int img_w = 0;
    float step_ = 8.0;
    float step_h = 0.0;
    float step_w = 0.0;
    float offset = 0.5;
};

// The code was taken from caffe and adopted to InferenceEngine reality
void refPriorBox(Blob::Ptr dst, const PriorBoxParams &p) {
    std::vector<float> aspect_ratios_;
    aspect_ratios_.push_back(1.0);
    for (int i = 0; i < p.aspect_ratio.size(); ++i) {
        float ar = p.aspect_ratio[i];
        bool already_exist = false;
        for (int j = 0; j < aspect_ratios_.size(); ++j) {
            if (fabs(ar - aspect_ratios_[j]) < 1e-6) {
                already_exist = true;
                break;
            }
        }
        if (!already_exist) {
            aspect_ratios_.push_back(ar);
            if (p.flip) {
                aspect_ratios_.push_back(1.0 / ar);
            }
        }
    }

    int num_priors_ = aspect_ratios_.size() * p.min_size.size() + p.max_size.size();

    int layer_width  = p.in1.w;
    int layer_height = p.in1.h;

    int32_t img_width  = p.img_w == 0 ? p.in2.w : p.img_w;
    int32_t img_height = p.img_h == 0 ? p.in2.h : p.img_h;

    float step_w = p.step_w == 0 ? p.step_ : p.step_w;
    float step_h = p.step_h == 0 ? p.step_ : p.step_h;
    if (step_w == 0 || step_h == 0) {
        step_w = static_cast<float>(img_width) / layer_width;
        step_h = static_cast<float>(img_height) / layer_height;
    }

    std::vector<float> top_data(dst->size());
    int dim = layer_height * layer_width * num_priors_ * 4;

    int idx = 0;
    for (int h = 0; h < layer_height; ++h) {
        for (int w = 0; w < layer_width;  ++w) {
            float center_x = (w + p.offset) * step_w;
            float center_y = (h + p.offset) * step_h;

            float box_width, box_height;
            for (int s = 0; s < p.min_size.size(); ++s) {
                float min_size_ = p.min_size[s];

                // first prior: aspect_ratio = 1, size = min_size
                box_width = box_height = min_size_;
                // xmin
                top_data[idx++] = (center_x - box_width / 2.) / img_width;
                // ymin
                top_data[idx++] = (center_y - box_height / 2.) / img_height;
                // xmax
                top_data[idx++] = (center_x + box_width / 2.) / img_width;
                // ymax
                top_data[idx++] = (center_y + box_height / 2.) / img_height;

                if (p.max_size.size() > 0) {
                    float max_size_ = p.max_size[s];

                    // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                    box_width = box_height = sqrt(min_size_ * max_size_);
                    // xmin
                    top_data[idx++] = (center_x - box_width / 2.) / img_width;
                    // ymin
                    top_data[idx++] = (center_y - box_height / 2.) / img_height;
                    // xmax
                    top_data[idx++] = (center_x + box_width / 2.) / img_width;
                    // ymax
                    top_data[idx++] = (center_y + box_height / 2.) / img_height;
                }

                // rest of priors
                for (int r = 0; r < aspect_ratios_.size(); ++r) {
                    float ar = aspect_ratios_[r];
                    if (fabs(ar - 1.) < 1e-6) {
                        continue;
                    }

                    box_width = min_size_ * sqrt(ar);
                    box_height = min_size_ / sqrt(ar);

                    // xmin
                    top_data[idx++] = (center_x - box_width / 2.) / img_width;
                    // ymin
                    top_data[idx++] = (center_y - box_height / 2.) / img_height;
                    // xmax
                    top_data[idx++] = (center_x + box_width / 2.) / img_width;
                    // ymax
                    top_data[idx++] = (center_y + box_height / 2.) / img_height;
                }
            }
        }
    }

    ie_fp16* output_data = static_cast<ie_fp16*>(dst->buffer());

    // clip the prior's coordidate such that it is within [0, 1]
    if (p.clip) {
        for (int d = 0; d < dim; ++d) {
            float val = std::min(std::max(top_data[d], 0.0f), 1.0f);
            output_data[d] = PrecisionUtils::f32tof16(val);
        }
    } else {
        for (int d = 0; d < dim; ++d) {
            output_data[d] = PrecisionUtils::f32tof16(top_data[d]);
        }
    }

    output_data += dst->dims()[0];

    // set the variance.
    if (p.variance.size() == 0) {
        // Set default to 0.1.
        for (int d = 0; d < dim; ++d) {
            output_data[d] = PrecisionUtils::f32tof16(0.1f);
        }
    } else if (p.variance.size() == 1) {
        for (int d = 0; d < dim; ++d) {
            output_data[d] = PrecisionUtils::f32tof16(p.variance[0]);
        }
    } else {
        // Must and only provide 4 variance.
        ASSERT_EQ(4u, p.variance.size());

        int idx = 0;
        for (int h = 0; h < layer_height; ++h) {
            for (int w = 0; w < layer_width; ++w) {
                for (int i = 0; i < num_priors_; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        output_data[idx++] = PrecisionUtils::f32tof16(p.variance[j]);
                    }
                }
            }
        }
    }
}

class myriadLayersPriorBoxTests_nightly : public myriadLayersTests_nightly {
public:
    Blob::Ptr getFp16Blob(const Blob::Ptr& in) {
        if (in->getTensorDesc().getPrecision() == Precision::FP16)
            return in;

        auto out = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, in->getTensorDesc().getLayout(), in->dims());
        out->allocate();

        if (in->getTensorDesc().getPrecision() == Precision::FP32) {
            PrecisionUtils::f32tof16Arrays(out->buffer().as<ie_fp16 *>(), in->cbuffer().as<float *>(), in->size());
        } else {
            ADD_FAILURE() << "Unsupported precision " << in->getTensorDesc().getPrecision();
        }

        return out;
    }

    void RunOnModel(const std::string& model, const std::string& outputName, Precision outPrec = Precision::FP16) {
        SetSeed(DEFAULT_SEED_VALUE + 5);

        StatusCode st;

        ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
        ASSERT_TRUE(_net_reader.isParseSuccess());

        CNNNetwork network = _net_reader.getNetwork();

        _inputsInfo = network.getInputsInfo();
        _inputsInfo["data1"]->setInputPrecision(Precision::FP16);
        _inputsInfo["data2"]->setInputPrecision(Precision::FP16);

        _outputsInfo = network.getOutputsInfo();
        _outputsInfo["data1_copy"]->setPrecision(Precision::FP16);
        _outputsInfo["data2_copy"]->setPrecision(Precision::FP16);
        _outputsInfo[outputName]->setPrecision(outPrec);

        ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(_exeNetwork, network, {}, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

        ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        Blob::Ptr data1;
        ASSERT_NO_THROW(st = _inferRequest->GetBlob("data1", data1, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        Blob::Ptr data2;
        ASSERT_NO_THROW(st = _inferRequest->GetBlob("data2", data2, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        GenRandomData(data1);
        GenRandomData(data2);

        ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        Blob::Ptr outputBlob;
        ASSERT_NO_THROW(_inferRequest->GetBlob(outputName.c_str(), outputBlob, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        _refBlob = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, ANY, outputBlob->dims());
        _refBlob->allocate();

        PriorBoxParams params;
        refPriorBox(_refBlob, params);

        Compare(getFp16Blob(outputBlob), _refBlob, 0.0);
    }
};

TEST_F(myriadLayersPriorBoxTests_nightly, NotLastLayer)
{
    std::string model = R"V0G0N(
        <net name="PriorBox" version="2" batch="1">
            <layers>
                <layer name="data1" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="11">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data1_copy" type="Power" precision="FP16" id="2">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="21">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="22">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data2" type="Input" precision="FP16" id="3">
                    <output>
                        <port id="31">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data2_copy" type="Power" precision="FP16" id="4">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="41">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </input>
                    <output>
                        <port id="42">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </output>
                </layer>
                <layer name="priorbox" type="PriorBox" precision="FP16" id="5">
                    <data
                        min_size="21.000000"
                        max_size="45.000000"
                        aspect_ratio="2.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="8.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="51">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                        <port id="52">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="53">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>23104</dim>
                        </port>
                    </output>
                </layer>
                <layer name="priorbox_copy" type="Power" precision="FP16" id="6">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="61">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>23104</dim>
                        </port>
                    </input>
                    <output>
                        <port id="62">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>23104</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="11" to-layer="2" to-port="21"/>
                <edge from-layer="3" from-port="31" to-layer="4" to-port="41"/>
                <edge from-layer="3" from-port="31" to-layer="5" to-port="51"/>
                <edge from-layer="1" from-port="11" to-layer="5" to-port="52"/>
                <edge from-layer="5" from-port="53" to-layer="6" to-port="61"/>
            </edges>
        </net>
    )V0G0N";

    RunOnModel(model, "priorbox_copy");
}

TEST_F(myriadLayersPriorBoxTests_nightly, LastLayer_FP16)
{
    std::string model = R"V0G0N(
        <net name="PriorBox" version="2" batch="1">
            <layers>
                <layer name="data1" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="11">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data1_copy" type="Power" precision="FP16" id="2">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="21">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="22">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data2" type="Input" precision="FP16" id="3">
                    <output>
                        <port id="31">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data2_copy" type="Power" precision="FP16" id="4">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="41">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </input>
                    <output>
                        <port id="42">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </output>
                </layer>
                <layer name="priorbox" type="PriorBox" precision="FP16" id="5">
                    <data
                        min_size="21.000000"
                        max_size="45.000000"
                        aspect_ratio="2.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="8.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="51">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                        <port id="52">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="53">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>23104</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="11" to-layer="2" to-port="21"/>
                <edge from-layer="3" from-port="31" to-layer="4" to-port="41"/>
                <edge from-layer="3" from-port="31" to-layer="5" to-port="51"/>
                <edge from-layer="1" from-port="11" to-layer="5" to-port="52"/>
            </edges>
        </net>
    )V0G0N";

    RunOnModel(model, "priorbox", Precision::FP16);
}

TEST_F(myriadLayersPriorBoxTests_nightly, LastLayer_FP32)
{
    std::string model = R"V0G0N(
        <net name="PriorBox" version="2" batch="1">
            <layers>
                <layer name="data1" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="11">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data1_copy" type="Power" precision="FP16" id="2">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="21">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="22">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data2" type="Input" precision="FP16" id="3">
                    <output>
                        <port id="31">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </output>
                </layer>
                <layer name="data2_copy" type="Power" precision="FP16" id="4">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="41">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </input>
                    <output>
                        <port id="42">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </output>
                </layer>
                <layer name="priorbox" type="PriorBox" precision="FP16" id="5">
                    <data
                        min_size="21.000000"
                        max_size="45.000000"
                        aspect_ratio="2.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="8.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="51">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                        <port id="52">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="53">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>23104</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="11" to-layer="2" to-port="21"/>
                <edge from-layer="3" from-port="31" to-layer="4" to-port="41"/>
                <edge from-layer="3" from-port="31" to-layer="5" to-port="51"/>
                <edge from-layer="1" from-port="11" to-layer="5" to-port="52"/>
            </edges>
        </net>
    )V0G0N";

    RunOnModel(model, "priorbox", Precision::FP32);
}

TEST_F(myriadLayersTests_nightly, PriorBox_WithConcat)
{
    std::string model = R"V0G0N(
        <net name="PriorBox_WithConcat" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>
                <layer name="input_copy" type="Power" precision="FP16" id="2">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </output>
                </layer>

                <layer name="conv4_3_norm" type="Input" precision="FP16" id="3">
                    <output>
                        <port id="4">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv4_3_norm_copy" type="Power" precision="FP16" id="4">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="5">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </input>
                    <output>
                        <port id="6">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv4_3_norm_mbox_priorbox" type="PriorBox" precision="FP16" id="5">
                    <data
                        min_size="21.000000"
                        max_size="45.000000"
                        aspect_ratio="2.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="8.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="7">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>38</dim>
                            <dim>38</dim>
                        </port>
                        <port id="8">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="9">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>23104</dim>
                        </port>
                    </output>
                </layer>

                <layer name="fc7" type="Input" precision="FP16" id="6">
                    <output>
                        <port id="10">
                            <dim>1</dim>
                            <dim>1024</dim>
                            <dim>19</dim>
                            <dim>19</dim>
                        </port>
                    </output>
                </layer>
                <layer name="fc7_copy" type="Power" precision="FP16" id="7">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="11">
                            <dim>1</dim>
                            <dim>1024</dim>
                            <dim>19</dim>
                            <dim>19</dim>
                        </port>
                    </input>
                    <output>
                        <port id="12">
                            <dim>1</dim>
                            <dim>1024</dim>
                            <dim>19</dim>
                            <dim>19</dim>
                        </port>
                    </output>
                </layer>
                <layer name="fc7_mbox_priorbox" type="PriorBox" precision="FP16" id="8">
                    <data
                        min_size="45.000000"
                        max_size="99.000000"
                        aspect_ratio="2.000000,3.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="16.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="13">
                            <dim>1</dim>
                            <dim>1024</dim>
                            <dim>19</dim>
                            <dim>19</dim>
                        </port>
                        <port id="14">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="15">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>8664</dim>
                        </port>
                    </output>
                </layer>

                <layer name="conv6_2" type="Input" precision="FP16" id="9">
                    <output>
                        <port id="16">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>10</dim>
                            <dim>10</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv6_2_copy" type="Power" precision="FP16" id="10">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="17">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>10</dim>
                            <dim>10</dim>
                        </port>
                    </input>
                    <output>
                        <port id="18">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>10</dim>
                            <dim>10</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv6_2_mbox_priorbox" type="PriorBox" precision="FP16" id="11">
                    <data
                        min_size="99.000000"
                        max_size="153.000000"
                        aspect_ratio="2.000000,3.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="32.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="19">
                            <dim>1</dim>
                            <dim>512</dim>
                            <dim>10</dim>
                            <dim>10</dim>
                        </port>
                        <port id="20">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="21">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>2400</dim>
                        </port>
                    </output>
                </layer>

                <layer name="conv7_2" type="Input" precision="FP16" id="12">
                    <output>
                        <port id="22">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv7_2_copy" type="Power" precision="FP16" id="13">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="23">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </input>
                    <output>
                        <port id="24">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv7_2_mbox_priorbox" type="PriorBox" precision="FP16" id="14">
                    <data
                        min_size="153.000000"
                        max_size="207.000000"
                        aspect_ratio="2.000000,3.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="64.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="25">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                        <port id="26">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="27">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>600</dim>
                        </port>
                    </output>
                </layer>

                <layer name="conv8_2" type="Input" precision="FP16" id="15">
                    <output>
                        <port id="28">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>3</dim>
                            <dim>3</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv8_2_copy" type="Power" precision="FP16" id="16">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="29">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>3</dim>
                            <dim>3</dim>
                        </port>
                    </input>
                    <output>
                        <port id="30">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>3</dim>
                            <dim>3</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv8_2_mbox_priorbox" type="PriorBox" precision="FP16" id="17">
                    <data
                        min_size="207.000000"
                        max_size="261.000000"
                        aspect_ratio="2.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="100.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="31">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>3</dim>
                            <dim>3</dim>
                        </port>
                        <port id="32">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="33">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>144</dim>
                        </port>
                    </output>
                </layer>)V0G0N";

    model += R"V0G0N(
                <layer name="conv9_2" type="Input" precision="FP16" id="18">
                    <output>
                        <port id="34">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv9_2_copy" type="Power" precision="FP16" id="19">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="35">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </input>
                    <output>
                        <port id="36">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv9_2_mbox_priorbox" type="PriorBox" precision="FP16" id="20">
                    <data
                        min_size="261.000000"
                        max_size="315.000000"
                        aspect_ratio="2.000000"
                        flip="1"
                        clip="0"
                        variance="0.100000,0.100000,0.200000,0.200000"
                        img_size="0"
                        img_h="0"
                        img_w="0"
                        step="300.000000"
                        step_h="0.000000"
                        step_w="0.000000"
                        offset="0.500000"
                        width="#"
                        height="#"
                    />
                    <input>
                        <port id="37">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                        <port id="38">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>300</dim>
                            <dim>300</dim>
                        </port>
                    </input>
                    <output>
                        <port id="39">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>16</dim>
                        </port>
                    </output>
                </layer>

                <layer name="mbox_priorbox" type="Concat" precision="FP16" id="21">
                    <concat_data axis="2"/>
                    <input>
                        <port id="40">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>23104</dim>
                        </port>
                        <port id="41">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>8664</dim>
                        </port>
                        <port id="42">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>2400</dim>
                        </port>
                        <port id="43">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>600</dim>
                        </port>
                        <port id="44">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>144</dim>
                        </port>
                        <port id="45">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>16</dim>
                        </port>
                    </input>
                    <output>
                        <port id="46">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>34928</dim>
                        </port>
                    </output>
                </layer>
                <layer name="mbox_priorbox_copy" type="Power" precision="FP16" id="22">
                    <power_data power="1" scale="1" shift="0"/>
                    <input>
                        <port id="47">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>34928</dim>
                        </port>
                    </input>
                    <output>
                        <port id="48">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>34928</dim>
                        </port>
                    </output>
                </layer>
            </layers>

            <edges>
                <!-- input > input_copy -->
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>

                <!-- conv4_3_norm > conv4_3_norm_copy -->
                <edge from-layer="3" from-port="4" to-layer="4" to-port="5"/>

                <!-- conv4_3_norm > conv4_3_norm_mbox_priorbox -->
                <edge from-layer="3" from-port="4" to-layer="5" to-port="7"/>
                <!-- input > conv4_3_norm_mbox_priorbox -->
                <edge from-layer="1" from-port="1" to-layer="5" to-port="8"/>

                <!-- fc7 > fc7_copy -->
                <edge from-layer="6" from-port="10" to-layer="7" to-port="11"/>

                <!-- fc7 > fc7_mbox_priorbox -->
                <edge from-layer="6" from-port="10" to-layer="8" to-port="13"/>
                <!-- input > fc7_mbox_priorbox -->
                <edge from-layer="1" from-port="1" to-layer="8" to-port="14"/>

                <!-- conv6_2 > conv6_2_copy -->
                <edge from-layer="9" from-port="16" to-layer="10" to-port="17"/>

                <!-- conv6_2 > conv6_2_mbox_priorbox -->
                <edge from-layer="9" from-port="16" to-layer="11" to-port="19"/>
                <!-- input > conv6_2_mbox_priorbox -->
                <edge from-layer="1" from-port="1" to-layer="11" to-port="20"/>

                <!-- conv7_2 > conv7_2_copy -->
                <edge from-layer="12" from-port="22" to-layer="13" to-port="23"/>

                <!-- conv7_2 > conv7_2_mbox_priorbox -->
                <edge from-layer="12" from-port="22" to-layer="14" to-port="25"/>
                <!-- input > conv7_2_mbox_priorbox -->
                <edge from-layer="1" from-port="1" to-layer="14" to-port="26"/>

                <!-- conv8_2 > conv8_2_copy -->
                <edge from-layer="15" from-port="28" to-layer="16" to-port="29"/>

                <!-- conv8_2 > conv8_2_mbox_priorbox -->
                <edge from-layer="15" from-port="28" to-layer="17" to-port="31"/>
                <!-- input > conv8_2_mbox_priorbox -->
                <edge from-layer="1" from-port="1" to-layer="17" to-port="32"/>

                <!-- conv9_2 > conv9_2_copy -->
                <edge from-layer="18" from-port="34" to-layer="19" to-port="35"/>

                <!-- conv9_2 > conv9_2_mbox_priorbox -->
                <edge from-layer="18" from-port="34" to-layer="20" to-port="37"/>
                <!-- input > conv9_2_mbox_priorbox -->
                <edge from-layer="1" from-port="1" to-layer="20" to-port="38"/>

                <!-- conv4_3_norm_mbox_priorbox > mbox_priorbox -->
                <edge from-layer="5" from-port="9" to-layer="21" to-port="40"/>
                <!-- fc7_mbox_priorbox > mbox_priorbox -->
                <edge from-layer="8" from-port="15" to-layer="21" to-port="41"/>
                <!-- conv6_2_mbox_priorbox > mbox_priorbox -->
                <edge from-layer="11" from-port="21" to-layer="21" to-port="42"/>
                <!-- conv7_2_mbox_priorbox > mbox_priorbox -->
                <edge from-layer="14" from-port="27" to-layer="21" to-port="43"/>
                <!-- conv8_2_mbox_priorbox > mbox_priorbox -->
                <edge from-layer="17" from-port="33" to-layer="21" to-port="44"/>
                <!-- conv9_2_mbox_priorbox > mbox_priorbox -->
                <edge from-layer="20" from-port="39" to-layer="21" to-port="45"/>

                <!-- mbox_priorbox > mbox_priorbox_copy -->
                <edge from-layer="21" from-port="46" to-layer="22" to-port="47"/>
            </edges>
        </net>
    )V0G0N";

    StatusCode st;

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.data(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    CNNNetwork network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    _inputsInfo["input"]->setInputPrecision(Precision::FP16);
    _inputsInfo["conv4_3_norm"]->setInputPrecision(Precision::FP16);
    _inputsInfo["fc7"]->setInputPrecision(Precision::FP16);
    _inputsInfo["conv6_2"]->setInputPrecision(Precision::FP16);
    _inputsInfo["conv7_2"]->setInputPrecision(Precision::FP16);
    _inputsInfo["conv8_2"]->setInputPrecision(Precision::FP16);
    _inputsInfo["conv9_2"]->setInputPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    _outputsInfo["input_copy"]->setPrecision(Precision::FP16);
    _outputsInfo["conv4_3_norm_copy"]->setPrecision(Precision::FP16);
    _outputsInfo["fc7_copy"]->setPrecision(Precision::FP16);
    _outputsInfo["conv6_2_copy"]->setPrecision(Precision::FP16);
    _outputsInfo["conv7_2_copy"]->setPrecision(Precision::FP16);
    _outputsInfo["conv8_2_copy"]->setPrecision(Precision::FP16);
    _outputsInfo["conv9_2_copy"]->setPrecision(Precision::FP16);
    _outputsInfo["mbox_priorbox_copy"]->setPrecision(Precision::FP16);

    ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(_exeNetwork, network, {}, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NE(_exeNetwork, nullptr) << _resp.msg;

    ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    // TODO: uncomment this code when GraphTransformer will be updated
    // to optimize out extra copies in case of PriorBox+Concat pair.
#if 0
    {
        std::map<std::string, InferenceEngineProfileInfo> perfMap;
        ASSERT_NO_THROW(st = _inferRequest->GetPerformanceCounts(perfMap, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        int count = 0;
        for (auto p : perfMap) {
            auto layerName = p.first;
            auto status = p.second.status;
            if (layerName.find("mbox_priorbox@copy") == 0) {
                EXPECT_EQ(InferenceEngineProfileInfo::OPTIMIZED_OUT, status) << layerName;
                ++count;
            }
        }
        EXPECT_EQ(6, count);
    }
#endif

    Blob::Ptr outputBlob;
    ASSERT_NO_THROW(_inferRequest->GetBlob("mbox_priorbox_copy", outputBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    auto conv4_3_norm_mbox_priorbox = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::ANY, {23104, 2, 1});
    {
        conv4_3_norm_mbox_priorbox->allocate();

        PriorBoxParams params;
        params.in1 = {1, 512, 38, 38};
        params.in2 = {1, 3, 300, 300};
        params.min_size = {21.0};
        params.max_size = {45.0};
        params.aspect_ratio = {2.0};
        params.flip = 1;
        params.clip = 0;
        params.variance = {0.1f, 0.1f, 0.2f, 0.2f};
        params.img_size = 0;
        params.img_h =  0;
        params.img_w = 0;
        params.step_ = 8.0;
        params.step_h = 0.0;
        params.step_w = 0.0;
        params.offset = 0.5;

        refPriorBox(conv4_3_norm_mbox_priorbox, params);
    }

    auto fc7_mbox_priorbox = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::ANY, {8664, 2, 1});
    {
        fc7_mbox_priorbox->allocate();

        PriorBoxParams params;
        params.in1 = {1, 1024, 19, 19};
        params.in2 = {1, 3, 300, 300};
        params.min_size = {45.0};
        params.max_size = {99.0};
        params.aspect_ratio = {2.0, 3.0};
        params.flip = 1;
        params.clip = 0;
        params.variance = {0.1f, 0.1f, 0.2f, 0.2f};
        params.img_size = 0;
        params.img_h =  0;
        params.img_w = 0;
        params.step_ = 16.0;
        params.step_h = 0.0;
        params.step_w = 0.0;
        params.offset = 0.5;

        refPriorBox(fc7_mbox_priorbox, params);
    }

    auto conv6_2_mbox_priorbox = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::ANY, {2400, 2, 1});
    {
        conv6_2_mbox_priorbox->allocate();

        PriorBoxParams params;
        params.in1 = {1, 512, 10, 10};
        params.in2 = {1, 3, 300, 300};
        params.min_size = {99.0};
        params.max_size = {153.0};
        params.aspect_ratio = {2.0, 3.0};
        params.flip = 1;
        params.clip = 0;
        params.variance = {0.1f, 0.1f, 0.2f, 0.2f};
        params.img_size = 0;
        params.img_h =  0;
        params.img_w = 0;
        params.step_ = 32.0;
        params.step_h = 0.0;
        params.step_w = 0.0;
        params.offset = 0.5;

        refPriorBox(conv6_2_mbox_priorbox, params);
    }

    auto conv7_2_mbox_priorbox = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::ANY, {600, 2, 1});
    {
        conv7_2_mbox_priorbox->allocate();

        PriorBoxParams params;
        params.in1 = {1, 256, 5, 5};
        params.in2 = {1, 3, 300, 300};
        params.min_size = {153.0};
        params.max_size = {207.0};
        params.aspect_ratio = {2.0, 3.0};
        params.flip = 1;
        params.clip = 0;
        params.variance = {0.1f, 0.1f, 0.2f, 0.2f};
        params.img_size = 0;
        params.img_h =  0;
        params.img_w = 0;
        params.step_ = 64.0;
        params.step_h = 0.0;
        params.step_w = 0.0;
        params.offset = 0.5;

        refPriorBox(conv7_2_mbox_priorbox, params);
    }

    auto conv8_2_mbox_priorbox = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::ANY, {144, 2, 1});
    {
        conv8_2_mbox_priorbox->allocate();

        PriorBoxParams params;
        params.in1 = {1, 256, 3, 3};
        params.in2 = {1, 3, 300, 300};
        params.min_size = {207.0};
        params.max_size = {261.0};
        params.aspect_ratio = {2.0};
        params.flip = 1;
        params.clip = 0;
        params.variance = {0.1f, 0.1f, 0.2f, 0.2f};
        params.img_size = 0;
        params.img_h =  0;
        params.img_w = 0;
        params.step_ = 100.0;
        params.step_h = 0.0;
        params.step_w = 0.0;
        params.offset = 0.5;

        refPriorBox(conv8_2_mbox_priorbox, params);
    }

    auto conv9_2_mbox_priorbox = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::ANY, {16, 2, 1});
    {
        conv9_2_mbox_priorbox->allocate();

        PriorBoxParams params;
        params.in1 = {1, 256, 1, 1};
        params.in2 = {1, 3, 300, 300};
        params.min_size = {261.0};
        params.max_size = {315.0};
        params.aspect_ratio = {2.0};
        params.flip = 1;
        params.clip = 0;
        params.variance = {0.1f, 0.1f, 0.2f, 0.2f};
        params.img_size = 0;
        params.img_h =  0;
        params.img_w = 0;
        params.step_ = 300.0;
        params.step_h = 0.0;
        params.step_w = 0.0;
        params.offset = 0.5;

        refPriorBox(conv9_2_mbox_priorbox, params);
    }

    _refBlob = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, ANY, {34928, 2, 1});
    _refBlob->allocate();
    {
        ie_fp16* dst_ptr = _refBlob->buffer().as<ie_fp16*>();
        int dst_stride = _refBlob->dims()[0];

        int dst_offset = 0;

        auto concat = [&](const Blob::Ptr& src) {
            const ie_fp16* src_ptr = src->cbuffer().as<const ie_fp16*>();
            int num = src->dims()[0];

            for (int y = 0; y < 2; ++y) {
                for (int x = 0; x < num; ++x) {
                    dst_ptr[dst_offset + x + y * dst_stride] = src_ptr[x + y * num];
                }
            }

            dst_offset += num;
        };

        concat(conv4_3_norm_mbox_priorbox);
        concat(fc7_mbox_priorbox);
        concat(conv6_2_mbox_priorbox);
        concat(conv7_2_mbox_priorbox);
        concat(conv8_2_mbox_priorbox);
        concat(conv9_2_mbox_priorbox);
    }

    Compare(_refBlob, outputBlob, 0.0);
}
