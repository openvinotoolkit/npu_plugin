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

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <parse_layers_helpers.hpp>
#include <single_layer_common.hpp>
#include <tests_common.hpp>

using namespace InferenceEngine;

struct xmlPriorBoxClusteredParam {
    xmlPriorBoxClusteredParam(float offset, int clip, float step, float step_w, float step_h, int img_size, int img_w,
        int img_h, int flip, const std::vector<float> width, const std::vector<float> height,
        const std::vector<float> variance, const InferenceEngine::SizeVector& data_dims,
        const InferenceEngine::SizeVector& image_dims, const InferenceEngine::SizeVector& out_dims)
        : _offset(offset),
          _clip(clip),
          _step(step),
          _step_w(step_w),
          _step_h(step_h),
          _img_size(img_size),
          _img_w(img_w),
          _img_h(img_h),
          _flip(flip),
          _width(width),
          _height(height),
          _variance(variance),
          _data_dims(data_dims),
          _image_dims(image_dims),
          _out_dims(out_dims) {}
    float _offset;
    int _clip;
    float _step;
    float _step_w;
    float _step_h;
    int _img_size;
    int _img_w;
    int _img_h;
    int _flip;
    std::vector<float> _width;
    std::vector<float> _height;
    std::vector<float> _variance;
    InferenceEngine::SizeVector _data_dims;
    InferenceEngine::SizeVector _image_dims;
    InferenceEngine::SizeVector _out_dims;
};

class KmbComputePriorboxClusteredTest :
    public TestsCommon,
    public testing::WithParamInterface<xmlPriorBoxClusteredParam> {
    std::string model_t = R"V0G0N(
<Net Name="PriorBoxClustered_Only" version="2" precision="FP32" batch="1">
    <layers>
        <layer name="input1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>_IC1_</dim>
                    <dim>_IH1_</dim>
                    <dim>_IW1_</dim>
                </port>
            </output>
        </layer>
        <layer name="input2" type="Input" precision="FP32" id="1">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>_IC2_</dim>
                    <dim>_IH2_</dim>
                    <dim>_IW2_</dim>
                </port>
            </output>
        </layer>
        <layer name="priorboxclustered" type="PriorBoxClustered" precision="FP32" id="2">
            <data
                min_size="#"
                max_size="#"
                aspect_ratio="#"
                flip="_FLIP_"
                clip="_CLIP_"
                variance="_VARIANCE_"
                img_size="_IMG_SIZE_"
                img_h="_IMG_H_"
                img_w="_IMG_W_"
                step="_STEP_"
                step_h="_STEP_H_"
                step_w="_STEP_W_"
                offset="_OFFSET_"
                width="_WIDTH_"
                height="_HEIGHT_"/>
            <input>
                <port id="2">
                    <dim>1</dim>
                    <dim>_IC1_</dim>
                    <dim>_IH1_</dim>
                    <dim>_IW1_</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>_IC2_</dim>
                    <dim>_IH2_</dim>
                    <dim>_IW2_</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>_OC_</dim>
                    <dim>_OH_</dim>
                    <dim>_OW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="2"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="3"/>
    </edges>

</Net>
)V0G0N";

    std::string getModel(xmlPriorBoxClusteredParam p) {
        std::string model = model_t;

        REPLACE_WITH_NUM(model, "_IW1_", p._data_dims[3]);
        REPLACE_WITH_NUM(model, "_IH1_", p._data_dims[2]);
        REPLACE_WITH_NUM(model, "_IC1_", p._data_dims[1]);

        REPLACE_WITH_NUM(model, "_IW2_", p._image_dims[3]);
        REPLACE_WITH_NUM(model, "_IH2_", p._image_dims[2]);
        REPLACE_WITH_NUM(model, "_IC2_", p._image_dims[1]);

        REPLACE_WITH_NUM(model, "_OW_", p._out_dims[2]);
        REPLACE_WITH_NUM(model, "_OH_", p._out_dims[1]);
        REPLACE_WITH_NUM(model, "_OC_", p._out_dims[0]);

        REPLACE_WITH_STR(model, "_WIDTH_", vpu::ParseLayersHelpers::vectorToStr(p._width));
        REPLACE_WITH_STR(model, "_HEIGHT_", vpu::ParseLayersHelpers::vectorToStr(p._height));
        REPLACE_WITH_STR(model, "_VARIANCE_", vpu::ParseLayersHelpers::vectorToStr(p._variance));

        REPLACE_WITH_NUM(model, "_CLIP_", p._clip);
        REPLACE_WITH_NUM(model, "_FLIP_", p._flip);
        REPLACE_WITH_NUM(model, "_OFFSET_", p._offset);
        REPLACE_WITH_NUM(model, "_STEP_W_", p._step_w);
        REPLACE_WITH_NUM(model, "_STEP_H_", p._step_h);
        REPLACE_WITH_NUM(model, "_STEP_", p._step);
        REPLACE_WITH_NUM(model, "_IMG_SIZE_", p._img_size);
        REPLACE_WITH_NUM(model, "_IMG_W_", p._img_w);
        REPLACE_WITH_NUM(model, "_IMG_H_", p._img_h);

        return model;
    }
    std::vector<double> kmbComputePriorboxClustered(const xmlPriorBoxClusteredParam& p) {
        const int clip = p._clip;
        const int img_h = p._img_h;
        const int img_w = p._img_w;
        std::vector<float> widths = p._width;
        std::vector<float> heights = p._height;
        const float step = p._step;
        const float offset = p._offset;

        float step_w = p._step_w;
        float step_h = p._step_h;

        int img_width = p._image_dims[3];
        int img_height = p._image_dims[2];
        img_width = img_w == 0 ? img_width : img_w;
        img_height = img_h == 0 ? img_height : img_h;

        int layer_width = p._data_dims[3];
        int layer_height = p._data_dims[2];

        IE_ASSERT(widths.size() == heights.size());
        int num_priors = widths.size();

        std::vector<float> variance = p._variance;
        if (variance.empty()) {
            variance.push_back(0.1f);
        }

        if (step_w == 0 && step_h == 0) {
            if (step == 0) {
                step_w = static_cast<float>(img_width) / layer_width;
                step_h = static_cast<float>(img_height) / layer_height;
            } else {
                step_w = step;
                step_h = step;
            }
        }

        const auto& dims = p._out_dims;

        IE_ASSERT(dims.size() == 3);
        int size = dims[0] * dims[1] * dims[2];

        vpu::ParseLayersHelpers::priorBoxClusteredParam param{offset, clip, step_w, step_h, layer_width, layer_height,
            img_width, img_height, num_priors, std::move(widths), std::move(heights), std::move(variance), size};
        std::vector<double> kmb_priorbox_clustered_result = vpu::ParseLayersHelpers::computePriorboxClustered(param);
        return kmb_priorbox_clustered_result;
    }

protected:
    virtual void SetUp() {
        try {
            // prepase model
            xmlPriorBoxClusteredParam p = ::testing::WithParamInterface<xmlPriorBoxClusteredParam>::GetParam();
            std::string model = getModel(p);

            // calc priorbox output using CPU plugin
            Core ie;
            CNNNetwork network;
            ASSERT_NO_THROW(network = ie.ReadNetwork(model, Blob::CPtr()));

            network.setBatchSize(1);

            InputsDataMap inputs = network.getInputsInfo();

            DataPtr inputPtr1 = inputs["input1"]->getInputData();
            DataPtr inputPtr2 = inputs["input2"]->getInputData();

            InferenceEngine::Blob::Ptr input1 = InferenceEngine::make_shared_blob<float>(inputPtr1->getTensorDesc());
            input1->allocate();

            InferenceEngine::Blob::Ptr input2 = InferenceEngine::make_shared_blob<float>(inputPtr2->getTensorDesc());
            input2->allocate();

            InferenceEngine::BlobMap inputBlobs;
            inputBlobs["input1"] = input1;
            inputBlobs["input2"] = input2;

            OutputsDataMap outputs = network.getOutputsInfo();

            InferenceEngine::TBlob<float>::Ptr output;
            output = InferenceEngine::make_shared_blob<float>(outputs["priorboxclustered"]->getTensorDesc());
            output->allocate();

            InferenceEngine::BlobMap outputBlobs;
            outputBlobs["priorboxclustered"] = output;

            ExecutableNetwork exeNetwork = ie.LoadNetwork(network, "CPU");
            InferRequest inferRequest = exeNetwork.CreateInferRequest();
            inferRequest.SetInput(inputBlobs);
            inferRequest.SetOutput(outputBlobs);
            inferRequest.Infer();

            // calc priorbox output using kmb plugin
            std::vector<double> kmb_priorbox_clustered_result = kmbComputePriorboxClustered(p);

            // compare CPU and KMB outputs

            auto moutputHolder = output->wmap();
            float* dst_ptr = moutputHolder.as<float*>();

            for (size_t i = 0; i < kmb_priorbox_clustered_result.size(); ++i) {
                EXPECT_EQ(dst_ptr[i], kmb_priorbox_clustered_result[i]);
            }

        } catch (const InferenceEngine::details::InferenceEngineException& e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(KmbComputePriorboxClusteredTest, TestsPriorBoxClustered) {}

#ifndef __aarch64__
INSTANTIATE_TEST_CASE_P(KmbTestsPriorBoxClustered, KmbComputePriorboxClusteredTest,
    ::testing::Values(
        xmlPriorBoxClusteredParam(0.5, 0, 16.0, 0.0, 0.0, 0, 0, 0, 1,
            {9.400000, 25.100000, 14.700000, 34.700001, 143.000000, 77.400002, 128.800003, 51.099998, 75.599998},
            {15.000000, 39.599998, 25.500000, 63.200001, 227.500000, 162.899994, 124.500000, 105.099998, 72.599998},
            {0.100000, 0.100000, 0.200000, 0.200000}, {1, 384, 19, 19}, {1, 3, 300, 300}, {1, 2, 12996}),
        xmlPriorBoxClusteredParam(0.5, 0, 16.0, 16.0, 16.0, 0, 0, 0, 1,
            {9.400000, 25.100000, 14.700000, 34.700001, 143.000000, 77.400002, 128.800003, 51.099998, 75.599998},
            {15.000000, 39.599998, 25.500000, 63.200001, 227.500000, 162.899994, 124.500000, 105.099998, 72.599998},
            {0.100000, 0.100000, 0.200000, 0.200000}, {1, 384, 19, 19}, {1, 3, 300, 300}, {1, 2, 12996})));
#endif
