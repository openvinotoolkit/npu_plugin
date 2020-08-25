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
#include <single_layer_common.hpp>
#include <tests_common.hpp>

#include "parse_layers_helpers.hpp"

using namespace InferenceEngine;

class KmbComputePriorboxTest :
    public TestsCommon,
    public testing::WithParamInterface<vpu::KmbPlugin::utils::priorBoxParam> {
    std::string model_t = R"V0G0N(
<Net Name="PriorBox_Only" version="2" precision="FP32" batch="1">
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
        <layer name="prior" type="PriorBox" precision="FP32" id="2">
                <data aspect_ratio="_ASP_RAT_" clip="_CLIP_" density="_DENSITY_" fixed_ratio="_F_RAT_" fixed_size="_F_SIZE_" flip="_FLIP_" max_size="_MAX_SIZE_" min_size="_MIN_SIZE_" offset="_OFFSET_" step="_STEP_" variance="_VARIANCE_"/>
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
                    <dim>1</dim>
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

    std::string getModel(vpu::KmbPlugin::utils::priorBoxParam p) {
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

        REPLACE_WITH_STR(model, "_ASP_RAT_", vpu::KmbPlugin::utils::vectorToStr(p._src_aspect_ratios));
        REPLACE_WITH_STR(model, "_DENSITY_", vpu::KmbPlugin::utils::vectorToStr(p._densitys));
        REPLACE_WITH_STR(model, "_F_RAT_", vpu::KmbPlugin::utils::vectorToStr(p._fixed_ratios));
        REPLACE_WITH_STR(model, "_F_SIZE_", vpu::KmbPlugin::utils::vectorToStr(p._densitys));
        REPLACE_WITH_STR(model, "_MAX_SIZE_", vpu::KmbPlugin::utils::vectorToStr(p._max_sizes));
        REPLACE_WITH_STR(model, "_MIN_SIZE_", vpu::KmbPlugin::utils::vectorToStr(p._min_sizes));
        REPLACE_WITH_STR(model, "_VARIANCE_", vpu::KmbPlugin::utils::vectorToStr(p._src_variance));

        REPLACE_WITH_NUM(model, "_CLIP_", p._clip ? 1 : 0);
        REPLACE_WITH_NUM(model, "_FLIP_", p._flip ? 1 : 0);

        REPLACE_WITH_NUM(model, "_OFFSET_", p._offset);
        REPLACE_WITH_NUM(model, "_STEP_", p._step);

        return model;
    }

protected:
    virtual void SetUp() {
        try {
            // prepase model
            vpu::KmbPlugin::utils::priorBoxParam p =
                ::testing::WithParamInterface<vpu::KmbPlugin::utils::priorBoxParam>::GetParam();
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
            output = InferenceEngine::make_shared_blob<float>(outputs["prior"]->getTensorDesc());
            output->allocate();

            InferenceEngine::BlobMap outputBlobs;
            outputBlobs["prior"] = output;

            ExecutableNetwork exeNetwork = ie.LoadNetwork(network, "CPU");
            InferRequest inferRequest = exeNetwork.CreateInferRequest();
            inferRequest.SetInput(inputBlobs);
            inferRequest.SetOutput(outputBlobs);
            inferRequest.Infer();

            // calc priorbox output using kmb plugin
            std::vector<double> kmb_priorbox_result = vpu::KmbPlugin::utils::computePriorbox(p);

            // compare CPU and KMB outputs

            const TBlob<float>::Ptr outputArray = std::dynamic_pointer_cast<TBlob<float>>(output);
            float* dst_ptr = outputArray->data();

            for (size_t i = 0; i < kmb_priorbox_result.size(); ++i) {
                EXPECT_EQ(dst_ptr[i], kmb_priorbox_result[i]);
            }

        } catch (const InferenceEngine::details::InferenceEngineException& e) {
            FAIL() << e.what();
        }
    }
};

TEST_P(KmbComputePriorboxTest, TestsPriorBox) {}

#ifndef __aarch64__
INSTANTIATE_TEST_CASE_P(KmbTestsPriorBox, KmbComputePriorboxTest,
    ::testing::Values(vpu::KmbPlugin::utils::priorBoxParam(0.5, 16, {76.8}, {153.6}, true, false, true, {}, {}, {},
                          {2, 3}, {0.1, 0.1, 0.2, 0.2}, {1, 1024, 32, 32}, {1, 3, 512, 512}, {2, 1, 24576}),
        vpu::KmbPlugin::utils::priorBoxParam(0.5, 32, {153.6}, {230.4}, true, false, true, {}, {}, {}, {2, 3},
            {0.1, 0.1, 0.2, 0.2}, {1, 512, 16, 16}, {1, 3, 512, 512}, {2, 1, 6144}),
        vpu::KmbPlugin::utils::priorBoxParam(0.5, 64, {230.4}, {307.2}, true, false, true, {}, {}, {}, {2, 3},
            {0.1, 0.1, 0.2, 0.2}, {1, 256, 8, 8}, {1, 3, 512, 512}, {2, 1, 1536}),
        vpu::KmbPlugin::utils::priorBoxParam(0.5, 128, {307.2}, {384.0}, true, false, true, {}, {}, {}, {2, 3},
            {0.1, 0.1, 0.2, 0.2}, {1, 256, 4, 4}, {1, 3, 512, 512}, {2, 1, 384}),
        vpu::KmbPlugin::utils::priorBoxParam(0.5, 256, {384.0}, {460.8}, true, false, true, {}, {}, {}, {2},
            {0.1, 0.1, 0.2, 0.2}, {1, 256, 2, 2}, {1, 3, 512, 512}, {2, 1, 64}),
        vpu::KmbPlugin::utils::priorBoxParam(0.5, 512, {460.8}, {537.6}, true, false, true, {}, {}, {}, {2},
            {0.1, 0.1, 0.2, 0.2}, {1, 256, 1, 1}, {1, 3, 512, 512}, {2, 1, 16})));
#endif
