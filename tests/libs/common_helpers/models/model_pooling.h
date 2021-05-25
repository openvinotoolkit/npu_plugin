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

#pragma once

#include "model_helper.h"
//------------------------------------------------------------------------------
class ModelPooling_Helper : public ModelHelper {
public:
    ModelPooling_Helper();

protected:
    InferenceEngine::Blob::CPtr _weights;

    const std::string _model = R"V0G0N(
    <net batch="1" name="POOLING_TEST" version="2">
    <layers>
        <layer id="0" name="input" precision="U8" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>224</dim>
                    <dim>224</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="pooling_test" precision="U8" type="Pooling">
            <data kernel="3,3" strides="2,2" exclude-pad="true" pool-method="max" pads_begin="0,0" pads_end="0,0"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>224</dim>
                    <dim>224</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>125</dim>
                    <dim>125</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
    </net>
        )V0G0N";
};

//------------------------------------------------------------------------------
inline ModelPooling_Helper::ModelPooling_Helper() {
    InferenceEngine::Core ie;
    _network = ie.ReadNetwork(_model, _weights);
}
