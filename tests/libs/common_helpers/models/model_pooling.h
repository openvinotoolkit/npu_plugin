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
