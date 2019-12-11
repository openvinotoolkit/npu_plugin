// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

static std::string full_quant_model = R"V0G0N(
    <?xml version="1.0" ?>
    <net batch="1" name="resnet50-int8-fragment" version="5">
    <layers>
        <layer id="0" name="input" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="conv1" precision="FP32" type="Convolution">
            <data auto_pad="same_upper" dilations="1,1" group="1" kernel="3,3" output="64" pads_begin="1,1" pads_end="1,1" strides="1,1"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                </port>
            </output>
            <blobs>
                <weights offset="0" size="147456"/>
                <biases offset="147456" size="256"/>
            </blobs>
        </layer>
        <layer id="2" name="conv1_relu" precision="FP32" type="ReLU">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="conv2" precision="FP32" type="Convolution">
            <data auto_pad="same_upper" dilations="1,1" group="1" kernel="1,1" output="256" pads_begin="0,0" pads_end="0,0" strides="1,1"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                </port>
            </output>
            <blobs>
                <weights offset="147712" size="65536"/>
                <biases offset="213248" size="1024"/>
            </blobs>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="3" to-layer="2" to-port="0"/>
        <edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>
    </edges>
    <statistics>
        <layer>
            <name>input</name>
            <min>0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0</min>
            <max>10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047, 10.544994354248047</max>
        </layer>
        <layer>
            <name>conv1_relu</name>
            <min>0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0</min>
            <max>12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559, 12.449431419372559</max>
        </layer>
        <layer>
            <name>conv2</name>
            <min>-11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415, -11.959489201564415</min>
            <max>10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074, 10.97035851572074</max>
        </layer>
    </statistics>
    </net>
    )V0G0N";

static std::string convolution_only = R"V0G0N(
    <net batch="1" name="CONVOLUTION_TEST" version="2">
        <layers>
            <layer id="0" name="input" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>224</dim>
                        <dim>224</dim>
                    </port>
                </output>
            </layer>
            <layer id="2" name="conv_test1" precision="FP16" type="Convolution">
                <data dilations="1,1" group="1" kernel="7,7" output="64" pads_begin="3,3" pads_end="3,3" strides="2,2"/>
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
                        <dim>64</dim>
                        <dim>112</dim>
                        <dim>112</dim>
                    </port>
                </output>
                <blobs>
                    <weights offset="0" size="18816"/>
                    <biases offset="18816" size="128"/>
                </blobs>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        </edges>
    </net>
        )V0G0N";

static std::string convolution_only_with_bias_template = R"V0G0N(
<net batch="1" name="CONVOLUTION_TEST" version="6">
    <layers>
        <layer id="0" name="input" precision="_NET_PRECISION_" type="Input">
            <output>
                <port id="1">
                    <dim>_INPUT_BATCH_</dim>
                    <dim>_INPUT_CHANNEL_</dim>
                    <dim>_INPUT_HEIGHT_</dim>
                    <dim>_INPUT_WIDTH_</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="conv_test1/weights" precision="_WEIGHTS_PRECISION_" type="Const">
            <output>
                <port id="1">
                    <dim>_OUTPUT_CHANNEL_</dim>
                    <dim>_INPUT_CHANNEL_</dim>
                    <dim>_KERNELY_</dim>
                    <dim>_KERNELX_</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="_WEIGHTS_BYTE_SIZE_"/>
            </blobs>
        </layer>
        <layer id="2" name="conv_test1/bias" precision="_BIAS_PRECISION_" type="Const">
            <output>
                <port id="1">
                    <dim>_OUTPUT_CHANNEL_</dim>
                </port>
            </output>
            <blobs>
                <custom offset="_BIAS_OFFSET_" size="_BIAS_BYTE_SIZE_"/>
            </blobs>
        </layer>
        <layer id="3" name="output" precision="_CONV_PRECISION_" type="Convolution">
            <data kernel="_KERNEL_" output="_OUTPUT_CHANNEL_" strides="_STRIDE_" dilations="1,1" group="1"   pads_begin="_PADS_BEGIN_" pads_end="_PADS_END_" />
            <input>
                <port id="0">
                    <dim>_INPUT_BATCH_</dim>
                    <dim>_INPUT_CHANNEL_</dim>
                    <dim>_INPUT_HEIGHT_</dim>
                    <dim>_INPUT_WIDTH_</dim>
                </port>
                <port id="1">
                    <dim>_OUTPUT_CHANNEL_</dim>
                    <dim>_INPUT_CHANNEL_</dim>
                    <dim>_KERNELY_</dim>
                    <dim>_KERNELX_</dim>
                </port>
                <port id="2">
                    <dim>_OUTPUT_CHANNEL_</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>_OUTPUT_BATCH_</dim>
                    <dim>_OUTPUT_CHANNEL_</dim>
                    <dim>_OUTPUT_HEIGHT_</dim>
                    <dim>_OUTPUT_WIDTH_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="1" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="1" to-layer="3" to-port="2"/>
    </edges>
</net>
            )V0G0N";

static std::string t_fq_convolution_only_slim = R"V0G0N(
<?xml version="1.0" ?>
<net batch="1" name="resnet50v1-int8-onnx-0001" version="6">
    <layers>
        <layer id="0" name="input_.2" precision="FP32" type="Input">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>224</dim>
                    <dim>224</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Copy_574/Output_0/Data_/copy_const" precision="FP32" type="Const">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="0" size="12"/>
            </blobs>
        </layer>
        <layer id="2" name="Copy_575/Output_0/Data_/copy_const" precision="FP32" type="Const">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="12" size="12"/>
            </blobs>
        </layer>
        <layer id="3" name="Copy_574/Output_0/Data_3126/copy_const" precision="FP32" type="Const">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="24" size="4"/>
            </blobs>
        </layer>
        <layer id="4" name="Copy_575/Output_0/Data_3090/copy_const" precision="FP32" type="Const">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="28" size="4"/>
            </blobs>
        </layer>
        <layer id="5" name="576" precision="FP32" type="FakeQuantize">
            <data levels="256"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>224</dim>
                    <dim>224</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="5">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>224</dim>
                    <dim>224</dim>
                </port>
            </output>
        </layer>
        <layer id="6" name="module.conv1.weight/Output_0/Data__const" precision="FP32" type="Const">
            <output>
                <port id="1">
                    <dim>64</dim>
                    <dim>3</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </output>
            <blobs>
                <custom offset="32" size="37632"/>
            </blobs>
        </layer>
        <layer id="7" name="Copy_571/Output_0/Data_/copy_const" precision="FP32" type="Const">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="37664" size="4"/>
            </blobs>
        </layer>
        <layer id="8" name="Copy_572/Output_0/Data_/copy_const" precision="FP32" type="Const">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="37668" size="4"/>
            </blobs>
        </layer>
        <layer id="9" name="Mul1_15869/Fused_Mul_2036720369_const" precision="FP32" type="Const">
            <output>
                <port id="1">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="37672" size="256"/>
            </blobs>
        </layer>
        <layer id="10" name="Mul1_15869/Fused_Mul_203672402824029_const" precision="FP32" type="Const">
            <output>
                <port id="1">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="37928" size="256"/>
            </blobs>
        </layer>
        <layer id="11" name="573" precision="FP32" type="FakeQuantize">
            <data levels="255"/>
            <input>
                <port id="0">
                    <dim>64</dim>
                    <dim>3</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="3">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="4">
                    <dim>64</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="5">
                    <dim>64</dim>
                    <dim>3</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </output>
        </layer>
        <layer id="12" name="bias_data20371_const" precision="FP32" type="Const">
            <output>
                <port id="1">
                    <dim>64</dim>
                </port>
            </output>
            <blobs>
                <custom offset="38184" size="256"/>
            </blobs>
        </layer>
        <layer id="13" name="577" precision="FP32" type="Convolution">
            <data dilations="1,1" group="1" kernel="7,7" output="64" pads_begin="3,3" pads_end="3,3" strides="2,2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>224</dim>
                    <dim>224</dim>
                </port>
                <port id="1">
                    <dim>64</dim>
                    <dim>3</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
                <port id="2">
                    <dim>64</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="14" name="Copy_580/Output_0/Data_/copy_const" precision="FP32" type="Const">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="38440" size="4"/>
            </blobs>
        </layer>
        <layer id="15" name="Copy_581/Output_0/Data_/copy_const" precision="FP32" type="Const">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="38444" size="4"/>
            </blobs>
        </layer>
        <layer id="16" name="Copy_580/Output_0/Data_3240/copy_const" precision="FP32" type="Const">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="38440" size="4"/>
            </blobs>
        </layer>
        <layer id="17" name="Copy_581/Output_0/Data_3372/copy_const" precision="FP32" type="Const">
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </output>
            <blobs>
                <custom offset="38444" size="4"/>
            </blobs>
        </layer>
        <layer id="18" name="582" precision="FP32" type="FakeQuantize">
            <data levels="256"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                    <dim>1</dim>
                </port>
            </input>
            <output>
                <port id="5">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </output>
        </layer>
        <layer id="19" name="583" precision="FP32" type="Pooling">
            <data exclude-pad="true" kernel="3,3" pads_begin="1,1" pads_end="1,1" pool-method="max" rounding_type="floor" strides="2,2"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>112</dim>
                    <dim>112</dim>
                </port>
            </input>
            <output>
                <port id="1">
                    <dim>1</dim>
                    <dim>64</dim>
                    <dim>56</dim>
                    <dim>56</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="5" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="5" to-port="1"/>
        <edge from-layer="2" from-port="1" to-layer="5" to-port="2"/>
        <edge from-layer="3" from-port="1" to-layer="5" to-port="3"/>
        <edge from-layer="4" from-port="1" to-layer="5" to-port="4"/>
        <edge from-layer="6" from-port="1" to-layer="11" to-port="0"/>
        <edge from-layer="7" from-port="1" to-layer="11" to-port="1"/>
        <edge from-layer="8" from-port="1" to-layer="11" to-port="2"/>
        <edge from-layer="9" from-port="1" to-layer="11" to-port="3"/>
        <edge from-layer="10" from-port="1" to-layer="11" to-port="4"/>
        <edge from-layer="5" from-port="5" to-layer="13" to-port="0"/>
        <edge from-layer="11" from-port="5" to-layer="13" to-port="1"/>
        <edge from-layer="12" from-port="1" to-layer="13" to-port="2"/>
        <edge from-layer="13" from-port="3" to-layer="18" to-port="0"/>
        <edge from-layer="14" from-port="1" to-layer="18" to-port="1"/>
        <edge from-layer="15" from-port="1" to-layer="18" to-port="2"/>
        <edge from-layer="16" from-port="1" to-layer="18" to-port="3"/>
        <edge from-layer="17" from-port="1" to-layer="18" to-port="4"/>
        <edge from-layer="18" from-port="5" to-layer="19" to-port="0"/>
    </edges>
    <meta_data>
        <MO_version value="unknown version"/>
        <cli_parameters>
            <blobs_as_inputs value="True"/>
            <caffe_parser_path value="DIR"/>
            <data_type value="FP32"/>
            <disable_nhwc_to_nchw value="False"/>
            <disable_omitting_optional value="False"/>
            <disable_resnet_optimization value="False"/>
            <enable_concat_optimization value="False"/>
            <enable_flattening_nested_params value="False"/>
            <enable_ssd_gluoncv value="False"/>
            <extensions value="DIR"/>
            <framework value="onnx"/>
            <freeze_placeholder_with_value value="{}"/>
            <generate_experimental_IR_V10 value="False"/>
            <input value="input_.2"/>
            <input_model value="DIR/resnet50-v1-int8.onnx"/>
            <input_model_is_text value="False"/>
            <input_shape value="[1,3,224,224]"/>
            <k value="DIR/CustomLayersMapping.xml"/>
            <keep_quantize_ops_in_IR value="True"/>
            <keep_shape_ops value="False"/>
            <legacy_mxnet_model value="False"/>
            <log_level value="ERROR"/>
            <mean_scale_values value="{'input_.2': {'mean': array([123.675, 116.28 , 103.53 ]), 'scale': array([58.47953216, 57.14285714, 57.47126437])}}"/>
            <mean_values value="input_.2[123.675,116.28,103.53]"/>
            <model_name value="resnet50v1-int8-onnx-0001"/>
            <move_to_preprocess value="False"/>
            <output value="['1127']"/>
            <output_dir value="DIR"/>
            <placeholder_shapes value="{'input_.2': array([  1,   3, 224, 224])}"/>
            <progress value="False"/>
            <remove_memory value="False"/>
            <remove_output_softmax value="False"/>
            <reverse_input_channels value="True"/>
            <save_params_from_nd value="False"/>
            <scale_values value="input_.2[58.47953216374269,57.14285714285714,57.4712643678161]"/>
            <silent value="False"/>
            <stream_output value="False"/>
            <version value="False"/>
            <unset unset_cli_parameters="batch, counts, disable_fusing, disable_gfusing, finegrain_fusing, generate_deprecated_IR_V2, input_checkpoint, input_meta_graph, input_proto, input_symbol, mean_file, mean_file_offsets, nd_prefix_name, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_operation_patterns, tensorflow_subgraph_patterns, tensorflow_use_custom_operations_config"/>
        </cli_parameters>
    </meta_data>
</net>
)V0G0N";

static std::string fq_convolution_only_slim = R"V0G0N(
<net batch="1" name="resnet50-int8" version="6">
	<layers>
		<layer id="36" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>_INPUT_BATCH_</dim>
					<dim>_INPUT_CHANNEL_</dim>
					<dim>_INPUT_HEIGHT_</dim>
					<dim>_INPUT_WIDTH_</dim>
				</port>
			</output>
		</layer>
        <layer id="37" name="input_input_min" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="32" size="4"/>
			</blobs>
		</layer>
		<layer id="38" name="input_input_max" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="36" size="4"/>
			</blobs>
		</layer>
		<layer id="39" name="input_output_min" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="40" size="4"/>
			</blobs>
		</layer>
		<layer id="40" name="input_output_max" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="44" size="4"/>
			</blobs>
		</layer>
		<layer id="41" name="input/quntize" precision="FP32" type="FakeQuantize">
			<data levels="256"/>
			<input>
				<port id="0">
					<dim>_INPUT_BATCH_</dim>
					<dim>_INPUT_CHANNEL_</dim>
					<dim>_INPUT_HEIGHT_</dim>
					<dim>_INPUT_WIDTH_</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="5">
					<dim>_INPUT_BATCH_</dim>
					<dim>_INPUT_CHANNEL_</dim>
					<dim>_INPUT_HEIGHT_</dim>
					<dim>_INPUT_WIDTH_</dim>
				</port>
			</output>
		</layer>
		<layer id="42" name="conv2/weights" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>_OUTPUT_CHANNEL_</dim>
					<dim>_INPUT_CHANNEL_</dim>
					<dim>_KERNEL_SIZE_</dim>
					<dim>_KERNEL_SIZE_</dim>
				</port>
			</output>
			<blobs>
				<custom offset="_WEIGHTS_OFFSET_" size="_WEIGHTS_BYTE_SIZE_"/>
			</blobs>
		</layer>
		<layer id="43" name="conv2/weights_quant/FakeQuantWithMinMaxVars/nudged_min" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" size="4"/>
			</blobs>
		</layer>
		<layer id="44" name="conv2/weights_quant/FakeQuantWithMinMaxVars/nudged_max" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="4" size="4"/>
			</blobs>
		</layer>
		<layer id="45" name="conv2/weights_quant/FakeQuantWithMinMaxVars/nudged_min2" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="8" size="4"/>
			</blobs>
		</layer>
		<layer id="46" name="conv2/weights_quant/FakeQuantWithMinMaxVars/nudged_max2" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="12" size="4"/>
			</blobs>
		</layer>
		<layer id="47" name="conv2/weights_quant/FakeQuantWithMinMaxVars" precision="FP32" type="FakeQuantize">
			<data levels="255"/>
			<input>
				<port id="0">
					<dim>_OUTPUT_CHANNEL_</dim>
					<dim>_INPUT_CHANNEL_</dim>
					<dim>_KERNEL_SIZE_</dim>
					<dim>_KERNEL_SIZE_</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="5">
					<dim>_OUTPUT_CHANNEL_</dim>
					<dim>_INPUT_CHANNEL_</dim>
					<dim>_KERNEL_SIZE_</dim>
					<dim>_KERNEL_SIZE_</dim>
				</port>
			</output>
		</layer>
		<layer id="48" name="bias2" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>_OUTPUT_CHANNEL_</dim>
				</port>
			</output>
			<blobs>
				<custom offset="_BIAS_OFFSET_" size="_BIAS_BYTE_SIZE_"/>
			</blobs>
		</layer>
		<layer id="49" name="conv2" precision="FP32" type="Convolution">
			<data output="_OUTPUT_CHANNEL_" kernel="_KERNEL_" strides="_STRIDE_" dilations="1,1" group="1" pads_begin="0,0" pads_end="0,0"/>
			<input>
				<port id="0">
					<dim>_INPUT_BATCH_</dim>
					<dim>_INPUT_CHANNEL_</dim>
					<dim>_INPUT_HEIGHT_</dim>
					<dim>_INPUT_WIDTH_</dim>
				</port>
				<port id="1">
					<dim>_OUTPUT_CHANNEL_</dim>
					<dim>_INPUT_CHANNEL_</dim>
					<dim>_KERNEL_SIZE_</dim>
					<dim>_KERNEL_SIZE_</dim>
				</port>
				<port id="2">
					<dim>_OUTPUT_CHANNEL_</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>_OUTPUT_BATCH_</dim>
					<dim>_OUTPUT_CHANNEL_</dim>
					<dim>_OUTPUT_HEIGHT_</dim>
					<dim>_OUTPUT_WIDTH_</dim>
				</port>
			</output>
		</layer>
        <layer id="50" name="conv2/after_quant/FakeQuantWithMinMaxVars/nudged_min" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="16" size="4"/>
			</blobs>
		</layer>
		<layer id="51" name="conv2/after_quant/FakeQuantWithMinMaxVars/nudged_max" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="20" size="4"/>
			</blobs>
		</layer>
		<layer id="52" name="conv2/after_quant/FakeQuantWithMinMaxVars/nudged_min2" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="24" size="4"/>
			</blobs>
		</layer>
		<layer id="53" name="conv2/after_quant/FakeQuantWithMinMaxVars/nudged_max2" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="28" size="4"/>
			</blobs>
		</layer>
		<layer id="54" name="583" precision="FP32" type="FakeQuantize">
			<data levels="256"/>
			<input>
				<port id="0">
					<dim>_OUTPUT_BATCH_</dim>
					<dim>_OUTPUT_CHANNEL_</dim>
					<dim>_OUTPUT_HEIGHT_</dim>
					<dim>_OUTPUT_WIDTH_</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="5">
					<dim>_OUTPUT_BATCH_</dim>
					<dim>_OUTPUT_CHANNEL_</dim>
					<dim>_OUTPUT_HEIGHT_</dim>
					<dim>_OUTPUT_WIDTH_</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="36" from-port="0" to-layer="41" to-port="0"/>
		<edge from-layer="37" from-port="1" to-layer="41" to-port="1"/>
		<edge from-layer="38" from-port="1" to-layer="41" to-port="2"/>
		<edge from-layer="39" from-port="1" to-layer="41" to-port="3"/>
		<edge from-layer="40" from-port="1" to-layer="41" to-port="4"/>
		<edge from-layer="41" from-port="5" to-layer="49" to-port="0"/>
		<edge from-layer="42" from-port="1" to-layer="47" to-port="0"/>
		<edge from-layer="43" from-port="1" to-layer="47" to-port="1"/>
		<edge from-layer="44" from-port="1" to-layer="47" to-port="2"/>
		<edge from-layer="45" from-port="1" to-layer="47" to-port="3"/>
		<edge from-layer="46" from-port="1" to-layer="47" to-port="4"/>
		<edge from-layer="47" from-port="5" to-layer="49" to-port="1"/>
		<edge from-layer="48" from-port="1" to-layer="49" to-port="2"/>
		<edge from-layer="49" from-port="3" to-layer="54" to-port="0"/>
		<edge from-layer="50" from-port="1" to-layer="54" to-port="1"/>
		<edge from-layer="51" from-port="1" to-layer="54" to-port="2"/>
		<edge from-layer="52" from-port="1" to-layer="54" to-port="3"/>
		<edge from-layer="53" from-port="1" to-layer="54" to-port="4"/>
	</edges>
	<meta_data>
		<MO_version value="unknown version"/>
		<cli_parameters>
			<blobs_as_inputs value="True"/>
			<caffe_parser_path value="DIR"/>
			<data_type value="FP32"/>
			<disable_nhwc_to_nchw value="False"/>
			<disable_omitting_optional value="False"/>
			<disable_resnet_optimization value="False"/>
			<enable_concat_optimization value="False"/>
			<enable_flattening_nested_params value="False"/>
			<enable_ssd_gluoncv value="False"/>
			<extensions value="DIR"/>
			<framework value="tf"/>
			<freeze_placeholder_with_value value="{}"/>
			<generate_experimental_IR_V10 value="False"/>
			<input value="input"/>
			<input_model value="DIR/resnet50-int8.pb"/>
			<input_model_is_text value="False"/>
			<input_shape value="[1, 224, 224, 3]"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_quantize_ops_in_IR value="True"/>
			<keep_shape_ops value="False"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<mean_scale_values value="{'input': {'scale': array([1.]), 'mean': array([123.68, 116.78, 103.94])}}"/>
			<mean_values value="input[123.68,116.78,103.94]"/>
			<move_to_preprocess value="False"/>
			<output_dir value="DIR"/>
			<placeholder_shapes value="{'input': array([  1, 224, 224,   3])}"/>
			<progress value="False"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="True"/>
			<save_params_from_nd value="False"/>
			<scale_values value="input[1.0]"/>
			<silent value="False"/>
			<stream_output value="False"/>
			<version value="False"/>
			<unset unset_cli_parameters="batch, counts, disable_fusing, disable_gfusing, finegrain_fusing, generate_deprecated_IR_V2, input_checkpoint, input_meta_graph, input_proto, input_symbol, mean_file, mean_file_offsets, model_name, nd_prefix_name, output, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_operation_patterns, tensorflow_subgraph_patterns, tensorflow_use_custom_operations_config"/>
		</cli_parameters>
	</meta_data>
</net>
        )V0G0N";

static std::string conv_after_scale_shift = R"V0G0N(
    <net batch="1" name="CONVOLUTION_TEST" version="2">
        <layers>
            <layer id="0" name="input" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>224</dim>
                        <dim>224</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="scale_shift1" precision="FP16" type="ScaleShift">
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
                        <dim>224</dim>
                        <dim>224</dim>
                    </port>
                </output>
                <blobs>
                    <weights offset="0" size="6"/>
                    <biases offset="6" size="6"/>
                </blobs>
            </layer>
            <layer id="2" name="conv_test1" precision="FP16" type="Convolution">
                <data dilations="1,1" group="1" kernel="7,7" output="64" pads_begin="3,3" pads_end="3,3" strides="2,2"/>
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
                        <dim>64</dim>
                        <dim>112</dim>
                        <dim>112</dim>
                    </port>
                </output>
                <blobs>
                    <weights offset="12" size="18816"/>
                    <biases offset="18828" size="128"/>
                </blobs>
            </layer>
        </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
            <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
        </edges>
    </net>
        )V0G0N";


static std::string pooling_test2 = R"V0G0N(
    <net batch="1" name="POOLING_TEST" version="2">
        <layers>
            <layer id="0" name="input" precision="FP16" type="Input">
                <output>
                    <port id="0">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>224</dim>
                        <dim>224</dim>
                    </port>
                </output>
            </layer>
            <layer id="1" name="pooling_test" precision="FP16" type="Pooling">
                <data auto_pad="same_upper" exclude-pad="true" kernel="3,3" pads_begin="0,0" pads_end="1,1" pool-method="max" strides="2,2"/>
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
                        <dim>224</dim>
                        <dim>224</dim>
                    </port>
                </output>
            </layer>
            </layers>
        <edges>
            <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        </edges>
    </net>
        )V0G0N";

static std::string convolution_u8_only = R"V0G0N(
<net batch="1" name="resnet50-int8" version="6">
	<layers>
		<layer id="36" name="input" precision="U8" type="Input">
			<output>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
		<layer id="42" name="conv2/weights" precision="U8" type="Const">
			<output>
				<port id="1">
					<dim>64</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" size="36864"/>
			</blobs>
		</layer>
		<layer id="48" name="bias2" precision="I32" type="Const">
			<output>
				<port id="1">
					<dim>64</dim>
				</port>
			</output>
			<blobs>
				<custom offset="36864" size="256"/>
			</blobs>
		</layer>
		<layer id="49" name="conv2" precision="U8" type="Convolution">
			<data kernel="3,3" output="64" strides="1,1" auto_pad="same_upper" dilations="1,1" group="1" pads_begin="1,1" pads_end="1,1" />
			<input>
				<port id="0">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
				<port id="1">
					<dim>64</dim>
					<dim>64</dim>
					<dim>3</dim>
					<dim>3</dim>
				</port>
				<port id="2">
					<dim>64</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>1</dim>
					<dim>64</dim>
					<dim>16</dim>
					<dim>16</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="36" from-port="0" to-layer="49" to-port="0"/>
		<edge from-layer="42" from-port="1" to-layer="49" to-port="1"/>
		<edge from-layer="48" from-port="1" to-layer="49" to-port="2"/>
	</edges>
</net>
        )V0G0N";


static std::string relu_test_2 = R"V0G0N(
        <net batch="1" name="RELU_TEST" version="2">
            <layers>
                <layer id="0" name="input" precision="FP16" type="Input">
                    <output>
                        <port id="0">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>112</dim>
                            <dim>112</dim>
                        </port>
                    </output>
                </layer>
                <layer id="3" name="relu_test" precision="FP16" type="ReLU">
                    <input>
                        <port id="0">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>112</dim>
                            <dim>112</dim>
                        </port>
                    </input>
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>112</dim>
                            <dim>112</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
            </edges>
        </net>
            )V0G0N";

static std::string conv_relu_u8_test = R"V0G0N(
        <net batch="1" name="RELU_TEST" version="2">
            <layers>
                <layer id="0" name="input" precision="U8" type="Input">
                    <output>
                        <port id="1">
                            <dim>_CONV_INPUT_BATCH_</dim>
                            <dim>_CONV_INPUT_CHANNEL_</dim>
                            <dim>_CONV_INPUT_HEIGHT_</dim>
                            <dim>_CONV_INPUT_WIDTH_</dim>
                        </port>
                    </output>
		        </layer>
                <layer id="1" name="weights" precision="U8" type="Const">
                    <output>
                        <port id="1">
                            <dim>_CONV_OUTPUT_CHANNEL_</dim>
                            <dim>_CONV_INPUT_CHANNEL_</dim>
                            <dim>_CONV_KERNEL_SIZE_</dim>
                            <dim>_CONV_KERNEL_SIZE_</dim>
                        </port>
                    </output>
                    <blobs>
                        <custom offset="0" size="_CONV_WEIGHTS_BYTE_SIZE_"/>
                    </blobs>
                </layer>
                <layer id="2" name="bias" precision="I32" type="Const">
                    <output>
                        <port id="1">
                            <dim>_CONV_OUTPUT_CHANNEL_</dim>
                        </port>
                    </output>
                    <blobs>
                        <custom offset="_CONV_BIAS_OFFSET_" size="_CONV_BIAS_BYTE_SIZE_"/>
                    </blobs>
                </layer>
                <layer id="3" name="conv" precision="U8" type="Convolution">
                    <data kernel="_CONV_KERNEL_" output="_CONV_OUTPUT_CHANNEL_" strides="_CONV_STRIDE_" dilations="1,1" group="1" pads_begin="0,0" pads_end="0,0" />
                    <input>
                        <port id="0">
                            <dim>_CONV_INPUT_BATCH_</dim>
                            <dim>_CONV_INPUT_CHANNEL_</dim>
                            <dim>_CONV_INPUT_HEIGHT_</dim>
                            <dim>_CONV_INPUT_WIDTH_</dim>
                        </port>
                        <port id="1">
                            <dim>_CONV_OUTPUT_CHANNEL_</dim>
                            <dim>_CONV_INPUT_CHANNEL_</dim>
                            <dim>_CONV_KERNEL_SIZE_</dim>
                            <dim>_CONV_KERNEL_SIZE_</dim>
                        </port>
                        <port id="2">
                            <dim>_CONV_OUTPUT_CHANNEL_</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>_CONV_OUTPUT_BATCH_</dim>
                            <dim>_CONV_OUTPUT_CHANNEL_</dim>
                            <dim>_CONV_OUTPUT_HEIGHT_</dim>
                            <dim>_CONV_OUTPUT_WIDTH_</dim>
                        </port>
                    </output>
                </layer>
                <layer id="4" name="relu" precision="U8" type="ReLU">
                    <input>
                        <port id="0">
                            <dim>_CONV_OUTPUT_BATCH_</dim>
                            <dim>_CONV_OUTPUT_CHANNEL_</dim>
                            <dim>_CONV_OUTPUT_HEIGHT_</dim>
                            <dim>_CONV_OUTPUT_WIDTH_</dim>
                        </port>
                    </input>
                    <output>
                        <port id="1">
                            <dim>_CONV_OUTPUT_BATCH_</dim>
                            <dim>_CONV_OUTPUT_CHANNEL_</dim>
                            <dim>_CONV_OUTPUT_HEIGHT_</dim>
                            <dim>_CONV_OUTPUT_WIDTH_</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="1" to-layer="3" to-port="0"/>
                <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
                <edge from-layer="2" from-port="1" to-layer="3" to-port="2"/>
                <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
            </edges>
        </net>
)V0G0N";

static std::string fq_fully_connected_only_slim = R"V0G0N(
<net batch="1" name="resnet50-int8" version="6">
	<layers>
		<layer id="36" name="input" precision="FP32" type="Input">
			<output>
				<port id="0">
					<dim>_INPUT_BATCH_</dim>
					<dim>_INPUT_CHANNEL_</dim>
				</port>
			</output>
		</layer>
        <layer id="37" name="conv1/act_quant/FakeQuantWithMinMaxVars/nudged_min" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="0" size="4"/>
			</blobs>
		</layer>
		<layer id="38" name="conv1/act_quant/FakeQuantWithMinMaxVars/nudged_max" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="4" size="4"/>
			</blobs>
		</layer>
		<layer id="39" name="conv1/act_quant/FakeQuantWithMinMaxVars/nudged_min2" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="8" size="4"/>
			</blobs>
		</layer>
		<layer id="40" name="conv1/act_quant/FakeQuantWithMinMaxVars/nudged_max2" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="12" size="4"/>
			</blobs>
		</layer>
		<layer id="41" name="conv1/act_quant/FakeQuantWithMinMaxVars" precision="FP32" type="FakeQuantize">
			<data levels="256"/>
			<input>
				<port id="0">
					<dim>_INPUT_BATCH_</dim>
					<dim>_INPUT_CHANNEL_</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="5">
					<dim>_INPUT_BATCH_</dim>
					<dim>_INPUT_CHANNEL_</dim>
				</port>
			</output>
		</layer>
		<layer id="42" name="FC/weights" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>_OUTPUT_CHANNEL_</dim>
					<dim>_INPUT_CHANNEL_</dim>
				</port>
			</output>
			<blobs>
				<custom offset="_WEIGHTS_OFFSET_" size="_WEIGHTS_BYTE_SIZE_"/>
			</blobs>
		</layer>
		<layer id="43" name="conv2/weights_quant/FakeQuantWithMinMaxVars/nudged_min" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="16" size="4"/>
			</blobs>
		</layer>
		<layer id="44" name="conv2/weights_quant/FakeQuantWithMinMaxVars/nudged_max" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="20" size="4"/>
			</blobs>
		</layer>
		<layer id="45" name="conv2/weights_quant/FakeQuantWithMinMaxVars/nudged_min2" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="24" size="4"/>
			</blobs>
		</layer>
		<layer id="46" name="conv2/weights_quant/FakeQuantWithMinMaxVars/nudged_max2" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="28" size="4"/>
			</blobs>
		</layer>
		<layer id="47" name="FC/weights/weights_quant/FakeQuantWithMinMaxVars" precision="FP32" type="FakeQuantize">
			<data levels="255"/>
			<input>
				<port id="0">
					<dim>_OUTPUT_CHANNEL_</dim>
					<dim>_INPUT_CHANNEL_</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="5">
					<dim>_OUTPUT_CHANNEL_</dim>
					<dim>_INPUT_CHANNEL_</dim>
				</port>
			</output>
		</layer>
		<layer id="48" name="bias2" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>_OUTPUT_CHANNEL_</dim>
				</port>
			</output>
			<blobs>
				<custom offset="_BIAS_OFFSET_" size="_BIAS_BYTE_SIZE_"/>
			</blobs>
		</layer>
		<layer id="49" name="conv2" precision="FP32" type="FullyConnected">
			<data out-size="_OUTPUT_CHANNEL_"/>
			<input>
				<port id="0">
					<dim>_INPUT_BATCH_</dim>
					<dim>_INPUT_CHANNEL_</dim>
				</port>
				<port id="1">
					<dim>_OUTPUT_CHANNEL_</dim>
					<dim>_INPUT_CHANNEL_</dim>
				</port>
				<port id="2">
					<dim>_OUTPUT_CHANNEL_</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>_OUTPUT_BATCH_</dim>
					<dim>_OUTPUT_CHANNEL_</dim>
				</port>
			</output>
		</layer>
		<layer id="50" name="conv2/after_quant/FakeQuantWithMinMaxVars/nudged_min" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="32" size="4"/>
			</blobs>
		</layer>
		<layer id="51" name="conv2/after_quant/FakeQuantWithMinMaxVars/nudged_max" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="36" size="4"/>
			</blobs>
		</layer>
		<layer id="52" name="conv2/after_quant/FakeQuantWithMinMaxVars/nudged_min2" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="40" size="4"/>
			</blobs>
		</layer>
		<layer id="53" name="conv2/after_quant/FakeQuantWithMinMaxVars/nudged_max2" precision="FP32" type="Const">
			<output>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</output>
			<blobs>
				<custom offset="44" size="4"/>
			</blobs>
		</layer>
		<layer id="54" name="conv2/after_quant/FakeQuantWithMinMaxVars" precision="FP32" type="FakeQuantize">
			<data levels="256"/>
			<input>
				<port id="0">
					<dim>_OUTPUT_BATCH_</dim>
					<dim>_OUTPUT_CHANNEL_</dim>
				</port>
				<port id="1">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="2">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="3">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
				<port id="4">
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="5">
					<dim>_OUTPUT_BATCH_</dim>
					<dim>_OUTPUT_CHANNEL_</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="36" from-port="0" to-layer="41" to-port="0"/>
		<edge from-layer="37" from-port="1" to-layer="41" to-port="1"/>
		<edge from-layer="38" from-port="1" to-layer="41" to-port="2"/>
		<edge from-layer="39" from-port="1" to-layer="41" to-port="3"/>
		<edge from-layer="40" from-port="1" to-layer="41" to-port="4"/>
		<edge from-layer="42" from-port="1" to-layer="47" to-port="0"/>
		<edge from-layer="43" from-port="1" to-layer="47" to-port="1"/>
		<edge from-layer="44" from-port="1" to-layer="47" to-port="2"/>
		<edge from-layer="45" from-port="1" to-layer="47" to-port="3"/>
		<edge from-layer="46" from-port="1" to-layer="47" to-port="4"/>
		<edge from-layer="41" from-port="5" to-layer="49" to-port="0"/>
		<edge from-layer="47" from-port="5" to-layer="49" to-port="1"/>
		<edge from-layer="48" from-port="1" to-layer="49" to-port="2"/>
		<edge from-layer="49" from-port="3" to-layer="54" to-port="0"/>
		<edge from-layer="50" from-port="1" to-layer="54" to-port="1"/>
		<edge from-layer="51" from-port="1" to-layer="54" to-port="2"/>
		<edge from-layer="52" from-port="1" to-layer="54" to-port="3"/>
		<edge from-layer="53" from-port="1" to-layer="54" to-port="4"/>
	</edges>
	<meta_data>
		<MO_version value="unknown version"/>
		<cli_parameters>
			<blobs_as_inputs value="True"/>
			<caffe_parser_path value="DIR"/>
			<data_type value="FP32"/>
			<disable_nhwc_to_nchw value="False"/>
			<disable_omitting_optional value="False"/>
			<disable_resnet_optimization value="False"/>
			<enable_concat_optimization value="False"/>
			<enable_flattening_nested_params value="False"/>
			<enable_ssd_gluoncv value="False"/>
			<extensions value="DIR"/>
			<framework value="tf"/>
			<freeze_placeholder_with_value value="{}"/>
			<generate_experimental_IR_V10 value="False"/>
			<input value="input"/>
			<input_model value="DIR/resnet50-int8.pb"/>
			<input_model_is_text value="False"/>
			<input_shape value="[1, 224, 224, 3]"/>
			<k value="DIR/CustomLayersMapping.xml"/>
			<keep_quantize_ops_in_IR value="True"/>
			<keep_shape_ops value="False"/>
			<legacy_mxnet_model value="False"/>
			<log_level value="ERROR"/>
			<move_to_preprocess value="False"/>
			<output_dir value="DIR"/>
			<placeholder_shapes value="{'input': array([  1, 224, 224,   3])}"/>
			<progress value="False"/>
			<remove_output_softmax value="False"/>
			<reverse_input_channels value="True"/>
			<save_params_from_nd value="False"/>
			<scale_values value="input[1.0]"/>
			<silent value="False"/>
			<stream_output value="False"/>
			<version value="False"/>
			<unset unset_cli_parameters="batch, counts, disable_fusing, disable_gfusing, finegrain_fusing, generate_deprecated_IR_V2, input_checkpoint, input_meta_graph, input_proto, input_symbol, mean_file, mean_file_offsets, model_name, nd_prefix_name, output, pretrained_model_name, saved_model_dir, saved_model_tags, scale, tensorboard_logdir, tensorflow_custom_layer_libraries, tensorflow_custom_operations_config_update, tensorflow_object_detection_api_pipeline_config, tensorflow_operation_patterns, tensorflow_subgraph_patterns, tensorflow_use_custom_operations_config"/>
		</cli_parameters>
	</meta_data>
</net>
        )V0G0N";

static std::string fc_u8_only = R"V0G0N(
<net batch="1" name="resnet50-int8" version="6">
	<layers>
		<layer id="36" name="input" precision="U8" type="Input">
			<output>
				<port id="0">
					<dim>_INPUT_BATCH_</dim>
					<dim>_INPUT_CHANNEL_</dim>
				</port>
			</output>
		</layer>
		<layer id="42" name="conv2/weights" precision="U8" type="Const">
			<output>
				<port id="1">
					<dim>_OUTPUT_CHANNEL_</dim>
					<dim>_INPUT_CHANNEL_</dim>
				</port>
			</output>
			<blobs>
				<custom offset="_WEIGHTS_OFFSET_" size="_WEIGHTS_BYTE_SIZE_"/>
			</blobs>
		</layer>
		<layer id="48" name="bias2" precision="I32" type="Const">
			<output>
				<port id="1">
					<dim>_OUTPUT_CHANNEL_</dim>
				</port>
			</output>
			<blobs>
				<custom offset="_BIAS_OFFSET_" size="_BIAS_BYTE_SIZE_"/>
			</blobs>
		</layer>
		<layer id="49" name="conv2" precision="U8" type="FullyConnected">
			<data out-size="_OUTPUT_CHANNEL_"/>
			<input>
				<port id="0">
					<dim>_INPUT_BATCH_</dim>
					<dim>_INPUT_CHANNEL_</dim>
				</port>
				<port id="1">
					<dim>_OUTPUT_CHANNEL_</dim>
					<dim>_INPUT_CHANNEL_</dim>
				</port>
				<port id="2">
					<dim>_OUTPUT_CHANNEL_</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>_OUTPUT_BATCH_</dim>
					<dim>_OUTPUT_CHANNEL_</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="36" from-port="0" to-layer="49" to-port="0"/>
		<edge from-layer="42" from-port="1" to-layer="49" to-port="1"/>
		<edge from-layer="48" from-port="1" to-layer="49" to-port="2"/>
	</edges>
</net>
        )V0G0N";

static std::string fc_u8_only_4d = R"V0G0N(
<net batch="1" name="resnet50-int8-4d" version="6">
	<layers>
		<layer id="36" name="input" precision="U8" type="Input">
			<output>
				<port id="0">
					<dim>_INPUT_BATCH_</dim>
					<dim>_INPUT_CHANNEL_</dim>
                                        <dim>_INPUT_HEIGHT_</dim>
                                        <dim>_INPUT_WIDTH_</dim>
				</port>
			</output>
		</layer>
		<layer id="42" name="conv2/weights" precision="U8" type="Const">
			<output>
				<port id="1">
					<dim>_OUTPUT_CHANNEL_</dim>
					<dim>_INPUT_CHANNEL_</dim>
				</port>
			</output>
			<blobs>
				<custom offset="_WEIGHTS_OFFSET_" size="_WEIGHTS_BYTE_SIZE_"/>
			</blobs>
		</layer>
		<layer id="48" name="bias2" precision="I32" type="Const">
			<output>
				<port id="1">
					<dim>_OUTPUT_CHANNEL_</dim>
				</port>
			</output>
			<blobs>
				<custom offset="_BIAS_OFFSET_" size="_BIAS_BYTE_SIZE_"/>
			</blobs>
		</layer>
		<layer id="49" name="conv2" precision="U8" type="FullyConnected">
			<data out-size="_OUTPUT_CHANNEL_"/>
			<input>
				<port id="0">
					<dim>_INPUT_BATCH_</dim>
					<dim>_INPUT_CHANNEL_</dim>
                                        <dim>_INPUT_HEIGHT_</dim>
                                        <dim>_INPUT_WIDTH_</dim>
				</port>
				<port id="1">
					<dim>_OUTPUT_CHANNEL_</dim>
					<dim>_INPUT_CHANNEL_</dim>
				</port>
				<port id="2">
					<dim>_OUTPUT_CHANNEL_</dim>
				</port>
			</input>
			<output>
				<port id="3">
					<dim>_OUTPUT_BATCH_</dim>
					<dim>_OUTPUT_CHANNEL_</dim>
				</port>
			</output>
		</layer>
	</layers>
	<edges>
		<edge from-layer="36" from-port="0" to-layer="49" to-port="0"/>
		<edge from-layer="42" from-port="1" to-layer="49" to-port="1"/>
		<edge from-layer="48" from-port="1" to-layer="49" to-port="2"/>
	</edges>
</net>
        )V0G0N";

static std::string conv_pool_u8_test = R"V0G0N(
        <net batch="1" name="POOL_TEST" version="2">
            <layers>
                <layer id="0" name="input" precision="U8" type="Input">
                    <output>
                        <port id="1">
                            <dim>_CONV_INPUT_BATCH_</dim>
                            <dim>_CONV_INPUT_CHANNEL_</dim>
                            <dim>_CONV_INPUT_HEIGHT_</dim>
                            <dim>_CONV_INPUT_WIDTH_</dim>
                        </port>
                    </output>
		        </layer>
                <layer id="1" name="weights" precision="_WEIGHT_PRECISION_" type="Const">
                    <output>
                        <port id="1">
                            <dim>_CONV_OUTPUT_CHANNEL_</dim>
                            <dim>_CONV_INPUT_CHANNEL_</dim>
                            <dim>_CONV_KERNEL_SIZE_</dim>
                            <dim>_CONV_KERNEL_SIZE_</dim>
                        </port>
                    </output>
                    <blobs>
                        <custom offset="0" size="_CONV_WEIGHTS_BYTE_SIZE_"/>
                    </blobs>
                </layer>
                <layer id="2" name="bias" precision="I32" type="Const">
                    <output>
                        <port id="1">
                            <dim>_CONV_OUTPUT_CHANNEL_</dim>
                        </port>
                    </output>
                    <blobs>
                        <custom offset="_CONV_BIAS_OFFSET_" size="_CONV_BIAS_BYTE_SIZE_"/>
                    </blobs>
                </layer>
                <layer id="3" name="conv" precision="U8" type="Convolution">
                    <data kernel="_CONV_KERNEL_" output="_CONV_OUTPUT_CHANNEL_" strides="_CONV_STRIDE_" dilations="1,1" group="1" pads_begin="0,0" pads_end="0,0" />
                    <input>
                        <port id="0">
                            <dim>_CONV_INPUT_BATCH_</dim>
                            <dim>_CONV_INPUT_CHANNEL_</dim>
                            <dim>_CONV_INPUT_HEIGHT_</dim>
                            <dim>_CONV_INPUT_WIDTH_</dim>
                        </port>
                        <port id="1">
                            <dim>_CONV_OUTPUT_CHANNEL_</dim>
                            <dim>_CONV_INPUT_CHANNEL_</dim>
                            <dim>_CONV_KERNEL_SIZE_</dim>
                            <dim>_CONV_KERNEL_SIZE_</dim>
                        </port>
                        <port id="2">
                            <dim>_CONV_OUTPUT_CHANNEL_</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>_CONV_OUTPUT_BATCH_</dim>
                            <dim>_CONV_OUTPUT_CHANNEL_</dim>
                            <dim>_CONV_OUTPUT_HEIGHT_</dim>
                            <dim>_CONV_OUTPUT_WIDTH_</dim>
                        </port>
                    </output>
                </layer>
                <layer id="4" name="pooling" precision="U8" type="Pooling">
                    <data kernel="_POOL_KERNEL_" strides="_POOL_STRIDE_" exclude-pad="_POOL_EXCLUDE_PAD_" pool-method="max" pads_begin="0,0" pads_end="0,0"/>
                    <input>
                        <port id="0">
                            <dim>_POOL_INPUT_BATCH_</dim>
                            <dim>_POOL_INPUT_CHANNEL_</dim>
                            <dim>_POOL_INPUT_HEIGHT_</dim>
                            <dim>_POOL_INPUT_WIDTH_</dim>
                        </port>
                    </input>
                    <output>
                        <port id="1">
                            <dim>_POOL_OUTPUT_BATCH_</dim>
                            <dim>_POOL_OUTPUT_CHANNEL_</dim>
                            <dim>_POOL_OUTPUT_HEIGHT_</dim>
                            <dim>_POOL_OUTPUT_WIDTH_</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="1" to-layer="3" to-port="0"/>
                <edge from-layer="1" from-port="1" to-layer="3" to-port="1"/>
                <edge from-layer="2" from-port="1" to-layer="3" to-port="2"/>
                <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
            </edges>
        </net>
)V0G0N";
