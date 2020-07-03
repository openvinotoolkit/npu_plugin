//
// Copyright 2019 Intel Corporation.
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

#include "kmb_test_model.hpp"
#include "kmb_test_utils.hpp"

struct ConvolutionParams final {
    size_t _outChannels = 0;
    Vec2D<size_t> _kernel {};
    Vec2D<size_t> _strides {};
    Pad2D _pad {};
    Vec2D<size_t> _dilation = {1, 1};

    ConvolutionParams& outChannels(const size_t& outChannels) {
        this->_outChannels = outChannels;
        return *this;
    }

    ConvolutionParams& kernel(const Vec2D<size_t>& kernel) {
        this->_kernel = kernel;
        return *this;
    }
    ConvolutionParams& kernel(size_t kernel) {
        this->_kernel = {kernel, kernel};
        return *this;
    }

    ConvolutionParams& strides(const Vec2D<size_t>& strides) {
        this->_strides = strides;
        return *this;
    }
    ConvolutionParams& strides(size_t strides) {
        this->_strides = {strides, strides};
        return *this;
    }

    ConvolutionParams& pad(const Pad2D& pad) {
        this->_pad = pad;
        return *this;
    }
    ConvolutionParams& pad(ptrdiff_t pad) {
        this->_pad = {pad, pad, pad, pad};
        return *this;
    }

    ConvolutionParams& dilation(const Vec2D<size_t>& dilation) {
        this->_dilation = dilation;
        return *this;
    }
    ConvolutionParams& dilation(size_t dilation) {
        this->_dilation = {dilation, dilation};
        return *this;
    }
};
inline std::ostream& operator<<(std::ostream& os, const ConvolutionParams& p) {
    vpu::formatPrint(os, "[outChannels:%v, kernel:%v, strides:%v, pad:%v, dilation:%v]",
        p._outChannels, p._kernel, p._strides, p._pad, p._dilation);
    return os;
}

struct ConvolutionLayerDef final {
    TestNetwork& testNet;

    std::string name;

    ConvolutionParams params;

    PortInfo inputPort;
    PortInfo weightsPort;
    PortInfo biasesPort;

    ConvolutionLayerDef(TestNetwork& testNet, std::string name, ConvolutionParams params)
        : testNet(testNet), name(std::move(name)), params(std::move(params)) {
    }

    ConvolutionLayerDef& input(const std::string& layerName, size_t index = 0) {
        inputPort = PortInfo(layerName, index);
        return *this;
    }

    ConvolutionLayerDef& weights(const std::string& layerName, size_t index = 0) {
        weightsPort = PortInfo(layerName, index);
        return *this;
    }
    ConvolutionLayerDef& weights(const Blob::Ptr& blob) {
        const auto weightsLayerName = name + "_weights";
        testNet.addConst(weightsLayerName, blob);
        return weights(weightsLayerName);
    }

    ConvolutionLayerDef& biases(const std::string& layerName, size_t index = 0) {
        biasesPort = {layerName, index};
        return *this;
    }
    ConvolutionLayerDef& biases(const Blob::Ptr& blob) {
        const auto biasesLayerName = name + "_biases";
        testNet.addConst(biasesLayerName, blob);
        return biases(biasesLayerName);
    }

    TestNetwork& build();
};

TensorDesc getConvWeightsDesc(const ConvolutionParams& params, size_t inChannels, Precision precision);
TensorDesc getConvBiasesDesc(const ConvolutionParams& params, Precision precision);
