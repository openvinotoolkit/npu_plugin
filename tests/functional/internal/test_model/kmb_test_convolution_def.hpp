//
// Copyright 2019 Intel Corporation.
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
        if (blob == nullptr) {
            return *this;
        }
        const auto biasesLayerName = name + "_biases";
        testNet.addConst(biasesLayerName, blob);
        return biases(biasesLayerName);
    }

    TestNetwork& build();
};

TensorDesc getConvWeightsDesc(const ConvolutionParams& params, size_t inChannels, Precision precision);
TensorDesc getConvBiasesDesc(const ConvolutionParams& params, Precision precision);
