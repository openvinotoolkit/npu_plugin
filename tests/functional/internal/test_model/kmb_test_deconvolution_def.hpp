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

#include "kmb_test_model.hpp"
#include "kmb_test_utils.hpp"

struct DeconvolutionParams final {
    size_t _outChannels = 0;
    size_t _group = 1;
    Vec2D<size_t> _kernel {};
    Vec2D<size_t> _strides {};
    Pad2D _pad {};
    Vec2D<size_t> _dilation = {1, 1};

    DeconvolutionParams& outChannels(const size_t& outChannels) {
        this->_outChannels = outChannels;
        return *this;
    }

    DeconvolutionParams& kernel(const Vec2D<size_t>& kernel) {
        this->_kernel = kernel;
        return *this;
    }
    DeconvolutionParams& kernel(size_t kernel) {
        this->_kernel = {kernel, kernel};
        return *this;
    }

    DeconvolutionParams& strides(const Vec2D<size_t>& strides) {
        this->_strides = strides;
        return *this;
    }
    DeconvolutionParams& strides(size_t strides) {
        this->_strides = {strides, strides};
        return *this;
    }

    DeconvolutionParams& pad(const Pad2D& pad) {
        this->_pad = pad;
        return *this;
    }
    DeconvolutionParams& pad(ptrdiff_t pad) {
        this->_pad = {pad, pad, pad, pad};
        return *this;
    }

    DeconvolutionParams& dilation(const Vec2D<size_t>& dilation) {
        this->_dilation = dilation;
        return *this;
    }
    DeconvolutionParams& dilation(size_t dilation) {
        this->_dilation = {dilation, dilation};
        return *this;
    }
    DeconvolutionParams& group(size_t group) {
        this->_group = group;
        return *this;
    }
};
inline std::ostream& operator<<(std::ostream& os, const DeconvolutionParams& p) {
    vpu::formatPrint(os, "[outChannels:%v, kernel:%v, strides:%v, pad:%v, dilation:%v]",
        p._outChannels, p._kernel, p._strides, p._pad, p._dilation);
    return os;
}

struct DeconvolutionLayerDef final {
    TestNetwork& testNet;

    std::string name;

    DeconvolutionParams params;

    PortInfo inputPort;
    PortInfo weightsPort;

    DeconvolutionLayerDef(TestNetwork& testNet, std::string name, DeconvolutionParams params)
        : testNet(testNet), name(std::move(name)), params(std::move(params)) {
    }

    DeconvolutionLayerDef& input(const std::string& layerName, size_t index = 0) {
        inputPort = PortInfo(layerName, index);
        return *this;
    }

    DeconvolutionLayerDef& weights(const std::string& layerName, size_t index = 0) {
        weightsPort = PortInfo(layerName, index);
        return *this;
    }

    DeconvolutionLayerDef& weights(const Blob::Ptr& blob) {
        const auto weightsLayerName = name + "_weights";
        testNet.addConst(weightsLayerName, blob);
        return weights(weightsLayerName);
    }

    TestNetwork& build();
};

TensorDesc getDeconvDwWeightsDesc(const DeconvolutionParams& params, Precision precision);
