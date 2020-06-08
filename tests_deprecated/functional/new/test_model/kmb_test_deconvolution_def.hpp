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

TensorDesc getDeconvDwWeightsDesc(const DeconvolutionParams& params, size_t inChannels, Precision precision);
