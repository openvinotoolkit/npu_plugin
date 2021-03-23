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

// IE
#include "ie_layouts.h"
#include "ie_remote_context.hpp"
// Plugin
#include "hddl2/hddl2_params.hpp"
#include "vpux_params.hpp"
#include "vpux_params_private_options.h"

namespace vpux {

namespace IE = InferenceEngine;
//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
void checkSupportedColorFormat(const IE::ColorFormat& colorFormat) {
    switch (colorFormat) {
    case IE::NV12:
    case IE::BGR:
        return;
    case IE::RAW:
        // TODO Supported?
    case IE::RGB:
    case IE::RGBX:
    case IE::BGRX:
    case IE::I420:
        IE_THROW() << "Unsupported color format.";
    }
}

//------------------------------------------------------------------------------
void ParsedRemoteBlobParams::update(const IE::ParamMap& updateParams) {
    IE::ParamMap mergedMap = updateParams;
    // Fill updated map with not provided keys
    mergedMap.insert(_paramMap.begin(), _paramMap.end());
    _paramMap = mergedMap;
    parse();
}

// TODO Avoid full parsing on each update
// TODO Refactor setParam approach
void ParsedRemoteBlobParams::parse() {
    if (_paramMap.find(IE::KMB_PARAM_KEY(ROI_PTR)) != _paramMap.end()) {
        try {
            _roiPtr = _paramMap.at(IE::KMB_PARAM_KEY(ROI_PTR)).as<std::shared_ptr<IE::ROI>>();
        } catch (...) {
            IE_THROW() << "ROI param is incorrect!";
        }
        // If we working with ROI, also need information about original tensor
        if (_paramMap.find(IE::KMB_PARAM_KEY(ORIGINAL_TENSOR_DESC)) != _paramMap.end()) {
            try {
                _originalTensorDesc =
                        _paramMap.at(IE::KMB_PARAM_KEY(ORIGINAL_TENSOR_DESC)).as<std::shared_ptr<IE::TensorDesc>>();
            } catch (...) {
                IE_THROW() << "Original tensor desc have incorrect type information";
            }
        }
    } else {
        _roiPtr = nullptr;
    }

    if (_paramMap.find(IE::HDDL2_PARAM_KEY(COLOR_FORMAT)) != _paramMap.end()) {
        try {
            _colorFormat = _paramMap.at(IE::HDDL2_PARAM_KEY(COLOR_FORMAT)).as<IE::ColorFormat>();
            checkSupportedColorFormat(_colorFormat);
        } catch (...) {
            IE_THROW() << "Color format param have incorrect type information";
        }
    } else {
        _colorFormat = IE::ColorFormat::BGR;
    }
}

}  // namespace vpux
