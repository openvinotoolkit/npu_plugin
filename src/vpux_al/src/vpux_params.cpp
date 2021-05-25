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

// IE
#include "ie_layouts.h"
#include "ie_remote_context.hpp"
// Plugin
// [Track number: E#12122]
// TODO Remove this header after removing KMB deprecated parameters in future releases
#include "vpux/kmb_params.hpp"
#include "vpux_params.hpp"
#include "vpux_params_private_options.hpp"

namespace vpux {

namespace IE = InferenceEngine;
//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
void ParsedRemoteBlobParams::update(const IE::ParamMap& updateParams) {
    IE::ParamMap mergedMap = updateParams;
    // Fill updated map with not provided keys
    mergedMap.insert(_paramMap.begin(), _paramMap.end());
    _paramMap = mergedMap;
    parse();
}

void ParsedRemoteBlobParams::updateFull(const IE::ParamMap& updateParams) {
    IE::ParamMap mergedMap = _paramMap;
    // Fill updated map with all provided keys
    for (const auto& elem : updateParams) {
        if (mergedMap.find(elem.first) != mergedMap.end()) {
            mergedMap[elem.first] = elem.second;
        } else {
            mergedMap.insert(elem);
        }
    }
    _paramMap = mergedMap;
    parse();
}

// TODO Avoid full parsing on each update
// TODO Refactor setParam approach
void ParsedRemoteBlobParams::parse() {
    if (_paramMap.find(IE::VPUX_PARAM_KEY(ROI_PTR)) != _paramMap.end()) {
        try {
            _roiPtr = _paramMap.at(IE::VPUX_PARAM_KEY(ROI_PTR)).as<std::shared_ptr<IE::ROI>>();
        } catch (...) {
            IE_THROW() << "ROI parameter is not correct";
        }
        // If we working with ROI, also need information about original tensor
        if (_paramMap.find(IE::VPUX_PARAM_KEY(ORIGINAL_TENSOR_DESC)) != _paramMap.end()) {
            try {
                _originalTensorDesc =
                        _paramMap.at(IE::VPUX_PARAM_KEY(ORIGINAL_TENSOR_DESC)).as<std::shared_ptr<IE::TensorDesc>>();
            } catch (...) {
                IE_THROW() << "Original tensor desc parameter has incorrect type information";
            }
        }
    } else {
        _roiPtr = nullptr;
    }

    // [Track number: E#12122]
    // TODO Remove KMB_PARAM_KEY part after removing deprecated KMB parameters in future releases
    if (_paramMap.find(IE::KMB_PARAM_KEY(MEM_OFFSET)) != _paramMap.end()) {
        try {
            _memoryOffset = _paramMap.at(IE::KMB_PARAM_KEY(MEM_OFFSET)).as<size_t>();
        } catch (...) {
            IE_THROW() << "Memory offset parameter is not correct";
        }
    } else if (_paramMap.find(IE::VPUX_PARAM_KEY(MEM_OFFSET)) != _paramMap.end()) {
        try {
            _memoryOffset = _paramMap.at(IE::VPUX_PARAM_KEY(MEM_OFFSET)).as<size_t>();
        } catch (...) {
            IE_THROW() << "Memory offset parameter is not correct";
        }
    }

    // [Track number: E#12122]
    // TODO Remove KMB_PARAM_KEY part after removing deprecated KMB parameters in future releases
    if (_paramMap.find(IE::KMB_PARAM_KEY(MEM_HANDLE)) != _paramMap.end()) {
        try {
            _memoryHandle = _paramMap.at(IE::KMB_PARAM_KEY(MEM_HANDLE)).as<VpuxHandleParam>();
        } catch (...) {
            IE_THROW() << "Memory handle parameter is not correct";
        }
    } else if (_paramMap.find(IE::VPUX_PARAM_KEY(MEM_HANDLE)) != _paramMap.end()) {
        try {
            _memoryHandle = _paramMap.at(IE::VPUX_PARAM_KEY(MEM_HANDLE)).as<VpuxHandleParam>();
        } catch (...) {
            IE_THROW() << "Memory handle parameter is not correct";
        }
    }

    // [Track number: E#12122]
    // TODO Remove KMB_PARAM_KEY part after removing deprecated KMB parameters in future releases
    if (_paramMap.find(IE::KMB_PARAM_KEY(REMOTE_MEMORY_FD)) != _paramMap.end()) {
        try {
            _remoteMemoryFD = _paramMap.at(IE::KMB_PARAM_KEY(REMOTE_MEMORY_FD)).as<VpuxRemoteMemoryFD>();
        } catch (...) {
            IE_THROW() << "Remote Memory FD parameter is not correct";
        }
    } else if (_paramMap.find(IE::VPUX_PARAM_KEY(REMOTE_MEMORY_FD)) != _paramMap.end()) {
        try {
            _remoteMemoryFD = _paramMap.at(IE::VPUX_PARAM_KEY(REMOTE_MEMORY_FD)).as<VpuxRemoteMemoryFD>();
        } catch (...) {
            IE_THROW() << "Remote Memory FD parameter is not correct";
        }
    }

    if (_paramMap.find(IE::VPUX_PARAM_KEY(BLOB_COLOR_FORMAT)) != _paramMap.end()) {
        try {
            _blobColorFormat = _paramMap.at(IE::VPUX_PARAM_KEY(BLOB_COLOR_FORMAT)).as<IE::ColorFormat>();
        } catch (...) {
            IE_THROW() << "Blob color format parameter is not correct";
        }
    }
}

}  // namespace vpux
