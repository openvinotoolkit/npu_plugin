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

/**
 * @brief Represent private params options, which should not be exposed to user and used only inside plugin
 */

#pragma once

#include <vpux/vpux_plugin_params.hpp>

namespace InferenceEngine {
namespace VpuxContextParams {

/** @brief Allow to store ROI provided by user on createROI call */
DECLARE_VPUX_PARAM_KEY(ROI_PTR, std::shared_ptr<InferenceEngine::ROI>);

/** @brief Information about original tensor desc, used with ROI to keep full frame information */
DECLARE_VPUX_PARAM_KEY(ORIGINAL_TENSOR_DESC, std::shared_ptr<InferenceEngine::TensorDesc>);

/** @brief Information about blob color format */
DECLARE_VPUX_PARAM_KEY(BLOB_COLOR_FORMAT, IE::ColorFormat);

/** @brief VPUSMM allocator need to know size of allocation */
DECLARE_VPUX_PARAM_KEY(ALLOCATION_SIZE, size_t);

}  // namespace VpuxContextParams
}  // namespace InferenceEngine
