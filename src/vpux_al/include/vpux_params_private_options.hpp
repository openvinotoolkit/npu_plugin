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
