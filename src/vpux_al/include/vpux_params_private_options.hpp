//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

/**
 * @brief Represent private params options, which should not be exposed to user and used only inside plugin
 *
 * @deprecated Configuration API v1.0 would be deprecated in 2023.1 release.
 * It was left due to backward compatibility needs.
 * As such usage of this version of API is discouraged.
 * Prefer Configuration API v2.0.
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
