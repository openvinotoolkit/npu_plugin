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

#include <vpux/kmb_params.hpp>
// TODO Refactor namespace and names in configs/params activity
namespace InferenceEngine {
namespace KmbContextParams {

/** @brief Memory handle stored inside blob */
DECLARE_KMB_PARAM_KEY(BLOB_MEMORY_HANDLE, void*);

/** @brief Allow to store ROI provided by user on createROI call */
DECLARE_KMB_PARAM_KEY(ROI_PTR, std::shared_ptr<InferenceEngine::ROI>);

/** @brief VPUSMM allocator need to know size of allocation */
DECLARE_KMB_PARAM_KEY(ALLOCATION_SIZE, size_t);
}  // namespace KmbContextParams
}  // namespace InferenceEngine
