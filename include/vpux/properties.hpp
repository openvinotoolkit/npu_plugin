//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/runtime/properties.hpp>

namespace ov {
namespace intel_vpux {

/**
 * @brief [Only for VPUX compiler]
 * Type: "YES", "NO", default is platform-dependent.
 * This option allows to use host based pre- and post- processing
 *
 * Note: Not only the preprocessing operations that are present in the
 * nGraph model are removed from IR, but also the ones introduced in the compiler itself.
 */
static constexpr ov::Property<bool> force_host_precision_layout_conversion{
        "NPU_FORCE_HOST_PRECISION_LAYOUT_CONVERSION"};

/**
 * @brief [Only for VPUX plugin]
 * Type: uint64_t
 * Read-only property to get size of available VPU DDR memory (both for discrete/integrated VPU devices)
 *
 * Note: Queries driver when device is discrete; returns host memory size when device is integrated
 */
static constexpr ov::Property<uint64_t> device_total_mem_size{"NPU_DEVICE_TOTAL_MEM_SIZE"};

/**
 * @brief [Only for VPUX plugin]
 * Type: uint32_t
 * Read-only property to get VPU driver version (for both discrete/integrated VPU devices)
 */
static constexpr ov::Property<uint32_t> driver_version{"NPU_DRIVER_VERSION"};

}  // namespace intel_vpux
}  // namespace ov
