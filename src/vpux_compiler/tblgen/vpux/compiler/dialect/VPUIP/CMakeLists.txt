#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

add_vpux_dialect(VPUIP)
add_vpux_ops_interface(VPUIP dialect/VPUIP/)
add_vpux_attr_interface(VPUIP dialect/VPUIP/)
add_vpux_attribute(VPUIP ENABLE_VPUX_ENUMS ENABLE_VPUX_STRUCTS ENABLE_VPUX_ATTR)
add_vpux_type(VPUIP)
add_vpux_pass(VPUIP VPUIP dialect/VPUIP/)
add_vpux_rewrite(convert_allocations_to_declarations VPUIP)
