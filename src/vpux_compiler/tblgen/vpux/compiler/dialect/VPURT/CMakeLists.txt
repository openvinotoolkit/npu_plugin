#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

add_vpux_dialect(VPURT)
add_vpux_ops_interface(VPURT dialect/VPURT/)
add_vpux_type(VPURT)
add_vpux_attribute(VPURT ENABLE_VPUX_ENUMS ENABLE_VPUX_ATTR)
add_vpux_pass(VPURT VPURT dialect/VPURT/)
