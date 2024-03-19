#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

add_vpux_dialect(VPUMI37XX)
add_vpux_ops_interface(VPUMI37XX dialect/VPUMI37XX/)
add_vpux_attribute(VPUMI37XX ENABLE_VPUX_ENUMS ENABLE_VPUX_ATTR)
add_vpux_type(VPUMI37XX)
add_vpux_pass(VPUMI37XX VPUMI37XX dialect/VPUMI37XX/)
