#
# Copyright (C) 2024 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

add_vpux_dialect(ELFNPU37XX)
add_vpux_ops_interface(ELFNPU37XX dialect/ELFNPU37XX/)
add_vpux_attribute(ELFNPU37XX ENABLE_VPUX_ENUMS ENABLE_VPUX_ATTR)
add_vpux_type(ELFNPU37XX)
add_vpux_pass(ELFNPU37XX ELFNPU37XX dialect/ELFNPU37XX/)
