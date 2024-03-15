#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

add_vpux_dialect(IE)
add_vpux_ops_interface(IE dialect/IE/)
add_vpux_attribute(IE ENABLE_VPUX_ENUMS ENABLE_VPUX_ATTR)
add_vpux_pass(IE IE dialect/IE/)
add_vpux_rewrite(convert IECvt)
add_vpux_rewrite(reorder IERdr)
