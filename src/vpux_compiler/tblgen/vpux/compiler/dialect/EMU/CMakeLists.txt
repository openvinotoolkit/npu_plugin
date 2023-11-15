#
# Copyright (C) 2023 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

add_vpux_dialect(EMU)
add_vpux_ops_interface(EMU dialect/EMU/)
add_vpux_pass(EMU EMU dialect/EMU/)
