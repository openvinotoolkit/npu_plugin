#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#

include-dirs-lrt-y += inc
include-dirs-shave-y += inc
include-dirs-shave_nn-y += inc
srcs-lrt-y += $(wildcard ./src/*.c*)
srcs-shave-y += $(wildcard ./src/*.c*)

