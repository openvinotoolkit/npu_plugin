#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

#

$(info Using Shavelib Stub for 3600)
include-dirs-lrt-$(CONFIG_HAS_LRT_SRCS) += inc
include-dirs-lnn-$(CONFIG_HAS_LNN_SRCS) += inc
srcs-lrt-$(CONFIG_HAS_LRT_SRCS) += $(wildcard src/3600/*.c*)
#TODO: which functions need stubs?
