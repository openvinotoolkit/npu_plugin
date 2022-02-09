//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#ifdef __leon__

#include <cassert>
#define VPUX_ELF_THROW(...) assert(__VA_ARGS__)

#define VPUX_ELF_THROW_UNLESS(_condition_, ...) \
    if(!(_condition_))                            \
    VPUX_ELF_THROW(__VA_ARGS__)

#define VPUX_ELF_THROW_WHEN(_condition_, ...) \
    if((_condition_))                             \
    VPUX_ELF_THROW(__VA_ARGS__)

#else

#include <vpux/utils/core/error.hpp>

#define VPUX_ELF_THROW(...) VPUX_THROW(__VA_ARGS__)
#define VPUX_ELF_THROW_UNLESS(__condition__, ...) VPUX_THROW_UNLESS(__condition__, __VA_ARGS__)
#define VPUX_ELF_THROW_WHEN(__condition__, ...) VPUX_THROW_WHEN(__condition__, __VA_ARGS__)

#endif
