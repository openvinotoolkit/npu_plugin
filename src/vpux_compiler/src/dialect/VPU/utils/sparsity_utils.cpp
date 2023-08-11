//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"

#include "vpux/utils/core/error.hpp"

#include <algorithm>

using namespace vpux;

static constexpr auto MODE_AUTO = "auto";
static constexpr auto MODE_TRUE = "true";
static constexpr auto MODE_FALSE = "false";

VPU::EnableActivationSparsityMode VPU::getActSparsityMode(std::string strMode) {
    std::transform(strMode.begin(), strMode.end(), strMode.begin(), ::tolower);

    if (strMode == MODE_AUTO) {
        return VPU::EnableActivationSparsityMode::AUTO;
    } else if (strMode == MODE_TRUE) {
        return VPU::EnableActivationSparsityMode::TRUE;
    } else if (strMode == MODE_FALSE) {
        return VPU::EnableActivationSparsityMode::FALSE;
    }

    VPUX_THROW("Unknown value for the enable activation sparsity option: {0}", strMode);
}

VPU::EnableActivationSparsityMode VPU::getActSparsityMode(const StrOption& enableActivationSparsityOption) {
    auto strOption = convertToOptional(enableActivationSparsityOption);
    if (!strOption.has_value()) {
        return VPU::EnableActivationSparsityMode::AUTO;
    }
    return getActSparsityMode(strOption.value());
}

bool VPU::isActSparsityEnabled(const StrOption& enableActivationSparsityOption) {
    const auto actSparsityMode = getActSparsityMode(enableActivationSparsityOption);
    return actSparsityMode == VPU::EnableActivationSparsityMode::TRUE ||
           actSparsityMode == VPU::EnableActivationSparsityMode::AUTO;
}
