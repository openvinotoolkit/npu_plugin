//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <vpux/compiler/utils/passes.hpp>

namespace vpux {
namespace VPU {

enum class EnableActivationSparsityMode { AUTO, TRUE, FALSE };

EnableActivationSparsityMode getActSparsityMode(std::string enableActivationSparsityOption);
EnableActivationSparsityMode getActSparsityMode(const StrOption& enableActivationSparsityOption);
bool isActSparsityEnabled(const StrOption& enableActivationSparsityOption);

}  // namespace VPU
}  // namespace vpux
