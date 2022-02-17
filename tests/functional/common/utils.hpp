// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/runtime/core.hpp>

std::string getBackendName(const ov::Core& core);
std::vector<std::string> getAvailableDevices(const ov::Core& core);
