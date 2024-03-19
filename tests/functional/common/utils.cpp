//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <string>

#include "utils.hpp"
#include "vpux/properties.hpp"

namespace ov {

namespace test {

namespace utils {

const char* DEVICE_NPU = "NPU";

}  // namespace utils

}  // namespace test

}  // namespace ov

std::string getBackendName(const ov::Core& core) {
    return core.get_property("NPU", ov::intel_vpux::backend_name.name()).as<std::string>();
}

std::vector<std::string> getAvailableDevices(const ov::Core& core) {
    return core.get_property("NPU", ov::available_devices.name()).as<std::vector<std::string>>();
}

std::string modelPriorityToString(const ov::hint::Priority priority) {
    std::ostringstream stringStream;

    stringStream << priority;

    return stringStream.str();
}
