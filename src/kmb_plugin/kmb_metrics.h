// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "kmb_executor.h"

using RangeType = std::tuple<unsigned int, unsigned int, unsigned int>;

namespace vpu {
namespace KmbPlugin {

//------------------------------------------------------------------------------
// class KmbMetrics
// Class to keep and extract metrics value.
//------------------------------------------------------------------------------

class KmbMetrics {
public:
    // Constructor
    KmbMetrics();

    // Accessors
    std::vector<std::string> AvailableDevicesNames() const;
    const std::vector<std::string>& SupportedMetrics() const;

    // Destructor
    ~KmbMetrics() = default;

private:
    // Data section
    std::vector<std::string> _supportedMetrics;
};

}  // namespace KmbPlugin
}  // namespace vpu
