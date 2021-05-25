//
// Copyright 2020 Intel Corporation.
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

#include <ie_common.h>

#include <map>
#include <string>
#include <unordered_set>
#include <vpux_config.hpp>

namespace vpux {

class VpualConfig final : public vpux::VPUXConfig {
public:
    VpualConfig();

    bool repackInputLayout() const { return _repackInputLayout; }

protected:
    void parse(const std::map<std::string, std::string>& config) override;

private:
    bool _repackInputLayout = false;
};

}  // namespace vpux
