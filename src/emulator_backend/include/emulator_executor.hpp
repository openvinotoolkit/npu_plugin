//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux.hpp"
#include "vpux/al/config/common.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux {

class EmulatorExecutor final : public vpux::Executor {
public:
    EmulatorExecutor(const vpux::NetworkDescription::Ptr& network, const vpux::Config& config)
            : _config(config), _network(network), _logger("EmulatorBackend", config.get<LOG_LEVEL>()) {
    }

    NetworkDescription& getNetworkDesc() {
        return *_network.get();
    }

private:
    Logger _logger;
    Config _config;
    vpux::NetworkDescription::Ptr _network;
};

}  // namespace vpux
