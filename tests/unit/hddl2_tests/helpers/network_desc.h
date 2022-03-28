//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "simple_graph.hpp"
#include "vpux/al/config/common.hpp"
#include "vpux/al/config/compiler.hpp"
#include "vpux/al/config/mcm_compiler.hpp"
#include "vpux_compiler.hpp"

namespace vpux {

class NetworkDescription_Helper {
public:
    NetworkDescription_Helper();
    ~NetworkDescription_Helper() = default;
    NetworkDescription::Ptr getNetworkDesc() {
        return _networkDescPtr;
    }

protected:
    NetworkDescription::Ptr _networkDescPtr = nullptr;
};

//------------------------------------------------------------------------------
inline NetworkDescription_Helper::NetworkDescription_Helper() {
    auto options = std::make_shared<OptionsDesc>();
    registerCommonOptions(*options);
    registerCompilerOptions(*options);
    registerMcmCompilerOptions(*options);

    Config config(options);

    auto compiler = Compiler::create(config);
    std::stringstream blobStream;
    utils::simpleGraph::getExeNetwork()->Export(blobStream);
    _networkDescPtr = compiler->parse(blobStream, config, "");
}

}  // namespace vpux
