//
// Copyright 2017-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include <vpu/graph_transformer.hpp>
#include <vpu/network_config.hpp>
#include <vpu/model/model.hpp>
#include <vpu/utils/logger.hpp>

namespace vpu {

struct CompileEnv final {
    Platform platform = Platform::UNKNOWN;
    Resources resources;

    CompilationConfig config;
    NetworkConfig netConfig;

    Logger::Ptr log;

    bool initialized = false;

    static const CompileEnv& get();

    static void init(
            Platform platform,
            const CompilationConfig& config,
            const Logger::Ptr& log);
    static void updateConfig(const CompilationConfig& config);
    static void free();
};

}  // namespace vpu
