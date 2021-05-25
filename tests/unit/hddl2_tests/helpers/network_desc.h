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
#include <vpux_config.hpp>
#include <vpux_compiler.hpp>

#include "vpux_compiler.hpp"
#include "models/precompiled_resnet.h"

namespace vpux {
class NetworkDescription_Helper {
public:
    NetworkDescription_Helper();
    ~NetworkDescription_Helper() = default;
    NetworkDescription::Ptr getNetworkDesc() { return _networkDescPtr; }

protected:
    const std::string _modelToImport = PrecompiledResNet_Helper::resnet50.graphPath;
    NetworkDescription::Ptr _networkDescPtr = nullptr;
};

//------------------------------------------------------------------------------
inline NetworkDescription_Helper::NetworkDescription_Helper() {
    auto compiler = Compiler::create();
    _networkDescPtr = compiler->parse(_modelToImport);
}
}  // namespace vpux
