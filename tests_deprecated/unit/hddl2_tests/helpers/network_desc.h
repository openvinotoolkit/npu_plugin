//
// Copyright 2020 Intel Corporation.
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
    // FIXME: Please take a note that _networkDescription should be destructed before _compiler,
    // due _compiler is opened as plugin and _networkDescription is created by _compiler
    // Need to design more accurate solution to avoid missunderstanding in future
    // [Track number: S#37571]
    ICompiler::Ptr _compiler = nullptr;
    NetworkDescription::Ptr _networkDescPtr = nullptr;
};

//------------------------------------------------------------------------------
inline NetworkDescription_Helper::NetworkDescription_Helper() {
    _compiler = ICompiler::create(CompilerType::MCMCompiler);
    _networkDescPtr = _compiler->parse(_modelToImport);
}
}  // namespace vpux
