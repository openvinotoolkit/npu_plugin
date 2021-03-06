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

#pragma once

#include <memory>

#include <include/mcm/compiler/compilation_unit.hpp>
#include <mcm_config.hpp>
#include <vpu/utils/logger.hpp>
#include <vpux_compiler.hpp>

namespace vpu {
namespace MCMAdapter {

class EmulatorNetworkDescription final : public vpux::INetworkDescription {
public:
    EmulatorNetworkDescription(std::unique_ptr<mv::CompilationUnit>&& compiler, const vpu::MCMConfig& config,
                               const std::string& name);

    const vpux::DataMap& getInputsInfo() const final;

    const vpux::DataMap& getOutputsInfo() const final;

    const vpux::DataMap& getDeviceInputsInfo() const final;

    const vpux::DataMap& getDeviceOutputsInfo() const final;

    const std::vector<char>& getCompiledNetwork() const final;

    const void* getNetworkModel() const final;

    std::size_t getNetworkModelSize() const final;  // not relevant information for this type

    const std::string& getName() const final;

private:
    std::string _name;
    std::unique_ptr<mv::CompilationUnit> _compiler;
    std::unique_ptr<vpu::Logger> _logger;
    vpux::DataMap _dataMapPlaceholder;
    std::vector<char> _compiledNetworkPlaceholder;
};

}  // namespace MCMAdapter
}  // namespace vpu
