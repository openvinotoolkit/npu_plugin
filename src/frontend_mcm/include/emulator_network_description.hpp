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
