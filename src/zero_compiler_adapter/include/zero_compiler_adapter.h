//
// Copyright Intel Corporation.
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
#include "icompiler_adapter.h"
#include "vpux_compiler.hpp"

namespace vpux {
namespace zeroCompilerAdapter {

/**
 * @brief Adapter for Compiler in driver
 * @details Wrap compiler in driver calls and do preliminary actions (like opset conversion)
 */
class ZeroCompilerAdapter final : public ICompiler {
public:
    ZeroCompilerAdapter();
    std::shared_ptr<INetworkDescription> compile(const std::shared_ptr<ngraph::Function>& func,
                                                 const std::string& netName,
                                                 const InferenceEngine::InputsDataMap& inputsInfo,
                                                 const InferenceEngine::OutputsDataMap& outputsInfo,
                                                 const VPUXConfig& config) override;

    InferenceEngine::QueryNetworkResult query(const InferenceEngine::CNNNetwork& network,
                                              const VPUXConfig& config) override;

    std::shared_ptr<vpux::INetworkDescription> parse(const std::vector<char>& network, const VPUXConfig& config,
                                                     const std::string& netName) final;

private:
    /**
     * @brief Separate externals calls to separate class
     */
    ICompiler_Adapter::Ptr apiAdapter;

    // TODO Switch log level to Debug when it will be production solution
    const std::unique_ptr<vpu::Logger> _logger = std::unique_ptr<vpu::Logger>(
            new vpu::Logger("ZeroCompilerAdapter", vpu::LogLevel::Debug /*_config.logLevel()*/, vpu::consoleOutput()));
};

}  // namespace zeroCompilerAdapter
}  // namespace vpux
