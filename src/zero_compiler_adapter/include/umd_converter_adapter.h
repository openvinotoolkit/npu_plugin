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
#include <vector>
#include "converter.h"
#include "icompiler_adapter.h"
#include "zero_compiler_adapter.h"

#if defined(_WIN32)
#define LIBTYPE HINSTANCE
#else
#define LIBTYPE void*
#endif

namespace vpux {
namespace zeroCompilerAdapter {
/**
 * Adapter for UMD Converter.
 */
class UMD_Converter final : public ICompiler_Adapter {
public:
    UMD_Converter();
    virtual ~UMD_Converter();

    Opset getSupportedOpset() override;

    Blob::Ptr compileIR(std::vector<char>& xml, std::vector<char>& weights) override;

    std::tuple<const std::string, const DataMap, const DataMap, const DataMap, const DataMap> getNetworkMeta(
            const Blob::Ptr compiledNetwork) override;

    std::tuple<const DataMap, const DataMap> getDeviceNetworkMeta(const Blob::Ptr compiledNetwork) override;

private:
    LIBTYPE handle;
    vpux_converter_t converter;

    // TODO Switch log level to Debug when it will be production solution
    const std::unique_ptr<vpu::Logger> _logger = std::unique_ptr<vpu::Logger>(
            new vpu::Logger("UMD_Converter", vpu::LogLevel::Debug /*_config.logLevel()*/, vpu::consoleOutput()));
};
}  // namespace zeroCompilerAdapter
}  // namespace vpux
