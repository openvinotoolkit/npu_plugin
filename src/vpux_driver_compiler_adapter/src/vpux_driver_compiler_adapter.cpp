//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux_driver_compiler_adapter.h"
#include "network_description.h"
#include "ngraph_transformations.h"
#include "vpux/al/config/common.hpp"
#include "ze_intel_vpu_uuid.h"
#include "zero_compiler_in_driver.h"

namespace vpux {
namespace driverCompilerAdapter {

LevelZeroCompilerAdapter::LevelZeroCompilerAdapter(): _logger("LevelZeroCompilerAdapter", LogLevel::Warning) {
    auto result = zeInit(ZE_INIT_FLAG_VPU_ONLY);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "LevelZeroCompilerAdapter: Failed to initialize zeAPI. Error code: " << std::hex << result;
    }
    uint32_t drivers = 0;
    result = zeDriverGet(&drivers, nullptr);

    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "LevelZeroCompilerAdapter: Failed to get information about zeDriver. Error code: " << std::hex
                   << result;
    }

    std::vector<ze_driver_handle_t> allDrivers(drivers);
    result = zeDriverGet(&drivers, allDrivers.data());
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "LevelZeroCompilerAdapter: Failed to get zeDriver. Error code: " << std::hex << result;
    }

    const ze_driver_uuid_t uuid = ze_intel_vpu_driver_uuid;
    ze_driver_properties_t props{};
    // Get our target driver
    for (uint32_t i = 0; i < drivers; ++i) {
        auto res = zeDriverGetProperties(allDrivers[i], &props);
        if (ZE_RESULT_SUCCESS != res) {
            IE_THROW() << "LevelZeroCompilerAdapter: Failed to get properties about zeDriver. Error code: " << std::hex
                       << res;
        }
        if (memcmp(&props.uuid, &uuid, sizeof(uuid)) == 0) {
            _driverHandle = allDrivers[i];
            break;
        }
    }

    if (_driverHandle == nullptr) {
        IE_THROW() << "LevelZeroCompilerAdapter: Failed to get properties about zeDriver";
        return;
    }

    // query the extension properties
    uint32_t count = 0;
    result = zeDriverGetExtensionProperties(_driverHandle, &count, nullptr);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "LevelZeroCompilerAdapter: Failed to query the extension properties count. Error code: "
                   << std::hex << result;
    }
    std::vector<ze_driver_extension_properties_t> extProps;
    extProps.resize(count);
    result = zeDriverGetExtensionProperties(_driverHandle, &count, extProps.data());
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "LevelZeroCompilerAdapter: Failed to query the extension properties. Error code: " << std::hex
                   << result;
    }
    const char* graphExtName = nullptr;
    uint32_t graphExtVersion = 0;
    for (uint32_t i = 0; i < count; ++i) {
        auto& property = extProps[i];

        if (strncmp(property.name, ZE_GRAPH_EXT_NAME, strlen(ZE_GRAPH_EXT_NAME)) != 0)
            continue;

        // If the driver version is latest, will just use its name.
        if (property.version == ZE_GRAPH_EXT_VERSION_CURRENT) {
            graphExtName = property.name;
            graphExtVersion = property.version;
            break;
        }

        // Use the latest version supported by the driver.
        if (property.version > graphExtVersion) {
            graphExtName = property.name;
            graphExtVersion = property.version;
        }
    }

    if (graphExtName == nullptr) {
        IE_THROW() << "LevelZeroCompilerAdapter: Failed to find Graph extension in VPU Driver";
    }

    const uint16_t adapterMajorVersion = 1;
    uint16_t driverMajorVersion = ZE_MAJOR_VERSION(graphExtVersion);
    if (adapterMajorVersion != driverMajorVersion) {
        IE_THROW() << "LevelZeroCompilerAdapter: adapterMajorVersion: " << adapterMajorVersion
                   << " and driverMajorVersion: " << driverMajorVersion << " mismatch!";
    }

    _logger.debug("Using extension {0}", graphExtName);
    if (strcmp(graphExtName, ZE_GRAPH_EXT_NAME_1_4) == 0) {
        apiAdapter =
                std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_4_t>>(graphExtName, _driverHandle);
    } else if (strcmp(graphExtName, ZE_GRAPH_EXT_NAME_1_1) == 0) {
        apiAdapter =
                std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_1_t>>(graphExtName, _driverHandle);
    } else {
        apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_t>>(graphExtName, _driverHandle);
    }
}

LevelZeroCompilerAdapter::LevelZeroCompilerAdapter(const IExternalCompiler::Ptr& compilerAdapter)
        : apiAdapter(compilerAdapter), _logger("LevelZeroCompilerAdapter", LogLevel::Warning) {
}

std::shared_ptr<INetworkDescription> LevelZeroCompilerAdapter::compile(
        const std::shared_ptr<ngraph::Function>& ngraphFunc, const std::string& netName,
        const InferenceEngine::InputsDataMap& inputsInfo, const InferenceEngine::OutputsDataMap& outputsInfo,
        const vpux::Config& config) {
    _logger.setLevel(config.get<LOG_LEVEL>());
    uint32_t adapterVersion = apiAdapter->getSupportedOpset();
    auto IR = ngraphTransformations::serializeToIR(ngraphFunc, adapterVersion);
    return apiAdapter->compileIR(netName, IR.xml, IR.weights, inputsInfo, outputsInfo, config);
}

// TODO #-29924: Implement query method
InferenceEngine::QueryNetworkResult LevelZeroCompilerAdapter::query(const InferenceEngine::CNNNetwork& /* network */,
                                                                    const vpux::Config& /* config */) {
    THROW_IE_EXCEPTION << "vpux::LevelZeroCompilerAdapter::query is not implemented.";
    return InferenceEngine::QueryNetworkResult();
}

std::shared_ptr<vpux::INetworkDescription> LevelZeroCompilerAdapter::parse(const std::vector<char>& blob,
                                                                           const vpux::Config& config,
                                                                           const std::string& netName) {
    _logger.setLevel(config.get<LOG_LEVEL>());
    return apiAdapter->parseBlob(netName, blob, config);
}

INFERENCE_PLUGIN_API(void)
CreateVPUXCompiler(std::shared_ptr<ICompiler>& compiler) {
    compiler = std::make_shared<LevelZeroCompilerAdapter>();
}

}  // namespace driverCompilerAdapter
}  // namespace vpux
