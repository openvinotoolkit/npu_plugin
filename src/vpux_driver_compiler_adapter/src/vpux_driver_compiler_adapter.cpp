//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux_driver_compiler_adapter.h"
#include "graph_transformations.h"
#include "network_description.h"
#include "vpux/al/config/common.hpp"
#include "ze_intel_vpu_uuid.h"
#include "zero_compiler_in_driver.h"

namespace vpux {
namespace driverCompilerAdapter {

LevelZeroCompilerAdapter::LevelZeroCompilerAdapter(): _logger("LevelZeroCompilerAdapter", LogLevel::Warning) {
    auto result = zeInit(ZE_INIT_FLAG_VPU_ONLY);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("LevelZeroCompilerAdapter: Failed to initialize zeAPI. Error code: ", std::hex, result,
                       "\nPlease make sure that the device is available.");
    }
    uint32_t drivers = 0;
    result = zeDriverGet(&drivers, nullptr);

    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("LevelZeroCompilerAdapter: Failed to get information about zeDriver. Error code: ", std::hex,
                       result);
    }

    std::vector<ze_driver_handle_t> allDrivers(drivers);
    result = zeDriverGet(&drivers, allDrivers.data());
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("LevelZeroCompilerAdapter: Failed to get zeDriver. Error code: ", std::hex, result);
    }

    const ze_driver_uuid_t uuid = ze_intel_vpu_driver_uuid;
    ze_driver_properties_t props{};
    // Get our target driver
    for (uint32_t i = 0; i < drivers; ++i) {
        auto res = zeDriverGetProperties(allDrivers[i], &props);
        if (ZE_RESULT_SUCCESS != res) {
            OPENVINO_THROW("LevelZeroCompilerAdapter: Failed to get properties about zeDriver. Error code: ", std::hex,
                           res);
        }
        if (memcmp(&props.uuid, &uuid, sizeof(uuid)) == 0) {
            _driverHandle = allDrivers[i];
            break;
        }
    }

    if (_driverHandle == nullptr) {
        OPENVINO_THROW("LevelZeroCompilerAdapter: Failed to get properties about zeDriver");
        return;
    }

    // query the extension properties
    uint32_t count = 0;
    result = zeDriverGetExtensionProperties(_driverHandle, &count, nullptr);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("LevelZeroCompilerAdapter: Failed to query the extension properties count. Error code: ",
                       std::hex, result);
    }
    std::vector<ze_driver_extension_properties_t> extProps;
    extProps.resize(count);
    result = zeDriverGetExtensionProperties(_driverHandle, &count, extProps.data());
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("LevelZeroCompilerAdapter: Failed to query the extension properties. Error code: ", std::hex,
                       result);
    }
    const char* graphExtName = nullptr;
    uint32_t targetVersion = 0;
    for (uint32_t i = 0; i < count; ++i) {
        auto& property = extProps[i];

        if (strncmp(property.name, ZE_GRAPH_EXT_NAME, strlen(ZE_GRAPH_EXT_NAME)) != 0)
            continue;

        // If the driver version is latest, will just use its name.
        if (property.version == ZE_GRAPH_EXT_VERSION_CURRENT) {
            graphExtName = property.name;
            targetVersion = property.version;
            break;
        }

        // Use the latest version supported by the driver.
        if (property.version > targetVersion) {
            graphExtName = property.name;
            targetVersion = property.version;
        }
    }

    if (graphExtName == nullptr) {
        OPENVINO_THROW("LevelZeroCompilerAdapter: Failed to find Graph extension in VPU Driver");
    }

    const uint16_t adapterMajorVersion = 1;
    uint16_t driverMajorVersion = ZE_MAJOR_VERSION(targetVersion);
    if (adapterMajorVersion != driverMajorVersion) {
        OPENVINO_THROW("LevelZeroCompilerAdapter: adapterMajorVersion: ", adapterMajorVersion,
                       " and driverMajorVersion: ", driverMajorVersion, " mismatch!");
    }

#if defined(VPUX_DEVELOPER_BUILD)
    auto adapterManualConfig = std::getenv("ADAPTER_MANUAL_CONFIG");
    if (adapterManualConfig != nullptr) {
        if (strcmp(adapterManualConfig, "ZE_extension_graph_1_5") == 0) {
            _logger.info("With ADAPTER_MANUAL_CONFIG. Using ZE_GRAPH_EXT_VERSION_1_5");
            targetVersion = ZE_GRAPH_EXT_VERSION_1_5;
        } else if (strcmp(adapterManualConfig, "ZE_extension_graph_1_4") == 0) {
            _logger.info("With ADAPTER_MANUAL_CONFIG. Using ZE_GRAPH_EXT_VERSION_1_4");
            targetVersion = ZE_GRAPH_EXT_VERSION_1_4;
        } else if (strcmp(adapterManualConfig, "ZE_extension_graph_1_3") == 0) {
            _logger.info("With ADAPTER_MANUAL_CONFIG. Using ZE_GRAPH_EXT_VERSION_1_3");
            targetVersion = ZE_GRAPH_EXT_VERSION_1_3;
        } else if (strcmp(adapterManualConfig, "ZE_extension_graph_1_2") == 0) {
            _logger.info("With ADAPTER_MANUAL_CONFIG. Using ZE_GRAPH_EXT_VERSION_1_2");
            targetVersion = ZE_GRAPH_EXT_VERSION_1_2;
        } else if (strcmp(adapterManualConfig, "ZE_extension_graph_1_1") == 0) {
            _logger.info("With ADAPTER_MANUAL_CONFIG. Using ZE_GRAPH_EXT_VERSION_1_1");
            targetVersion = ZE_GRAPH_EXT_VERSION_1_1;
        } else if (strcmp(adapterManualConfig, "ZE_extension_graph_1_0") == 0) {
            _logger.info("With ADAPTER_MANUAL_CONFIG. Using ZE_GRAPH_EXT_VERSION_1_0");
            targetVersion = ZE_GRAPH_EXT_VERSION_1_0;
        } else {
            OPENVINO_THROW("Using unsupported ADAPTER_MANUAL_CONFIG!");
        }
    }
#endif
    if (ZE_GRAPH_EXT_VERSION_1_1 == targetVersion) {
        _logger.info("Using ZE_GRAPH_EXT_VERSION_1_1");
        apiAdapter =
                std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_1_t>>(graphExtName, _driverHandle);
    } else if (ZE_GRAPH_EXT_VERSION_1_2 == targetVersion) {
        _logger.info("Using ZE_GRAPH_EXT_VERSION_1_2");
        apiAdapter =
                std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_2_t>>(graphExtName, _driverHandle);
    } else if (strcmp(graphExtName, ZE_GRAPH_EXT_NAME_1_3) == 0) {
        apiAdapter =
                std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_3_t>>(graphExtName, _driverHandle);
    } else if (ZE_GRAPH_EXT_VERSION_1_4 == targetVersion) {
        _logger.info("Using ZE_GRAPH_EXT_VERSION_1_4");
        apiAdapter =
                std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_4_t>>(graphExtName, _driverHandle);
    } else if (ZE_GRAPH_EXT_VERSION_1_5 == targetVersion) {
        _logger.info("Using ZE_GRAPH_EXT_VERSION_1_5");
        apiAdapter =
                std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_5_t>>(graphExtName, _driverHandle);
    } else {
        _logger.info("Using ZE_GRAPH_EXT_VERSION_1_0");
        apiAdapter = std::make_shared<LevelZeroCompilerInDriver<ze_graph_dditable_ext_t>>(graphExtName, _driverHandle);
    }
}

LevelZeroCompilerAdapter::LevelZeroCompilerAdapter(const IExternalCompiler::Ptr& compilerAdapter)
        : apiAdapter(compilerAdapter), _logger("LevelZeroCompilerAdapter", LogLevel::Warning) {
}

std::shared_ptr<INetworkDescription> LevelZeroCompilerAdapter::compile(std::shared_ptr<ov::Model>& model,
                                                                       const std::string& networkName,
                                                                       const vpux::Config& config) {
    _logger.setLevel(config.get<LOG_LEVEL>());
    uint32_t adapterVersion = apiAdapter->getSupportedOpset();

    // Depending on the driver version, the compiler attached to it may request this information as an indicator of the
    // precision/layout preprocessing requirement. We are setting this value to "true" since the API version is no
    // longer a cause for altering the metadata. This is due to the preprocessing performed in the OpenVINO framework's
    // implementaion, the "ov::Model" object is preprocessed before reaching the NPU plugin.
    model->set_rt_info(true, "is_new_api");

    auto IR = graphTransformations::serializeToIR(model, adapterVersion);
    return apiAdapter->compileIR(model, networkName, IR.xml, IR.weights, config);
}

ov::SupportedOpsMap LevelZeroCompilerAdapter::query(const std::shared_ptr<const ov::Model>& model,
                                                    const vpux::Config& config) {
    _logger.setLevel(config.get<LOG_LEVEL>());
    ov::SupportedOpsMap result;
    const std::string deviceName = "NPU";

    std::shared_ptr<ov::Model> clonedModel = model->clone();
    auto IR = graphTransformations::serializeToIR(clonedModel);
    try {
        const auto supportedLayers = apiAdapter->getQueryResult(IR.xml, IR.weights, config);
        for (auto&& layerName : supportedLayers) {
            result.emplace(layerName, deviceName);
        }
        _logger.info("For given model, there are {0} supported layers", supportedLayers.size());
    } catch (std::exception& e) {
        OPENVINO_THROW("Fail in calling querynetwork : ", e.what());
    }

    return result;
}

std::shared_ptr<vpux::INetworkDescription> LevelZeroCompilerAdapter::parse(const std::vector<char>& blob,
                                                                           const vpux::Config& config,
                                                                           const std::string& netName) {
    _logger.setLevel(config.get<LOG_LEVEL>());
    return apiAdapter->parseBlob(netName, blob, config);
}

OPENVINO_PLUGIN_API void CreateVPUXCompiler(std::shared_ptr<ICompiler>& compiler) {
    compiler = std::make_shared<LevelZeroCompilerAdapter>();
}

}  // namespace driverCompilerAdapter
}  // namespace vpux
