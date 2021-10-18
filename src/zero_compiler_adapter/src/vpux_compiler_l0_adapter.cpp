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

#include "vpux_compiler_l0_adapter.h"
#include <iostream>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#if defined(_WIN32)
#define OPENLIB(libName) LoadLibrary((libName))
#define LIBFUNC(lib, fn) GetProcAddress((lib), (fn))
#define CLOSELIB(handle) FreeLibrary(handle)
#else
#define OPENLIB(libName) dlopen((libName), RTLD_LAZY)
#define LIBFUNC(lib, fn) dlsym((lib), (fn))
#define CLOSELIB(handle) dlclose(handle)
#endif

#if defined(_WIN32)
#define LIBNAME "VPUXCompilerL0.dll"
#else
#define LIBNAME "libVPUXCompilerL0.so"
#endif

// Extracted from vpux_compiler_l0.h. Otherwise, compilation error. (multiple definition of `getVPUXCompilerL0')
/** The entrance of VPUXCompilerL0.dll */
const char* GET_VPUX_COMPILER_L0 = "getCompiler";
vcl_result_t (*getVPUXCompilerL0)(vcl_t* vcl);

namespace vpux {
namespace zeroCompilerAdapter {

Blob::Blob(const std::vector<char>& data): data(data) {
}

using SerializedIR = std::vector<uint8_t>;

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
LIBTYPE VPUXCompilerL0::initLib() {
    LIBTYPE handle = OPENLIB(LIBNAME);
    if (handle == NULL) {
        _logger->error("Failed to open %s", LIBNAME);
#if defined(_WIN32)
        _logger->error("Error: %d\n", GetLastError());
#else
        _logger->error("Error: %s\n", dlerror());
#endif
        exit(EXIT_FAILURE);
    }

    getVPUXCompilerL0 = (vcl_result_t(*)(vcl_t*))LIBFUNC(handle, GET_VPUX_COMPILER_L0);
    if (!getVPUXCompilerL0) {
        _logger->error("can not find %s in %s\n", GET_VPUX_COMPILER_L0, LIBNAME);
#if defined(_WIN32)
        _logger->error("Error: %d\n", GetLastError());
#else
        _logger->error("Error: %s\n", dlerror());
#endif
        exit(EXIT_FAILURE);
    }

    return handle;
}

void safelyCloseLib(LIBTYPE handle) {
#if defined(_WIN32)
    // Let the c runtime to free the library when the process terminates
    // to avoid memory leaking issue in Application verifier tests.
    (void)(handle);
#else
    CLOSELIB(handle);
#endif
}


/**
 * @brief Place xml + weights in sequential memory
 * @details Format of the memory:
 *  1. Number of data element (now only xml + weights = 2)
 *  2. Size of data 1 (xml)
 *  3. Data 1
 *  4. Size of data 2 (weights)
 *  5. Data 2
 */
SerializedIR serializeIR(std::vector<char>& xml, std::vector<char>& weights) {
    const uint32_t numberOfInputData = 2;
    const uint32_t xmlSize = static_cast<uint32_t>(xml.size());
    const uint32_t weightsSize = static_cast<uint32_t>(weights.size());

    const size_t sizeOfSerializedIR =
            sizeof(numberOfInputData) + sizeof(xmlSize) + xml.size() + sizeof(weightsSize) + weights.size();

    std::vector<uint8_t> serializedIR;
    serializedIR.resize(sizeOfSerializedIR);

    uint32_t offset = 0;
    ie_memcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, &numberOfInputData, sizeof(numberOfInputData));
    offset += sizeof(numberOfInputData);
    ie_memcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, &xmlSize, sizeof(xmlSize));
    offset += sizeof(xmlSize);
    ie_memcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, xml.data(), xmlSize);
    offset += xmlSize;
    ie_memcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, &weightsSize, sizeof(weightsSize));
    offset += sizeof(weightsSize);
    ie_memcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, weights.data(), weightsSize);
    offset += weightsSize;

    IE_ASSERT(offset == sizeOfSerializedIR);

    return serializedIR;
}

/**
 * @brief Pair function for IR serialization
 */
void deserializeIR(const SerializedIR& serializedIR, std::vector<uint8_t>& xml, std::vector<uint8_t>& weights) {
    /* Few validation values to make sure we are working with valid data */
    const uint32_t maxNumberOfElements = 10;
    const uint32_t maxSizeOfXML = (uint32_t)1 * 1024 * 1024 * 1024;      // 1GB
    const uint32_t maxSizeOfWeights = (uint32_t)2 * 1024 * 1024 * 1024;  // 2GB

    size_t offset = 0;

    const uint32_t numberOfElements = *(reinterpret_cast<const uint32_t *>(serializedIR.data()));
    offset += sizeof(numberOfElements);
    // VPUX_THROW_WHEN(numberOfElements >= maxNumberOfElements, "IR was corrupted");

    const uint32_t sizeOfXML = *(reinterpret_cast<const uint32_t *>((serializedIR.data() + offset)));
    offset += sizeof(sizeOfXML);
    // VPUX_THROW_WHEN(sizeOfXML == 0 || sizeOfXML >= maxSizeOfXML, "IR was corrupted");

    xml.resize(sizeOfXML);
    ie_memcpy(xml.data(), sizeOfXML, serializedIR.data() + offset, sizeOfXML);
    offset += sizeOfXML;
    // VPUX_THROW_WHEN(xml.empty(), "IR was corrupted");

    const uint32_t sizeOfWeights = *(reinterpret_cast<const uint32_t *>(serializedIR.data() + offset));
    offset += sizeof(sizeOfWeights);
    // VPUX_THROW_WHEN(sizeOfWeights >= maxSizeOfWeights, "IR was corrupted");  // Graph can have weights of size 0

    weights.resize(sizeOfWeights);
    ie_memcpy(weights.data(), sizeOfWeights, serializedIR.data() + offset, sizeOfWeights);
    offset += sizeOfWeights;

    // VPUX_THROW_WHEN(offset != serializedIR.size(), "IR was corrupted");
}

//------------------------------------------------------------------------------
//      VPUXCompilerL0
//------------------------------------------------------------------------------
VPUXCompilerL0::VPUXCompilerL0() {
    _handle = initLib();

    vcl_result_t ret = VCL_RESULT_SUCCESS;

    ret = getVPUXCompilerL0(&vcl);
    if (ret) {
        THROW_IE_EXCEPTION << "Failed to create compiler!";
    }

    // Default compiler info
    // _logger->debug("\n############################################\n\n");
    // _logger->debug("Package prop:\n");
    // _logger->debug("ID: %s\n", vcl.id);
    // _logger->debug("Product famliy:%x\n", vcl.compilerDesc.family);
    // _logger->debug("Supported opsets:\n\
    //           \tOV6:%d\n\
    //           \tOV7:%d\n",
    //        vcl.compilerDesc.supportedOpsets & VCL_EXECUTABLE_OPSET_TYPE_OV6 && 1,
    //        vcl.compilerDesc.supportedOpsets & VCL_EXECUTABLE_OPSET_TYPE_OV7 && 1);
    // _logger->debug("Log level:%d\n", vcl.executableDesc.logLevel);
    // _logger->debug("Platform:%d\n", vcl.executableDesc.platform);
    // _logger->debug("Compilation mode:%d\n", vcl.executableDesc.compilationMode);

    vcl_compiler_desc_t compilerDesc = {VCL_PRODUCT_FAMILY_KEEMBAY, VCL_EXECUTABLE_OPSET_TYPE_OV6};
    
    ret = vcl.methods.createCompiler(compilerDesc, &compiler);
    if (ret) {
        THROW_IE_EXCEPTION << "Failed to create compiler!";
    }
}

VPUXCompilerL0::~VPUXCompilerL0() {
    safelyCloseLib(_handle);
}

// This function based on convertTest.c sample
Blob::Ptr VPUXCompilerL0::compileIR(std::vector<char>& xml, std::vector<char>& weights) {
    _logger->debug("VPUXCompilerL0::compileIR start");
    vcl_result_t ret = VCL_RESULT_SUCCESS;

    // Prepare blob
    auto serializedIR = serializeIR(xml, weights);

    // std::vector<char> blob;
    // vpux_compiler_l0_model_ir modelIR = {static_cast<uint32_t>(xml.size()), static_cast<uint32_t>(weights.size()),
    //                                      reinterpret_cast<uint8_t*>(xml.data()),
    //                                      reinterpret_cast<uint8_t*>(weights.data())};

    vcl_executable_handle_t executable = NULL;
    vcl_executable_desc_t exeDesc = {VCL_NONE, VCL_VPU3700, VCL_HW};
    uint32_t blobSize = 0;
    std::vector<char> blob;

    ret = vcl.methods.generateSerializableBlob(compiler, exeDesc, reinterpret_cast<uint8_t*>(serializedIR.data()), &blobSize, &executable);

    if (ret != VCL_RESULT_SUCCESS  || blobSize == 0) {
        THROW_IE_EXCEPTION << "Failed to get blob size!\n";
    } else {
        blob.resize(blobSize);
        ret = vcl.methods.getSerializableBlob(executable, reinterpret_cast<uint8_t*>(blob.data()), blobSize);
    }

    ret = vcl.methods.destroyExecutable(executable);
    if (ret != VCL_RESULT_SUCCESS) {
        THROW_IE_EXCEPTION << "Failed to destroy executable!\n";
    }
    executable = NULL;

    ret = vcl.methods.destroyCompiler(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        THROW_IE_EXCEPTION << "Failed to deinit compiler!\n";
    }

    _logger->debug("VPUXCompilerL0::compileIR end");
    return std::make_shared<Blob>(blob);
}

std::tuple<const std::string, const DataMap, const DataMap, const DataMap, const DataMap>
VPUXCompilerL0::getNetworkMeta(const std::vector<char>& blob) {
    // _logger->debug("VPUXCompilerL0::getNetworkMeta start");
    // vpux::Compiler::Ptr compiler = std::make_shared<Compiler>(getLibFilePath("vpux_compiler"));
    // static const auto networkDesc = compiler->parse(compiledNetwork->data);
    // _logger->debug("VPUXCompilerL0::getNetworkMeta end");
    // return std::make_tuple(networkDesc->getName(), networkDesc->getInputsInfo(), networkDesc->getOutputsInfo(),
    //                        networkDesc->getDeviceInputsInfo(), networkDesc->getDeviceOutputsInfo());
    return std::make_tuple(std::string(), DataMap(), DataMap(), DataMap(), DataMap());
}

std::tuple<const DataMap, const DataMap> VPUXCompilerL0::getDeviceNetworkMeta(const Blob::Ptr compiledNetwork) {
    vpux::Compiler::Ptr compiler = std::make_shared<Compiler>(getLibFilePath("vpux_compiler"));
    const auto networkDesc = compiler->parse(compiledNetwork->data);
    return std::make_tuple(networkDesc->getDeviceInputsInfo(), networkDesc->getDeviceOutputsInfo());
}

std::string getEnvVarDefault(const std::string& varName, const std::string& defaultValue) {
    const char* value = getenv(varName.c_str());
    return value ? value : defaultValue;
}

Opset VPUXCompilerL0::getSupportedOpset() {
    vcl_result_t ret = VCL_RESULT_SUCCESS;
    vcl_properties_t compilerProp;
    ret = vcl.methods.getCompilerProperties(compiler, &compilerProp);
    if (ret) {
        // _logger->debug("Failed to query compiler props! result: %d\n", ret);
        THROW_IE_EXCEPTION << "Failed to create compiler!";
    } else {
        // _logger->debug("\n############################################\n\n");
        // _logger->debug("Current compiler info:\n");
        // _logger->debug("Compiler version:%d.%d\n", compilerProp.version.major, compilerProp.version.minor);
        // _logger->debug("\tProduct family:%d\n", compilerProp.desc.family);
        // _logger->debug("\tSupported opsets:\n\
        //       \t\tOV6:%d\n\
        //       \t\tOV7:%d\n",
        //        compilerProp.desc.supportedOpsets & VCL_EXECUTABLE_OPSET_TYPE_OV6 && 1,
        //        compilerProp.desc.supportedOpsets & VCL_EXECUTABLE_OPSET_TYPE_OV7 && 1);
        // _logger->debug("\n############################################\n\n");
    }

    if (compilerProp.desc.supportedOpsets) {
        if (compilerProp.desc.supportedOpsets == VCL_EXECUTABLE_OPSET_TYPE_OV6) {
            return {7};
        } else if (compilerProp.desc.supportedOpsets == VCL_EXECUTABLE_OPSET_TYPE_OV7) {
            return {6};
        } else {
            return {1};
        }
    }
    return {1};
}

}  // namespace zeroCompilerAdapter
}  // namespace vpux
