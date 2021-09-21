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
#include <vpux/utils/core/error.hpp>
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
vpux_compiler_l0_result_t (*getVPUXCompilerL0)(vpux_compiler_l0_t* vcl);

namespace vpux {
namespace zeroCompilerAdapter {

Blob::Blob(const std::vector<char>& data): data(data) {
}

using CompressedIR = std::vector<uint8_t>;

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
void VPUXCompilerL0::initLib() {
    handle = OPENLIB(LIBNAME);
    if (handle == NULL) {
        _logger->error("Failed to open %s", LIBNAME);
#if defined(_WIN32)
        _logger->error("Error: %d\n", GetLastError());
#else
        _logger->error("Error: %s\n", dlerror());
#endif
        exit(EXIT_FAILURE);
    }

    getVPUXCompilerL0 = (vpux_compiler_l0_result_t(*)(vpux_compiler_l0_t*))LIBFUNC(handle, GET_VPUX_COMPILER_L0);
    if (!getVPUXCompilerL0) {
        _logger->error("can not find %s in %s\n", GET_VPUX_COMPILER_L0, LIBNAME);
#if defined(_WIN32)
        _logger->error("Error: %d\n", GetLastError());
#else
        _logger->error("Error: %s\n", dlerror());
#endif
        exit(EXIT_FAILURE);
    }
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
CompressedIR createCompressedIR(std::vector<char>& xml, std::vector<char>& weights) {
    const uint32_t numberOfInputData = 2;
    const uint32_t xmlSize = static_cast<uint32_t>(xml.size());
    const uint32_t weightsSize = static_cast<uint32_t>(weights.size());

    const size_t sizeOfCompressedIR =
            sizeof(numberOfInputData) + sizeof(xmlSize) + xml.size() + sizeof(weightsSize) + weights.size();

    std::vector<uint8_t> compressedIR;
    compressedIR.resize(sizeOfCompressedIR);

    uint32_t offset = 0;
    ie_memcpy(compressedIR.data() + offset, sizeOfCompressedIR - offset, &numberOfInputData, sizeof(numberOfInputData));
    offset += sizeof(numberOfInputData);
    ie_memcpy(compressedIR.data() + offset, sizeOfCompressedIR - offset, &xmlSize, sizeof(xmlSize));
    offset += sizeof(xmlSize);
    ie_memcpy(compressedIR.data() + offset, sizeOfCompressedIR - offset, xml.data(), xmlSize);
    offset += xmlSize;
    ie_memcpy(compressedIR.data() + offset, sizeOfCompressedIR - offset, &weightsSize, sizeof(weightsSize));
    offset += sizeof(weightsSize);
    ie_memcpy(compressedIR.data() + offset, sizeOfCompressedIR - offset, weights.data(), weightsSize);
    offset += weightsSize;

    IE_ASSERT(offset == sizeOfCompressedIR);

    return compressedIR;
}

/**
 * @brief Pair function for IR compression
 */
void decompressIR(const CompressedIR& compressedIR, std::vector<uint8_t>& xml, std::vector<uint8_t>& weights) {
    /* Few validation values to make sure we are working with valid data */
    const uint32_t maxNumberOfElements = 10;
    const uint32_t maxSizeOfXML = (uint32_t)1 * 1024 * 1024 * 1024;      // 1GB
    const uint32_t maxSizeOfWeights = (uint32_t)2 * 1024 * 1024 * 1024;  // 2GB

    size_t offset = 0;

    const uint32_t numberOfElements = *(uint32_t*)compressedIR.data();
    offset += sizeof(numberOfElements);
    VPUX_THROW_WHEN(numberOfElements >= maxNumberOfElements, "IR was corrupted");

    const uint32_t sizeOfXML = *(uint32_t*)(compressedIR.data() + offset);
    offset += sizeof(sizeOfXML);
    VPUX_THROW_WHEN(sizeOfXML == 0 || sizeOfXML >= maxSizeOfXML, "IR was corrupted");

    xml.resize(sizeOfXML);
    ie_memcpy(xml.data(), sizeOfXML, compressedIR.data() + offset, sizeOfXML);
    offset += sizeOfXML;
    VPUX_THROW_WHEN(xml.empty(), "IR was corrupted");

    const uint32_t sizeOfWeights = *(uint32_t*)(compressedIR.data() + offset);
    offset += sizeof(sizeOfWeights);
    VPUX_THROW_WHEN(sizeOfWeights >= maxSizeOfWeights, "IR was corrupted");  // Graph can have weights of size 0

    weights.resize(sizeOfWeights);
    ie_memcpy(weights.data(), sizeOfWeights, compressedIR.data() + offset, sizeOfWeights);
    offset += sizeOfWeights;

    VPUX_THROW_WHEN(offset != compressedIR.size(), "IR was corrupted");
}

//------------------------------------------------------------------------------
//      VPUXCompilerL0
//------------------------------------------------------------------------------
VPUXCompilerL0::VPUXCompilerL0() {
    initLib();

    vpux_compiler_l0_result_t ret = RESULT_SUCCESS;

    ret = getVPUXCompilerL0(&vcl);
    if (ret) {
        THROW_IE_EXCEPTION << "Failed to create compiler!";
    }
}

VPUXCompilerL0::~VPUXCompilerL0() {
    // FIXME Cause segfault on second run for some reason
    //    gc_result_t ret = GC_RESULT_SUCCESS;
    //    ret = converter.methods.deinitCompiler();
    //    if(ret != GC_RESULT_SUCCESS) {
    //        _logger->error("Failed to deinit compiler!\n");
    //        exit(2);
    //    }
    CLOSELIB(handle);
}

// This function based on convertTest.c sample
Blob::Ptr VPUXCompilerL0::compileIR(std::vector<char>& xmlBeforeCompression,
                                    std::vector<char>& weightsBeforeCompression) {
    _logger->debug("VPUXCompilerL0::compileIR start");

    /**
     * Compress and decompress back, since VPUXCompilerL0 doesn't support compressed IR for now
     */
    const auto compressedIR = createCompressedIR(xmlBeforeCompression, weightsBeforeCompression);
    std::vector<uint8_t> xml;
    std::vector<uint8_t> weights;
    decompressIR(compressedIR, xml, weights);

    IE_ASSERT(xml.size() == xmlBeforeCompression.size());
    IE_ASSERT(weights.size() == weightsBeforeCompression.size());

    vpux_compiler_l0_result_t ret = RESULT_SUCCESS;

    uint32_t blobSize = 0;
    std::vector<char> blob;

    vpux_compiler_l0_model_ir modelIR = {static_cast<uint32_t>(xml.size()), static_cast<uint32_t>(weights.size()),
                                         reinterpret_cast<uint8_t*>(xml.data()),
                                         reinterpret_cast<uint8_t*>(weights.data())};
    ret = vcl.methods.generateSerializableBlob(&modelIR, &blobSize);
    if (ret != RESULT_SUCCESS || blobSize == 0) {
        THROW_IE_EXCEPTION << "Failed to get blob size!\n";
    } else {
        blob.resize(blobSize);
        ret = vcl.methods.getSerializableBlob(reinterpret_cast<uint8_t*>(blob.data()), blobSize);
    }

    _logger->debug("VPUXCompilerL0::compileIR end");
    return std::make_shared<Blob>(blob);
}

std::tuple<const std::string, const DataMap, const DataMap, const DataMap, const DataMap>
VPUXCompilerL0::getNetworkMeta(const Blob::Ptr compiledNetwork) {
    _logger->debug("VPUXCompilerL0::getNetworkMeta start");
    vpux::Compiler::Ptr compiler = std::make_shared<Compiler>(getLibFilePath("vpux_compiler"));
    const auto networkDesc = compiler->parse(compiledNetwork->data);
    _logger->debug("VPUXCompilerL0::getNetworkMeta end");
    return std::make_tuple(networkDesc->getName(), networkDesc->getInputsInfo(), networkDesc->getOutputsInfo(),
                           networkDesc->getDeviceInputsInfo(), networkDesc->getDeviceOutputsInfo());
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
    vpux_compiler_l0_result_t ret = RESULT_SUCCESS;
    vpux_compiler_l0_properties_t compilerInfo;
    ret = vcl.methods.getCompilerProperties(&compilerInfo);

    if (ret) {
        CLOSELIB(handle);
        THROW_IE_EXCEPTION << "Failed to query compiler props! result: " << ret;
    } else {
        _logger->info("Compiler version:%d.%d\n", compilerInfo.compiler_version.major,
                      compilerInfo.compiler_version.minor);
        _logger->info("\tSupported format:\n\
          \t\tNATIVE:%d\n\
        \t\tNGRAPH_LITE:%d\n",
                      compilerInfo.supported_formats & EXECUTABLE_INPUT_TYPE_NATIVE && 1,
                      compilerInfo.supported_formats & EXECUTABLE_INPUT_TYPE_NGRAPH_LITE && 1);
        _logger->info("\tSupported opsets:\n\
              \t\tOV6:%d\n\
        \t\tOV7:%d\n",
                      compilerInfo.supported_opsets & EXECUTABLE_OPSET_TYPE_OV6 && 1,
                      compilerInfo.supported_opsets & EXECUTABLE_OPSET_TYPE_OV7 && 1);
    }
    // Set custom opset
    const size_t customOpset = std::atoi(getEnvVarDefault("CUSTOM_OPSET", "0").c_str());
    if (customOpset != 0) {
        return {customOpset};
    }

    if (compilerInfo.supported_opsets) {
        if (compilerInfo.supported_opsets == EXECUTABLE_OPSET_TYPE_OV7) {
            return {7};
        } else if (compilerInfo.supported_opsets == EXECUTABLE_OPSET_TYPE_OV6) {
            return {6};
        } else {
            return {1};
        }
    }
    return {1};
}

}  // namespace zeroCompilerAdapter
}  // namespace vpux
