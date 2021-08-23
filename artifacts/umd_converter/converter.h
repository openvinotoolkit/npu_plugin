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
#include "gc_api.h"
// clang-format off

typedef struct vpux_converter_methods_t {
    /** Get a compiler properties, include the version, supported formats, supported_opsets */
    gc_result_t (*getCompilerProperties)(gc_compiler_properties_t* compilerInfo);

    /**
     * Set new config to the compiler.
     *
     * If the configs are supported, will create a new compiler, else will keep using the old compiler/
     *
     */
    gc_result_t (*updateCompilerConfig)(gc_compiler_desc_t* compilerDesc, gc_executable_desc_t* executableDesc);

    /**
     * Get a blob from the input buffer(xml) and weights(bin).
     *
     * The first call to get blobSize and the second call to fill the blob.
     */
    gc_result_t (*getSerializableBlob)(uint8_t* buffer, uint32_t bufferSize, uint8_t* weights, uint32_t weightsSize,
            uint8_t* blob, uint32_t* blobSize);

    /**
     * Release compiler resources.
     *
     * Call before close dll.
     */
    gc_result_t (*deinitCompiler)();
} vpux_converter_methods_t;

typedef struct vpux_converter_t {
    /** Identifier of converter, use to show release info*/
    const char* id;

    /** Converter methods */
    vpux_converter_methods_t methods;
} vpux_converter_t;

// clang-format on
/** The entrance of umd_converter.dll */
