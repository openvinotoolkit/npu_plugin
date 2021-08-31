#ifndef VPUX_COMPILER_L0_H
#define VPUX_COMPILER_L0_H

#include "gc_api.h"
typedef struct vpux_compiler_l0_methods_t {
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
} vpux_compiler_l0_methods_t;

typedef struct vpux_compiler_l0_t {
    /** Identifier of compiler, use to show release info*/
    const char* id;

    /** Compiler methods */
    vpux_compiler_l0_methods_t methods;
} vpux_compiler_l0_t;

#endif  // VPUX_COMPILER_L0_H
