#ifndef VPUX_COMPILER_L0_H
#define VPUX_COMPILER_L0_H

#include <stdint.h>

typedef struct _vpux_compiler_l0_model_ir{
    uint32_t bufferSize;
    uint32_t weightsSize;
    uint8_t* buffer;
    uint8_t* weights;
} vpux_compiler_l0_model_ir;

typedef struct _vpux_compiler_l0_info_t {
    struct {
        uint16_t major;
        uint16_t minor;
    };
} vpux_compiler_l0_info_t;

typedef enum _vpux_compiler_l0_result_t {
    RESULT_SUCCESS = 0,                             ///< [Core] success
    RESULT_ERROR_OUT_OF_MEMORY = 0x70000002,        ///< [Core] insufficient memory to satisfy call
    RESULT_ERROR_INVALID_ARGUMENT = 0x78000004,     ///< [Validation] generic error code for invalid arguments
    RESULT_ERROR_INVALID_NULL_HANDLE = 0x78000005,  ///< [Validation] handle argument is not valid
    RESULT_ERROR_IO = 0x78000006,                   ///< [Core] IO error
    RESULT_ERROR_UNKNOWN = 0x7ffffffe,              ///< [Core] unknown or internal error
} vpux_compiler_l0_result_t;

typedef enum _vpux_compiler_l0_executable_input_type_t {
    EXECUTABLE_INPUT_TYPE_NATIVE = 0x1,
    EXECUTABLE_INPUT_TYPE_NGRAPH_LITE = 0x2,
} vpux_compiler_l0_executable_input_type_t;

typedef enum _vpux_compiler_l0_executable_opset_type_t {
    EXECUTABLE_OPSET_TYPE_OV6 = 0x1,
    EXECUTABLE_OPSET_TYPE_OV7 = 0x2,
} vpux_compiler_l0_executable_opset_type_t;

typedef struct _vpux_compiler_l0_properties_t {
    vpux_compiler_l0_info_t compiler_version;
    vpux_compiler_l0_executable_input_type_t supported_formats;
    vpux_compiler_l0_executable_opset_type_t supported_opsets;
} vpux_compiler_l0_properties_t;

typedef enum _vpux_compiler_l0_product_family_t {
    PRODUCT_FAMILY_UNKNOWN = 0,
    PRODUCT_FAMILY_MYRIADX,
    PRODUCT_FAMILY_KEEMBAY,
    PRODUCT_FAMILY_METEORLAKE,
    PRODUCT_FAMILY_LUNARLAKE,
} vpux_compiler_l0_product_family_t;

typedef enum _vpux_compiler_l0_revision_id_t {
    REVISION_ID_A0 = 0,
    REVISION_ID_A1,
    REVISION_ID_A3,
    REVISION_ID_B,
    REVISION_ID_C,
    REVISION_ID_D,
    REVISION_ID_K,
} vpux_compiler_l0_revision_id_t;

typedef struct _vpux_compiler_l0_compiler_desc_t {
    vpux_compiler_l0_product_family_t family;
    vpux_compiler_l0_revision_id_t revision_id;
    uint32_t debug_level;
} vpux_compiler_l0_compiler_desc_t;

typedef struct _vpux_compiler_l0_executable_desc_t {
    const char* options;    // Includes profiling and other options
    const uint8_t* buffer;  // Will this be the serialized or deserialized form? Having serialized will have a
                            // performance and memory impact
    uint32_t buffer_size;
    vpux_compiler_l0_executable_input_type_t type;
} vpux_compiler_l0_executable_desc_t;

typedef struct vpux_compiler_l0_methods_t {
    /** Get a compiler properties, include the version, supported formats, supported_opsets */
    vpux_compiler_l0_result_t (*getCompilerProperties)(vpux_compiler_l0_properties_t* compilerInfo);

    /**
     * Set new config to the compiler.
     *
     * If the configs are supported, will create a new compiler, else will keep using the old compiler/
     *
     */
    vpux_compiler_l0_result_t (*updateCompilerConfig)(vpux_compiler_l0_compiler_desc_t* compilerDesc,
                                                      vpux_compiler_l0_executable_desc_t* executableDesc);

    /**
     * Generate a blob from the input buffer(xml) and weights(bin).
     *
     * Will set the blob size.
     */
    vpux_compiler_l0_result_t (*generateSerializableBlob)(void* modelIR, uint32_t* blobSize);

    /**
     * Get the latest blob.
     *
     * Fill the blob buffer.
     */
    vpux_compiler_l0_result_t (*getSerializableBlob)(uint8_t* blob, uint32_t blobSize);

    /**
     * Release compiler resources.
     *
     * Call before close dll.
     */
    vpux_compiler_l0_result_t (*deinitCompiler)();
} vpux_compiler_l0_methods_t;

typedef struct vpux_compiler_l0_t {
    /** Identifier of compiler, use to show release info*/
    const char* id;

    /** Compiler methods */
    vpux_compiler_l0_methods_t methods;
} vpux_compiler_l0_t;

/** The entrance of VPUXCompilerL0.dll */
const char* GET_VPUX_COMPILER_L0 = "getCompiler";
vpux_compiler_l0_result_t (*getVPUXCompilerL0)(vpux_compiler_l0_t* vcl);

#endif  // VPUX_COMPILER_L0_H
