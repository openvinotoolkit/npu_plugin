#ifndef VPUX_COMPILER_L0_H
#define VPUX_COMPILER_L0_H

#include <stdint.h>

typedef struct _vcl_compiler_handle_t* vcl_compiler_handle_t;
typedef struct _vcl_executable_handle_t* vcl_executable_handle_t;

typedef enum _vcl_result_t {
    VCL_RESULT_SUCCESS = 0,                       ///< [Core] success
    VCL_RESULT_ERROR_UNSUPPORT_COMPILER,          ///< [Validation] can not create compiler based on the compiler desc
    VCL_RESULT_ERROR_COMPILE_FAILED,              ///<  [Core] can not create executable
    VCL_RESULT_ERROR_OUT_OF_MEMORY = 0x70000002,  ///< [Core] insufficient memory to satisfy call
    VCL_RESULT_ERROR_INVALID_ARGUMENT = 0x78000004,     ///< [Validation] generic error code for invalid arguments
    VCL_RESULT_ERROR_INVALID_NULL_HANDLE = 0x78000005,  ///< [Validation] handle argument is not valid
    VCL_RESULT_ERROR_IO = 0x78000006,                   ///< [Core] IO error
    VCL_RESULT_ERROR_INVALID_IR = 0x78000007,           ///< [Validation] the member of modelIR is not valid
    VCL_RESULT_ERROR_UNKNOWN = 0x7ffffffe,              ///< [Core] unknown or internal error
} vcl_result_t;

typedef struct _vcl_version_t {
    uint16_t major;
    uint16_t minor;
} vcl_version_t;

typedef enum _vcl_opset_type_t {
    VCL_EXECUTABLE_OPSET_TYPE_OV6 = 0x1,
    VCL_EXECUTABLE_OPSET_TYPE_OV7 = 0x2,
} vcl_opset_type_t;

typedef enum _vcl_product_family_t {
    VCL_PRODUCT_FAMILY_UNKNOWN = 0,
    VCL_PRODUCT_FAMILY_MYRIADX,
    VCL_PRODUCT_FAMILY_KEEMBAY,
    VCL_PRODUCT_FAMILY_METEORLAKE,
    VCL_PRODUCT_FAMILY_LUNARLAKE,
} vcl_product_family_t;

typedef struct _vcl_compiler_desc_t {
    vcl_product_family_t family;
    vcl_opset_type_t supportedOpsets;
} vcl_compiler_desc_t;

typedef struct _vcl_properties_t {
    vcl_version_t version;
    vcl_compiler_desc_t desc;
} vcl_properties_t;

typedef enum _vcl_log_level_t {
    VCL_NONE = 0,
    VCL_ERROR,
    VCL_WARNING,
    VCL_INFO,
    VCL_DEBUG,
    VCL_TRACE,
} vcl_log_level_t;

typedef enum _vcl_platform_t {
    VCL_VPU3400,  // KMB
    VCL_VPU3700,  // KMB
    VCL_VPU3900,  // TBH
    VCL_VPU3720,  // MTL
} vcl_platform_t;

typedef enum _vcl_compilation_mode {
    VCL_HW,
    VCL_SW,
} vcl_compilation_mode;

typedef struct _vcl_executable_desc_t {
    // Runtime options
    vcl_log_level_t logLevel;
    // Compile options
    vcl_platform_t platform;
    vcl_compilation_mode compilationMode;
} vcl_executable_desc_t;

typedef struct vcl_methods_t {
    /**
     * Create compiler with the config.
     *
     * Return error when the compiler desc is not supported.
     *
     */
    vcl_result_t (*createCompiler)(vcl_compiler_desc_t desc, vcl_compiler_handle_t* compiler);

    /** Get the properties of the compiler
     */
    vcl_result_t (*getCompilerProperties)(vcl_compiler_handle_t compiler, vcl_properties_t* compilerProp);

    /**
     * Generate a blob from the input buffer(xml) and weights(bin).
     *
     * Will set the blob size.
     *
     * Format of modelIR:
     * 1. Num of data elements (now only xml + weights = 2)
     * 2. Size of data 1 (xml)
     * 3. Data 1
     * 4. Size of data 2 (weights)
     * 5. Data 2
     */
    vcl_result_t (*generateSerializableBlob)(vcl_compiler_handle_t compiler, vcl_executable_desc_t exeDesc,
                                             uint8_t* modelIR, uint32_t* blobSize, vcl_executable_handle_t* exe);

    /**
     * Get the blob from the executable.
     *
     * Fill the blob buffer.
     */
    vcl_result_t (*getSerializableBlob)(vcl_executable_handle_t exe, uint8_t* blob, uint32_t blobSize);

    /**
     * Release the executable.
     *
     * Use the executable handle after this is illegal.
     */
    vcl_result_t (*destroyExecutable)(vcl_executable_handle_t exe);

    /**
     * Release compiler resources.
     *
     * Call before close dll.
     */
    vcl_result_t (*destroyCompiler)(vcl_compiler_handle_t compiler);
} vcl_methods_t;

typedef struct vcl_t {
    /** Identifier of compiler, use to show release info*/
    const char* id;

    /** Show compiler capabilities*/
    vcl_compiler_desc_t compilerDesc;

    /** Show executable switchs*/
    vcl_executable_desc_t executableDesc;

    /** Compiler methods */
    vcl_methods_t methods;
} vcl_t;

/** The entrance of VPUXCompilerL0.dll */
const char* GET_VPUX_COMPILER_L0 = "getCompiler";
vcl_result_t (*getVPUXCompilerL0)(vcl_t* vcl);

#endif  // vcl_H
