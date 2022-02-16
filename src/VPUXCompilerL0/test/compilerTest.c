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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "VPUXCompilerL0.h"

vcl_tensor_precision_t getPrecision(const char* value) {
    if (!strcmp(value, "fp32") || !strcmp(value, "FP32")) {
        return VCL_TENSOR_PRECISION_FP32;
    } else if (!strcmp(value, "fp16") || !strcmp(value, "FP16")) {
        return VCL_TENSOR_PRECISION_FP16;
    } else if (!strcmp(value, "i32") || !strcmp(value, "I32")) {
        return VCL_TENSOR_PRECISION_INT32;
    } else if (!strcmp(value, "u32") || !strcmp(value, "U32")) {
        return VCL_TENSOR_PRECISION_UINT32;
    } else if (!strcmp(value, "i8") || !strcmp(value, "I8")) {
        return VCL_TENSOR_PRECISION_INT8;
    } else if (!strcmp(value, "u8") || !strcmp(value, "U8")) {
        return VCL_TENSOR_PRECISION_UINT8;
    } else {
        printf("The precison of %s is unknown, use UNKNOWN precision now!\n", value);
        return VCL_TENSOR_PRECISION_UNKNOWN;
    }
}

vcl_tensor_layout_t getLayout(const char* value) {
    if (!strcmp(value, "ncdhw") || !strcmp(value, "NCDHW")) {
        return VCL_TENSOR_LAYOUT_NCDHW;
    } else if (!strcmp(value, "ndhwc") || !strcmp(value, "NDHWC")) {
        return VCL_TENSOR_LAYOUT_NDHWC;
    } else if (!strcmp(value, "nchw") || !strcmp(value, "NCHW")) {
        return VCL_TENSOR_LAYOUT_NCHW;
    } else if (!strcmp(value, "nhwc") || !strcmp(value, "NHWC")) {
        return VCL_TENSOR_LAYOUT_NHWC;
    } else if (!strcmp(value, "chw") || !strcmp(value, "CHW")) {
        return VCL_TENSOR_LAYOUT_CHW;
    } else if (!strcmp(value, "hwc") || !strcmp(value, "HWC")) {
        return VCL_TENSOR_LAYOUT_HWC;
    } else if (!strcmp(value, "nc") || !strcmp(value, "NC")) {
        return VCL_TENSOR_LAYOUT_NC;
    } else if (!strcmp(value, "c") || !strcmp(value, "C")) {
        return VCL_TENSOR_LAYOUT_C;
    } else {
        printf("The layout of %s is unknown, use ANY precision now!\n", value);
        return VCL_TENSOR_LAYOUT_ANY;
    }
}

vcl_result_t testCompiler(int argc, char** argv) {
    if (argc != 4 && argc != 9) {
        printf("usage:\n\tcompilerTest net.xml weight.bin output.net\n");
        printf("usage:\n\tcompilerTest net.xml weight.bin output.net $IP $IO $OP $OO $configFile\n");
        return -1;
    }

    vcl_result_t ret = VCL_RESULT_SUCCESS;
    vcl_compiler_desc_t compilerDesc = {VCL_PLATFORM_VPU3700, 0};
    vcl_compiler_handle_t compiler = NULL;
    ret = vclCompilerCreate(compilerDesc, &compiler);
    if (ret) {
        printf("Failed to create compiler! Result:%x\n", ret);
        return ret;
    }

    vcl_compiler_properties_t compilerProp;
    ret = vclCompilerGetProperties(compiler, &compilerProp);
    if (ret) {
        printf("Failed to query compiler props! Result: %x\n", ret);
        vclCompilerDestroy(compiler);
        return ret;
    } else {
        printf("\n############################################\n\n");
        printf("Current compiler info:\n");
        printf("ID: %s\n", compilerProp.id);
        printf("Version:%d.%d\n", compilerProp.version.major, compilerProp.version.minor);
        printf("\tSupported opsets:%d\n", compilerProp.supportedOpsets);
        printf("\n############################################\n\n");
    }

    // Read buffer, add net.xml
    size_t bytesRead = 0;
    char* netName = argv[1];
    FILE* fpN = fopen(netName, "rb");
    if (!fpN) {
        printf("Cannot open file %s\n", netName);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    fseek(fpN, 0, SEEK_END);
    uint64_t xmlSize = ftell(fpN);
    fseek(fpN, 0, SEEK_SET);

    // Read weights size, add weight.bin
    char* weightName = argv[2];
    FILE* fpW = fopen(weightName, "rb");
    if (!fpW) {
        printf("Cannot open file %s\n", weightName);
        fclose(fpN);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    fseek(fpW, 0, SEEK_END);
    uint64_t weightsSize = ftell(fpW);

    // Init modelIR
    vcl_version_info_t version = compilerProp.version;
    uint32_t numberOfInputData = 2;
    uint64_t modelIRSize =
            sizeof(version) + sizeof(numberOfInputData) + sizeof(xmlSize) + xmlSize + sizeof(weightsSize) + weightsSize;
    uint8_t* modelIR = (uint8_t*)malloc(modelIRSize);
    if (!modelIR) {
        printf("Failed to alloc memory for IR!\n");
        fclose(fpW);
        fclose(fpN);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_OUT_OF_MEMORY;
    }
    uint64_t offset = 0;
    memcpy(modelIR, &version, sizeof(version));
    offset += sizeof(version);
    memcpy(modelIR + offset, &numberOfInputData, sizeof(numberOfInputData));
    offset += sizeof(numberOfInputData);
    memcpy(modelIR + offset, &xmlSize, sizeof(xmlSize));
    offset += sizeof(xmlSize);
    uint8_t* xmlData = modelIR + offset;
    bytesRead = fread(xmlData, 1, xmlSize, fpN);
    if ((uint64_t)bytesRead != xmlSize) {
        printf("Short read on network buffer!!!\n");
        free(modelIR);
        fclose(fpW);
        fclose(fpN);
        vclCompilerDestroy(compiler);
        return VCL_RESULT_ERROR_IO;
    }
    int cret = fclose(fpN);
    if (cret) {
        printf("Failed to close %s. Result:%d\n", netName, cret);
        free(modelIR);
        fclose(fpW);
        vclCompilerDestroy(compiler);
        return cret;
    }

    offset += xmlSize;
    memcpy(modelIR + offset, &weightsSize, sizeof(weightsSize));
    offset += sizeof(weightsSize);
    uint8_t* weights = NULL;
    if (weightsSize != 0) {
        weights = modelIR + offset;
        fseek(fpW, 0, SEEK_SET);
        bytesRead = fread(weights, 1, weightsSize, fpW);
        if ((uint64_t)bytesRead != weightsSize) {
            printf("Short read on weights file!!!\n");
            free(modelIR);
            fclose(fpW);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_IO;
        }
    }
    cret = fclose(fpW);
    if (cret) {
        printf("Failed to close %s. Result:%d\n", weightName, cret);
        free(modelIR);
        vclCompilerDestroy(compiler);
        return cret;
    }

    vcl_executable_handle_t executable = NULL;
    if (argc != 9) {
        vcl_tensor_precision_t inPrc = VCL_TENSOR_PRECISION_FP32;
        vcl_tensor_layout_t inLayout = VCL_TENSOR_LAYOUT_ANY;
        vcl_tensor_precision_t outPrc = VCL_TENSOR_PRECISION_FP32;
        vcl_tensor_layout_t outLayout = VCL_TENSOR_LAYOUT_ANY;
        char options[] = "VPUX_PLATFORM 3700 LOG_LEVEL LOG_INFO ";
        vcl_executable_desc_t exeDesc = {modelIR, modelIRSize, inPrc,   inLayout,
                                         outPrc,  outLayout,   options, sizeof(options)};
        ret = vclExecutableCreate(compiler, exeDesc, &executable);
    } else {
        vcl_tensor_precision_t inPrc = getPrecision(argv[4]);
        vcl_tensor_layout_t inLayout = getLayout(argv[5]);
        vcl_tensor_precision_t outPrc = getPrecision(argv[6]);
        vcl_tensor_layout_t outLayout = getLayout(argv[7]);
        char* configFile = argv[8];
        FILE* fpC = fopen(configFile, "rb");
        if (!fpC) {
            printf("Cannot open file %s\n", configFile);
            free(modelIR);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_IO;
        }
        fseek(fpC, 0, SEEK_END);
        uint32_t configSize = ftell(fpC);
        fseek(fpC, 0, SEEK_SET);
        if (configSize == 0) {
            cret = fclose(fpC);
            if (cret) {
                printf("Failed to close %s. Result:%d\n", configFile, cret);
                free(modelIR);
                vclCompilerDestroy(compiler);
                return cret;
            }
            char options[] = "VPUX_PLATFORM 3700 LOG_LEVEL LOG_INFO ";
            vcl_executable_desc_t exeDesc = {modelIR, modelIRSize, inPrc,   inLayout,
                                             outPrc,  outLayout,   options, sizeof(options)};
            ret = vclExecutableCreate(compiler, exeDesc, &executable);
        } else {
            char* options = (char*)malloc(configSize);
            if (!options) {
                printf("Failed to alloc memory for options\n");
                fclose(fpC);
                free(modelIR);
                vclCompilerDestroy(compiler);
                return VCL_RESULT_ERROR_OUT_OF_MEMORY;
            }
            bytesRead = fread(options, 1, configSize, fpC);
            if ((uint32_t)bytesRead != configSize) {
                printf("Short read on config file buffer!!!\n");
                free(options);
                fclose(fpC);
                free(modelIR);
                vclCompilerDestroy(compiler);
                return VCL_RESULT_ERROR_IO;
            }
            cret = fclose(fpC);
            if (cret) {
                printf("Failed to close %s. Result:%d\n", configFile, cret);
                free(options);
                free(modelIR);
                vclCompilerDestroy(compiler);
                return cret;
            }

            vcl_executable_desc_t exeDesc = {modelIR, modelIRSize, inPrc,   inLayout,
                                             outPrc,  outLayout,   options, configSize};
            ret = vclExecutableCreate(compiler, exeDesc, &executable);
            free(options);
        }
    }
    free(modelIR);
    if (ret != VCL_RESULT_SUCCESS) {
        printf("Failed to create executable handle! Result:%x\n", ret);
        vclCompilerDestroy(compiler);
        return ret;
    }
    uint64_t blobSize = 0;
    ret = vclExecutableGetSerializableBlob(executable, NULL, &blobSize);
    if (ret != VCL_RESULT_SUCCESS || blobSize == 0) {
        printf("Failed to get blob size! Result:%x\n", ret);
        vclExecutableDestroy(executable);
        vclCompilerDestroy(compiler);
        return ret;
    } else {
        uint8_t* blob = (uint8_t*)malloc(blobSize);
        if (!blob) {
            printf("Failed to alloc memory for blob!\n");
            vclExecutableDestroy(executable);
            vclCompilerDestroy(compiler);
            return VCL_RESULT_ERROR_OUT_OF_MEMORY;
        }
        ret = vclExecutableGetSerializableBlob(executable, blob, &blobSize);
        if (ret == VCL_RESULT_SUCCESS) {
            char* blobName = argv[3];
            FILE* fpB = fopen(blobName, "wb");
            if (!fpB) {
                printf("Can not open %s, skip dump!\n", blobName);
            } else {
                uint64_t bytesWrite = fwrite(blob, 1, blobSize, fpB);
                if (bytesWrite != blobSize) {
                    printf("Short write to %s, the file is invalid!\n", blobName);
                }
                int cret = fclose(fpB);
                if (cret) {
                    printf("Failed to close %s. Result:%d\n", blobName, cret);
                } else {
                    printf("The output name:%s\n", blobName);
                }
            }
        }
        free(blob);
    }

    ret = vclExecutableDestroy(executable);
    if (ret != VCL_RESULT_SUCCESS) {
        printf("Failed to destroy executable! Result:%x\n", ret);
        ret = vclCompilerDestroy(compiler);
        return ret;
    }
    executable = NULL;

    ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        printf("Failed to destroy compiler! Result:%x\n", ret);
        return ret;
    }
    return ret;
}

int main(int argc, char** argv) {
    testCompiler(argc, argv);
    // Multiple tests
    // testCompiler(lib, argc, argv);
    return 0;
}
