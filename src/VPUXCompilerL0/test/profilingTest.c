//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#include <stdio.h>
#include <stdlib.h>

#include "VPUXCompilerL0.h"

int readFile(const char* fileName, char** buffer, size_t* size) {
    FILE* file = fopen(fileName, "rb");
    if (!file) {
        perror("Can't open blob file");
        return EXIT_FAILURE;
    }

    fseek(file, 0L, SEEK_END);
    size_t filesize = (size_t)ftell(file);
    fseek(file, 0L, SEEK_SET);

    char* binaryBuffer = (char*)malloc(filesize);
    if (!binaryBuffer) {
        fprintf(stderr, "Can't allocate %zu bytes to read %s.", filesize, fileName);
        return EXIT_FAILURE;
    }

    const size_t numOfReadBytes = fread(binaryBuffer, 1, filesize, file);
    if (numOfReadBytes != filesize) {
        free(binaryBuffer);
        fprintf(stderr, "Can't read whole file %s.", fileName);
        return EXIT_FAILURE;
    }

    if (fclose(file) != 0) {
        fprintf(stderr, "Can't close file stream %s. Will continue execution.", fileName);
    }

    *buffer = binaryBuffer;
    *size = filesize;

    return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("usage:\n"
               "\tprofilingTest network.blob profiling_output.bin\n"
               "where\n"
               "\tnetwork.blob - blob with profiling enabled ('PERF_COUNT YES' parameter in the compiler)"
               "\tprofiling_output.bin - raw profiling output acquired from InferenceManagerDemo according to "
               "guides/how-to-use-profiling.md\n");
        return EXIT_FAILURE;
    }

    const char* blobFileName = argv[1];
    const char* profFileName = argv[2];

    int result = EXIT_SUCCESS;

    char* blobBuffer = NULL;
    size_t blobSize = 0;
    result = readFile(blobFileName, &blobBuffer, &blobSize);
    if (result != EXIT_SUCCESS) {
        return result;
    }

    char* profBuffer = NULL;
    size_t profSize = 0;
    result = readFile(profFileName, &profBuffer, &profSize);
    if (result != EXIT_SUCCESS) {
        free(blobBuffer);
        return result;
    }

    vcl_result_t ret = VCL_RESULT_SUCCESS;
    vcl_profiling_input_t profilingApiInput = {
        .blobData = blobBuffer,
        .blobSize = blobSize,
        .profData = profBuffer,
        .profSize = profSize
    };
    vcl_profiling_handle_t profHandle = NULL;
    ret = vclProfilingCreate(&profilingApiInput, &profHandle);
    if (ret != VCL_RESULT_SUCCESS) {
        result = EXIT_FAILURE;
        goto exit;
    }

    vcl_profiling_properties_t profProperties;
    ret = vclProfilingGetProperties(profHandle, &profProperties);
    if (ret != VCL_RESULT_SUCCESS) {
        result = EXIT_FAILURE;
        goto exit_destroy;
    }
    printf("Using profiling version %hu.%hu\n", profProperties.version.major, profProperties.version.minor);

    vcl_profiling_output_t profOutput;
    profOutput.data = NULL;
    ret = vclGetDecodedProfilingBuffer(profHandle, VCL_PROFILING_LAYER_LEVEL, &profOutput);
    if (ret != VCL_RESULT_SUCCESS || profOutput.data == NULL) {
        result = EXIT_FAILURE;
        goto exit_destroy;
    }

    profOutput.data = NULL;
    ret = vclGetDecodedProfilingBuffer(profHandle, VCL_PROFILING_TASK_LEVEL, &profOutput);
    if (ret != VCL_RESULT_SUCCESS || profOutput.data == NULL) {
        result = EXIT_FAILURE;
        goto exit_destroy;
    }

    profOutput.data = NULL;
    ret = vclGetDecodedProfilingBuffer(profHandle, VCL_PROFILING_RAW, &profOutput);
    if (ret != VCL_RESULT_SUCCESS || profOutput.data == NULL) {
        result = EXIT_FAILURE;
        goto exit_destroy;
    }

    printf("Test passed. Profiling API works! Great success!\n");

exit_destroy:
    vclProfilingDestroy(profHandle);

exit:
    free(blobBuffer);
    free(profBuffer);

    return result;
}
