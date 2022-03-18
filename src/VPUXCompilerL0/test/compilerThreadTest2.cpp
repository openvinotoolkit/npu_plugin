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
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>
#include "VPUXCompilerL0.h"

class CompilerTest {
public:
    CompilerTest(): modelIR(nullptr), modelIRSize(0), numCompilationThreads(0), numGetBlobThreads(0) {
        outputs.clear();
    }
    CompilerTest(const CompilerTest& ct) = delete;
    CompilerTest(CompilerTest&& ct) = delete;
    CompilerTest& operator=(const CompilerTest& ct) = delete;
    CompilerTest& operator=(CompilerTest&& ct) = delete;
    ~CompilerTest() {
        if (modelIR != nullptr) {
            free(modelIR);
        }
    }
    vcl_result_t init(char* netName, char* weightName);
    void setThreadCount(int compilationThreads, int getBlobThreads) {
        numCompilationThreads = compilationThreads;
        numGetBlobThreads = getBlobThreads;
    }
    vcl_result_t run();
    bool check() const;
    size_t getOutputSize() const {
        return outputs.size();
    }

private:
    uint8_t* modelIR;
    uint64_t modelIRSize;
    int numCompilationThreads;
    int numGetBlobThreads;
    std::vector<std::string> outputs;
    std::mutex lock;
};

vcl_result_t CompilerTest::init(char* netName, char* weightName) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;

    vcl_version_info_t version;
    vcl_compiler_desc_t compilerDesc = {VCL_PLATFORM_VPU3700, 0};
    vcl_compiler_handle_t compiler = NULL;
    ret = vclCompilerCreate(compilerDesc, &compiler);
    if (ret) {
        std::cerr << "Failed to create compiler! Result:0x" << std::hex << uint64_t(ret) << std::dec << std::endl;
        return ret;
    }

    vcl_compiler_properties_t compilerProp;
    ret = vclCompilerGetProperties(compiler, &compilerProp);
    if (ret) {
        std::cerr << "Failed to query compiler props! Result:0x" << std::hex << uint64_t(ret) << std::dec << std::endl;
        vclCompilerDestroy(compiler);
        return ret;
    } else {
        version.major = compilerProp.version.major;
        version.minor = compilerProp.version.minor;
        vclCompilerDestroy(compiler);
    }

    // Read buffer, add net.xml
    size_t bytesRead = 0;
    FILE* fpN = fopen(netName, "rb");
    if (!fpN) {
        std::cerr << "Cannot open file " << netName << std::endl;
        return VCL_RESULT_ERROR_IO;
    }
    fseek(fpN, 0, SEEK_END);
    uint64_t xmlSize = ftell(fpN);
    fseek(fpN, 0, SEEK_SET);

    // Read weights size, add weight.bin
    FILE* fpW = fopen(weightName, "rb");
    if (!fpW) {
        std::cerr << "Cannot open file " << weightName << std::endl;
        fclose(fpN);
        return VCL_RESULT_ERROR_IO;
    }
    fseek(fpW, 0, SEEK_END);
    uint64_t weightsSize = ftell(fpW);

    // Init modelIR
    uint32_t numberOfInputData = 2;
    modelIRSize =
            sizeof(version) + sizeof(numberOfInputData) + sizeof(xmlSize) + xmlSize + sizeof(weightsSize) + weightsSize;
    modelIR = (uint8_t*)malloc(modelIRSize);
    if (!modelIR) {
        std::cerr << "Failed to malloc memory for IR!" << std::endl;
        fclose(fpW);
        fclose(fpN);
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
        std::cerr << "Short read on network buffer!" << std::endl;
        free(modelIR);
        fclose(fpW);
        fclose(fpN);
        return VCL_RESULT_ERROR_IO;
    }
    fclose(fpN);

    offset += xmlSize;
    memcpy(modelIR + offset, &weightsSize, sizeof(weightsSize));
    offset += sizeof(weightsSize);
    uint8_t* weights = NULL;
    if (weightsSize != 0) {
        weights = modelIR + offset;
        fseek(fpW, 0, SEEK_SET);
        bytesRead = fread(weights, 1, weightsSize, fpW);
        if ((uint64_t)bytesRead != weightsSize) {
            std::cerr << "Short read on weights file!" << std::endl;
            free(modelIR);
            fclose(fpW);
            return VCL_RESULT_ERROR_IO;
        }
    }
    fclose(fpW);
    return VCL_RESULT_SUCCESS;
}

vcl_result_t CompilerTest::run() {
    static int count = 0;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto id = std::this_thread::get_id();
    std::stringstream ss;
    ss << id;
    std::string threadName = ss.str();
    vcl_result_t ret = VCL_RESULT_SUCCESS;

    vcl_compiler_desc_t compilerDesc = {VCL_PLATFORM_VPU3700, 0};
    vcl_compiler_handle_t compiler = NULL;
    ret = vclCompilerCreate(compilerDesc, &compiler);
    if (ret) {
        std::cerr << "Failed to create compiler! Result:0x" << std::hex << uint64_t(ret) << std::dec << std::endl;
        return ret;
    }

    vcl_compiler_properties_t compilerProp;
    ret = vclCompilerGetProperties(compiler, &compilerProp);
    if (ret) {
        std::cerr << "Failed to query compiler props! Result:0x" << std::hex << uint64_t(ret) << std::dec << std::endl;
        vclCompilerDestroy(compiler);
        return ret;
    } else {
        std::cout << "############################################" << std::endl;
        std::cout << threadName.c_str() << " Current compiler info:" << std::endl;
        std::cout << threadName.c_str() << " ID: " << compilerProp.id << std::endl;
        std::cout << threadName.c_str() << " Version:" << compilerProp.version.major << "."
                  << compilerProp.version.minor << std::endl;
        std::cout << threadName.c_str() << "\tSupported opsets:" << compilerProp.supportedOpsets << std::endl;
        std::cout << "############################################" << std::endl;
    }
    char options[] = "--inputs_precisions=\"input:U8\" --inputs_layouts=\"input:NCHW\" "
                     "--outputs_precisions=\"InceptionV1/Logits/Predictions/Softmax:FP32\" "
                     "--outputs_layouts=\"InceptionV1/Logits/Predictions/Softmax:NC\" --config  "
                     "VPUX_COMPILATION_MODE_PARAMS=\"use-user-precision=false propagate-quant-dequant=0\"";
    vcl_executable_desc_t exeDesc = {modelIR, modelIRSize, options, sizeof(options)};

    // Get the outputs from multiple threads env;
    std::vector<std::thread> compilationThreads;
    std::vector<std::pair<vcl_executable_handle_t*, uint64_t>> exeHandles;
    for (int i = 0; i < numCompilationThreads; i++) {
        vcl_executable_handle_t* exeHandle = new vcl_executable_handle_t();
        uint64_t blobSize = 0;
        std::thread thread(vclExecutableCreate, compiler, exeDesc, exeHandle);
        exeHandles.push_back(std::make_pair(exeHandle, blobSize));
        compilationThreads.push_back(move(thread));
    }
    for (auto& compilationThread : compilationThreads) {
        compilationThread.join();
    }
    std::vector<std::thread> compilationThreads2;
    for (auto& pair : exeHandles) {
        vcl_executable_handle_t exeHandle = *(pair.first);
        uint64_t& blobSize = pair.second;
        uint8_t* blob = NULL;
        std::thread thread(vclExecutableGetSerializableBlob, exeHandle, blob, &blobSize);
        compilationThreads2.push_back(move(thread));
    }
    for (auto& compilationThread2 : compilationThreads2) {
        compilationThread2.join();
    }
    std::vector<std::thread> getBlobThreads;
    std::vector<std::pair<uint8_t*, uint64_t>> blobs;
    for (int i = 0; i < numGetBlobThreads; i++) {
        int idx = i % numCompilationThreads;
        vcl_executable_handle_t exe = *(exeHandles[idx].first);
        uint64_t blobsize = exeHandles[idx].second;
        uint8_t* blob = (uint8_t*)malloc(blobsize);
        std::thread thread(vclExecutableGetSerializableBlob, exe, blob, &blobsize);
        blobs.push_back(std::make_pair(blob, blobsize));
        getBlobThreads.push_back(move(thread));
    }
    for (auto& getBlobThread : getBlobThreads) {
        getBlobThread.join();
    }
    for (auto pair : blobs) {
        auto blob = pair.first;
        auto blobSize = pair.second;
        std::string blobName = "ct2_" + std::to_string(count) + "_" + threadName + ".net";
        FILE* fpB = fopen(blobName.c_str(), "wb");
        if (!fpB) {
            std::cerr << "Failed to open " << blobName << ", skip dump!" << std::endl;
        } else {
            uint64_t bytesWrite = fwrite(blob, 1, blobSize, fpB);
            if (bytesWrite != blobSize) {
                std::cerr << "Short write to " << blobName << ", the file is invalid!" << std::endl;
            }
            int cret = fclose(fpB);
            if (cret) {
                std::cerr << "Failed to close " << blobName << ". Result:" << cret << std::endl;
            } else {
                std::cout << "The output name:" << blobName << std::endl;
            }
        }
        std::string output(reinterpret_cast<char*>(blob), blobSize);
        outputs.push_back(output);
        free(blob);
        count++;
    }
    for (auto& pair : exeHandles) {
        ret = vclExecutableDestroy(*(pair.first));
        if (ret != VCL_RESULT_SUCCESS) {
            std::cerr << "Failed to destroy executable! Result:0x" << std::hex << uint64_t(ret) << std::dec
                      << std::endl;
            vclCompilerDestroy(compiler);
            return ret;
        }
    }
    ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        std::cerr << "Failed to destroy compiler! Result:0x" << std::hex << uint64_t(ret) << std::dec << std::endl;
        return ret;
    }
    return ret;
}

bool CompilerTest::check() const {
    const size_t count = outputs.size();
    if (count == 0) {
        std::cerr << "No outputs!" << std::endl;
        return false;
    }
    const std::string& ref = outputs[0];
    for (size_t i = 1; i < count; i++) {
        if (ref != outputs[i]) {
            std::cerr << "The " << i << " output is different!" << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage:" << std::endl;
        std::cerr << "\tcompilerThreadTest2 net.xml weight.bin" << std::endl;
        return -1;
    }

    CompilerTest test;
    if (test.init(argv[1], argv[2]) != VCL_RESULT_SUCCESS)
        return -1;

    // Get the ref output from single thread env;
    test.setThreadCount(1, 1);
    vcl_result_t ret = test.run();
    if (ret) {
        std::cerr << "Failed to run test to create ref! Result:0x" << std::hex << uint64_t(ret) << std::dec
                  << std::endl;
    }

    // Get the outputs from multiple threads env;
    int numCompilationThreads = 5;
    int numGetBlobThreads = 17;
    test.setThreadCount(numCompilationThreads, numGetBlobThreads);
    ret = test.run();
    if (ret) {
        std::cerr << "Failed to run thread test! Result:0x" << std::hex << uint64_t(ret) << std::dec << std::endl;
    }

    if (test.getOutputSize() != 18) {
        std::cerr << "Not get all outputs successfully!" << std::endl;
    } else {
        std::cout << "Add threads get same outputs: " << std::boolalpha << test.check() << std::endl;
    }

    return 0;
}
