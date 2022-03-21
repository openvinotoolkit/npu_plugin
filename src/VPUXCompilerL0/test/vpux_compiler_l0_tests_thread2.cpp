//
// Copyright 2022 Intel Corporation.
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

#include "VPUXCompilerL0.h"
#include "vpux_compiler_l0_tests_common.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>
#include <fstream>
#include <regex>

using compilerThread2Params = std::tuple<std::vector<std::string>>;

class CompilerTestThread2 :public VpuxCompilerL0TestsUtils::VpuxCompilerL0TestsCommon, public testing::WithParamInterface<compilerThread2Params>, public testing::Test {
public:
    CompilerTestThread2(): modelIR(), modelIRSize(0), numCompilationThreads(0), numGetBlobThreads(0) {
        outputs.clear();
    }
    CompilerTestThread2(const CompilerTestThread2& ct) = delete;
    CompilerTestThread2(CompilerTestThread2&& ct) = delete;
    CompilerTestThread2& operator=(const CompilerTestThread2& ct) = delete;
    CompilerTestThread2& operator=(CompilerTestThread2&& ct) = delete;
    ~CompilerTestThread2() {
    }
    vcl_result_t init(const char* netName, const char* weightName);
    void setThreadCount(int compilationThreads, int getBlobThreads) {
        numCompilationThreads = compilationThreads;
        numGetBlobThreads = getBlobThreads;
    }
    vcl_result_t run(const std::string& options);

    bool check() const;
    size_t getOutputSize() const {
        return outputs.size();
    }
    void TestBody() override{};
    static std::string getTestCaseName(const testing::TestParamInfo<compilerThread2Params>& obj) {
        auto param = obj.param;
        auto netInfo = std::get<0>(param);
        const std::string testCaseName = netInfo[0];
        return testCaseName;
    }
private:
    std::vector<uint8_t> modelIR;
    size_t modelIRSize;
    int numCompilationThreads;
    int numGetBlobThreads;
    std::vector<std::string> outputs;
    std::mutex lock;
};

vcl_result_t CompilerTestThread2::init(const char* netName, const char* weightName) {
    vcl_result_t ret = VCL_RESULT_SUCCESS;

    vcl_version_info_t version;
    vcl_compiler_desc_t compilerDesc = {VCL_PLATFORM_VPU3700, 5};
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
    std::ifstream nifs(netName, std::ios::binary);
    if (!nifs.is_open()) {
        std::cerr << "Cannot open file " << netName << std::endl;
        return VCL_RESULT_ERROR_IO;
    }

    nifs.seekg(0, nifs.end);
    uint64_t xmlSize = nifs.tellg();
    nifs.seekg(0, nifs.beg);

    // Read weights size, add weight.bin
    std::ifstream wifs(weightName, std::ios::binary);

    if (!wifs.is_open()) {
        std::cerr << "Cannot open file " << weightName << std::endl;
        return VCL_RESULT_ERROR_IO;
    }
    wifs.seekg(0, wifs.end);
    uint64_t weightsSize = wifs.tellg();

    // Init modelIR
    uint32_t numberOfInputData = 2;
    modelIRSize =
            sizeof(version) + sizeof(numberOfInputData) + sizeof(xmlSize) + xmlSize + sizeof(weightsSize) + weightsSize;
    modelIR.resize(modelIRSize);

    uint64_t offset = 0;
    memcpy(modelIR.data(), &version, sizeof(version));
    offset += sizeof(version);
    memcpy(modelIR.data() + offset, &numberOfInputData, sizeof(numberOfInputData));
    offset += sizeof(numberOfInputData);
    memcpy(modelIR.data() + offset, &xmlSize, sizeof(xmlSize));
    offset += sizeof(xmlSize);
    uint8_t* xmlData = modelIR.data() + offset;
    nifs.read(reinterpret_cast<char*>(xmlData), xmlSize);

    if (nifs.fail()) {
        std::cerr << "Short read on network buffer!" << std::endl;
        return VCL_RESULT_ERROR_IO;
    }

    offset += xmlSize;
    memcpy(modelIR.data() + offset, &weightsSize, sizeof(weightsSize));
    offset += sizeof(weightsSize);
    uint8_t* weights = NULL;
    if (weightsSize != 0) {
        weights = modelIR.data() + offset;
        wifs.seekg(0, wifs.beg);
        wifs.read(reinterpret_cast<char*>(weights), weightsSize);

        if (wifs.fail()) {
            std::cerr << "Short read on weights file!" << std::endl;
            return VCL_RESULT_ERROR_IO;
        }
    }
    return VCL_RESULT_SUCCESS;
}

vcl_result_t CompilerTestThread2::run(const std::string& options) {
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
    vcl_executable_desc_t exeDesc = {modelIR.data(), modelIRSize, options.c_str(), options.size() + 1};

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
        uint64_t blobSize = exeHandles[idx].second;
        uint8_t* blob = (uint8_t*)malloc(blobSize);
        std::thread thread(vclExecutableGetSerializableBlob, exe, blob, &blobSize);
        blobs.push_back(std::make_pair(blob, blobSize));

        getBlobThreads.push_back(move(thread));
    }
    for (auto& getBlobThread : getBlobThreads) {
        getBlobThread.join();
    }

    for (auto pair : blobs) {
        auto blob = pair.first;
        auto blobSize = pair.second;
#ifdef BLOB_DUMP
        std::string blobName = "ct2_" + std::to_string(count) + "_" + threadName + ".net";
        std::ofstream bfos(blobName, std::ios::binary);
        if (!bfos.is_open()) {
            std::cerr << "Failed to open " << blobName << ", skip dump!" << std::endl;
        } else {
            bfos.write(reinterpret_cast<char*>(blob), blobSize);
            if (bfos.fail()) {
                std::cerr << "Short write to " << blobName << ", the file is invalid!" << std::endl;
            }
        }
        bfos.close();
#endif  // BLOB_DUMP
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

bool CompilerTestThread2::check() const {
    const size_t count = outputs.size();
    if (count == 0) {
        std::cerr << "No outputs!" << std::endl;
        return false;
    }
    const std::string& ref = outputs[0];
    for (size_t i = 1; i < count; i++) {
        if (ref != outputs[i]) {
            std::cerr << "The " << i << " output is differnt!" << std::endl;
            return false;
        }
    }
    return true;
}

TEST_P(CompilerTestThread2, threadTest2) {
    CompilerTestThread2 test;
    auto ir = test.GetParam();
    auto netInfo = std::get<0>(ir);
    const std::string net = netInfo[0];
    std::string netOptions = netInfo[1];

    if (net != "simple_function") {
        std::string PATH = test.getTestModelsBasePath();
        if (PATH.empty()) {
            GTEST_SKIP() << "POR_PATH is empty. Skipping this test";
        }
        const std::string interPath = netInfo[2];

        std::string netName = PATH + "/" + interPath + "/" + net + ".xml";
        std::string weightName = PATH + "/" + interPath + "/" + net + ".bin";
        std::cout << "netName:" << netName << " weightName:" << weightName << std::endl;

    #if defined(_WIN32)
        bool netExist = PathFileExists(netName.c_str());
        EXPECT_EQ(netExist, true) << "The network " << netName.c_str() << " does not exist!" << std::endl;
    #else
        struct stat buffer;
        int netExist = stat(netName.c_str(), &buffer);
        EXPECT_EQ(netExist, 0) << "The network " << netName.c_str() << " does not exist!" << std::endl;
    #endif

        EXPECT_EQ(test.init(netName.c_str(), weightName.c_str()), VCL_RESULT_SUCCESS);
    } else {
        auto function = test.create_simple_function();
        InferenceEngine::CNNNetwork cnnSimpleNet(function);
        auto inputs = cnnSimpleNet.getInputsInfo();
        auto outputs = cnnSimpleNet.getOutputsInfo();

        // Get input/output names dynamically.
        // In VpuxCompilerL0TestsCommon::create_simple_function, the created simple network only has one input/output, so no need to iterate through iterators.
        std::string inputName = inputs.begin()->first;
        std::string outputName = outputs.begin()->first;

        // Update netOptions by replacing input/output placeholders with dynamic names.
        netOptions = std::regex_replace(netOptions, std::regex("SIMPLE_IN_PLACEHOLDER"), inputName);
        netOptions = std::regex_replace(netOptions, std::regex("SIMPLE_OUT_PLACEHOLDER"), outputName);

        std::string m_out_xml_path = "./simple_func_thread2.xml";
        std::string m_out_bin_path = "./simple_func_thread2.bin";

        ngraph::pass::Manager passManager;
        passManager.register_pass<ngraph::pass::Serialize>(m_out_xml_path, m_out_bin_path);
        passManager.run_passes(function);

        EXPECT_EQ(test.init(m_out_xml_path.c_str(), m_out_bin_path.c_str()), VCL_RESULT_SUCCESS);
        int xmlStatus = std::remove(m_out_xml_path.c_str());
        int binStatus = std::remove(m_out_bin_path.c_str());

        if (xmlStatus == 0 && binStatus == 0) {
            std::cout << "Temp IRs" << m_out_xml_path << " and " << m_out_bin_path << " deleting succeeded!" << std::endl;;
        } else {
            std::cerr << "Temp IRs" << m_out_xml_path << " and " << m_out_bin_path << " deleting failed!" << std::endl;;
        }
    }

    // Get the ref output from single thread env;
    test.setThreadCount(1, 1);
    vcl_result_t ret = test.run(netOptions);
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS) << "Failed to run test to create ref! Result:0x" << std::hex << uint64_t(ret)
                                       << std::dec << std::endl;

    // Get the outputs from multiple threads env;
    int numCompilationThreads = 5;
    int numGetBlobThreads = 17;
    test.setThreadCount(numCompilationThreads, numGetBlobThreads);
    ret = test.run(netOptions);
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS) << "Failed to run thread test! Result:0x" << std::hex << uint64_t(ret)
                                       << std::dec << std::endl;
    EXPECT_EQ(test.getOutputSize(), 18) << "Not get all outputs successfully!" << std::endl;
    EXPECT_EQ(test.check(), true);
}

// You need to export POR_PATH manually. E.g. export POR_PATH=/path/to/om-vpu-models-por-ww46
const std::vector<std::vector<std::string>> smoke_ir_names = {{"googlenet-v1", "--inputs_precisions=\"input:U8\" --inputs_layouts=\"input:NCHW\" "
                                                                "--outputs_precisions=\"InceptionV1/Logits/Predictions/Softmax:FP32\" --outputs_layouts=\"InceptionV1/Logits/Predictions/Softmax:NC\" "
                                                                "--config LOG_LEVEL=\"LOG_INFO\" VPUX_COMPILATION_MODE_PARAMS=\"use-user-precision=false propagate-quant-dequant=0\"", "googlenet-v1/tf/FP16"},
                                                               {"mobilenet-v2", "--inputs_precisions=\"result.1:fp16\" --inputs_layouts=\"result.1:NCHW\" "
                                                                "--outputs_precisions=\"473:fp16\" --outputs_layouts=\"473:NC\" "
                                                                "--config VPUX_PLATFORM=\"3700\" DEVICE_ID=\"VPUX.3700\" VPUX_COMPILATION_MODE=\"DefaultHW\"", "mobilenet-v2/onnx/FP16"},
                                                               {"simple_function", "--inputs_precisions=\"SIMPLE_IN_PLACEHOLDER:fp16\" --inputs_layouts=\"SIMPLE_IN_PLACEHOLDER:NC\" "
                                                                "--outputs_precisions=\"SIMPLE_OUT_PLACEHOLDER:fp16\" --outputs_layouts=\"SIMPLE_OUT_PLACEHOLDER:NC\" "
                                                                "--config LOG_LEVEL=\"LOG_INFO\" VPUX_PLATFORM=\"3700\" DEVICE_ID=\"VPUX.3700\" VPUX_COMPILATION_MODE=\"DefaultHW\"", ""},
                                                                };

const std::vector<std::vector<std::string>> ir_names = {{"resnet-50-pytorch", "--inputs_precisions=\"result.1:fp16\" --inputs_layouts=\"result.1:NCHW\" "
                                                        "--outputs_precisions=\"495:fp16\" --outputs_layouts=\"495:NC\" "
                                                        "--config VPUX_PLATFORM=\"3700\" DEVICE_ID=\"VPUX.3700\" VPUX_COMPILATION_MODE=\"DefaultHW\"", "resnet-50-pytorch/onnx/FP16"},
                                                        {"yolo_v4", "--inputs_precisions=\"image_input:fp16\" --inputs_layouts=\"image_input:NCHW\" "
                                                        "--outputs_precisions=\"conv2d_93/BiasAdd/Add:fp16 conv2d_101/BiasAdd/Add:fp16 conv2d_109/BiasAdd/Add:fp16\" "
                                                        "--outputs_layouts=\"conv2d_93/BiasAdd/Add:NCHW conv2d_101/BiasAdd/Add:NCHW conv2d_109/BiasAdd/Add:NCHW\" "
                                                        "--config VPUX_PLATFORM=\"3700\" DEVICE_ID=\"VPUX.3700\" VPUX_COMPILATION_MODE=\"DefaultHW\"", "yolo_v4/tf/FP16-INT8"},
                                                        };

const auto smoke_params = testing::Combine(testing::ValuesIn(smoke_ir_names));

const auto params = testing::Combine(testing::ValuesIn(ir_names));

INSTANTIATE_TEST_SUITE_P(smoke_Compiler_test, CompilerTestThread2, smoke_params, CompilerTestThread2::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Compiler_test, CompilerTestThread2, params, CompilerTestThread2::getTestCaseName);
