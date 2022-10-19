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
#include <fstream>
#include <iostream>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using compilerThreadParams = std::tuple<std::vector<std::string>>;

class CompilerTestThread :
        public VpuxCompilerL0TestsUtils::VpuxCompilerL0TestsCommon,
        public testing::WithParamInterface<compilerThreadParams>,
        public testing::Test {
public:
    CompilerTestThread(): modelIR(), modelIRSize(0) {
    }
    ~CompilerTestThread() {
    }
    vcl_result_t init(const char* netName, const char* weightName);
    vcl_result_t run(const std::string& options);

    bool check() const;
    size_t getOutputSize() const {
        return outputs.size();
    }
    void TestBody() override{};
    static std::string getTestCaseName(const testing::TestParamInfo<compilerThreadParams>& obj) {
        auto param = obj.param;
        auto netInfo = std::get<0>(param);
        const std::string testCaseName = netInfo[0];
        return testCaseName;
    }

private:
    std::vector<uint8_t> modelIR;
    size_t modelIRSize;
    std::vector<std::string> outputs;
    std::mutex lock;
};

vcl_result_t CompilerTestThread::init(const char* netName, const char* weightName) {
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

vcl_result_t CompilerTestThread::run(const std::string& options) {
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

    vcl_executable_handle_t executable = NULL;
    vcl_executable_desc_t exeDesc = {modelIR.data(), modelIRSize, options.c_str(), options.size() + 1};

    ret = vclExecutableCreate(compiler, exeDesc, &executable);
    if (ret != VCL_RESULT_SUCCESS) {
        std::cerr << "Failed to create executable handle! Result:0x" << std::hex << uint64_t(ret) << std::dec
                  << std::endl;
        vclCompilerDestroy(compiler);
        return ret;
    }
    uint64_t blobSize = 0;
    ret = vclExecutableGetSerializableBlob(executable, NULL, &blobSize);
    if (ret != VCL_RESULT_SUCCESS || blobSize == 0) {
        std::cerr << "Failed to get blob size! Result:0x" << std::hex << uint64_t(ret) << std::dec << std::endl;
        vclExecutableDestroy(executable);
        vclCompilerDestroy(compiler);
        return ret;
    } else {
        uint8_t* blob = (uint8_t*)malloc(blobSize);
        ret = vclExecutableGetSerializableBlob(executable, blob, &blobSize);
        if (ret == VCL_RESULT_SUCCESS) {
#ifdef BLOB_DUMP
            std::string blobName = "ct1_" + threadName + ".net";
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
            lock.lock();
            outputs.push_back(output);
            lock.unlock();
        }
        free(blob);
    }

    ret = vclExecutableDestroy(executable);
    if (ret != VCL_RESULT_SUCCESS) {
        std::cerr << "Failed to destroy executable! Result:0x" << std::hex << uint64_t(ret) << std::dec << std::endl;
        vclCompilerDestroy(compiler);
        return ret;
    }
    executable = NULL;

    ret = vclCompilerDestroy(compiler);
    if (ret != VCL_RESULT_SUCCESS) {
        std::cerr << "Failed to destroy compiler! Result:0x" << std::hex << uint64_t(ret) << std::dec << std::endl;
        return ret;
    }
    return ret;
}

bool CompilerTestThread::check() const {
    const size_t count = outputs.size();
    if (count == 0) {
        std::cerr << "No outputs!" << std::endl;
        return false;
    }

    for (size_t i = 1; i < count; i++) {
        if (outputs[i].size() == 0) {
            std::cerr << "blob " << i << "'s size is zero." << std::endl;
            return false;
        }
    }
    return true;
}

TEST_P(CompilerTestThread, threadTest) {
    CompilerTestThread test;
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
        // In VpuxCompilerL0TestsCommon::create_simple_function, the created simple network only has one input/output,
        // so no need to iterate through iterators.
        std::string inputName = inputs.begin()->first;
        std::string outputName = outputs.begin()->first;

        // Update netOptions by replacing input/output placeholders with dynamic names.
        netOptions = std::regex_replace(netOptions, std::regex("SIMPLE_IN_PLACEHOLDER"), inputName);
        netOptions = std::regex_replace(netOptions, std::regex("SIMPLE_OUT_PLACEHOLDER"), outputName);

        std::string m_out_xml_path = "./simple_func_thread.xml";
        std::string m_out_bin_path = "./simple_func_thread.bin";

        ngraph::pass::Manager passManager;
        passManager.register_pass<ngraph::pass::Serialize>(m_out_xml_path, m_out_bin_path);
        passManager.run_passes(function);

        EXPECT_EQ(test.init(m_out_xml_path.c_str(), m_out_bin_path.c_str()), VCL_RESULT_SUCCESS);

        int xmlStatus = std::remove(m_out_xml_path.c_str());
        int binStatus = std::remove(m_out_bin_path.c_str());
        if (xmlStatus == 0 && binStatus == 0) {
            std::cout << "Temp IRs" << m_out_xml_path << " and " << m_out_bin_path << " deleting succeeded!"
                      << std::endl;
        } else {
            std::cerr << "Temp IRs" << m_out_xml_path << " and " << m_out_bin_path << " deleting failed!" << std::endl;
        }
    }

    std::vector<vcl_result_t> res(5, VCL_RESULT_SUCCESS);
    // Get the ref output from single thread env;

    auto runFunc = [&test] (std::string& options) {
        return test.run(options);
    };

    std::thread t0{[&] {
        res[0] = runFunc(netOptions);
    }};

    t0.join();
    // Get the outputs from multiple threads env;
    std::thread t1{[&] {
        res[1] = runFunc(netOptions);
    }};
    std::thread t2{[&] {
        res[2] = runFunc(netOptions);
    }};
    std::thread t3{[&] {
        res[3] = runFunc(netOptions);
    }};
    std::thread t4{[&] {
        res[4] = runFunc(netOptions);
    }};

    t1.join();
    t2.join();
    t3.join();
    t4.join();

    for (auto i = res.begin(); i != res.end(); ++i) {
        if (*i != VCL_RESULT_SUCCESS) {
            std::cerr << "Failed to run " << std::distance(res.begin(), i) << " thread!" << std::endl;
            std::cerr << "Result:0x" << std::hex << uint64_t(*i) << std::dec << std::endl;
        }
    }

    EXPECT_EQ(test.getOutputSize(), 5) << "Not get all outputs successfully!" << std::endl;
    EXPECT_EQ(test.check(), true);
}

// You need to export POR_PATH manually. E.g. export POR_PATH=/path/to/om-vpu-models-por-ww46
const std::vector<std::vector<std::string>> smoke_ir_names = {
        {"googlenet-v1",
        R"(--inputs_precisions="input:U8" --inputs_layouts="input:NCHW" )"
        R"(--outputs_precisions="InceptionV1/Logits/Predictions/Softmax:FP32" )"
        R"(--outputs_layouts="InceptionV1/Logits/Predictions/Softmax:NC" )"
        R"(--config LOG_LEVEL="LOG_INFO" VPUX_COMPILATION_MODE_PARAMS="use-user-precision=false propagate-quant-dequant=0")",
         "googlenet-v1/tf/FP16"},
        {"mobilenet-v2",
        R"(--inputs_precisions="result.1:fp16" --inputs_layouts="result.1:NCHW" )"
        R"(--outputs_precisions="473:fp16" --outputs_layouts="473:NC" )"
        R"(--config VPUX_PLATFORM="3700" DEVICE_ID="VPUX.3700" VPUX_COMPILATION_MODE="DefaultHW")",
         "mobilenet-v2/onnx/FP16"},
        {"simple_function",
        R"(--inputs_precisions="SIMPLE_IN_PLACEHOLDER:fp16" --inputs_layouts="SIMPLE_IN_PLACEHOLDER:NC" )"
        R"(--outputs_precisions="SIMPLE_OUT_PLACEHOLDER:fp16" --outputs_layouts="SIMPLE_OUT_PLACEHOLDER:NC" )"
        R"(--config LOG_LEVEL="LOG_NONE" VPUX_PLATFORM="3700" DEVICE_ID="VPUX.3700" )"
        R"(VPUX_COMPILATION_MODE="DefaultHW")",
         ""},
};

const std::vector<std::vector<std::string>> ir_names = {
        {"resnet-50-pytorch",
        R"(--inputs_precisions="result.1:fp16" --inputs_layouts="result.1:NCHW" )"
        R"(--outputs_precisions="495:fp16" --outputs_layouts="495:NC" )"
        R"(--config VPUX_PLATFORM="3700" DEVICE_ID="VPUX.3700" VPUX_COMPILATION_MODE="DefaultHW")",
         "resnet-50-pytorch/onnx/FP16"},
        {"yolo_v4",
        R"(--inputs_precisions="image_input:fp16" --inputs_layouts="image_input:NCHW" )"
        R"(--outputs_precisions="conv2d_93/BiasAdd/Add:fp16 conv2d_101/BiasAdd/Add:fp16 conv2d_109/BiasAdd/Add:fp16" )"
        R"(--outputs_layouts="conv2d_93/BiasAdd/Add:NCHW conv2d_101/BiasAdd/Add:NCHW conv2d_109/BiasAdd/Add:NCHW" )"
        R"(--config VPUX_PLATFORM="3700" DEVICE_ID="VPUX.3700" VPUX_COMPILATION_MODE="DefaultHW")",
         "yolo_v4/tf/FP16-INT8"},
};

const auto smoke_params = testing::Combine(testing::ValuesIn(smoke_ir_names));

const auto params = testing::Combine(testing::ValuesIn(ir_names));

INSTANTIATE_TEST_SUITE_P(smoke_Compiler_test, CompilerTestThread, smoke_params, CompilerTestThread::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(Compiler_test, CompilerTestThread, params, CompilerTestThread::getTestCaseName);
