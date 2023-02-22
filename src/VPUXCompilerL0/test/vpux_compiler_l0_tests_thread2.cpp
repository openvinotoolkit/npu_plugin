//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
#include <thread>
#include <tuple>
#include <vector>

using compilerThread2Params = std::tuple<std::unordered_map<std::string, std::string>>;

class CompilerTestThread2 :
        public VpuxCompilerL0TestsUtils::VpuxCompilerL0TestsCommon,
        public testing::WithParamInterface<compilerThread2Params>,
        public testing::Test {
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
    void Run();
    void SetUp() override;

    static std::string getTestCaseName(const testing::TestParamInfo<compilerThread2Params>& obj) {
        auto param = obj.param;
        auto netInfo = std::get<0>(param);
        const std::string netName = netInfo.at("network");
        const std::string deviceID = netInfo.at("device");
        const std::string testCaseName = netName + std::string("_VPUX.") + deviceID;
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
    std::vector<vcl_result_t> resCreate(numCompilationThreads, VCL_RESULT_SUCCESS);

    for (int i = 0; i < numCompilationThreads; i++) {
        vcl_executable_handle_t* exeHandle = new vcl_executable_handle_t();
        uint64_t blobSize = 0;
        std::thread thread{[&resCreate, &compiler, &exeDesc, exeHandle, i] {
            resCreate[i] = vclExecutableCreate(compiler, exeDesc, exeHandle);
        }};
        exeHandles.push_back(std::make_pair(exeHandle, blobSize));

        compilationThreads.push_back(move(thread));
    }
    for (auto& compilationThread : compilationThreads) {
        compilationThread.join();
    }

    for (auto i = resCreate.begin(); i != resCreate.end(); ++i) {
        if (*i != VCL_RESULT_SUCCESS) {
            std::cerr << "Failed to vclExecutableCreate with " << std::distance(resCreate.begin(), i) << " thread!"
                      << std::endl;
            std::cerr << "Result:0x" << std::hex << uint64_t(*i) << std::dec << std::endl;
            return *i;
        }
    }

    std::vector<std::thread> compilationThreads2;
    std::vector<vcl_result_t> resGetBlobInit(numCompilationThreads, VCL_RESULT_SUCCESS);
    unsigned idx = 0;
    for (auto& pair : exeHandles) {
        vcl_executable_handle_t exeHandle = *(pair.first);
        uint64_t& blobSize = pair.second;
        uint8_t* blob = NULL;
        std::thread thread{[&blobSize, idx, &resGetBlobInit, exeHandle, blob] {
            resGetBlobInit[idx] = vclExecutableGetSerializableBlob(exeHandle, blob, &blobSize);
        }};
        compilationThreads2.push_back(move(thread));
        idx++;
    }
    for (auto& compilationThread2 : compilationThreads2) {
        compilationThread2.join();
    }

    for (auto i = resGetBlobInit.begin(); i != resGetBlobInit.end(); i++) {
        if (*i != VCL_RESULT_SUCCESS) {
            std::cerr << "Failed to vclExecutableGetSerializableBlob initially with " << i - resGetBlobInit.begin()
                      << " thread!" << std::endl;
            std::cerr << "Result:0x" << std::hex << uint64_t(*i) << std::dec << std::endl;
            return *i;
        }
    }

    std::vector<std::thread> getBlobThreads;
    std::vector<std::pair<uint8_t*, uint64_t>> blobs;
    std::vector<vcl_result_t> resGetBlob(numGetBlobThreads, VCL_RESULT_SUCCESS);

    for (int i = 0; i < numGetBlobThreads; i++) {
        int idx = i % numCompilationThreads;

        vcl_executable_handle_t exe = *(exeHandles[idx].first);
        uint64_t blobSize = exeHandles[idx].second;
        uint8_t* blob = (uint8_t*)malloc(blobSize);
        std::thread thread{[&resGetBlob, &exeHandles, exe, i, blob, idx] {
            resGetBlob[i] = vclExecutableGetSerializableBlob(exe, blob, &(exeHandles[idx].second));
        }};
        blobs.push_back(std::make_pair(blob, exeHandles[idx].second));

        getBlobThreads.push_back(move(thread));
    }
    for (auto& getBlobThread : getBlobThreads) {
        getBlobThread.join();
    }

    for (auto i = resGetBlob.begin(); i != resGetBlob.end(); i++) {
        if (*i != VCL_RESULT_SUCCESS) {
            std::cerr << "Failed to vclExecutableGetSerializableBlob with " << i - resGetBlob.begin() << " thread!"
                      << std::endl;
            std::cerr << "Result:0x" << std::hex << uint64_t(*i) << std::dec << std::endl;
            return *i;
        }
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
    for (size_t i = 1; i < count; i++) {
        if (outputs[i].size() == 0) {
            std::cerr << "blob " << i << "'s size is zero." << std::endl;
            return false;
        }
    }
    return true;
}

void CompilerTestThread2::SetUp() {
    auto ir = GetParam();
    auto netInfo = std::get<0>(ir);
    const std::string net = netInfo.at("network");
    netOptions = netInfo.at("info");
    postProcessNetOptions(netInfo.at("device"));
    if (net != "simple_function") {
        std::string PATH = getTestModelsBasePath();
        if (PATH.empty()) {
            GTEST_SKIP() << "POR_PATH is empty. Skipping this test";
        }
        const std::string interPath = netInfo.at("path");

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

        EXPECT_EQ(init(netName.c_str(), weightName.c_str()), VCL_RESULT_SUCCESS);
    } else {
        auto function = create_simple_function();
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

        std::string m_out_xml_path = "./simple_func_thread2.xml";
        std::string m_out_bin_path = "./simple_func_thread2.bin";

        ngraph::pass::Manager passManager;
        passManager.register_pass<ngraph::pass::Serialize>(m_out_xml_path, m_out_bin_path);
        passManager.run_passes(function);

        EXPECT_EQ(init(m_out_xml_path.c_str(), m_out_bin_path.c_str()), VCL_RESULT_SUCCESS);
        int xmlStatus = std::remove(m_out_xml_path.c_str());
        int binStatus = std::remove(m_out_bin_path.c_str());

        if (xmlStatus == 0 && binStatus == 0) {
            std::cout << "Temp IRs" << m_out_xml_path << " and " << m_out_bin_path << " deleting succeeded!"
                      << std::endl;
        } else {
            std::cerr << "Temp IRs" << m_out_xml_path << " and " << m_out_bin_path << " deleting failed!" << std::endl;
        }
    }
}

void CompilerTestThread2::Run() {
    setThreadCount(1, 1);
    vcl_result_t ret = run(getNetOptions());
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS) << "Failed to run test to create ref! Result:0x" << std::hex << uint64_t(ret)
                                       << std::dec << std::endl;

    // Get the outputs from multiple threads env;
    int numCompilationThreads = 5;
    int numGetBlobThreads = 17;
    setThreadCount(numCompilationThreads, numGetBlobThreads);
    ret = run(getNetOptions());
    EXPECT_EQ(ret, VCL_RESULT_SUCCESS) << "Failed to run thread test! Result:0x" << std::hex << uint64_t(ret)
                                       << std::dec << std::endl;
    EXPECT_EQ(getOutputSize(), 18) << "Not get all outputs successfully!" << std::endl;
    EXPECT_EQ(check(), true);
}

TEST_P(CompilerTestThread2, threadTest2) {
    Run();
}

const auto cidTool = CompilerTestThread2::getCidToolPath();
const auto smoke_ir_infos = CompilerTestThread2::readJson2Vec(cidTool + VpuxCompilerL0TestsUtils::SMOKE_TEST_CONFIG);
const auto ir_infos = CompilerTestThread2::readJson2Vec(cidTool + VpuxCompilerL0TestsUtils::TEST_CONFIG);

const auto smoke_params = testing::Combine(testing::ValuesIn(smoke_ir_infos));

const auto params = testing::Combine(testing::ValuesIn(ir_infos));

INSTANTIATE_TEST_SUITE_P(smoke_Compiler_test, CompilerTestThread2, smoke_params, CompilerTestThread2::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Compiler_test, CompilerTestThread2, params, CompilerTestThread2::getTestCaseName);
