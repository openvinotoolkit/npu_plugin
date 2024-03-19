//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <common_test_utils/test_common.hpp>
#include "vpux/al/config/common.hpp"
#include "vpux/al/config/compiler.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux_compiler.hpp"

#include "llvm/Support/SHA256.h"

#include <gtest/gtest.h>
#include <openvino/openvino.hpp>

#include <array>
#include <fstream>
#include <future>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using namespace vpux;

using CompilationParams = std::tuple<std::vector<std::string>,  // model paths
                                     size_t,                    // number of threads per model
                                     size_t                     // number of compilation iterations per model thread
                                     >;

// The compilation status for each iteration of each thread, for every model
// Value true represents a successful compilation, while the string represents the failure message if it is present
using Status = std::tuple<bool, std::string>;
using CompilationStatus = std::vector<std::vector<std::vector<Status>>>;

using Checksum = std::array<uint8_t, 32>;
using CompilationChecksums = std::vector<std::vector<std::vector<Checksum>>>;

namespace {

class ChecksumException : public std::exception {
public:
    ChecksumException(const std::string& message): _message(message) {
    }
    const char* what() const noexcept override {
        return _message.c_str();
    }

private:
    std::string _message;
};

std::string stringifyChecksum(const Checksum& checksum) {
    std::string checksumStr;
    checksumStr.reserve(checksum.size() * 2);
    for (const auto byte : checksum) {
        char byteStr[3];
        sprintf(byteStr, "%02x", byte);
        checksumStr += byteStr;
    }
    return checksumStr;
};
}  // namespace

class CompilationTest : public testing::WithParamInterface<CompilationParams>, virtual public ov::test::TestsCommon {
public:
    CompilationTest()
            : _options{std::make_shared<OptionsDesc>()},
              _config{_options},
              _compiler{nullptr},
              _modelPaths{},
              _numThreads{},
              _log(Logger::global()) {
        _log.setName("CompilationTest");
    }

    static std::string getTestCaseName(const testing::TestParamInfo<CompilationParams>& obj) {
        std::vector<std::string> modelPaths;
        size_t numThreads;
        size_t numIterations;
        std::tie(modelPaths, numThreads, numIterations) = obj.param;
        const auto numModels = modelPaths.size();

        std::ostringstream result;
        result << "modelPaths=[";
        for (size_t i = 0; i < numModels - 1; ++i) {
            result << modelPaths[i] << ",";
        }
        result << modelPaths[numModels - 1] << "]_";
        result << "threads=" << numThreads << "_";
        result << "iterations=" << numIterations;
        return result.str();
    }

protected:
    void SetUp() override {
        _modelPaths = std::get<0>(GetParam());
        _numThreads = std::get<1>(GetParam());
        _numIterations = std::get<2>(GetParam());

        validateAndExpandModelPaths(_modelPaths);

        registerCommonOptions(*_options);
        registerCompilerOptions(*_options);
        _compiler = Compiler::create(_config);
    }

    void SetPlatform(const std::string& platform) {
        _config.update({{PLATFORM::key().data(), platform}});
    }

    void UseGraphFileBackend() {
        _config.update({{USE_ELF_COMPILER_BACKEND::key().data(), "NO"}});
    }

    void UseElfBackend() {
        _config.update({{USE_ELF_COMPILER_BACKEND::key().data(), "YES"}});
    }

    void Run() const {
        const auto compileNetwork = [](const Compiler::Ptr& compiler, std::shared_ptr<ov::Model>& model,
                                       const Config& config) -> Checksum {
            const auto netDesc = compiler->compile(model, model->get_name(), config);
            const auto blob = netDesc->getCompiledNetwork();
            const auto blobReference = llvm::ArrayRef(reinterpret_cast<const uint8_t*>(blob.data()), blob.size());
            return llvm::SHA256::hash(blobReference);
        };

        const auto threadFunction = [compileNetwork](const Compiler::Ptr compiler, std::shared_ptr<ov::Model> model,
                                                     const Config& config, const size_t iterationsCount) -> Checksum {
            const auto previousHash = compileNetwork(compiler, model, config);
            for (auto i : irange(iterationsCount - 1)) {
                const auto currentHash = compileNetwork(compiler, model, config);
                if (previousHash != currentHash) {
                    std::stringstream ss;
                    ss << "Checksum for iteration " << i + 1 << " (" << stringifyChecksum(currentHash)
                       << ") different than previous hash (" << stringifyChecksum(previousHash) << ")";
                    throw ChecksumException(ss.str());
                }
            }
            return previousHash;
        };

        std::vector<std::future<Checksum>> futures;
        size_t futureIdx = 0;

        ov::Core core;
        std::unordered_map<size_t, size_t> modelForFutures;
        for (auto p : _modelPaths | indexed) {
            const auto modelIdx = p.index();
            const auto& modelPath = p.value();

            _log.trace("Model {0} with path '{1}'", modelIdx + 1, modelPath);

            std::vector<std::shared_ptr<ov::Model>> models;
            for (const size_t tIt : irange(_numThreads)) {
                const auto model = core.read_model(modelPath);
                _log.nest().trace("Read model with name '{0}' for thread {1} / {2}", model->get_friendly_name(),
                                  tIt + 1, _numThreads);
                models.push_back(model);
            }

            for (const size_t tIt : irange(_numThreads)) {
                modelForFutures[futureIdx++] = modelIdx;

                const auto& model = models.at(tIt);

                _log.nest().trace("Starting thread {0} / {1} with {2} iteration(s)", tIt + 1, _numThreads,
                                  _numIterations);
                futures.push_back(
                        std::async(std::launch::async, threadFunction, _compiler, model, _config, _numIterations));
            }
        }

        bool anyFailure = false;
        std::unordered_map<size_t, Checksum> modelChecksums;
        for (const auto& future : futures | indexed) {
            try {
                const auto threadChecksum = future.value().get();

                const auto modelIdx = modelForFutures[future.index()];
                if (modelChecksums.find(modelIdx) == modelChecksums.end()) {
                    modelChecksums[modelIdx] = threadChecksum;
                    continue;
                }
                const auto& modelChecksum = modelChecksums.at(modelIdx);
                if (threadChecksum != modelChecksum) {
                    std::stringstream ss;
                    ss << "Checksum " << stringifyChecksum(threadChecksum)
                       << " different than hash of other threads: " << stringifyChecksum(modelChecksum);
                    throw ChecksumException(ss.str());
                }

                continue;
            } catch (const ChecksumException& hashException) {
                std::cout << "Checksum error for thread " << future.index() << ":" << std::endl;
                std::cout << "    " << hashException.what() << std::endl;
            } catch (const std::exception& compileException) {
                std::cout << "Compilation error for thread " << future.index() << ":" << std::endl;
                std::cout << "    " << compileException.what() << std::endl;
            } catch (...) {
                std::cout << "General error for thread " << future.index() << std::endl;
            }
            anyFailure = true;
        }

        ASSERT_EQ(anyFailure, false);
    }

private:
    // Finds instances of environmental variables (e.g. ${MODELS_PATH}) and expands them into the path string
    // In case any path does not point to a valid .xml file (and associated .bin file),
    // the function returns false
    void validateAndExpandModelPaths(std::vector<std::string>& modelPaths) {
        for (auto& path : modelPaths) {
            std::vector<std::tuple<size_t, size_t, std::string>> vars;

            size_t index = 0;
            while ((index = path.find("$", index)) != std::string::npos) {
                const auto indexLPar = index + 1;
                if (indexLPar >= path.size() || path[indexLPar] != '{') {
                    ++index;
                }
                size_t indexRPar = path.find("}", indexLPar);
                if (indexRPar >= path.size()) {
                    ++index;
                    continue;
                }
                const auto envVar = path.substr(indexLPar + 1, indexRPar - indexLPar - 1);
                if (const auto env = std::getenv(envVar.c_str())) {
                    vars.push_back(std::make_tuple(index, indexRPar, env));
                }
                index += envVar.length();
            }

            for (const auto& var : vars) {
                const auto indexStart = std::get<0>(var);
                const auto indexEnd = std::get<1>(var);
                const auto envVarValue = std::get<2>(var);
                path.replace(indexStart, indexEnd - indexStart + 1, envVarValue.c_str());
            }
        }

        for (const auto& path : modelPaths) {
            std::ifstream xmlFile(path.c_str());
            ASSERT_TRUE(xmlFile.good()) << "Invalid model xml path: " << path;

            auto binFilePath = path;
            const std::string binExt(".bin");
            binFilePath.replace(binFilePath.size() - binExt.size(), binExt.size(), binExt);
            std::ifstream binFile(path.c_str());
            ASSERT_TRUE(binFile.good()) << "Invalid model bin path: " << binFilePath;
        }
    }

private:
    std::shared_ptr<OptionsDesc> _options;
    Config _config;
    Compiler::Ptr _compiler;

    std::vector<std::string> _modelPaths;
    size_t _numIterations;
    size_t _numThreads;

    Logger _log;
};

TEST_P(CompilationTest, VPU3720) {
    SetPlatform("VPU3720");
    UseGraphFileBackend();
    Run();
}

using CompilationTest_ELF = CompilationTest;

TEST_P(CompilationTest_ELF, VPU3720) {
    SetPlatform("VPU3720");
    UseElfBackend();
    Run();
}

//
// GraphFile backend
//

INSTANTIATE_TEST_SUITE_P(
        DISABLED_precommit_multithreaded, CompilationTest,
        ::testing::Combine(::testing::Values(std::vector<std::string>{
                                   "${MODELS_PATH}/models/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224_i8.xml"}),
                           ::testing::ValuesIn(std::vector<size_t>{4}),  // num threads per model
                           ::testing::Values(1)                          // num iterations per thread
                           ),
        CompilationTest::getTestCaseName);

//
// ELF backend
//

INSTANTIATE_TEST_SUITE_P(
        DISABLED_precommit_single_thread, CompilationTest_ELF,
        ::testing::Combine(::testing::Values(std::vector<std::string>{
                                   "${MODELS_PATH}/models/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224_i8.xml"}),
                           ::testing::ValuesIn(std::vector<size_t>{1}),  // num threads per model
                           ::testing::Values(3)                          // num iterations per thread
                           ),
        CompilationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        DISABLED_precommit_multithreaded, CompilationTest_ELF,
        ::testing::Combine(::testing::Values(std::vector<std::string>{
                                   "${MODELS_PATH}/models/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224_i8.xml"}),
                           ::testing::ValuesIn(std::vector<size_t>{4}),  // num threads per model
                           ::testing::Values(1)                          // num iterations per thread
                           ),
        CompilationTest_ELF::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        DISABLED_precommit_multithreaded_two_models, CompilationTest_ELF,
        ::testing::Combine(::testing::Values(std::vector<std::string>{
                                   "${MODELS_PATH}/models/mobilenet_v2_1.4_224/mobilenet_v2_1.4_224_i8.xml",
                                   "${MODELS_PATH}/models/resnet_v1_50/resnet_v1_50_i8.xml",
                           }),
                           ::testing::ValuesIn(std::vector<size_t>{1}),  // num threads per model
                           ::testing::Values(2)                          // num iterations per thread
                           ),
        CompilationTest_ELF::getTestCaseName);
