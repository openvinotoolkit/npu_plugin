// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_test_tool.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>

#include <vpu/utils/error.hpp>

using namespace InferenceEngine;

namespace LayerTestsUtils {

KmbTestTool::KmbTestTool(const KmbTestEnvConfig& envCfg): envConfig(envCfg),
    DEVICE_NAME(envConfig.IE_KMB_TESTS_DEVICE_NAME.empty() ? "VPUX" : envConfig.IE_KMB_TESTS_DEVICE_NAME) {
}

void KmbTestTool::exportNetwork(ExecutableNetwork& exeNet, const std::string& fsName) {
    IE_ASSERT(!envConfig.IE_KMB_TESTS_DUMP_PATH.empty());

    const auto fileName = vpu::formatString("%v/%v", envConfig.IE_KMB_TESTS_DUMP_PATH, fsName);
    std::cout << "Exporting nn into " << (envConfig.IE_KMB_TESTS_RAW_EXPORT ? "" : "not ") << "raw file " << fileName
        << ", device " << DEVICE_NAME << std::endl;

    if (envConfig.IE_KMB_TESTS_RAW_EXPORT) {
        exeNet.Export(fileName);
    } else {
        std::ofstream file(fileName, std::ios_base::out | std::ios_base::binary);
        if (!file.is_open())
            IE_THROW() << "exportNetwork(). Can't open file " << fileName;

        exeNet.Export(file);
    }
    std::cout << "Exported nn into file " << fileName << std::endl;
}

ExecutableNetwork KmbTestTool::importNetwork(const std::shared_ptr<InferenceEngine::Core>& core, const std::string& fsName) {
    IE_ASSERT(!envConfig.IE_KMB_TESTS_DUMP_PATH.empty());

    const auto fileName = vpu::formatString("%v/%v", envConfig.IE_KMB_TESTS_DUMP_PATH, fsName);
    std::cout << "Importing nn from " << (envConfig.IE_KMB_TESTS_RAW_EXPORT ? "" : "not ") << "raw file " << fileName
        << ", device " << DEVICE_NAME << std::endl;

    if (envConfig.IE_KMB_TESTS_RAW_EXPORT) {
        return core->ImportNetwork(fileName, DEVICE_NAME);
    } else {
        std::ifstream file(fileName, std::ios_base::in | std::ios_base::binary);
        if (!file.is_open())
            IE_THROW() << "importNetwork(). Can't open file " << fileName;

        return core->ImportNetwork(file, DEVICE_NAME);
    }
}

void KmbTestTool::importBlob(InferenceEngine::Blob::Ptr blob, const std::string& fsName) {
    IE_ASSERT(!envConfig.IE_KMB_TESTS_DUMP_PATH.empty());

    const auto fileName = vpu::formatString("%v/%v", envConfig.IE_KMB_TESTS_DUMP_PATH, fsName);
    std::ifstream file(fileName, std::ios_base::in | std::ios_base::binary);
    if (!file.is_open())
        IE_THROW() << "importBlob(). Can't open file " << fileName;

    file.read(blob->cbuffer().as<char*>(), static_cast<std::streamsize>(blob->byteSize()));
    if (!file)
        IE_THROW() << "exportBlob(). Error when reading file " << fileName;
}

void KmbTestTool::exportBlob(const InferenceEngine::Blob::Ptr blob, const std::string& fsName) {
    IE_ASSERT(!envConfig.IE_KMB_TESTS_DUMP_PATH.empty());

    const auto fileName = vpu::formatString("%v/%v", envConfig.IE_KMB_TESTS_DUMP_PATH, fsName);
    std::ofstream file(fileName, std::ios_base::out | std::ios_base::binary);
    if (!file.is_open())
        IE_THROW() << "exportBlob(). Can't open file " << fileName;

    file.write(blob->cbuffer().as<const char*>(), static_cast<std::streamsize>(blob->byteSize()));
    if (!file)
        IE_THROW() << "exportBlob(). Error when writing file " << fileName;
}

unsigned long int FNV_hash(const std::string &str)
{
  const unsigned char* p = reinterpret_cast<const unsigned char *>(str.c_str());
  unsigned long int h = 2166136261UL;

  for(size_t i = 0; i < str.size(); i++)
    h = (h * 16777619) ^ p[i];

  return h;
}

std::string cleanName(std::string name) {
    std::replace_if(
        name.begin(), name.end(),
        [](char c) {
            return !(std::isalnum(c) || c == '.');
        },
        '_');
    return name;
}

std::string filesysName(const testing::TestInfo* testInfo, const std::string& ext, bool limitAbsPathLength) {

    const size_t maxExpectedFileNameLen = 256, maxExpectedDirLen = 100, extLen = ext.size();
    const size_t maxFileNameLen = (limitAbsPathLength ? maxExpectedFileNameLen - maxExpectedDirLen : maxExpectedFileNameLen),
        maxNoExtLen = maxFileNameLen - extLen, maxNoExtShortenedLen = maxNoExtLen - 20 - 1;
    const auto testName = vpu::formatString("%v_%v", testInfo->test_case_name(), testInfo->name());
    auto fnameNoExt = (testName.size() < maxNoExtLen) ? testName : vpu::formatString("%v_%v", testName.substr(0, maxNoExtShortenedLen), FNV_hash(testName));

    return cleanName(fnameNoExt.append(ext));
}

}  // namespace LayerTestsUtils
