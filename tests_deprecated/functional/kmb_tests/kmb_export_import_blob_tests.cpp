//
// Copyright 2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <fstream>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/vpu_compiler_config.hpp>

#include "kmb_layers_tests.hpp"
#include "kmb_xml_tests.hpp"

#define ERROR_BOUND (.1f)

using namespace InferenceEngine;
using namespace details;

enum class FileIOResult { FileNotOpened = -1, FilesWithDifferentSize = -2, FilesHaveEqualSize = 1 };

#ifdef ENABLE_MCM_COMPILER
size_t getFileSize(const std::string& fileName) {
    std::ifstream file(fileName.c_str(), std::ifstream::in | std::ifstream::binary);

    if (!file.is_open()) {
        return static_cast<size_t>(FileIOResult::FileNotOpened);
    }

    file.seekg(0, std::ios::end);
    size_t sizeOfFile = file.tellg();
    file.close();

    return sizeOfFile;
}

FileIOResult isContentOfFilesEqual(const std::string& fileName1, const std::string& fileName2) {
    std::ifstream file1(fileName1.c_str(), std::ifstream::in | std::ifstream::binary);
    std::ifstream file2(fileName2.c_str(), std::ifstream::in | std::ifstream::binary);

    if (!file1.is_open() || !file2.is_open()) {
        return FileIOResult ::FileNotOpened;
    }

    char x, y;

    while (!file1.eof() || !file2.eof()) {
        file1.read(&x, 1);
        file2.read(&y, 1);
        if (x != y) return FileIOResult ::FilesWithDifferentSize;
    }
    return FileIOResult ::FilesHaveEqualSize;
}

void ExportImportBlobToFromFile(std::string deviceName, const CNNNetwork& network,
    std::map<std::string, std::string>& config, const std::string& testDescription) {
    Core ie;
    ExecutableNetwork exeNetwork;
    ASSERT_NO_THROW(exeNetwork = ie.LoadNetwork(network, deviceName, config));

    std::string blobFileName1 = "TestExportImportBlob_" + testDescription + "_file01.blob";
    ASSERT_NO_THROW(exeNetwork.Export(blobFileName1));
    ASSERT_GT(getFileSize(blobFileName1), 0) << "Alarm! Alarm! We have gotten blob file with zero size!!!";
    config["VPU_KMB_KMB_EXECUTOR"] = CONFIG_VALUE(NO);

    ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(blobFileName1, deviceName, config));
    std::string blobFileName2 = "TestExportImportBlob_" + testDescription + "_file02.blob";
    ASSERT_NO_THROW(importedNetwork.Export(blobFileName2));

    ASSERT_GT(getFileSize(blobFileName1), 0);  // Test to be sure that first file size is not zero.
    ASSERT_GT(getFileSize(blobFileName2), 0);  // Test to be sure that second file size is not zero.
    ASSERT_EQ(getFileSize(blobFileName1), getFileSize(blobFileName2));  // And now compare size of first and second file

    ASSERT_EQ(isContentOfFilesEqual(blobFileName1, blobFileName2), FileIOResult::FilesHaveEqualSize);
}

// Disabled because LoadNetwork fails to initialize device
// [Track number: S#21379]
TEST_F(kmbLayersTests_nightly, DISABLED_TestExportImportBlob_Convolution_After_Scale_Shift) {
    extern std::string conv_after_scale_shift;
    std::string model = conv_after_scale_shift;
    REPLACE_WITH_STR(model, "<biases offset=\"6\" size=\"6\"/>", " ");
    REPLACE_WITH_STR(model, "<biases offset=\"18828\" size=\"128\"/>", " ");

    std::size_t weightSize = 6 + 18816;
    std::size_t biasSize = 6 + 128;
    TBlob<uint8_t>::Ptr weightsBlob(GenWeights<uint16_t>(weightSize + biasSize));
    Core ie;
    CNNNetwork network;
    ASSERT_NO_THROW(network = ie.ReadNetwork(model, weightsBlob));

    std::map<std::string, std::string> config;
    setCommonConfig(config);

    ExportImportBlobToFromFile(deviceName, network, config, "Convolution_After_Scale_Shift");
}

// Disabled because LoadNetwork fails to initialize device
// [Track number: S#21379]
TEST_F(kmbLayersTests_nightly, DISABLED_TestExportImportBlob_Pooling) {
    extern std::string pooling_test2;
    const std::string model = pooling_test2;

    Core ie;
    CNNNetwork network;
    ASSERT_NO_THROW(network = ie.ReadNetwork(model, Blob::CPtr()));

    std::map<std::string, std::string> config;
    setCommonConfig(config);

    ExportImportBlobToFromFile(deviceName, network, config, "Pooling");
}

// Disabled because LoadNetwork fails to initialize device
// [Track number: S#21379]
TEST_F(kmbLayersTests_nightly, DISABLED_TestExportImportBlob_ReLU) {
    extern std::string relu_test_2;
    const std::string model = relu_test_2;

    Core ie;
    CNNNetwork network;
    ASSERT_NO_THROW(network = ie.ReadNetwork(model, Blob::CPtr()));

    std::map<std::string, std::string> config;
    setCommonConfig(config);

    ExportImportBlobToFromFile(deviceName, network, config, "ReLU");
}

#endif
