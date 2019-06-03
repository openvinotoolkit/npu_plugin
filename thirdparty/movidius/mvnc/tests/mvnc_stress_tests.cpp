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

#include <gtest/gtest.h>

#include "mvnc.h"
#include "mvnc_tests_common.hpp"
#include "ncPrivateTypes.h"

class MvncStressTests : public MvncTestCommon {
public:
    int available_devices = 0;
protected:
    virtual ~MvncStressTests() {}
    void SetUp() override {
        MvncTestCommon::SetUp();
        available_devices = getAmountOfDevices();
        ASSERT_TRUE(available_devices > 0);
        ASSERT_NO_ERROR(setLogLevel(MVLOG_WARN));

#ifdef NO_BOOT
        // In case already booted device exist, do nothing
        if (getAmountOfBootedDevices() == 0) {
            MvncTestCommon::bootOneDevice();
        }
#endif
    }
};

/**
* @brief Open and close device for 1001 times
*/
TEST_F(MvncStressTests, OpenClose1001) {
    const int iterations = 1001;
    ncDeviceHandle_t *deviceHandle = nullptr;
    for (int i = 0; i < iterations; ++i) {
        fprintf(stderr, "Iteration %d of %d\n", i, iterations);
        ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, ANY_PLATFORM, NC_USB, watchdogInterval, _firmwarePath));
        ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));
        deviceHandle = nullptr;
    }
}

/**
* @brief Allocate and deallocate graph on device for 1001 times
*/
TEST_F(MvncStressTests, AllocateDeallocateGraph1001) {
    const int iterations = 1001;

    // Load graph
    const std::string blobPath = "bvlc_googlenet_fp16.blob";
    std::vector<char> _blob;
    ASSERT_NO_ERROR(readBINFile(blobPath, _blob));

    // Open device
    ncDeviceHandle_t *deviceHandle = nullptr;
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, ANY_PLATFORM, NC_USB, watchdogInterval, _firmwarePath));
    for (int i = 0; i < iterations; ++i) {
        fprintf(stderr, "Iteration %d of %d\n", i, iterations);
        // Create graph handlers
        ncGraphHandle_t*  graphHandle = nullptr;
        std::string graphName = "graph";

        ASSERT_NO_ERROR(ncGraphCreate(graphName.c_str(), &graphHandle));
        ASSERT_TRUE(graphHandle != nullptr);

        // Allocate graph
        ASSERT_NO_ERROR(ncGraphAllocate(deviceHandle, graphHandle,
                                        _blob.data(), _blob.size(),     // Blob
                                        _blob.data(), sizeof(ElfN_Ehdr) + sizeof(blob_header_v2)) );   // Header

        // Destroy graph
        ASSERT_NO_ERROR(ncGraphDestroy(&graphHandle));
    }
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));
}


/**
* @brief Run the full cycle of inference 101 times.
* It includes opening device, allocating graph and fifos, inference, destroying graph and fifos, closing device
*/
TEST_F(MvncStressTests, FullCycleOfWork101Times) {
    const int iterations = 101;

    const std::string blobPath = "bvlc_googlenet_fp16.blob";
    std::vector<char> blob;
    ASSERT_NO_ERROR(readBINFile(blobPath, blob));

    for (int i = 0; i < iterations; i++) {
        ncDeviceHandle_t *deviceHandle = nullptr;
        ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, ANY_PLATFORM, NC_USB, watchdogInterval, _firmwarePath));


        ncGraphHandle_t*  graphHandle = nullptr;
        std::string graphName = "graph";
        ASSERT_NO_ERROR(ncGraphCreate(graphName.c_str(), &graphHandle));
        ASSERT_TRUE(graphHandle != nullptr);

        ASSERT_NO_ERROR(ncGraphAllocate(deviceHandle, graphHandle,
                                        blob.data(), blob.size(),     // Blob
                                        blob.data(), sizeof(ElfN_Ehdr) + sizeof(blob_header_v2) ));


        unsigned int dataLength = sizeof(int);

        int numInputs = 0;
        ASSERT_NO_ERROR(ncGraphGetOption(graphHandle, NC_RO_GRAPH_INPUT_COUNT, &numInputs, &dataLength));

        int numOutputs = 0;
        ASSERT_NO_ERROR(ncGraphGetOption(graphHandle, NC_RO_GRAPH_OUTPUT_COUNT, &numOutputs, &dataLength));

        dataLength = sizeof(ncTensorDescriptor_t);

        ncTensorDescriptor_t inputDesc = {};
        ASSERT_NO_ERROR(ncGraphGetOption(graphHandle, NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS, &inputDesc,
                                         &dataLength));


        ncTensorDescriptor_t outputDesc = {};
        ASSERT_NO_ERROR(ncGraphGetOption(graphHandle, NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS, &outputDesc,
                                         &dataLength));

        unsigned int fifo_elements = 4;

        ncFifoHandle_t *inputFifoHandle = nullptr;
        ASSERT_NO_ERROR(ncFifoCreate("input", NC_FIFO_HOST_WO, &inputFifoHandle));

        ASSERT_NO_ERROR(ncFifoAllocate(inputFifoHandle, deviceHandle, &inputDesc, fifo_elements));

        ncFifoHandle_t *outputFifoHandle = nullptr;
        ASSERT_NO_ERROR(ncFifoCreate("output", NC_FIFO_HOST_RO, &outputFifoHandle));

        ASSERT_NO_ERROR(ncFifoAllocate(outputFifoHandle, deviceHandle, &outputDesc, fifo_elements));

        uint8_t *input_data = new uint8_t[inputDesc.totalSize];
        uint8_t *result_data = new uint8_t[outputDesc.totalSize];
        ASSERT_NO_ERROR(ncGraphQueueInferenceWithFifoElem(graphHandle,
                                                          inputFifoHandle, outputFifoHandle,
                                                          input_data, &inputDesc.totalSize, nullptr));

        void *userParam = nullptr;
        ASSERT_NO_ERROR(ncFifoReadElem(outputFifoHandle, result_data, &outputDesc.totalSize, &userParam));

        delete[] input_data;
        delete[] result_data;
        ASSERT_NO_ERROR(ncFifoDestroy(&inputFifoHandle));
        ASSERT_NO_ERROR(ncFifoDestroy(&outputFifoHandle));

        ASSERT_NO_ERROR(ncGraphDestroy(&graphHandle));

        ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));
    }

}
