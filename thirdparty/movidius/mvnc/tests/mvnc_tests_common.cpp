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

#include "mvnc.h"
#include "mvnc_tests_common.hpp"

//  ***********************************************  //
//              Open Device TESTS                    //
class MvncOpenDevice : public MvncTestCommon {
public:
    int available_devices = 0;
protected:
    virtual ~MvncOpenDevice() {}
    void SetUp() override {
        MvncTestCommon::SetUp();
        available_devices = getAmountOfDevices();
        ASSERT_TRUE(available_devices > 0);
    }
};

TEST_F(MvncOpenDevice, OpenUSBStickThenPCIeAndClose_USB_PCIE) {
    ncDeviceHandle_t *deviceHandle_MX = nullptr;
    ncDeviceHandle_t *deviceHandle_PCIe = nullptr;
    std::string actDeviceName;

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle_MX, ANY_PLATFORM, NC_USB, watchdogInterval, _firmwarePath));

    actDeviceName = deviceHandle_MX->private_data->dev_addr;
    ASSERT_TRUE(actDeviceName.size());
    ASSERT_TRUE(!isMyriadPCIeDevice(actDeviceName));

    // Second open
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle_PCIe, ANY_PLATFORM, NC_PCIE, watchdogInterval, _firmwarePath));

    actDeviceName = deviceHandle_PCIe->private_data->dev_addr;
    ASSERT_TRUE(isMyriadPCIeDevice(actDeviceName));

    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle_PCIe));
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle_MX));
}

/**
* @brief Open any device and close it
* @warn  Depend on WithCustomFirmware test
*/
TEST_F(MvncOpenDevice, OpenAndClose) {
    ncDeviceHandle_t *deviceHandle = nullptr;

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, ANY_PLATFORM, NC_USB, watchdogInterval, _firmwarePath));
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));
}

/**
* @brief Try to open device twice. DeviceHandle shouldn't be overwritten
*/
TEST_F(MvncOpenDevice, OpenTwiceSameHandler) {
    ncDeviceHandle_t *deviceHandle = nullptr;

    char dev_addr_first_open[MAX_DEV_NAME];
    unsigned int data_lenght_first = MAX_DEV_NAME;

    char dev_addr_second_open[MAX_DEV_NAME];
    unsigned int data_lenght_second = MAX_DEV_NAME;

    // First open, get device name
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, ANY_PLATFORM, NC_USB, watchdogInterval, _firmwarePath));
    ASSERT_NO_ERROR(ncDeviceGetOption(deviceHandle, NC_RO_DEVICE_NAME,
                                      dev_addr_first_open, &data_lenght_first));

    // Second open, get device name
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, ANY_PLATFORM, NC_USB, watchdogInterval, _firmwarePath));
    ASSERT_NO_ERROR(ncDeviceGetOption(deviceHandle, NC_RO_DEVICE_NAME,
                                      dev_addr_second_open, &data_lenght_second));

    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));
    // Should be the same device
    ASSERT_STREQ(dev_addr_first_open, dev_addr_second_open);
}

//  *************************************************** //
//              GRAPH ALLOCATION TESTS                  //
/**
 * @brief Test transfer data from host to device
 * @detail Allocate 2 devices and test some graph allocate cases
 * @warning For correct testing should be used blob with size more than 30mb
 */
class MvncGraphAllocations: public MvncOpenDevice {
public:
    // Devices
    ncDeviceHandle_t * _deviceHandle[MAX_DEVICES] = {nullptr};
    int _availableDevices = 0;

    // Graphs
    ncGraphHandle_t*  _graphHandle[MAX_DEVICES] = {nullptr};
    int _bootedDevices = 0;

    // Blob
    const std::string blobPath = "bvlc_googlenet_fp16.blob";
    std::vector<char> _blob;

protected:
    void SetUp() override {
        MvncOpenDevice::SetUp();
        // Load blob
        ASSERT_NO_ERROR(readBINFile(blobPath, _blob));
    }

    void TearDown() override {
        for (int index = 0; index < _bootedDevices; ++index) {
            ASSERT_NO_ERROR(ncDeviceClose(&_deviceHandle[index]));
        }
        _bootedDevices = 0;
    }

    void bootDevices(const int devicesToAllocate) {
        _availableDevices = getAmountOfDevices();
        if (_availableDevices < devicesToAllocate) {
            GTEST_SKIP_("Not enough devices");
        }

        for (int index = 0; index < devicesToAllocate; ++index) {
            ASSERT_NO_ERROR(ncDeviceOpen(&_deviceHandle[index], ANY_PLATFORM, NC_USB,
                                         watchdogInterval, _firmwarePath));
            ASSERT_TRUE(_deviceHandle[index] != nullptr);
            ++_bootedDevices;
        }
        ASSERT_EQ(_bootedDevices, devicesToAllocate);
    }

    ~MvncGraphAllocations() override = default;
};

/**
 * @brief Allocate graph for one device
 */
TEST_F(MvncGraphAllocations, DISABLED_OneGraph) {
    bootDevices(1);

    // Create graph handlers
    std::string graphName = "graph";
    ASSERT_NO_ERROR(ncGraphCreate(graphName.c_str(), &_graphHandle[0]));
    ASSERT_TRUE(_graphHandle[0] != nullptr);

    // Allocate graph
    ASSERT_NO_ERROR(ncGraphAllocate(_deviceHandle[0], _graphHandle[0],
                                    _blob.data(), _blob.size(),     // Blob
                                    _blob.data(), sizeof(ElfN_Ehdr) + sizeof(blob_header_v2) ));   // Header
}

/**
 * @brief Allocate graphs for 2 device (serial)
 */
TEST_F(MvncGraphAllocations, DISABLED_AllocateGraphsOn2DevicesSerial) {
    bootDevices(2);

    // Create graphs handlers
    for (int index = 0; index < _bootedDevices; ++index) {
        std::string graphName = "graph";
        graphName += std::to_string(index);
        ASSERT_NO_ERROR(ncGraphCreate(graphName.c_str(), &_graphHandle[index]));
        ASSERT_TRUE(_graphHandle[index] != nullptr);
    }

    // Allocate graphs in serial mode
    ncStatus_t rc[MAX_DEVICES];

    for (int i = 0; i < _bootedDevices; ++i) {
        rc[i] = ncGraphAllocate(_deviceHandle[0], _graphHandle[0],
                                _blob.data(), _blob.size(),     // Blob
                                _blob.data(), sizeof(ElfN_Ehdr) + sizeof(blob_header_v2) );  // Header
    }

    for (int i = 0; i < _bootedDevices; ++i) {
        ASSERT_NO_ERROR(rc[i]);
    }
}

/**
* @brief Allocate graphs for 2 device (parallel)
* @detail Open devices and then in parallel threads try to load graphs to it
*         The error easy appear, if USBLINK_TRANSFER_SIZE is (1024 * 1024 * 20)
* @warning It's depend on USBLINK_TRANSFER_SIZE constant from UsbLinkPlatform.c file
* @warning Need blob to use this tests
*/
TEST_F(MvncGraphAllocations, DISABLED_AllocateGraphsOn2DevicesParallel) {
    bootDevices(2);

    // Create graphs handlers
    for (int index = 0; index < _bootedDevices; ++index) {
        std::string graphName = "graph";
        graphName += std::to_string(index);
        ASSERT_NO_ERROR(ncGraphCreate(graphName.c_str(), &_graphHandle[index]));
        ASSERT_TRUE(_graphHandle[index] != nullptr);
    }

    // Allocate graphs in parallel threads
    std::thread requests[MAX_DEVICES];
    ncStatus_t rc[MAX_DEVICES];
    for (int i = 0; i < _bootedDevices; ++i) {
        requests[i] = std::thread([i, &rc, this]() {
            rc[i] = ncGraphAllocate(_deviceHandle[0], _graphHandle[0],
                                    _blob.data(), _blob.size(),     // Blob
                                    _blob.data(), sizeof(ElfN_Ehdr) + sizeof(blob_header_v2) );
        });
    }

    for (int i = 0; i < _bootedDevices; ++i) {
        requests[i].join();
        ASSERT_NO_ERROR(rc[i]);
    }
}

//  ***********************************************  //
//               Close device tests                  //

class MvncCloseDevice : public MvncTestCommon {
protected:
    ~MvncCloseDevice() override = default;
};

/**
* @brief Correct closing if handle is empty
*/
TEST_F(MvncCloseDevice, EmptyDeviceHandler) {
    ncDeviceHandle_t *deviceHandle = nullptr;
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));
}

/**
* @brief Correct closing if some handler fields is null
*/
TEST_F(MvncCloseDevice, EmptyFieldsOfDeviceHandle) {

    ncDeviceHandle_t *deviceHandlePtr;
    ncDeviceHandle_t *dH = (ncDeviceHandle_t*)calloc(1, sizeof(*dH));
    _devicePrivate_t *d = (_devicePrivate_t*)calloc(1, sizeof(*d));

    if (dH && d) {
        dH->private_data = d;
        d->dev_addr = nullptr;
        d->dev_addr_booted = nullptr;
        d->device_mon_stream_id = INVALID_LINK_ID;
        d->graph_monitor_stream_id = INVALID_LINK_ID;
        d->wd_interval = watchdogInterval;
        deviceHandlePtr = dH;
    }

    ASSERT_EQ(ncDeviceClose(&deviceHandlePtr), NC_INVALID_PARAMETERS);
}

//  *************************************************** //
//              TESTS WITH INFERENCE                    //

using MvncInference = MvncGraphAllocations;

TEST_F(MvncInference, DoOneIterationOfInference) {
    bootDevices(1);

    std::string graphName = "graph";
    ASSERT_NO_ERROR(ncGraphCreate(graphName.c_str(), &_graphHandle[0]));
    ASSERT_TRUE(&_graphHandle[0] != nullptr);

    ASSERT_NO_ERROR(ncGraphAllocate(_deviceHandle[0], _graphHandle[0],
                                    _blob.data(), _blob.size(),     // Blob
                                    _blob.data(), sizeof(ElfN_Ehdr) + sizeof(blob_header_v2) ));


    unsigned int dataLength = sizeof(int);

    int numInputs = 0;
    ASSERT_NO_ERROR(ncGraphGetOption(_graphHandle[0], NC_RO_GRAPH_INPUT_COUNT, &numInputs, &dataLength));

    int numOutputs = 0;
    ASSERT_NO_ERROR(ncGraphGetOption(_graphHandle[0], NC_RO_GRAPH_OUTPUT_COUNT, &numOutputs, &dataLength));

    dataLength = sizeof(ncTensorDescriptor_t);

    ncTensorDescriptor_t inputDesc = {};
    ASSERT_NO_ERROR(ncGraphGetOption(_graphHandle[0], NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS, &inputDesc,
                                     &dataLength));


    ncTensorDescriptor_t outputDesc = {};
    ASSERT_NO_ERROR(ncGraphGetOption(_graphHandle[0], NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS, &outputDesc,
                                     &dataLength));

    unsigned int fifo_elements = 4;

    ncFifoHandle_t *inputFifoHandle = nullptr;
    ASSERT_NO_ERROR(ncFifoCreate("input", NC_FIFO_HOST_WO, &inputFifoHandle));

    ASSERT_NO_ERROR(ncFifoAllocate(inputFifoHandle, _deviceHandle[0], &inputDesc, fifo_elements));

    ncFifoHandle_t *outputFifoHandle = nullptr;
    ASSERT_NO_ERROR(ncFifoCreate("output", NC_FIFO_HOST_RO, &outputFifoHandle));

    ASSERT_NO_ERROR(ncFifoAllocate(outputFifoHandle, _deviceHandle[0], &outputDesc, fifo_elements));

    uint8_t *input_data = new uint8_t[inputDesc.totalSize];
    uint8_t *result_data = new uint8_t[outputDesc.totalSize];
    ASSERT_NO_ERROR(ncGraphQueueInferenceWithFifoElem(_graphHandle[0],
                                                      inputFifoHandle, outputFifoHandle,
                                                      input_data, &inputDesc.totalSize, nullptr));

    void *userParam = nullptr;
    ASSERT_NO_ERROR(ncFifoReadElem(outputFifoHandle, result_data, &outputDesc.totalSize, &userParam));

    delete[] input_data;
    delete[] result_data;
    ASSERT_NO_ERROR(ncFifoDestroy(&inputFifoHandle));
    ASSERT_NO_ERROR(ncFifoDestroy(&outputFifoHandle));

    ASSERT_NO_ERROR(ncGraphDestroy(&_graphHandle[0]));

    ASSERT_NO_ERROR(ncDeviceClose(&_deviceHandle[0]));
}
