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

#include <map>
#include <string>
#include <memory>

#include <cstdarg>
#include <cstring>
#include <asm/ioctl.h>

#include <xlink_uapi.h>

#include <VpualMessage.h>
#include <GraphManagerPlg.h>
#include <NN_Types.h>
#include <NNFlicPlg.h>

#include <schema/graphfile/graphfile_generated.h>

#include <vpu/utils/logger.hpp>
#include <zconf.h>

#include "vpual-model.h"

using namespace vpu;

// a copy-paste from thirdparty/movidius/vpualHost/NN/NNFlicPlg/NNFlicPlg.cpp
// should be aligned with the same structure
enum NNFlicMethod : uint32_t {
    UNKNOWN = 0,
    CREATE = 1,
    // DELETE = 2,
    // STOP = 3,
    SET_NUMBER_OF_THREADS = 4,
    SET_NUMBER_OF_SHAVES = 5,
    GET_LATEST_STATE = 6,
    // SET_CONFIG = 7,
    GET_NUMBER_OF_INPUTS = 8,
    GET_NUMBER_OF_OUTPUTS = 9,
    GET_MAX_NUMBER_OF_THREADS = 10,
    GET_MAX_NUMBER_OF_SHAVES = 11,
    GET_NUMBER_OF_STAGES = 12,
    GET_BLOB_VERSION = 13,
    GET_INPUT_TENSOR_DESCRIPTOR = 14,
    GET_OUTPUT_TENSOR_DESCRIPTOR = 15,
};

// a copy-paste from thirdparty/movidius/vpualHost/NN/GraphManagerPlg/GraphManagerPlg.cpp
enum GraphManagerMethod : uint32_t {
    NN_GRAPH_CHECK_AVAILABLE = 2,
    NN_GRAPH_ALLOCATE = 3,
    NN_GRAPH_ALLOCATE_EXISTING_BLOB = 4,
    NN_DEALLOCATE_GRAPH = 5,
};

std::string operationModeToStr(xlink_opmode operationMode) {
    switch (operationMode) {
        case RXB_TXB:
            return "RXB_TXB";
        case RXN_TXN:
            return "RXN_TXN";
        case RXB_TXN:
            return "RXB_TXN";
        case RXN_TXB:
            return "RXN_TXB";
    }
    return "Unknown operation mode";
}


class FakeDevice {
    flicTensorDescriptor_t descOut{}, descIn{};
    int msgType{-1};
    std::map<int, std::string> decoders{};
    uint32_t currentStubID{0};
    uint32_t currentState;
    std::shared_ptr<Logger> logger;

    public:
        explicit FakeDevice(LogLevel logLevel = LogLevel::None)
            : logger(std::make_shared<Logger>("FakeDevice", logLevel, consoleOutput())) {
        }

        void sendRequest(const void *data, uint32_t size) {
            if (size == sizeof(VpualCmdHeader_t)) {
                const VpualCmdHeader_t *header = reinterpret_cast<const VpualCmdHeader_t *>(data);
                msgType = header->type;
                currentStubID = header->stubID;
            }

            if (msgType == VpualCmdHeader_t::DECODER_CREATE) {
                std::string decoderName(reinterpret_cast<const char *>(data), size);
                decoders[decoders.size()] = decoderName;
            } else if (msgType == VpualCmdHeader_t::DECODER_DECODE) {
                const uint32_t method = *reinterpret_cast<const uint32_t *>(data);
                currentState = method;
                auto currentDecoder = decoders[currentStubID - 1];

                if (currentState == GraphManagerMethod::NN_GRAPH_ALLOCATE
                    && currentDecoder.find("GraphMPDecoder") != currentDecoder.npos) {
                    const BlobHandle_t *blobHandle = reinterpret_cast<const BlobHandle_t *>(reinterpret_cast<const char *>(data) + sizeof(uint32_t));
                    unsigned long graphFileLU = 0x700000000000;

                    auto getShiftBase2 = [](long num) -> size_t {
                        size_t shiftCount = 0;
                        while (num >>= 1) shiftCount++;
                        return shiftCount;
                    };
                    // use assumption that we have a pointer aligned on pagesize to restore virtual address
                    // Please refer to kmb_native_allocator.cpp
                    unsigned long graphBuffLU = blobHandle->graphBuff;
                    graphFileLU |= (graphBuffLU << getShiftBase2(getpagesize()));

                    const void *graphFile = reinterpret_cast<const void *>(graphFileLU);

                    auto file = MVCNN::GetGraphFile(graphFile);
                    auto header = file->header();

                    auto inputs = header->net_input();
                    if (inputs->size() != 1) throw std::runtime_error("The fake device support only one input");

                    auto outputs = header->net_output();
                    if (outputs->size() != 1) throw std::runtime_error("The fake device support only one output");

                    auto output = (*outputs)[0];
                    auto dimsOut = output->dimensions();

                    if (dimsOut->size() != 4) throw std::runtime_error("The fake device support only 4D tensors");
                    descOut.n = (*dimsOut)[0]; descOut.c = (*dimsOut)[1]; descOut.h = (*dimsOut)[2]; descOut.w = (*dimsOut)[3];
                    descOut.totalSize = descOut.n * descOut.c * descOut.h * descOut.w * sizeof(uint8_t);

                    auto input = (*inputs)[0];
                    auto dimsIn = input->dimensions();

                    if (dimsIn->size() != 4) throw std::runtime_error("The fake device support only 4D tensors");
                    descIn.n = (*dimsIn)[0]; descIn.c = (*dimsIn)[1]; descIn.h = (*dimsIn)[2]; descIn.w = (*dimsIn)[3];
                    descIn.totalSize = descIn.n * descIn.c * descIn.h * descIn.w * sizeof(uint8_t);
                }
            }
        }

        uint32_t getCurrentState() {
            return currentState;
        }

        void getResponse(void *data, uint32_t *size) {
            if (msgType == VpualCmdHeader_t::DECODER_CREATE) {
                *size = sizeof(uint32_t);
                uint32_t *stubID = reinterpret_cast<uint32_t *>(data);
                *stubID = decoders.size();
            } else if (msgType == VpualCmdHeader_t::DECODER_DECODE) {
                switch (currentState) {
                    case NNFlicMethod::GET_LATEST_STATE: {
                        *size = sizeof(NNPlgState);
                        NNPlgState *state = reinterpret_cast<NNPlgState *>(data);
                        *state = NNPlgState::SUCCESS;
                        logger->debug("Send response NNPlgState::SUCCESS on NNFlicMethod::GET_LATEST_STATE ");
                        break;
                    }
                    case NNFlicMethod::GET_OUTPUT_TENSOR_DESCRIPTOR: {
                        *size = sizeof(flicTensorDescriptor_t);
                        std::memset(data, 0, *size);
                        flicTensorDescriptor_t *desc = reinterpret_cast<flicTensorDescriptor_t *>(data);
                        *desc = descOut;
                        logger->debug("Send response on NNFlicMethod::GET_OUTPUT_TENSOR_DESCRIPTOR");
                        break;
                    }
                    case NNFlicMethod::GET_INPUT_TENSOR_DESCRIPTOR: {
                        *size = sizeof(flicTensorDescriptor_t);
                        std::memset(data, 0, *size);
                        flicTensorDescriptor_t *desc = reinterpret_cast<flicTensorDescriptor_t *>(data);
                        *desc = descIn;
                        logger->debug("Send response on NNFlicMethod::GET_INPUT_TENSOR_DESCRIPTOR");
                        break;
                    }
                    case GraphManagerMethod::NN_GRAPH_ALLOCATE: {
                        *size = sizeof(GraphStatus);
                        GraphStatus *status = reinterpret_cast<GraphStatus *>(data);
                        *status = GraphStatus::Success;
                        logger->debug("Send response GraphStatus::Success on GraphManagerMethod::NN_GRAPH_ALLOCATE");
                        break;
                    }
                    case GraphManagerMethod::NN_GRAPH_CHECK_AVAILABLE: {
                        *size = sizeof(GraphStatus);
                        GraphStatus *status = reinterpret_cast<GraphStatus *>(data);
                        *status = GraphStatus::No_GraphId_Found;
                        logger->debug("Send response GraphStatus::No_GraphId_Found on GraphManagerMethod::NN_GRAPH_CHECK_AVAILABLE");
                        break;
                    }
                    default:
                        logger->warning("Unsupported operation");
                }
            }
        }
};

#ifdef NDEBUG
FakeDevice fakeDevice(LogLevel::None);
std::shared_ptr<Logger> logger = std::make_shared<Logger>("Fake_ioctl", LogLevel::None, consoleOutput());
#else
FakeDevice fakeDevice(LogLevel::Debug);
std::shared_ptr<Logger> logger = std::make_shared<Logger>("Fake_ioctl", LogLevel::Debug, consoleOutput());
#endif

int ioctl(int __fd, unsigned long int __request, ...) {
    va_list args;
    va_start(args, __request);

    switch (__request) {
        case XL_OPEN_CHANNEL: {
            logger->debug("Doing XL_OPEN_CHANNEL...");
            xlinkopenchannel *data = va_arg(args, xlinkopenchannel *);

            logger->debug("  Operation mode: %s", operationModeToStr(data->mode));
            logger->debug("  Data size: %u", data->data_size);
            data->return_code = 0;

            logger->debug("Done.");
            break;
        }
        case XL_READ_DATA: {
            logger->debug("Doing XL_READ_DATA...");
            xlinkreaddata *data = va_arg(args, xlinkreaddata *);

            logger->debug("  Size: %u", *data->size);
            data->return_code = 0;

            logger->debug("Done.");
            break;
        }
        case XL_WRITE_DATA: {
            logger->debug("Doing XL_WRITE_DATA...");
            xlinkwritedata *data = va_arg(args, xlinkwritedata *);

            logger->debug("  Size: %u", data->size);
            data->return_code = 0;

            logger->debug("Done.");
            break;
        }
        case XL_CLOSE_CHANNEL: {
            // FIXME: for some reason usage of logger here causes a segfault
            std::cout << "Doing XL_CLOSE_CHANNEL...\n";
            xlinkopenchannel *data = va_arg(args, xlinkopenchannel *);

            std::cout << "  Operation mode: " << operationModeToStr(data->mode) << '\n';
            std::cout << "  Data size: " << data->data_size << '\n';
            data->return_code = 0;

            std::cout << "Done." << '\n';
            break;
        }
        case XL_WRITE_VOLATILE: {
            logger->debug("Doing XL_WRITE_VOLATILE...");
            xlinkwritedata *data = va_arg(args, xlinkwritedata *);

            logger->debug("  Size: %u", data->size);

            fakeDevice.sendRequest(data->pmessage, data->size);
            fakeDevice.getCurrentState();
            data->return_code = 0;

            logger->debug("Done.");
            break;
        }
        case XL_READ_TO_BUFFER: {
            logger->debug("Doing XL_READ_TO_BUFFER...");
            xlinkreadtobuffer *data = va_arg(args, xlinkreadtobuffer *);

            logger->debug("  Size: %u", *data->size);
            fakeDevice.getResponse(data->pmessage, data->size);
            data->return_code = 0;
            logger->debug("Done.");
            break;
        }
        case XL_START_VPU: {
            logger->debug("Doing XL_START_VPU...");
            xlinkstartvpu *data = va_arg(args, xlinkstartvpu *);

            logger->debug("  Filename: %s", data->filename);
            logger->debug("  Namesize: %u", data->namesize);
            data->return_code = 0;

            logger->debug("Done.");
            break;
        }
        case XL_STOP_VPU: {
            // FIXME: for some reason usage of logger here causes a segfault
            std::cout << "Doing XL_STOP_VPU...\n";
            xlinkstopvpu *data = va_arg(args, xlinkstopvpu *);

            data->return_code = 0;

            std::cout << "Done." << '\n';
            break;
        }
        case XL_RESET_VPU: {
            logger->debug("Doing XL_RESET_VPU...");
            xlinkstopvpu *data = va_arg(args, xlinkstopvpu *);
            data->return_code = 0;
            logger->debug("Done.");
            break;
        }
        default:
            logger->error("Error: cannot recognize XLink operation!");
    }

    va_end(args);
    return 0;
}

