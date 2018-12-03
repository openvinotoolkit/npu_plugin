#include "include/mcm/deployer/executor/executor.hpp"

namespace mv
{
    Executor::Executor(mv::Configuration& config) : configuration_(config)
    {
        log(mv::Logger::MessageType::Info, "Initialize Executor");
        openDevice();
        loadGraph();
        allocateFifos();
    }

    bool Executor::checkTargetMatches(mv::Target target, ncDeviceHwVersion_t hwVersion)
    {
        switch(target)
        {
            case mv::Target::ma2480:
                if (hwVersion == NC_MA2480)
                    return true;
                break;
            case mv::Target::Unknown:
                return true;
        }
        return false;
    }

    void Executor::openDevice()
    {
        int loglevel = 2; //TODO make it configuratable?
        int idx = 0;
        log(mv::Logger::MessageType::Info, "openning device");

        ncStatus_t retCode = ncGlobalSetOption(NC_RW_LOG_LEVEL, &loglevel, sizeof(loglevel));
        while (retCode == NC_OK)
        {
            retCode = ncDeviceCreate(idx, &deviceHandle_);
            if (retCode == NC_DEVICE_NOT_FOUND)
                throw RuntimeError(*this, "Requested device is not found! please connect your device!");
            if (retCode != NC_OK)
                throw RuntimeError(*this, "ncDeviceCreate failed");

            ncDeviceHwVersion_t hwVersion;
            unsigned int size = sizeof(hwVersion);
            retCode = ncDeviceGetOption(deviceHandle_, NC_RO_DEVICE_HW_VERSION, &hwVersion, &size);
            if(retCode != NC_OK)
                throw RuntimeError(*this, "ncDeviceGetOption failed on NC_RO_DEVICE_HW_VERSION");

            if (checkTargetMatches(configuration_.getTarget(), hwVersion))
                break;
            idx++;
        }

        retCode = ncDeviceOpen(deviceHandle_);
        if(retCode != NC_OK)
            throw RuntimeError(*this, "ncDeviceOpen failed");
        log(mv::Logger::MessageType::Info, "Device Opened successfully!");
    }

    void Executor::getInputData(unsigned int imageSize, char* imageData)
    {

        if (configuration_.getInputMode() == mv::InputMode::FILE)
        {
            std::ifstream inputFile (configuration_.getInputFilePath(), std::ios::in | std::ios::binary);
            if (inputFile.fail())
                throw ArgumentError(*this, "Input File",
                    configuration_.getInputFilePath(), " Input file not found!");
            if (!inputFile.read (imageData, imageSize))
                throw RuntimeError(*this, "input file doesn't have enough data!");
        }
        else
        {
            if (configuration_.getInputMode() == mv::InputMode::ALL_ONE)
            {
                std::vector<unsigned short> myvector(imageSize/2);
                std::fill_n(myvector.begin(), myvector.size(), 1);
                memcpy(imageData, &myvector[0], imageSize);
            }
            else
            {
                memset(imageData, 0, imageSize);
            }
        }
    }

    void Executor::loadGraph()
    {
        log(mv::Logger::MessageType::Info, "loading graph");

        ncStatus_t retCode = ncGraphCreate("graph", &graphHandle_);
        if(retCode != NC_OK)
            throw RuntimeError(*this, "ncGraphCreate failed");

        void* graphFileBuf = configuration_.getRuntimePointer()->getDataPointer();
        unsigned int graphFileLen = configuration_.getRuntimePointer()->getBufferSize();

        retCode = ncGraphAllocate(deviceHandle_, graphHandle_, graphFileBuf, graphFileLen);
        if(retCode != NC_OK)
        {
            std::string errorReason = "";
            if (retCode == NC_INVALID_HANDLE || retCode == NC_INVALID_PARAMETERS)
                errorReason = "invalide arguments";
            else if (retCode == NC_OUT_OF_MEMORY)
                errorReason = "not enough memory";
            else if (retCode == NC_ERROR)
                errorReason = "XLink Error";
            else if (retCode == NC_MYRIAD_ERROR)
                errorReason = "myriad error";
            else if (retCode == NC_UNSUPPORTED_GRAPH_FILE)
                errorReason = "graph version incompatible";
            throw RuntimeError(*this, "ncGraphCreate failed: " + errorReason);
        }
        log(mv::Logger::MessageType::Info, "Graph Loading done successfully!");
    }

    void Executor::allocateFifos()
    {
        log(mv::Logger::MessageType::Info, "Allocating Fifos");

        // Get number of inputs/outputs
        unsigned int size = sizeof(int);
        ncStatus_t retCode = ncGraphGetOption(graphHandle_, NC_RO_GRAPH_INPUT_COUNT, &numInputs_,  &size);
        if(retCode != NC_OK)
            throw RuntimeError(*this, "ncGraphGetOption on NC_RO_GRAPH_INPUT_COUNT failed");
        retCode = ncGraphGetOption(graphHandle_, NC_RO_GRAPH_OUTPUT_COUNT, &numOutputs_,  &size);
        if(retCode != NC_OK)
            throw RuntimeError(*this, "ncGraphGetOption on NC_RO_GRAPH_INPUT_COUNT failed");

        // Read tensor descriptors
        size = sizeof(struct ncTensorDescriptor_t) * numInputs_;
        inputTensorDesc_ = new ncTensorDescriptor_t[numInputs_];
        retCode = ncGraphGetOption(graphHandle_, NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS, &inputTensorDesc_[0],  &size);
        if(retCode != NC_OK)
            throw RuntimeError(*this, "ncGraphGetOption on NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS failed");

        size = sizeof(struct ncTensorDescriptor_t) * numOutputs_;
        outputTensorDesc_ = new ncTensorDescriptor_t[numOutputs_];
        retCode = ncGraphGetOption(graphHandle_, NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS, &outputTensorDesc_[0],  &size);
        if(retCode != NC_OK)
            throw RuntimeError(*this, "ncGraphGetOption on NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS failed");

        //Allocate Fifos
        ncFifoDataType_t dataType = NC_FIFO_FP16;
        buffersIn_ = new ncFifoHandle_t*[numInputs_];
        for (int i = 0; i < numInputs_; i++)
        {
            std::string fifoname = "FifoIn" + std::to_string(i);
            retCode = ncFifoCreate(fifoname.c_str(), NC_FIFO_HOST_WO, &buffersIn_[i]);
            if(retCode != NC_OK)
                throw RuntimeError(*this, "ncFifoCreate failed");
            retCode = ncFifoSetOption(buffersIn_[i], NC_RW_FIFO_DATA_TYPE, &dataType,
                         sizeof(dataType));
            if(retCode != NC_OK)
                throw RuntimeError(*this, "ncFifoSetOption of NC_RW_FIFO_DATA_TYPE failed");
            retCode = ncFifoAllocate(buffersIn_[i], deviceHandle_, &inputTensorDesc_[i], 2);
            if(retCode != NC_OK)
                throw RuntimeError(*this, "ncFifoAllocate failed");
        }
        buffersOut_ = new ncFifoHandle_t*[numOutputs_];
        for (int i = 0; i < numOutputs_; i++)
        {
            std::string fifoname = "FifoOut" + std::to_string(i);
            retCode = ncFifoCreate(fifoname.c_str(), NC_FIFO_HOST_RO, &buffersOut_[i]);
            if(retCode != NC_OK)
                throw RuntimeError(*this, "ncFifoCreate failed");
            retCode = ncFifoSetOption(buffersOut_[i], NC_RW_FIFO_DATA_TYPE, &dataType,
                         sizeof(dataType));
            if(retCode != NC_OK)
                throw RuntimeError(*this, "ncFifoSetOption of NC_RW_FIFO_DATA_TYPE failed");
            retCode = ncFifoAllocate(buffersOut_[i], deviceHandle_, &outputTensorDesc_[i], 2);
            if(retCode != NC_OK)
                throw RuntimeError(*this, "ncFifoAllocate failed");
        }

        log(mv::Logger::MessageType::Info, "Fifos Allocated Successfully!");
    }

    void Executor::destroyAll()
    {
        log(mv::Logger::MessageType::Info, "Starting to destroy all!");
        ncStatus_t retCode;
        try {
            for (int i = 0; i < numInputs_; i++)
            {
                retCode = ncFifoDestroy(&buffersIn_[i]);
                if(retCode != NC_OK)
                    throw RuntimeError(*this, "Input fifo ncFifoDestroy failed");
            }
            for (int i = 0; i < numOutputs_; i++)
            {
                retCode = ncFifoDestroy(&buffersOut_[i]);
                if(retCode != NC_OK)
                    throw RuntimeError(*this, "output Fifo ncFifoDestroy failed");
            }
            retCode = ncGraphDestroy(&graphHandle_);
            if(retCode != NC_OK)
                throw RuntimeError(*this, "ncGraphDestroy failed");

            delete buffersIn_;
            delete buffersOut_;
            retCode = ncDeviceClose(deviceHandle_);
            if(retCode != NC_OK)
                throw RuntimeError(*this, "ncDeviceClose failed");
        } catch (RuntimeError e) {
            log(mv::Logger::MessageType::Error, e.what());
        }
        //delete allocated memory
        delete inputTensorDesc_;
        delete outputTensorDesc_;
    }

    Executor::~Executor()
    {
        destroyAll();
    }
    //mv::Tensor Executor::execute() {
    //    mv::Tensor res = new mv::Tensor();
    //    return res;

    char* Executor::execute()
    {
        // Assume one input for now
        unsigned int imageSize = inputTensorDesc_[0].totalSize;
        char* imageData = new char[imageSize];
        getInputData(imageSize, imageData);

        // Write tensor to input fifo
        ncStatus_t retCode = ncFifoWriteElem(buffersIn_[0], imageData, &imageSize, 0);
        if(retCode != NC_OK)
            throw RuntimeError(*this, "ncFifoWriteElem failed");
        log(mv::Logger::MessageType::Info, "Load input fifo successfully!");

        // queue inference
        retCode = ncGraphQueueInference(graphHandle_, &buffersIn_[0], 1, &buffersOut_[0], 1);
        if(retCode != NC_OK)
            throw RuntimeError(*this, "ncGraphQueueInference failed");

        // Read output results
        unsigned int elementDataSize;
        unsigned int length = sizeof(elementDataSize);
        retCode = ncFifoGetOption(buffersOut_[0], NC_RO_FIFO_ELEMENT_DATA_SIZE, &elementDataSize, &length);
        if(retCode != NC_OK)
            throw RuntimeError(*this, "ncFifoGetOption failed");
        char *result = new char[elementDataSize];

        void *userParam;
        unsigned int outputDataSize;
        retCode = ncFifoReadElem(buffersOut_[0], result, &outputDataSize, &userParam);
        //TODO Convert to mvTensor
        //TODO need to delete result
        return result;
    }

    std::string Executor::getLogID() const
    {
        return "Executor";
    }

}
