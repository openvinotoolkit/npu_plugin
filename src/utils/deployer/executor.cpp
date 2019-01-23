#include "include/mcm/utils/deployer/executor.hpp"
#include "include/mcm/utils/deployer/deployer_utils.hpp"

mv::exe::Executor::Executor(Target target,
    Protocol protocol):
    target_(target),
    protocol_(protocol)
    {

    }

bool mv::exe::Executor::checkTargetMatches(ncDeviceHwVersion_t hwVersion)
{

    switch(target_)
    {
        case Target::ma2480:
            if (hwVersion == NC_MA2480)
                return true;
            break;

        case Target::Unknown:
            return true;
    }

    return false;

}

void mv::exe::Executor::openDevice()
{

    int loglevel = 2; //TODO make it configuratable?
    int idx = 0;
    log(Logger::MessageType::Info, "openning device");

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

        if (checkTargetMatches(hwVersion))
            break;
        idx++;
    }

    retCode = ncDeviceOpen(deviceHandle_);
    if(retCode != NC_OK)
        throw RuntimeError(*this, "ncDeviceOpen failed");
    log(Logger::MessageType::Info, "Device Opened successfully!");

}

void mv::exe::Executor::loadGraph(void* graphFileBuf, int graphLen)
{

    log(Logger::MessageType::Info, "loading graph");

    ncStatus_t retCode = ncGraphCreate("graph", &graphHandle_);
    if (retCode != NC_OK)
        throw RuntimeError(*this, "ncGraphCreate failed");

    retCode = ncGraphAllocate(deviceHandle_, graphHandle_, graphFileBuf, graphLen);

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

    log(Logger::MessageType::Info, "Graph Loading done successfully!");

}

void mv::exe::Executor::allocateFifos()
{

    log(Logger::MessageType::Info, "Allocating Fifos");

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

    log(Logger::MessageType::Info, "Fifos Allocated Successfully!");

}

void mv::exe::Executor::destroyAll()
{

    log(Logger::MessageType::Info, "Starting to destroy all!");
    ncStatus_t retCode;
    try
    {

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
    }
    catch (RuntimeError e)
    {
        log(Logger::MessageType::Error, e.what());
    }

    //delete allocated memory
    delete inputTensorDesc_;
    delete outputTensorDesc_;

}

mv::Tensor mv::exe::Executor::execute(std::shared_ptr<mv::RuntimeBinary> binaryPointer,
    Tensor& inputTensor)
{
    void* graphFileBuf = binaryPointer->getDataPointer();;
    unsigned int graphFileLen = binaryPointer->getBufferSize();;

    return execute_(graphFileBuf, graphFileLen, inputTensor);
}

mv::Tensor mv::exe::Executor::execute(const std::string& graphFilePath,
    Tensor& inputTensor)
{
    utils::checkFileExists(getLogID(), "graph file ", graphFilePath);

    std::ifstream inputFile (graphFilePath, std::ios::in | std::ios::binary);

    inputFile.seekg (0, inputFile.end);
    unsigned int graphFileLen = inputFile.tellg();
    inputFile.seekg (0, inputFile.beg);
    char* graphFileBuf = new char[graphFileLen];
    if (!inputFile.read (graphFileBuf, graphFileLen))
        throw RuntimeError(*this, "Error reading graph file");

    Tensor res = execute_(graphFileBuf, graphFileLen, inputTensor);
    delete graphFileBuf;
    return res;
}

mv::Tensor mv::exe::Executor::execute_(void* graphFileBuf, int graphLen, Tensor& inputTensor)
{
    log(Logger::MessageType::Info, "Initialize Executor");
    openDevice();
    loadGraph(graphFileBuf, graphLen);
    allocateFifos();

    // Assume one input for now
    //Check size and order of the input tensor
    unsigned int imageSize = inputTensorDesc_[0].totalSize;
    if (imageSize/2 != inputTensor.getShape().totalSize())
        throw RuntimeError(*this, "size of input tensor doesn't match expected size by blob");

    Order graphInputOrder = utils::getTensorOrder(inputTensorDesc_[0]);
    if (graphInputOrder != inputTensor.getOrder())
        throw RuntimeError(*this, "Order of input tensor doesn't match expected order by blob");
    // Write tensor to input fifo
    std::vector<double> temp = inputTensor.getData();
    unsigned short* imageData = new unsigned short[imageSize/2];
    std::copy(std::begin(temp), std::end(temp), imageData);

    ncStatus_t retCode = ncFifoWriteElem(buffersIn_[0],imageData, &imageSize, 0);
    if(retCode != NC_OK)
        throw RuntimeError(*this, "ncFifoWriteElem failed");
    log(Logger::MessageType::Info, "Load input fifo successfully!");

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
    retCode = ncFifoReadElem(buffersOut_[0], result, &elementDataSize, &userParam);
    if(retCode != NC_OK)
        throw RuntimeError(*this, "ncFifoReadElem failed");
    log(Logger::MessageType::Info, "read result from fifo successfully!");

    //Convert to mvTensor
    unsigned int numberOfElements = elementDataSize / 2; //fp16 == 2 bytes
    std::vector<double> tensorData(numberOfElements);
    unsigned short* resultUS = (unsigned short*) result;
    for (unsigned int i = 0; i < numberOfElements; i++)
        tensorData[i] = (double) resultUS[i];

    Shape shape({outputTensorDesc_[0].w, outputTensorDesc_[0].h, outputTensorDesc_[0].c, outputTensorDesc_[0].n});
    Order order = mv::exe::utils::getTensorOrder(outputTensorDesc_[0]);

    Tensor resultTensor("result", shape, DType("Float16"), order);
    resultTensor.populate(tensorData);

    delete result;
    delete imageData;
    destroyAll();
    return resultTensor;

}

void mv::exe::Executor::setTarget(Target target)
{
    target_ = target;
}

void mv::exe::Executor::setProtocol(Protocol protocol)
{
    if (protocol == Protocol::Unknown)
        throw ArgumentError(*this, "protocol", "unknown", "Defining protocol as unknown is illegal");
    protocol_ = protocol;
}

mv::Target mv::exe::Executor::getTarget() const
{
    return target_;
}

mv::exe::Protocol mv::exe::Executor::getProtocol() const
{
    return protocol_;
}

std::string mv::exe::Executor::getLogID() const
{
    return "Executor";
}
