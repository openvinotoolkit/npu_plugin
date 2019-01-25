#include "include/mcm/target/keembay/runtime_model/runtime_model.hpp"
#include "contrib/flatbuffers/include/flatbuffers/util.h"
#include "include/mcm/base/exception/argument_error.hpp"
#include <fstream>

mv::RuntimeModel::RuntimeModel()
    :graphFile_(MVCNN::GraphFileT())
{

}

mv::RuntimeModel::~RuntimeModel()
{

}

const std::unordered_map<mv::DTypeType, MVCNN::DType> mv::RuntimeModel::dTypeMapping_ =
{
    {mv::DTypeType::Float64, MVCNN::DType::DType_FP64},
    {mv::DTypeType::Float32, MVCNN::DType::DType_FP32},
    {mv::DTypeType::Float16, MVCNN::DType::DType_FP16},
    {mv::DTypeType::Float8, MVCNN::DType::DType_FP8},
    {mv::DTypeType::UInt64, MVCNN::DType::DType_U64},
    {mv::DTypeType::UInt32, MVCNN::DType::DType_U32},
    {mv::DTypeType::UInt16, MVCNN::DType::DType_U16},
    {mv::DTypeType::UInt8, MVCNN::DType::DType_U8},
    {mv::DTypeType::Int64, MVCNN::DType::DType_I64},
    {mv::DTypeType::Int32, MVCNN::DType::DType_I32},
    {mv::DTypeType::Int16, MVCNN::DType::DType_I16},
    {mv::DTypeType::Int8, MVCNN::DType::DType_I8},
    {mv::DTypeType::Int4, MVCNN::DType::DType_I4},
    {mv::DTypeType::Int2, MVCNN::DType::DType_I2},
    {mv::DTypeType::Int2X, MVCNN::DType::DType_I2X},
    {mv::DTypeType::Int4X, MVCNN::DType::DType_I4X},
    {mv::DTypeType::Bin, MVCNN::DType::DType_BIN},
    {mv::DTypeType::Log, MVCNN::DType::DType_LOG}
};

const std::unordered_map<std::string, MVCNN::MemoryLocation> mv::RuntimeModel::memoryLocationMapping_ =
{
    {"ProgrammableInput", MVCNN::MemoryLocation::MemoryLocation_ProgrammableInput},
    {"ProgrammableOutput", MVCNN::MemoryLocation::MemoryLocation_ProgrammableOutput},
    {"VPU_DDR_Heap", MVCNN::MemoryLocation::MemoryLocation_VPU_DDR_Heap},
    {"GraphFile", MVCNN::MemoryLocation::MemoryLocation_GraphFile},
    {"VPU_CMX_NN", MVCNN::MemoryLocation::MemoryLocation_VPU_CMX_NN},
    {"VPU_CMX_UPA", MVCNN::MemoryLocation::MemoryLocation_VPU_CMX_UPA},
    {"VPU_DDR_BSS", MVCNN::MemoryLocation::MemoryLocation_VPU_DDR_BSS}
};


MVCNN::DType mv::RuntimeModel::convertDtype(const mv::DType& dtype)
{
    return dTypeMapping_.at(static_cast<mv::DTypeType>(dtype));
}

MVCNN::MemoryLocation mv::RuntimeModel::convertAllocatorToMemoryLocale(const std::string& allocatorName)
{
    return memoryLocationMapping_.at(allocatorName);
}

MVCNN::LinkT mv::RuntimeModel::convertOperationToLink(mv::ComputationModel& cm, mv::Data::OpListIterator op)
{
    MVCNN::LinkT toReturn;
    toReturn.name = op->getName();
    //toReturn.thisID = op->getId();

    //cm.getDataFlow()
    return toReturn;
}

MVCNN::TensorReferenceT mv::RuntimeModel::convertTensorRepresentation(MemoryAllocator &allocator, mv::Data::TensorIterator t)
{
    MVCNN::TensorReferenceT toReturn;

    mv::Data::BufferIterator it = allocator.getBuffer(0, t); //0 is the only stage for now, but this will probably change in the future

    toReturn.dimensions = it->getData()->getShape(); // Padded or not?
    toReturn.strides = it->getData()->computeStrides(); //Maybe directly it->computeStrides() in the future std::vector<uint32_t> strides;

    auto strides = it->getStrides();
    toReturn.leading_offset = strides[0];//    uint32_t leading_offset;
    toReturn.trailing_offset = strides[strides.size()-1] + it->getPostAlign();//    uint32_t trailing_offset;

    toReturn.data->data_index = it->getOffset();//    std::unique_ptr<IndirectDataReferenceT> data;
    toReturn.locale = convertAllocatorToMemoryLocale(allocator.getAllocatorName());//    MemoryLocation locale;
    toReturn.data_dtype = convertDtype(it->getData()->getDType());//    DType data_dtype;

    //UNSUPPORTED FOR NOW
    //toReturn.quant_scale;//    std::vector<int8_t> quant_scale;
    //toReturn.quant_zero;//    std::vector<int8_t> quant_zero;
    //toReturn.quant_shift;//    std::vector<int8_t> quant_shift;

    return toReturn;
}



char * mv::RuntimeModel::serialize(int& bufferSize)
{
    flatbuffers::FlatBufferBuilder fbb;
    auto offset = MVCNN::CreateGraphFile(fbb, &graphFile_);
    MVCNN::FinishGraphFileBuffer(fbb, offset);
    bufferSize = fbb.GetSize();
    char * buffer = new char[bufferSize];
    std::memcpy(buffer, (char*)fbb.GetBufferPointer(), bufferSize);
    return buffer;
}

void mv::RuntimeModel::serialize(const std::string& path)
{
    int bufferSize;
    char * dataBuffer = serialize(bufferSize);
    flatbuffers::SaveFile(path.c_str(), dataBuffer, bufferSize, true);
    delete [] dataBuffer;
}

void mv::RuntimeModel::deserialize(const std::string& path)
{
    std::ifstream ifs(path.c_str(), std::ifstream::binary|std::ifstream::in);
    ifs.seekg(0, std::ios::end);
    int length = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    char * dataBuffer = new char[length];
    ifs.read(dataBuffer, length);
    ifs.close();
    deserialize(dataBuffer, length);
    delete [] dataBuffer;
}

void mv::RuntimeModel::deserialize(char * dataBuffer, int length)
{
    flatbuffers::Verifier verifier(reinterpret_cast<const unsigned char*>(dataBuffer), length);
    if (!MVCNN::VerifyGraphFileBuffer(verifier))
        throw ArgumentError("tools:GraphComparator", "file:content", "invalid", "GraphFile verification failed");
    Logger::log(mv::Logger::MessageType::Info, "RuntimeModel", "GraphFile verification successful");
    const MVCNN::GraphFile *graphPtr = MVCNN::GetGraphFile(dataBuffer);
    graphPtr->UnPackTo(&graphFile_);
}
