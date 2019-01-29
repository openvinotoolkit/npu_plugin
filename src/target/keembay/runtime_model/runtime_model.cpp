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

const std::unordered_map<std::string, MVCNN::DType> mv::RuntimeModel::dTypeMapping_ =
{
    {"Float64", MVCNN::DType::DType_FP64},
    {"Float32", MVCNN::DType::DType_FP32},
    {"Float16", MVCNN::DType::DType_FP16},
    {"Float8", MVCNN::DType::DType_FP8},
    {"UInt64", MVCNN::DType::DType_U64},
    {"UInt32", MVCNN::DType::DType_U32},
    {"UInt16", MVCNN::DType::DType_U16},
    {"UInt8", MVCNN::DType::DType_U8},
    {"Int64", MVCNN::DType::DType_I64},
    {"Int32", MVCNN::DType::DType_I32},
    {"Int16", MVCNN::DType::DType_I16},
    {"Int8", MVCNN::DType::DType_I8},
    {"Int4", MVCNN::DType::DType_I4},
    {"Int2", MVCNN::DType::DType_I2},
    {"Int2X", MVCNN::DType::DType_I2X},
    {"Int4X", MVCNN::DType::DType_I4X},
    {"Bin", MVCNN::DType::DType_BIN},
    {"Log", MVCNN::DType::DType_LOG}
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
    return dTypeMapping_.at(dtype.toString());
}

MVCNN::MemoryLocation mv::RuntimeModel::convertAllocatorToMemoryLocale(const std::string& allocatorName)
{
    return memoryLocationMapping_.at(allocatorName);
}

void mv::RuntimeModel::buildGraphNodeT(mv::BaseOpModel &om, mv::Data::OpListIterator op, MVCNN::GraphNodeT &toBuild)
{
    toBuild.name = op->getName();
    toBuild.thisID = op->get<unsigned>("opId");

    for (auto nextChildOp = op.leftmostChild(); nextChildOp != om.opEnd(); ++nextChildOp)
        toBuild.sourceID.push_back(nextChildOp->get<unsigned>("opId"));

    for (auto nextParentOp = op.leftmostParent(); nextParentOp != om.opEnd(); ++nextParentOp)
        toBuild.sinkID.push_back(nextParentOp->get<unsigned>("opId"));

}

void mv::RuntimeModel::buildSourceStructureT(mv::BaseOpModel &om, MVCNN::SourceStructureT& toBuild)
{
    toBuild.first_ID.push_back(om.getInput()->get<unsigned>("opId"));
    toBuild.nodes = std::vector<std::unique_ptr<MVCNN::GraphNodeT>>(om.opsCount());
    unsigned i = 0;
    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
        buildGraphNodeT(om, opIt, *(toBuild.nodes[i++]));
}


void mv::RuntimeModel::buildTensorReferenceT(mv::BaseOpModel &om, mv::Data::TensorIterator t, MVCNN::TensorReferenceT& toBuild)
{
    mv::DataModel dm(om);
    auto allocator = dm.getAllocator(t->get<std::string>("allocator"));
    mv::Data::BufferIterator bufferIt = allocator.getBuffer(0, t); //0 is the only stage for now, but this will probably change in the future

    toBuild.dimensions = bufferIt->getData()->getShape(); // Padded or not?
    toBuild.strides = bufferIt->getData()->computeNumericStrides(); //NOTE: Maybe directly bufferIt->computeStrides() in the future?

    auto strides = bufferIt->getStrides();
    toBuild.leading_offset = strides[0];
    toBuild.trailing_offset = strides[strides.size()-1] + bufferIt->getPostAlign();

    toBuild.data->data_index = bufferIt->getOffset();
    toBuild.locale = convertAllocatorToMemoryLocale(allocator.getAllocatorName());
    toBuild.data_dtype = convertDtype(bufferIt->getData()->getDType());

    //UNSUPPORTED FOR NOW
    //toBuild.quant_scale;//    std::vector<int8_t> quant_scale;
    //toBuild.quant_zero;//    std::vector<int8_t> quant_zero;
    //toBuild.quant_shift;//    std::vector<int8_t> quant_shift;
}


void mv::RuntimeModel::buildSummaryHeaderT(BaseOpModel& om, MVCNN::SummaryHeaderT& toBuild)
{
    // TODO: probably Target/CompilationDescriptor needs to be passed
    buildVersionT(om, *toBuild.version);

    // Just one input for now
    toBuild.net_input = std::vector<std::unique_ptr<MVCNN::TensorReferenceT>>(1);
    buildTensorReferenceT(om, om.getInput()->getOutputTensor(0), *toBuild.net_input[0]);
    // Just one output for now
    toBuild.net_output = std::vector<std::unique_ptr<MVCNN::TensorReferenceT>>(1);
    buildTensorReferenceT(om, om.getOutput()->getInputTensor(0), *toBuild.net_output[0]);


    //TODO: om.taskCount() needs to be implemented
    toBuild.layer_count = om.opsCount();
    toBuild.task_count = om.opsCount();

    buildResourcesT(om, *toBuild.resources);
    buildSourceStructureT(om, *toBuild.original_structure);
}

void mv::RuntimeModel::buildVersionT(BaseOpModel& om, MVCNN::VersionT& toBuild)
{

}

void mv::RuntimeModel::buildResourcesT(BaseOpModel& om, MVCNN::ResourcesT& toBuild)
{

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
