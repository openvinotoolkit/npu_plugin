#include "include/mcm/target/keembay/runtime_model/runtime_model.hpp"
#include "meta/include/mcm/op_model.hpp"
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

void mv::RuntimeModel::buildGraphNodeT(mv::ComputationModel &cm, mv::Data::OpListIterator op, std::unique_ptr<MVCNN::GraphNodeT> toBuild)
{
    mv::OpModel opModel(cm);
    toBuild->name = op->getName();
    toBuild->thisID = op->get<unsigned>("opId");

    for (auto nextChildOp = op.leftmostChild(); nextChildOp != opModel.opEnd(); ++nextChildOp)
        toBuild->sourceID.push_back(nextChildOp->get<unsigned>("opId"));

    for (auto nextParentOp = op.leftmostParent(); nextParentOp != opModel.opEnd(); ++nextParentOp)
        toBuild->sinkID.push_back(nextParentOp->get<unsigned>("opId"));

}

void mv::RuntimeModel::buildSourceStructureT(mv::ComputationModel &cm, std::unique_ptr<MVCNN::SourceStructureT> toBuild)
{
    mv::OpModel opModel(cm);
    toBuild->first_ID.push_back(opModel.getInput()->get<unsigned>("opId"));
    toBuild->nodes = std::vector<std::unique_ptr<MVCNN::GraphNodeT>>(opModel.opsCount());
    unsigned i = 0;
    for(auto opIt = opModel.opBegin(); opIt != opModel.opEnd(); ++opIt)
    {
        toBuild->nodes[i] = std::unique_ptr<MVCNN::GraphNodeT>(new MVCNN::GraphNodeT());
        buildGraphNodeT(cm, opIt, std::move(toBuild->nodes[i++]));
    }
}


void mv::RuntimeModel::buildTensorReferenceT(mv::ComputationModel &cm, mv::Data::TensorIterator t, std::unique_ptr<MVCNN::TensorReferenceT> toBuild)
{
    mv::DataModel dm(cm);
    auto allocator = dm.getAllocator(t->get<std::string>("allocator"));

    //NOTE: With auto strangely it doesn't work
    mv::Data::BufferIterator bufferIt = allocator.getBuffer(0, t); //0 is the only stage for now, but this will probably change in the future

    toBuild->dimensions = bufferIt->getData()->getShape(); // Padded or not?
    toBuild->strides = bufferIt->getData()->computeNumericStrides(); //NOTE: Maybe directly bufferIt->computeStrides() in the future?

    auto strides = bufferIt->getStrides();
    toBuild->leading_offset = strides[0];
    toBuild->trailing_offset = strides[strides.size()-1] + bufferIt->getPostAlign();

    toBuild->data = std::unique_ptr<MVCNN::IndirectDataReferenceT>(new MVCNN::IndirectDataReferenceT());
    toBuild->data->data_index = bufferIt->getOffset();
    toBuild->locale = convertAllocatorToMemoryLocale(allocator.getAllocatorName());
    toBuild->data_dtype = convertDtype(bufferIt->getData()->getDType());

    //UNSUPPORTED FOR NOW
    //toBuild.quant_scale;//    std::vector<int8_t> quant_scale;
    //toBuild.quant_zero;//    std::vector<int8_t> quant_zero;
    //toBuild.quant_shift;//    std::vector<int8_t> quant_shift;
}


void mv::RuntimeModel::buildSummaryHeaderT(ComputationModel& cm, json::Object& compilationDescriptor, std::unique_ptr<MVCNN::SummaryHeaderT> toBuild)
{
    mv::OpModel opModel(cm);
    toBuild->version = std::unique_ptr<MVCNN::VersionT>(new MVCNN::VersionT());
    buildVersionT(compilationDescriptor, std::move(toBuild->version));

    // Just one input for now
    toBuild->net_input = std::vector<std::unique_ptr<MVCNN::TensorReferenceT>>(1);
    toBuild->net_input[0] = std::unique_ptr<MVCNN::TensorReferenceT>(new MVCNN::TensorReferenceT());
    buildTensorReferenceT(cm, opModel.getInput()->getOutputTensor(0), std::move(toBuild->net_input[0]));
    // Just one output for now
    toBuild->net_output = std::vector<std::unique_ptr<MVCNN::TensorReferenceT>>(1);
    toBuild->net_output[0] = std::unique_ptr<MVCNN::TensorReferenceT>(new MVCNN::TensorReferenceT());
    buildTensorReferenceT(cm, opModel.getOutput()->getInputTensor(0), std::move(toBuild->net_output[0]));

    //TODO: opModel.taskCount() needs to be implemented
    toBuild->layer_count = opModel.opsCount();
    toBuild->task_count = opModel.opsCount();

    toBuild->resources = std::unique_ptr<MVCNN::ResourcesT>(new MVCNN::ResourcesT());
    buildResourcesT(compilationDescriptor, std::move(toBuild->resources));

    toBuild->original_structure = std::unique_ptr<MVCNN::SourceStructureT>(new MVCNN::SourceStructureT());
    buildSourceStructureT(cm, std::move(toBuild->original_structure));
}

void mv::RuntimeModel::buildVersionT(json::Object& compilationDescriptor, std::unique_ptr<MVCNN::VersionT> toBuild)
{
    toBuild->majorV = compilationDescriptor["Version"]["Major"].get<long long>();
    toBuild->minorV = compilationDescriptor["Version"]["Minor"].get<long long>();
    toBuild->patchV = compilationDescriptor["Version"]["Patch"].get<long long>();
    toBuild->hash = compilationDescriptor["Version"]["Hash"].get<std::string>();
}

void mv::RuntimeModel::buildResourcesT(json::Object& compilationDescriptor, std::unique_ptr<MVCNN::ResourcesT> toBuild)
{
    toBuild->shave_mask = compilationDescriptor["Resources"]["ShaveMask"].get<long long>();
    toBuild->nce1_mask = compilationDescriptor["Resources"]["NCE1Mask"].get<long long>();
    toBuild->dpu_mask = compilationDescriptor["Resources"]["DPUMask"].get<long long>();
    toBuild->leon_cmx = compilationDescriptor["Resources"]["LeonCMX"].get<long long>();
    toBuild->nn_cmx = compilationDescriptor["Resources"]["NNCMX"].get<long long>();
    toBuild->ddr_scratch = compilationDescriptor["Resources"]["DDRScratch"].get<long long>();
}

void mv::RuntimeModel::buildBinaryDataT(Data::TensorIterator t, std::unique_ptr<MVCNN::BinaryDataT> toBuild)
{
    // NOTE: In the future tensor->toBinary() will probably handle also the sparsity map associated to the tensor.
    // Or maybe not, we will see
    auto binaryData = t->toBinary();

    toBuild->fp64 = binaryData.fp64();
    toBuild->fp32 = binaryData.fp32();
    toBuild->fp16 = binaryData.fp16();
    toBuild->f8 = binaryData.fp8();
    toBuild->u64 = binaryData.u64();
    toBuild->u32 = binaryData.u32();
    toBuild->u16 = binaryData.u16();
    toBuild->u8 = binaryData.u8();
    toBuild->i64 = binaryData.i64();
    toBuild->i32 = binaryData.i32();
    toBuild->i16 = binaryData.i16();
    toBuild->i8 = binaryData.i8();
    toBuild->i4 = binaryData.i4();
    toBuild->i2 = binaryData.i2();
    toBuild->i2x = binaryData.i2x();
    toBuild->i4x = binaryData.i4x();
    toBuild->bin = binaryData.bin();
    toBuild->log = binaryData.log();
}

void mv::RuntimeModel::buildGraphFileT(ComputationModel& cm, json::Object& compilationDescriptor)
{
    mv::OpModel om(cm);

    // HEADER
    graphFile_.header = std::unique_ptr<MVCNN::SummaryHeaderT>(new MVCNN::SummaryHeaderT());
    buildSummaryHeaderT(cm, compilationDescriptor, std::move(graphFile_.header)); //std::unique_ptr<SummaryHeaderT>

    // TASKS
    //    std::vector<std::unique_ptr<TaskListT>> task_lists;

    // BARRIERS
    //    std::vector<std::unique_ptr<BarrierT>> barrier_table;

    // BINARY DATA
    graphFile_.binary_data = std::vector<std::unique_ptr<MVCNN::BinaryDataT>>();
    unsigned i = 0;
    for(auto tensorIt = om.tensorBegin(); tensorIt != om.tensorBegin(); ++tensorIt)
    {
        if(tensorIt->isPopulated())
        {
            graphFile_.binary_data.push_back(std::unique_ptr<MVCNN::BinaryDataT>(new MVCNN::BinaryDataT()));
            buildBinaryDataT(tensorIt, std::move(graphFile_.binary_data[i++]));
        }
    }
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
