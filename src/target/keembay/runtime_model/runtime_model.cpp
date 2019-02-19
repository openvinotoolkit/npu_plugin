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

const std::unordered_map<std::string, MVCNN::DPULayerType> mv::RuntimeModel::dpuLayerMapping_ =
{
    {"Conv",MVCNN::DPULayerType::DPULayerType_CONV},
    {"DepthWiseConv",MVCNN::DPULayerType::DPULayerType_DWCONV},
    {"MaxPool",MVCNN::DPULayerType::DPULayerType_MAXPOOL},
    {"AveragePool",MVCNN::DPULayerType::DPULayerType_AVEPOOL},
    {"FullyConnected",MVCNN::DPULayerType::DPULayerType_FCL},
    {"ElementWise",MVCNN::DPULayerType::DPULayerType_ELTWISE},
    {"Identity",MVCNN::DPULayerType::DPULayerType_IDENTITY},
    {"ChannelMajorConvolution",MVCNN::DPULayerType::DPULayerType_CMCONV}
};

const std::unordered_map<mv::PpeLayerTypeEnum, MVCNN::PPELayerType, mv::EnumClassHash> mv::RuntimeModel::ppeLayerTypeMapping_ =
{
   {PPELayerType_STORE, MVCNN::PPELayerType::PPELayerType_STORE},
   {PPELayerType_LOAD, MVCNN::PPELayerType::PPELayerType_LOAD},
   {PPELayerType_CLEAR, MVCNN::PPELayerType::PPELayerType_CLEAR},
   {PPELayerType_NOOP, MVCNN::PPELayerType::PPELayerType_NOOP},
   {PPELayerType_HALT, MVCNN::PPELayerType::PPELayerType_HALT},
   {PPELayerType_ADD, MVCNN::PPELayerType::PPELayerType_ADD},
   {PPELayerType_SUB, MVCNN::PPELayerType::PPELayerType_SUB},
   {PPELayerType_MULT, MVCNN::PPELayerType::PPELayerType_MULT},
   {PPELayerType_LRELU, MVCNN::PPELayerType::PPELayerType_LRELU},
   {PPELayerType_LRELUX, MVCNN::PPELayerType::PPELayerType_LRELUX},
   {PPELayerType_LPRELU, MVCNN::PPELayerType::PPELayerType_LPRELU},
   {PPELayerType_MAXIMUM, MVCNN::PPELayerType::PPELayerType_MAXIMUM},
   {PPELayerType_MINIMUM, MVCNN::PPELayerType::PPELayerType_MINIMUM},
   {PPELayerType_CEIL, MVCNN::PPELayerType::PPELayerType_CEIL},
   {PPELayerType_FLOOR, MVCNN::PPELayerType::PPELayerType_FLOOR},
   {PPELayerType_AND, MVCNN::PPELayerType::PPELayerType_AND},
   {PPELayerType_OR, MVCNN::PPELayerType::PPELayerType_OR},
   {PPELayerType_XOR, MVCNN::PPELayerType::PPELayerType_XOR},
   {PPELayerType_NOT, MVCNN::PPELayerType::PPELayerType_NOT},
   {PPELayerType_ABS, MVCNN::PPELayerType::PPELayerType_ABS},
   {PPELayerType_NEG, MVCNN::PPELayerType::PPELayerType_NEG},
   {PPELayerType_POW, MVCNN::PPELayerType::PPELayerType_POW},
   {PPELayerType_EXP, MVCNN::PPELayerType::PPELayerType_EXP},
   {PPELayerType_SIGMOID, MVCNN::PPELayerType::PPELayerType_SIGMOID},
   {PPELayerType_TANH, MVCNN::PPELayerType::PPELayerType_TANH},
   {PPELayerType_SQRT, MVCNN::PPELayerType::PPELayerType_SQRT},
   {PPELayerType_RSQRT, MVCNN::PPELayerType::PPELayerType_RSQRT},
   {PPELayerType_FLEXARB, MVCNN::PPELayerType::PPELayerType_FLEXARB}
};

MVCNN::DType mv::RuntimeModel::convertDtype(const mv::DType& dtype)
{
    return dTypeMapping_.at(dtype.toString());
}

MVCNN::MemoryLocation mv::RuntimeModel::convertAllocatorToMemoryLocale(const std::string& allocatorName)
{
    return memoryLocationMapping_.at(allocatorName);
}

MVCNN::PPELayerType mv::RuntimeModel::convertPPELayerType(PpeLayerTypeEnum ppe)
{
    return ppeLayerTypeMapping_.at(ppe);
}


void mv::RuntimeModel::buildGraphNodeT(mv::ComputationModel &cm, mv::Element&, mv::Data::OpListIterator op, std::unique_ptr<MVCNN::GraphNodeT> toBuild)
{
    mv::OpModel opModel(cm);
    toBuild->name = op->getName();
    toBuild->thisID = op->get<unsigned>("opId");

    for (auto nextChildOp = op.leftmostChild(); nextChildOp != opModel.opEnd(); ++nextChildOp)
        toBuild->sourceID.push_back(nextChildOp->get<unsigned>("opId"));

    for (auto nextParentOp = op.leftmostParent(); nextParentOp != opModel.opEnd(); ++nextParentOp)
        toBuild->sinkID.push_back(nextParentOp->get<unsigned>("opId"));

}

void mv::RuntimeModel::buildSourceStructureT(mv::ComputationModel &cm, mv::Element &compilationDescriptor, std::unique_ptr<MVCNN::SourceStructureT> toBuild)
{
    mv::OpModel opModel(cm);
    auto inputOp = opModel.getInput();
    toBuild->first_ID.push_back(inputOp->get<unsigned>("opId"));
    toBuild->nodes = std::vector<std::unique_ptr<MVCNN::GraphNodeT>>(opModel.opsCount());
    unsigned i = 0;
    for(auto opIt = opModel.opBegin(); opIt != opModel.opEnd(); ++opIt)
    {
        toBuild->nodes[i] = std::unique_ptr<MVCNN::GraphNodeT>(new MVCNN::GraphNodeT());
        buildGraphNodeT(cm, compilationDescriptor, opIt, std::move(toBuild->nodes[i++]));
    }
}

void mv::RuntimeModel::buildTensorReferenceT(mv::ComputationModel& cm, mv::Element&, mv::Data::TensorIterator t, std::unique_ptr<MVCNN::TensorReferenceT> toBuild)
{    
    mv::DataModel dm(cm);

    auto tensorAllocatorName = t->get<std::set<std::string>>("allocators").begin();
    auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);
    mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, t); // 0 is the only stage for now, but this will probably change in the future

    toBuild->dimensions = tensorBufferIt->getData()->getShape(); // Padded or not?
    toBuild->strides = tensorBufferIt->getData()->computeNumericStrides(); //NOTE: Maybe directly bufferIt->computeStrides() in the future?

    auto strides = tensorBufferIt->getStrides();
    toBuild->leading_offset = strides[0];
    toBuild->trailing_offset = strides[strides.size()-1] + tensorBufferIt->getPostAlign();

    toBuild->data = std::unique_ptr<MVCNN::IndirectDataReferenceT>(new MVCNN::IndirectDataReferenceT());
    toBuild->data->data_index = tensorBufferIt->getOffset();
    // UNSUPPORTED FOR NOW
    // toBuild->sparsity_index
    toBuild->locale = convertAllocatorToMemoryLocale(*tensorAllocatorName);
    toBuild->data_dtype = convertDtype(tensorBufferIt->getData()->getDType());

    // UNSUPPORTED FOR NOW
    // toBuild.quant_scale;//    std::vector<int8_t> quant_scale;
    // toBuild.quant_zero; //    std::vector<int8_t> quant_zero;
    // toBuild.quant_shift;//    std::vector<int8_t> quant_shift;
}

// NOTE: Remember to Ask Ian which allocator should be used.
// Suspect: ProgrammableInput and ProgrammableOutput are really special zones of DDR
// If that's the case, the multiple allocator paradigm will be used only to handle populated tensors

// The following code is being developed on the above assumption
void mv::RuntimeModel::buildSummaryHeaderT(ComputationModel& cm, mv::Element& compilationDescriptor, std::unique_ptr<MVCNN::SummaryHeaderT> toBuild)
{
    mv::OpModel om(cm);
    mv::DataModel dm(dm);

    toBuild->version = std::unique_ptr<MVCNN::VersionT>(new MVCNN::VersionT());
    buildVersionT(cm, compilationDescriptor, std::move(toBuild->version));

    // Just one input for now
    toBuild->net_input = std::vector<std::unique_ptr<MVCNN::TensorReferenceT>>(1);
    toBuild->net_input[0] = std::unique_ptr<MVCNN::TensorReferenceT>(new MVCNN::TensorReferenceT());
    buildTensorReferenceT(cm, compilationDescriptor, om.getInput()->getOutputTensor(0), std::move(toBuild->net_input[0]));

    toBuild->net_output = std::vector<std::unique_ptr<MVCNN::TensorReferenceT>>(1);
    toBuild->net_output[0] = std::unique_ptr<MVCNN::TensorReferenceT>(new MVCNN::TensorReferenceT());
    buildTensorReferenceT(cm, compilationDescriptor, om.getOutput()->getInputTensor(0), std::move(toBuild->net_output[0]));

    //TODO: om.taskCount() needs to be implemented
    toBuild->layer_count = om.opsCount();
    toBuild->task_count = om.opsCount();

    toBuild->resources = std::unique_ptr<MVCNN::ResourcesT>(new MVCNN::ResourcesT());
    buildResourcesT(cm, compilationDescriptor, std::move(toBuild->resources));

    toBuild->original_structure = std::unique_ptr<MVCNN::SourceStructureT>(new MVCNN::SourceStructureT());
    buildSourceStructureT(cm, compilationDescriptor, std::move(toBuild->original_structure));
}

void mv::RuntimeModel::buildVersionT(ComputationModel&, mv::Element& compilationDescriptor, std::unique_ptr<MVCNN::VersionT> toBuild)
{
    toBuild->majorV = compilationDescriptor.get<int>("VersionMajor");
    toBuild->minorV = compilationDescriptor.get<int>("VersionMinor");
    toBuild->patchV = compilationDescriptor.get<int>("VersionPatch");
    toBuild->hash = compilationDescriptor.get<std::string>("VersionHash");
}

void mv::RuntimeModel::buildResourcesT(ComputationModel&, mv::Element& compilationDescriptor, std::unique_ptr<MVCNN::ResourcesT> toBuild)
{
    toBuild->upa_shaves = compilationDescriptor.get<int>("ResourcesUpaShaves");
    toBuild->nce1_blocks = compilationDescriptor.get<int>("ResourcesNCE1Mask");
    toBuild->nce2_blocks = compilationDescriptor.get<int>("ResourcesNCE2Mask");
    toBuild->upa_shared_cmx = compilationDescriptor.get<int>("ResourcesUPASharedCMX");
    toBuild->nn_cmx_per_slice = compilationDescriptor.get<int>("ResourcesNNCMXPerSlice");
    toBuild->nn_cmx_slice_amount = compilationDescriptor.get<int>("ResourcesNNCMXSliceAmount");
    toBuild->ddr_scratch = compilationDescriptor.get<int>("ResourcesDDRScratch");
}

void mv::RuntimeModel::buildBinaryDataT(ComputationModel&, mv::Element&, Data::TensorIterator t, std::unique_ptr<MVCNN::BinaryDataT> toBuild)
{
    // NOTE: In the future tensor->toBinary() will probably handle also the sparsity map associated to the tensor.
    // Or maybe not, we will see

    // OLD approach
//    auto binaryData = t->toBinary();

//    toBuild->fp64 = binaryData.fp64();
//    toBuild->fp32 = binaryData.fp32();
//    toBuild->fp16 = binaryData.fp16();
//    toBuild->f8 = binaryData.fp8();
//    toBuild->u64 = binaryData.u64();
//    toBuild->u32 = binaryData.u32();
//    toBuild->u16 = binaryData.u16();
//    toBuild->u8 = binaryData.u8();
//    toBuild->i64 = binaryData.i64();
//    toBuild->i32 = binaryData.i32();
//    toBuild->i16 = binaryData.i16();
//    toBuild->i8 = binaryData.i8();
//    toBuild->i4 = binaryData.i4();
//    toBuild->i2 = binaryData.i2();
//    toBuild->i2x = binaryData.i2x();
//    toBuild->i4x = binaryData.i4x();
//    toBuild->bin = binaryData.bin();
//    toBuild->log = binaryData.log();


    // NEW approach
    toBuild->data = t->getData();
    toBuild->length = t->getShape().totalSize();
    toBuild->underlying_type = convertDtype(t->getDType());
}

// NOTE: Only 1 TaskList for now, we will see in the future
void mv::RuntimeModel::buildTaskListT(ComputationModel& cm, mv::Element& compilationDescriptor, std::unique_ptr<MVCNN::TaskListT> toBuild)
{
    mv::OpModel om(cm);

    unsigned i = 0;
    for(auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        toBuild->content.push_back(std::unique_ptr<MVCNN::TaskT>(new MVCNN::TaskT()));
        buildTaskT(cm, compilationDescriptor, opIt, std::move(toBuild->content[i++]));
    }
}

void mv::RuntimeModel::buildSpecificTaskUnion(ComputationModel& cm, mv::Element &compilationDescriptor, Data::OpListIterator opIt, MVCNN::SpecificTaskUnion& specificTask)
{
    std::string taskType(opIt->getOpType());

    //NOTE: This if conditions of this big switch statements are not definitive and could change in the future
    if(taskType == "MvTensorTask")
    {
        specificTask.type = MVCNN::SpecificTask_MvTensorTask;
        specificTask.value = new MVCNN::MvTensorTaskT();
        buildMvTensorTaskT(cm, compilationDescriptor, opIt, reinterpret_cast<MVCNN::MvTensorTaskT*>(specificTask.value));
    }
    else if(taskType == "UPADMATask")
    {
        specificTask.type = MVCNN::SpecificTask_UPADMATask;
        specificTask.value = new MVCNN::UPADMATaskT();
        buildUPADMATaskT(cm, compilationDescriptor, opIt, reinterpret_cast<MVCNN::UPADMATaskT*>(specificTask.value));
    }
    else if(taskType == "NNDMATask")
    {
        specificTask.type = MVCNN::SpecificTask_NNDMATask;
        specificTask.value = new MVCNN::NNDMATaskT();
        buildNNDMATaskT(cm, compilationDescriptor, opIt, reinterpret_cast<MVCNN::NNDMATaskT*>(specificTask.value));
    }
    else if(taskType == "NCE1Task")
    {
        specificTask.type = MVCNN::SpecificTask_NCE1Task;
        specificTask.value = new MVCNN::NCE1TaskT();
        buildNCE1TaskT(cm, compilationDescriptor, opIt, reinterpret_cast<MVCNN::NCE1TaskT*>(specificTask.value));
    }
    else if(taskType == "NCE2Task")
    {
        specificTask.type = MVCNN::SpecificTask_NCE2Task;
        specificTask.value = new MVCNN::NCE2TaskT();
        buildNCE2TaskT(cm, compilationDescriptor, opIt, reinterpret_cast<MVCNN::NCE2TaskT*>(specificTask.value));
    }
    else if(taskType == "NNTensorTask")
    {
        specificTask.type = MVCNN::SpecificTask_NNTensorTask;
        specificTask.value = new MVCNN::NNTensorTaskT();
        buildNNTensorTaskT(cm, compilationDescriptor, opIt, reinterpret_cast<MVCNN::NNTensorTaskT*>(specificTask.value));
    }
    else if(taskType == "ControllerTask")
    {
        specificTask.type = MVCNN::SpecificTask_ControllerTask;
        specificTask.value = new MVCNN::ControllerTaskT();
        buildControllerTaskT(cm, compilationDescriptor, opIt, reinterpret_cast<MVCNN::ControllerTaskT*>(specificTask.value));
    }
}

void mv::RuntimeModel::buildMvTensorTaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Data::OpListIterator opIt, MVCNN::MvTensorTaskT* toBuild)
{

}

void mv::RuntimeModel::buildUPADMATaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Data::OpListIterator opIt, MVCNN::UPADMATaskT* toBuild)
{

    toBuild->src = std::unique_ptr<MVCNN::TensorReferenceT>(new MVCNN::TensorReferenceT());
    toBuild->dst = std::unique_ptr<MVCNN::TensorReferenceT>(new MVCNN::TensorReferenceT());

    buildTensorReferenceT(cm, compilationDescriptor, opIt->getInputTensor(0), std::move(toBuild->src));
    buildTensorReferenceT(cm, compilationDescriptor, opIt->getOutputTensor(0), std::move(toBuild->dst));
}

void mv::RuntimeModel::buildNNDMATaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Data::OpListIterator opIt, MVCNN::NNDMATaskT* toBuild)
{
    toBuild->src = std::unique_ptr<MVCNN::TensorReferenceT>(new MVCNN::TensorReferenceT());
    toBuild->dst = std::unique_ptr<MVCNN::TensorReferenceT>(new MVCNN::TensorReferenceT());
    toBuild->broadcast_mask = opIt->get<unsigned>("BroadcastMask");
    toBuild->compression = opIt->get<bool>("Compression");

    buildTensorReferenceT(cm, compilationDescriptor, opIt->getInputTensor(0), std::move(toBuild->src));
    buildTensorReferenceT(cm, compilationDescriptor, opIt->getOutputTensor(0), std::move(toBuild->dst));
}

void mv::RuntimeModel::buildNCE1TaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Data::OpListIterator opIt, MVCNN::NCE1TaskT* toBuild)
{

}

MVCNN::DPULayerType mv::RuntimeModel::convertTaskOp(const std::string& opName)
{
    return dpuLayerMapping_.at(opName);
}


void mv::RuntimeModel::buildPPEFixedFunctionT(ComputationModel&, mv::Element&, const mv::PPEFixedFunction& ppe, std::unique_ptr<MVCNN::PPEFixedFunctionT> toBuild)
{
    auto layers = ppe.getLayers();
    unsigned n = layers.size();
    toBuild->Ops = std::vector<MVCNN::PPELayerType>(n);
    for(unsigned i = 0; i < n; ++i)
        toBuild->Ops[i] = convertPPELayerType(layers[i]);
    toBuild->Clamp_Low = ppe.getLowClamp();
    toBuild->Clamp_High = ppe.getHighClamp();
}

void mv::RuntimeModel::buildPPETaskT(ComputationModel& cm, mv::Element& compilationDescriptor, Data::OpListIterator opIt, std::unique_ptr<MVCNN::PPETaskT> toBuild)
{
    if(opIt->hasAttr("scale"))
    {
        toBuild->scale_data = std::unique_ptr<MVCNN::TensorReferenceT>(new MVCNN::TensorReferenceT);
        Data::TensorIterator tensorIt = opIt->get<Data::TensorIterator>("scale");
        buildTensorReferenceT(cm, compilationDescriptor, tensorIt, std::move(toBuild->scale_data));
    }

    // If this function has been called, this part must be built for sure
    auto fixed_functions = opIt->get<std::vector<PPEFixedFunction>>("PPETask");
    unsigned n = fixed_functions.size();
    toBuild->fixed_function = std::vector<std::unique_ptr<MVCNN::PPEFixedFunctionT>>(n);
    for(unsigned i = 0; i < n; ++i)
    {
        toBuild->fixed_function[i] = std::unique_ptr<MVCNN::PPEFixedFunctionT>(new MVCNN::PPEFixedFunctionT());
        buildPPEFixedFunctionT(cm, compilationDescriptor, fixed_functions[i], std::move(toBuild->fixed_function[i]));
    }
}

void mv::RuntimeModel::buildNCEInvariantFieldsT(ComputationModel& cm, mv::Element &compilationDescriptor, Data::OpListIterator opIt, std::unique_ptr<MVCNN::NCEInvariantFieldsT> toBuild)
{
    toBuild->dpu_task_type = convertTaskOp(opIt->get<std::string>("taskOp"));

    if(opIt->hasAttr("PPETask"))
    {
        toBuild->ppe_task = std::unique_ptr<MVCNN::PPETaskT>(new MVCNN::PPETaskT());
        buildPPETaskT(cm, compilationDescriptor, opIt, std::move(toBuild->ppe_task));
    }
    // TODO
    // std::vector<std::unique_ptr<NNTensorTaskT>> nnshv_task;

    switch (toBuild->dpu_task_type)
    {
        case MVCNN::DPULayerType_CONV:
        case MVCNN::DPULayerType_DWCONV:
        case MVCNN::DPULayerType_CMCONV:
        {
            auto weightsShape = opIt->getInputTensor(1)->getShape();
            toBuild->kernelW = weightsShape[0];
            toBuild->kernelH = weightsShape[1];
            break;
        }
        case MVCNN::DPULayerType_MAXPOOL:
        case MVCNN::DPULayerType_AVEPOOL:
        {
            auto kernelShape = opIt->get<std::array<unsigned short, 2>>("kSize");
            toBuild->kernelW = kernelShape[0];
            toBuild->kernelH = kernelShape[1];
            break;
        }
        default:
            break;
    }

    //Stride is always an attribute
    auto kernelStride = opIt->get<std::array<unsigned short, 2>>("stride");
    toBuild->kernel_strideW = kernelStride[0];
    toBuild->kernel_strideH = kernelStride[1];

    toBuild->input_data = std::unique_ptr<MVCNN::TensorReferenceT>(new MVCNN::TensorReferenceT());
    buildTensorReferenceT(cm, compilationDescriptor, opIt->getInputTensor(0), std::move(toBuild->input_data));
    toBuild->output_data = std::unique_ptr<MVCNN::TensorReferenceT>(new MVCNN::TensorReferenceT());
    buildTensorReferenceT(cm, compilationDescriptor, opIt->getOutputTensor(0), std::move(toBuild->output_data));

    switch (toBuild->dpu_task_type)
    {
        case MVCNN::DPULayerType_CONV:
        case MVCNN::DPULayerType_DWCONV:
        case MVCNN::DPULayerType_CMCONV:
        case MVCNN::DPULayerType_FCL:
            toBuild->weights_data = std::unique_ptr<MVCNN::TensorReferenceT>(new MVCNN::TensorReferenceT());
            buildTensorReferenceT(cm, compilationDescriptor, opIt->getInputTensor(1), std::move(toBuild->weights_data));
            // NOTE: Bias should be handled as well here.
            // std::unique_ptr<TensorReferenceT> bias_data;
            break;
        default:
            break;
    }
}

MVCNN::MPE_Mode mv::RuntimeModel::convertMPEMode(mv::MPE_Mode mpe)
{
    switch (mpe)
    {
        case mv::MPE_Mode::Matrix:
            return MVCNN::MPE_Mode::MPE_Mode_MATRIX;
        case mv::MPE_Mode::Vector:
            return MVCNN::MPE_Mode::MPE_Mode_VECTOR;
        default:
            return MVCNN::MPE_Mode::MPE_Mode_VECTOR;
    }
}

void mv::RuntimeModel::buildNCEVariantFieldsT(ComputationModel& cm, mv::Element &compilationDescriptor, Data::OpListIterator opIt, Workload workload, std::unique_ptr<MVCNN::NCEVariantFieldsT> toBuild)
{
    toBuild->clusterID = workload.clusterID;
    toBuild->workloadID = workload.workloadID;
    if(compilationDescriptor.get<std::string>("Scheduling") == "Dynamic")
    {
        // NOTE: Ignoring barriers for now
        // std::unique_ptr<BarrierReferenceT> associated_barriers;
    }
    toBuild->mpe_mode = convertMPEMode(workload.MPEMode);
    toBuild->padLeft = workload.padLeft;
    toBuild->padRight = workload.padRight;
    toBuild->padTop = workload.padTop;
    toBuild->padBottom = workload.padBottom;
    toBuild->workload_start_X = workload.MinX;
    toBuild->workload_start_Y = workload.MinY;
    toBuild->workload_start_Z = workload.MinZ;
    toBuild->workload_end_X = workload.MaxX;
    toBuild->workload_end_Y = workload.MaxY;
    toBuild->workload_end_Z = workload.MaxZ;
}

void mv::RuntimeModel::buildNCEVariantFieldsTVector(ComputationModel& cm, mv::Element &compilationDescriptor, Data::OpListIterator opIt, std::vector<std::unique_ptr<MVCNN::NCEVariantFieldsT>>& toBuild)
{
    auto workloads = opIt->get<mv::Workloads>("workloads").getWorkloads();
    unsigned n = workloads.size();
    toBuild = std::vector<std::unique_ptr<MVCNN::NCEVariantFieldsT>>(n);
    for(unsigned i = 0; i < n; ++i)
    {
        toBuild[i] = std::unique_ptr<MVCNN::NCEVariantFieldsT>(new MVCNN::NCEVariantFieldsT());
        buildNCEVariantFieldsT(cm, compilationDescriptor, opIt, workloads[i], std::move(toBuild[i]));
    }
}

void mv::RuntimeModel::buildNCE2TaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Data::OpListIterator opIt, MVCNN::NCE2TaskT* toBuild)
{
    toBuild->invariant = std::unique_ptr<MVCNN::NCEInvariantFieldsT>(new MVCNN::NCEInvariantFieldsT());
    buildNCEInvariantFieldsT(cm, compilationDescriptor, opIt, std::move(toBuild->invariant));
    buildNCEVariantFieldsTVector(cm, compilationDescriptor, opIt, toBuild->variant);
}

void mv::RuntimeModel::buildNNTensorTaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Data::OpListIterator opIt, MVCNN::NNTensorTaskT* toBuild)
{

}

void mv::RuntimeModel::buildControllerTaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Data::OpListIterator opIt, MVCNN::ControllerTaskT* toBuild)
{

}

void mv::RuntimeModel::buildTaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Data::OpListIterator opIt, std::unique_ptr<MVCNN::TaskT> toBuild)
{
    toBuild->nodeID = opIt->get<unsigned>("taskId");

    // NOTE: This might change in the future
    toBuild->sourceTaskIDs = {opIt->get<unsigned>("opId")};

    if(compilationDescriptor.get<std::string>("Scheduling") == "Dynamic")
    {
        // NOTE: Ignoring barriers for now
        // std::unique_ptr<BarrierReferenceT> associated_barriers;
    }

    buildSpecificTaskUnion(cm, compilationDescriptor, opIt, toBuild->task);
}

void mv::RuntimeModel::buildGraphFileT(ComputationModel& cm, mv::Element& compilationDescriptor)
{
    mv::OpModel om(cm);

    // HEADER
    graphFile_.header = std::unique_ptr<MVCNN::SummaryHeaderT>(new MVCNN::SummaryHeaderT());
    buildSummaryHeaderT(cm, compilationDescriptor, std::move(graphFile_.header)); //std::unique_ptr<SummaryHeaderT>

    // TASKS
    // BUG: A task list must be built only if there is at least one task.
    // Otherwise it has no sense.
    graphFile_.task_lists = std::vector<std::unique_ptr<MVCNN::TaskListT>>(1);
    graphFile_.task_lists[0] = std::unique_ptr<MVCNN::TaskListT>(new MVCNN::TaskListT());
    buildTaskListT(cm, compilationDescriptor, std::move(graphFile_.task_lists[0]));

    // BARRIERS
    // std::vector<std::unique_ptr<BarrierT>> barrier_table;

    // BINARY DATA
    graphFile_.binary_data = std::vector<std::unique_ptr<MVCNN::BinaryDataT>>();
    unsigned i = 0;
    for(auto tensorIt = om.tensorBegin(); tensorIt != om.tensorBegin(); ++tensorIt)
    {
        if(tensorIt->isPopulated())
        {
            graphFile_.binary_data.push_back(std::unique_ptr<MVCNN::BinaryDataT>(new MVCNN::BinaryDataT()));
            buildBinaryDataT(cm, compilationDescriptor, tensorIt, std::move(graphFile_.binary_data[i++]));
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
