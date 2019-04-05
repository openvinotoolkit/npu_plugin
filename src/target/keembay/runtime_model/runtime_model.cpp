#include "include/mcm/target/keembay/runtime_model/runtime_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "contrib/flatbuffers/include/flatbuffers/util.h"
#include "include/mcm/base/exception/argument_error.hpp"
#include <fstream>
#include <iostream>

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
    {"DepthwiseConv",MVCNN::DPULayerType::DPULayerType_DWCONV},
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


std::unique_ptr<MVCNN::GraphNodeT> mv::RuntimeModel::buildGraphNodeT(mv::ComputationModel &cm, mv::Element&, mv::Data::OpListIterator op)
{
    std::unique_ptr<MVCNN::GraphNodeT> toBuild = std::unique_ptr<MVCNN::GraphNodeT>(new MVCNN::GraphNodeT());

    mv::OpModel opModel(cm);
    toBuild->name = op->getName();
    toBuild->thisID = op->get<unsigned>("opId");

    for (auto nextChildOp = op.leftmostChild(); nextChildOp != opModel.opEnd(); ++nextChildOp)
        toBuild->sourceID.push_back(nextChildOp->get<unsigned>("opId"));

    for (auto nextParentOp = op.leftmostParent(); nextParentOp != opModel.opEnd(); ++nextParentOp)
        toBuild->sinkID.push_back(nextParentOp->get<unsigned>("opId"));

    return toBuild;
}

std::unique_ptr<MVCNN::SourceStructureT> mv::RuntimeModel::buildSourceStructureT(mv::ComputationModel &cm, mv::Element &compilationDescriptor)
{
    std::unique_ptr<MVCNN::SourceStructureT> toBuild = std::unique_ptr<MVCNN::SourceStructureT>(new MVCNN::SourceStructureT());

    mv::OpModel opModel(cm);
    auto inputOp = opModel.getInput();
    toBuild->first_ID.push_back(inputOp->get<unsigned>("opId"));
    toBuild->nodes = std::vector<std::unique_ptr<MVCNN::GraphNodeT>>(opModel.opsCount());
    unsigned i = 0;
    for(auto opIt = opModel.opBegin(); opIt != opModel.opEnd(); ++opIt)
        toBuild->nodes[i++] = buildGraphNodeT(cm, compilationDescriptor, opIt);

    return toBuild;
}

std::unique_ptr<MVCNN::TensorReferenceT> mv::RuntimeModel::buildTensorReferenceT(mv::ComputationModel& cm, mv::Element&, mv::Data::TensorIterator t)
{    
    mv::DataModel dm(cm);
    std::unique_ptr<MVCNN::TensorReferenceT> toBuild = std::unique_ptr<MVCNN::TensorReferenceT>(new MVCNN::TensorReferenceT());

    auto tensorAllocatorName = t->get<std::set<std::string>>("allocators").begin();
    auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);
    mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, t); // 0 is the only stage for now, but this will probably change in the future

    toBuild->dimensions = tensorBufferIt->getData()->getShape(); // Padded or not?
    toBuild->strides = tensorBufferIt->getData()->computeNumericStrides(); //NOTE: Maybe directly bufferIt->computeStrides() in the future?

    auto strides = tensorBufferIt->getStrides();
    toBuild->leading_offset = strides[0];
    toBuild->trailing_offset = strides[strides.size()-1] + tensorBufferIt->getPostAlign();

    toBuild->data = std::unique_ptr<MVCNN::IndirectDataReferenceT>(new MVCNN::IndirectDataReferenceT());
    auto allocators = t->get<std::set<std::string>>("allocators");
    if (allocators.count("GraphFile") != 0)
    {
        toBuild->data->data_index = t->get<unsigned>("graphFileIndex");
        // UNSUPPORTED FOR NOW
        // toBuild->sparsity_index
    }
    else
    {
        toBuild->data->data_index = tensorBufferIt->getOffset();
        // UNSUPPORTED FOR NOW
        // toBuild->sparsity_index
    }
    toBuild->locale = convertAllocatorToMemoryLocale(*tensorAllocatorName);
    toBuild->data_dtype = convertDtype(tensorBufferIt->getData()->getDType());

    // UNSUPPORTED FOR NOW
    // toBuild.quant_scale;//    std::vector<int8_t> quant_scale;
    // toBuild.quant_zero; //    std::vector<int8_t> quant_zero;
    // toBuild.quant_shift;//    std::vector<int8_t> quant_shift;

    return toBuild;
}

std::unique_ptr<MVCNN::SummaryHeaderT> mv::RuntimeModel::buildSummaryHeaderMetaInformations(ComputationModel& cm, mv::Element& compilationDescriptor)
{
    std::unique_ptr<MVCNN::SummaryHeaderT> toBuild = std::unique_ptr<MVCNN::SummaryHeaderT>(new MVCNN::SummaryHeaderT());

    toBuild->version = buildVersionT(cm, compilationDescriptor);
    toBuild->resources = buildResourcesT(cm, compilationDescriptor);
    toBuild->original_structure = buildSourceStructureT(cm, compilationDescriptor);

    return toBuild;
}


std::unique_ptr<MVCNN::SummaryHeaderT> mv::RuntimeModel::buildSummaryHeaderT(ComputationModel& cm, mv::Element& compilationDescriptor, std::unique_ptr<MVCNN::SummaryHeaderT> originalHeader)
{
    mv::OpModel om(cm);

    std::unique_ptr<MVCNN::SummaryHeaderT> toBuild = std::unique_ptr<MVCNN::SummaryHeaderT>(new MVCNN::SummaryHeaderT());

    toBuild->version = std::move(originalHeader->version);
    toBuild->original_structure = std::move(originalHeader->original_structure);
    toBuild->resources = std::move(originalHeader->resources);

    // Just one input for now
    toBuild->net_input = std::vector<std::unique_ptr<MVCNN::TensorReferenceT>>(1);
    toBuild->net_input[0] = buildTensorReferenceT(cm, compilationDescriptor, om.getInput()->getOutputTensor(0));

    toBuild->net_output = std::vector<std::unique_ptr<MVCNN::TensorReferenceT>>(1);
    toBuild->net_output[0] = buildTensorReferenceT(cm, compilationDescriptor, om.getOutput()->getInputTensor(0));

    auto taskCount = [](mv::OpModel m)
    {
        unsigned i = 0;
        for(auto opIt = m.opBegin(); opIt != m.opEnd(); ++opIt)
            if(opIt->getOpType().find("Task") != std::string::npos)
                ++i;
        return i;
    };

    toBuild->layer_count = om.opsCount();
    toBuild->task_count = taskCount(om);

    return toBuild;
}

std::unique_ptr<MVCNN::VersionT> mv::RuntimeModel::buildVersionT(ComputationModel&, mv::Element& compilationDescriptor)
{
    std::unique_ptr<MVCNN::VersionT> toBuild = std::unique_ptr<MVCNN::VersionT>(new MVCNN::VersionT());

    toBuild->majorV = compilationDescriptor.get<int>("VersionMajor");
    toBuild->minorV = compilationDescriptor.get<int>("VersionMinor");
    toBuild->patchV = compilationDescriptor.get<int>("VersionPatch");
    toBuild->hash = compilationDescriptor.get<std::string>("VersionHash");

    return toBuild;
}

std::unique_ptr<MVCNN::ResourcesT> mv::RuntimeModel::buildResourcesT(ComputationModel&, mv::Element& compilationDescriptor)
{
    std::unique_ptr<MVCNN::ResourcesT> toBuild = std::unique_ptr<MVCNN::ResourcesT>(new MVCNN::ResourcesT());

    toBuild->upa_shaves = compilationDescriptor.get<int>("ResourcesUpaShaves");
    toBuild->nce1_blocks = compilationDescriptor.get<int>("ResourcesNCE1Mask");
    toBuild->nce2_blocks = compilationDescriptor.get<int>("ResourcesNCE2Mask");
    toBuild->upa_shared_cmx = compilationDescriptor.get<int>("ResourcesUPASharedCMX");
    toBuild->nn_cmx_per_slice = compilationDescriptor.get<int>("ResourcesNNCMXPerSlice");
    toBuild->nn_cmx_slice_amount = compilationDescriptor.get<int>("ResourcesNNCMXSliceAmount");
    toBuild->ddr_scratch = compilationDescriptor.get<int>("ResourcesDDRScratch");

    return toBuild;
}

std::unique_ptr<MVCNN::BinaryDataT> mv::RuntimeModel::buildBinaryDataT(ComputationModel&, mv::Element&, mv::Tensor& t)
{
    std::unique_ptr<MVCNN::BinaryDataT> toBuild = std::unique_ptr<MVCNN::BinaryDataT>(new MVCNN::BinaryDataT());

    // NEW approach
    if (t.isDoubleType())
    {
        auto tensorData = t.getDoubleData();
        toBuild->data = std::vector<long unsigned int>(tensorData.begin(), tensorData.end());
    }
    else
    {
        auto tensorData = t.getIntData();
        toBuild->data = std::vector<long unsigned int>(tensorData.begin(), tensorData.end());
    }
    toBuild->length = t.getShape().totalSize();
    toBuild->underlying_type = convertDtype(t.getDType());

    return toBuild;
}

// We have three taskslist for POC:
// Tasklist 0: Contains all the tasks
// We need to topologically sort the control model graph to get the tasks in the correct order.

std::vector<std::unique_ptr<MVCNN::TaskListT>> mv::RuntimeModel::buildTaskListT(ComputationModel& cm, mv::Element& compilationDescriptor)
{
    mv::OpModel om(cm);
    mv::ControlModel controlModel(cm);
    std::vector<std::unique_ptr<MVCNN::TaskListT>> toBuild = std::vector<std::unique_ptr<MVCNN::TaskListT>>(1);
    toBuild[0] = std::unique_ptr<MVCNN::TaskListT>(new MVCNN::TaskListT());

    auto topologicallySortedOps = controlModel.topologicalSort();

    for(auto vecIt = topologicallySortedOps.begin(); vecIt != topologicallySortedOps.end(); ++vecIt)
    {
        auto opIt = *vecIt;
        std::string opType = opIt->getOpType();
        //Only Tasks in TaskLists
        if(opType.find("Task") != std::string::npos)
            toBuild[0]->content.push_back(buildTaskT(cm, compilationDescriptor, opIt));
    }

    return toBuild;
}

void mv::RuntimeModel::buildBarrierTaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt, MVCNN::ControllerTaskT* toBuild)
{
    toBuild->task.type = MVCNN::ControllerSubTask_BarrierConfigurationTask;

    auto tmp = new MVCNN::BarrierConfigurationTaskT();
    tmp->target = buildBarrierT(cm, compilationDescriptor, opIt);
    toBuild->task.value = tmp;
}

void mv::RuntimeModel::buildSpecificTaskUnion(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt, MVCNN::SpecificTaskUnion& specificTask)
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
    else if(taskType == "DMATask")
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
    else if(taskType == "DPUTask")
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
    else if(taskType == "BarrierTask")
    {
        specificTask.type = MVCNN::SpecificTask_ControllerTask;
        specificTask.value = new MVCNN::ControllerTaskT();
        buildBarrierTaskT(cm, compilationDescriptor, opIt, reinterpret_cast<MVCNN::ControllerTaskT*>(specificTask.value));
    }
}

void mv::RuntimeModel::buildMvTensorTaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt, MVCNN::MvTensorTaskT* toBuild)
{

}

void mv::RuntimeModel::buildUPADMATaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt, MVCNN::UPADMATaskT* toBuild)
{
    toBuild->src = buildTensorReferenceT(cm, compilationDescriptor, opIt->getInputTensor(0));
    toBuild->dst = buildTensorReferenceT(cm, compilationDescriptor, opIt->getOutputTensor(0));
}

void mv::RuntimeModel::buildNNDMATaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt, MVCNN::NNDMATaskT* toBuild)
{
    toBuild->src = buildTensorReferenceT(cm, compilationDescriptor, opIt->getInputTensor(0));
    toBuild->dst = buildTensorReferenceT(cm, compilationDescriptor, opIt->getOutputTensor(0));
    if(opIt->hasAttr("Compression"))
        toBuild->compression = opIt->get<bool>("Compression");
}

void mv::RuntimeModel::buildNCE1TaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt, MVCNN::NCE1TaskT* toBuild)
{

}

MVCNN::DPULayerType mv::RuntimeModel::convertTaskOp(const std::string& opName)
{
    if (opName == "Add" || opName == "Subtract" || opName == "Multiply")
        return dpuLayerMapping_.at("ElementWise");
    return dpuLayerMapping_.at(opName);
}


std::unique_ptr<MVCNN::PPEFixedFunctionT> mv::RuntimeModel::buildPPEFixedFunctionT(ComputationModel&, mv::Element&, const mv::PPEFixedFunction& ppe)
{
    std::unique_ptr<MVCNN::PPEFixedFunctionT> toBuild = std::unique_ptr<MVCNN::PPEFixedFunctionT>(new MVCNN::PPEFixedFunctionT());

    auto layers = ppe.getLayers();
    unsigned n = layers.size();
    toBuild->Ops = std::vector<MVCNN::PPELayerType>(n);
    for(unsigned i = 0; i < n; ++i)
        toBuild->Ops[i] = convertPPELayerType(layers[i]);
    toBuild->Clamp_Low = ppe.getLowClamp();
    toBuild->Clamp_High = ppe.getHighClamp();

    return toBuild;
}

std::unique_ptr<MVCNN::PPETaskT> mv::RuntimeModel::buildPPETaskT(ComputationModel& cm, mv::Element& compilationDescriptor, Control::OpListIterator opIt)
{
    std::unique_ptr<MVCNN::PPETaskT> toBuild = std::unique_ptr<MVCNN::PPETaskT>(new MVCNN::PPETaskT());

//    if(opIt->hasAttr("scale"))
//    {
//        Data::TensorIterator tensorIt = opIt->get<Data::TensorIterator>("scale");
//        toBuild->scale_data = buildTensorReferenceT(cm, compilationDescriptor, tensorIt);
//    }

//    // If this function has been called, this part must be built for sure
//    auto fixed_functions = opIt->get<PPEFixedFunction>("PPETask");
//    toBuild->fixed_function = buildPPEFixedFunctionT(cm, compilationDescriptor, fixed_functions);

    return toBuild;
}

std::unique_ptr<MVCNN::NCEInvariantFieldsT> mv::RuntimeModel::buildNCEInvariantFieldsT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt)
{
    std::unique_ptr<MVCNN::NCEInvariantFieldsT> toBuild = std::unique_ptr<MVCNN::NCEInvariantFieldsT>(new MVCNN::NCEInvariantFieldsT());

    toBuild->dpu_task_type = convertTaskOp(opIt->get<std::string>("taskOp"));

    if(opIt->hasAttr("PPETask"))
        toBuild->ppe_task = buildPPETaskT(cm, compilationDescriptor, opIt);
    // TODO
    // std::vector<std::unique_ptr<NNTensorTaskT>> nnshv_task;

    if (opIt->hasAttr("kSize"))
    {
        auto kernelShape = opIt->get<std::array<unsigned short, 2>>("kSize");
        toBuild->kernelW = kernelShape[0];
        toBuild->kernelH = kernelShape[1];
    }

    if (opIt->hasAttr("stride"))
    {
        auto kernelStride = opIt->get<std::array<unsigned short, 2>>("stride");
        toBuild->kernel_strideW = kernelStride[0];
        toBuild->kernel_strideH = kernelStride[1];
    }
    toBuild->input_data = buildTensorReferenceT(cm, compilationDescriptor, opIt->getInputTensor(0));
    toBuild->output_data = buildTensorReferenceT(cm, compilationDescriptor, opIt->getOutputTensor(0));

    unsigned num_inputs = opIt->getInputTensor().size();

    if(opIt->hasAttr("fakeSparsity"))
        toBuild->activation_window = buildTensorReferenceT(cm, compilationDescriptor, opIt->getInputTensor(num_inputs - 2));

    toBuild->weights_table = buildTensorReferenceT(cm, compilationDescriptor, opIt->getInputTensor(num_inputs - 1));

    switch (toBuild->dpu_task_type)
    {
        case MVCNN::DPULayerType_CONV:
        case MVCNN::DPULayerType_DWCONV:
        case MVCNN::DPULayerType_CMCONV:
        case MVCNN::DPULayerType_FCL:
            toBuild->weights_data = buildTensorReferenceT(cm, compilationDescriptor, opIt->getInputTensor(1));
            // NOTE: Bias should be handled as well here.
            // std::unique_ptr<TensorReferenceT> bias_data;
            break;
        default:
            break;
    }

    return toBuild;
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

std::unique_ptr<MVCNN::NCEVariantFieldsT> mv::RuntimeModel::buildNCEVariantFieldsT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt, Workload workload)
{
    std::unique_ptr<MVCNN::NCEVariantFieldsT> toBuild = std::unique_ptr<MVCNN::NCEVariantFieldsT>(new MVCNN::NCEVariantFieldsT());

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
    return toBuild;
}

std::vector<std::unique_ptr<MVCNN::NCEVariantFieldsT>> mv::RuntimeModel::buildNCEVariantFieldsTVector(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt)
{
    auto workloads = opIt->get<mv::Workloads>("Workloads").getWorkloads();
    unsigned n = workloads.size();
    std::vector<std::unique_ptr<MVCNN::NCEVariantFieldsT>> toBuild = std::vector<std::unique_ptr<MVCNN::NCEVariantFieldsT>>(n);
    for(unsigned i = 0; i < n; ++i)
        toBuild[i] = buildNCEVariantFieldsT(cm, compilationDescriptor, opIt, workloads[i]);

    return toBuild;
}

void mv::RuntimeModel::buildNCE2TaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt, MVCNN::NCE2TaskT* toBuild)
{
    toBuild->invariant = buildNCEInvariantFieldsT(cm, compilationDescriptor, opIt);
    toBuild->variant = buildNCEVariantFieldsTVector(cm, compilationDescriptor, opIt);
}

void mv::RuntimeModel::buildNNTensorTaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt, MVCNN::NNTensorTaskT* toBuild)
{

}

void mv::RuntimeModel::buildControllerTaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt, MVCNN::ControllerTaskT* toBuild)
{

}

std::unique_ptr<MVCNN::BarrierReferenceT> mv::RuntimeModel::buildBarrierReferenceT(ComputationModel& cm, Element& compilationDescription, BarrierDependencies dep)
{
    std::unique_ptr<MVCNN::BarrierReferenceT> toBuild = std::unique_ptr<MVCNN::BarrierReferenceT>(new MVCNN::BarrierReferenceT());
    int waitBarrier = dep.getWait();
    if(waitBarrier != -1)
        toBuild->wait_barriers = {waitBarrier};
    toBuild->update_barriers = dep.getUpdate();
    return toBuild;
}

std::unique_ptr<MVCNN::TaskT> mv::RuntimeModel::buildTaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt)
{
    std::unique_ptr<MVCNN::TaskT> toBuild = std::unique_ptr<MVCNN::TaskT>(new MVCNN::TaskT());

    toBuild->nodeID = opIt->get<unsigned>("taskId");
    toBuild->name = opIt->getName();

    // NOTE: This might change in the future
    if(opIt->hasAttr("opId"))
        toBuild->sourceTaskIDs = {opIt->get<unsigned>("opId")};

    if(opIt->hasAttr("BarrierDeps"))
        toBuild->associated_barriers = buildBarrierReferenceT(cm, compilationDescriptor, opIt->get<mv::BarrierDependencies>("BarrierDeps"));

    buildSpecificTaskUnion(cm, compilationDescriptor, opIt, toBuild->task);
    return toBuild;
}

std::unique_ptr<MVCNN::BarrierT> mv::RuntimeModel::buildBarrierT(mv::ComputationModel& cm, mv::Element& compilationDescriptor, mv::Control::OpListIterator opIt)
{
    std::unique_ptr<MVCNN::BarrierT> toBuild = std::unique_ptr<MVCNN::BarrierT>(new MVCNN::BarrierT());
    toBuild->barrier_id = opIt->get<mv::Barrier>("Barrier").getIndex();
    toBuild->consumer_count = opIt->get<mv::Barrier>("Barrier").getNumConsumers();
    toBuild->producer_count = opIt->get<mv::Barrier>("Barrier").getNumProducers();
    return toBuild;
}

std::vector<std::unique_ptr<MVCNN::BarrierT>> mv::RuntimeModel::buildBarrierTable(mv::ComputationModel& cm, mv::Element& compilationDescriptor)
{
    mv::OpModel om(cm);
    mv::ControlModel controlModel(cm);

    auto barrierTasks = om.getOps("BarrierTask");
    std::sort(
        barrierTasks.begin(),
        barrierTasks.end(),
        [](const mv::Data::OpListIterator& a, const mv::Data::OpListIterator& b) -> bool { return a->getName() < b->getName(); }
        );

    unsigned n = barrierTasks.size();
    std::vector<std::unique_ptr<MVCNN::BarrierT>> toBuild = std::vector<std::unique_ptr<MVCNN::BarrierT>>(n);
    for(unsigned i = 0; i < n; ++i)
        toBuild[i] = buildBarrierT(cm, compilationDescriptor, controlModel.switchContext(barrierTasks[i]));
    return toBuild;
}

void mv::RuntimeModel::buildHeader(ComputationModel &cm, Element &compilationDescriptor)
{
    //HEADER
    graphFile_.header = buildSummaryHeaderMetaInformations(cm, compilationDescriptor);
}

void mv::RuntimeModel::buildGraphFile(ComputationModel& cm, mv::Element& compilationDescriptor)
{
    mv::OpModel om(cm);

    graphFile_.header = buildSummaryHeaderT(cm, compilationDescriptor, std::move(graphFile_.header));

    // TASKS
    graphFile_.task_lists = buildTaskListT(cm, compilationDescriptor);

    // Barrier Table must be build only on dynamic scheduling
    if(compilationDescriptor.get<std::string>("barrier_index_assignment") == "Dynamic")
        graphFile_.barrier_table = buildBarrierTable(cm, compilationDescriptor);

    // Binary Data
    graphFile_.binary_data = std::vector<std::unique_ptr<MVCNN::BinaryDataT>>();
    for(auto tensorIt = om.tensorBegin(); tensorIt != om.tensorEnd(); ++tensorIt)
    {
        auto allocators = tensorIt->get<std::set<std::string>>("allocators");
        if (allocators.count("GraphFile") != 0)
            buildBinaryDataT(cm, compilationDescriptor, *tensorIt);
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
