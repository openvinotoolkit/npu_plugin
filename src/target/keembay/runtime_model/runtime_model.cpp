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

const std::unordered_map<mv::PPELayerTypeEnum, MVCNN::PPELayerType, mv::EnumClassHash> mv::RuntimeModel::ppeLayerTypeMapping_ =
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

template <typename T1, typename T2>
void setIfPresent(T1& fieldToFill, mv::Element& compilationDescriptor, const std::string& key)
{
    if(compilationDescriptor.hasAttr(key))
        fieldToFill = compilationDescriptor.get<T2>(key);
}

MVCNN::DType mv::RuntimeModel::convertDtype(const mv::DType& dtype)
{
    return dTypeMapping_.at(dtype.toString());
}

MVCNN::MemoryLocation mv::RuntimeModel::convertAllocatorToMemoryLocale(const std::string& allocatorName)
{
    return memoryLocationMapping_.at(allocatorName);
}

MVCNN::PPELayerType mv::RuntimeModel::convertPPELayerType(PPELayerTypeEnum ppe)
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
        toBuild->sinkID.push_back(nextChildOp->get<unsigned>("opId"));

    for (auto nextParentOp = op.leftmostParent(); nextParentOp != opModel.opEnd(); ++nextParentOp)
        toBuild->sourceID.push_back(nextParentOp->get<unsigned>("opId"));

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

    //auto ops = opModel.topologicalSort();
    for(auto opIt = opModel.opBegin(); opIt != opModel.opEnd(); ++opIt)
        toBuild->nodes[i++] = buildGraphNodeT(cm, compilationDescriptor, opIt);

    return toBuild;
}

std::vector<unsigned> mv::RuntimeModel::reduceQuantVector_(std::vector<unsigned> inVec)
{
    if (inVec.size() > 1)
    {
        auto firstVal = inVec[0];
        auto onlyOneValue = true;
        for (size_t i = 1; i < inVec.size(); i++)
            if (firstVal != inVec[i])
                onlyOneValue = false;
        if (onlyOneValue)
        {
            inVec.clear();
            inVec.push_back(firstVal);
        }
    }
    return inVec;
}

std::unique_ptr<MVCNN::TensorReferenceT> mv::RuntimeModel::buildTensorReferenceT(mv::ComputationModel& model, mv::Element&, mv::Data::TensorIterator t)
{
    mv::DataModel dm(model);
    mv::OpModel om(model);
    mv::ControlModel cm(model);


    mv::Control::StageIterator stg = cm.getStage(0);

    std::unique_ptr<MVCNN::TensorReferenceT> toBuild = std::unique_ptr<MVCNN::TensorReferenceT>(new MVCNN::TensorReferenceT());

    toBuild->name = t->getName();

    auto tensorAllocatorName = t->get<std::set<std::string>>("allocators").begin();
    auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);
    mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, t); // 0 is the only stage for now, but this will probably change in the future

    auto underlyingTensor = tensorBufferIt->getData();
    std::vector<uint32_t> dimensions = underlyingTensor->getShape();
    std::vector<uint32_t> numericStrides = underlyingTensor->computeNumericStrides();

    auto masterBuffer = tensorBufferIt->getMaster();
    if (masterBuffer != dm.bufferEnd(*tensorAllocatorName, stg))
        numericStrides = (*masterBuffer)->getData()->computeNumericStrides();

    numericStrides.push_back(underlyingTensor->getDType().getSizeInBits() / 8);

    //Because according to graphfile order is given as NCHW, which is exactly the reverse of our shape assumption WHCN
    std::reverse(dimensions.begin(), dimensions.end());
    std::reverse(numericStrides.begin(), numericStrides.end());

    toBuild->dimensions = dimensions;
    toBuild->strides = numericStrides; // NOTE: Maybe directly bufferIt->computeStrides() in the future?

    // NOTE: not sure anymore about this
//    auto strides = tensorBufferIt->getStrides();
//    toBuild->leading_offset = strides[0]/2; //for some reason we get double the value, for now take the proper one.
//    toBuild->trailing_offset = strides[strides.size()-1] + tensorBufferIt->getPostAlign();
//    toBuild->trailing_offset = toBuild->trailing_offset / 2;
    toBuild->data = std::unique_ptr<MVCNN::IndirectDataReferenceT>(new MVCNN::IndirectDataReferenceT());
    if (*tensorAllocatorName == "GraphFile")
    {
        toBuild->data->data_index = t->get<unsigned>("graphFileIndex");
        // No need to set sparsity_index for tensor stored in graphfile
    }
    else if(*tensorAllocatorName == "ProgrammableInput" || *tensorAllocatorName == "ProgrammableOutput")
    {
        toBuild->data->data_index = 0;
        if (toBuild->leading_offset)
            toBuild->data->data_index += toBuild->leading_offset;
        // No need to set sparsity_index for input/output tensor of the network
    }
    else
    {
        toBuild->data->data_index = tensorBufferIt->getOffset();
        if (toBuild->leading_offset)
            toBuild->data->data_index += toBuild->leading_offset;

        // VERY IMPORTANT NOTE: Sparsity index is not used by populated tensors
        // as populated tensor represent weights, and all the information we need
        // about sparsity is contained in the weights table. This was confirmed
        // after a chat with Levi
        if(t->isSparse())
        {
            if(!t->isPopulated())
            {
                toBuild->data->sparsity_index = t->getSparsityMap()->getAddress();
                toBuild->data->storage_element_index = t->getStorageElement()->getAddress();
            }
        }
    }
    toBuild->locale = convertAllocatorToMemoryLocale(*tensorAllocatorName);

    // NOTE: Will probably change in the future
    toBuild->locale_index = std::vector<unsigned int>(1,0);

    toBuild->data_dtype = convertDtype(tensorBufferIt->getData()->getDType());

    // could also be t->hasAttr("quantizationParameters")
    // but in my opinion quantization for a tensor of floats makes very little sense
    // leaving this comment here for future generations
    if(t->isQuantized())
    {
        auto quantizationParams = t->get<mv::QuantizationParams>("quantParams");
        auto quantZero = quantizationParams.getZeroPoint();
        toBuild->quant_zero = std::vector<unsigned char>(quantZero.begin(), quantZero.end());
        std::vector<unsigned> quantScale = {};
        if (quantizationParams.hasAttr("mult"))
            quantScale = quantizationParams.getMult();

        quantScale = reduceQuantVector_(quantScale);
        toBuild->quant_scale = std::vector<unsigned short int>(quantScale.begin(), quantScale.end());
        std::vector<unsigned> quantShift;
        if (quantizationParams.hasAttr("shift"))
            quantShift = quantizationParams.getShift();
        quantShift = reduceQuantVector_(quantShift);
        toBuild->quant_shift = std::vector<unsigned char>(quantShift.begin(), quantShift.end());

    }

    return toBuild;
}

std::unique_ptr<MVCNN::SummaryHeaderT> mv::RuntimeModel::buildSummaryHeaderMetaInformations(ComputationModel& cm, mv::Element& compilationDescriptor)
{
    mv::OpModel om(cm);

    std::unique_ptr<MVCNN::SummaryHeaderT> toBuild = std::unique_ptr<MVCNN::SummaryHeaderT>(new MVCNN::SummaryHeaderT());

    toBuild->version = buildVersionT(cm, compilationDescriptor);
    toBuild->resources = buildResourcesT(cm, compilationDescriptor);
    toBuild->original_structure = buildSourceStructureT(cm, compilationDescriptor);
    toBuild->layer_count = om.opsCount();

    return toBuild;
}


std::unique_ptr<MVCNN::SummaryHeaderT> mv::RuntimeModel::buildSummaryHeaderT(ComputationModel& cm, mv::Element& compilationDescriptor, std::unique_ptr<MVCNN::SummaryHeaderT> originalHeader)
{
    mv::OpModel om(cm);

    std::unique_ptr<MVCNN::SummaryHeaderT> toBuild = std::unique_ptr<MVCNN::SummaryHeaderT>(new MVCNN::SummaryHeaderT());

    auto globalConfigurationParameters = cm.getGlobalConfigParams();

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

    toBuild->options = std::vector<MVCNN::ExecutionFlag>();
    //toBuild->options.push_back(MVCNN::ExecutionFlag_Compiled_For_VPU3);
    if(globalConfigurationParameters->get<std::string>("barrier_index_assignment") == "Dynamic")
        toBuild->options.push_back(MVCNN::ExecutionFlag_DynamicBarriers);

    toBuild->layer_count = originalHeader->layer_count;
    toBuild->task_count = taskCount(om);

    return toBuild;
}

std::unique_ptr<MVCNN::VersionT> mv::RuntimeModel::buildVersionT(ComputationModel&, mv::Element& compilationDescriptor)
{
    std::unique_ptr<MVCNN::VersionT> toBuild = std::unique_ptr<MVCNN::VersionT>(new MVCNN::VersionT());

    setIfPresent<uint32_t, int>(toBuild->majorV, compilationDescriptor, "VersionMajor");
    setIfPresent<uint32_t, int>(toBuild->minorV, compilationDescriptor, "VersionMinor");
    setIfPresent<uint32_t, int>(toBuild->patchV, compilationDescriptor, "VersionPatch");
    setIfPresent<std::string, std::string>(toBuild->hash, compilationDescriptor, "VersionHash");

    return toBuild;
}

std::unique_ptr<MVCNN::ResourcesT> mv::RuntimeModel::buildResourcesT(ComputationModel& cm, mv::Element& compilationDescriptor)
{
    std::unique_ptr<MVCNN::ResourcesT> toBuild = std::unique_ptr<MVCNN::ResourcesT>(new MVCNN::ResourcesT());

    auto globalConfigurationParams = cm.getGlobalConfigParams();

    setIfPresent<uint32_t, int>(toBuild->upa_shaves, *globalConfigurationParams , "UpaShaves");
    setIfPresent<int8_t, int>(toBuild->nce1_blocks, *globalConfigurationParams, "NCE1Mask");
    setIfPresent<uint32_t, int>(toBuild->nce2_blocks, *globalConfigurationParams, "Number_of_DPUs");
    setIfPresent<uint32_t, int>(toBuild->upa_shared_cmx, *globalConfigurationParams, "UPASharedCMX");
    setIfPresent<uint32_t, unsigned>(toBuild->nn_cmx_per_slice, *globalConfigurationParams, "cmx");
    setIfPresent<uint32_t, unsigned>(toBuild->nn_cmx_slice_amount, *globalConfigurationParams, "clusters");
    setIfPresent<uint32_t, int>(toBuild->ddr_scratch, *globalConfigurationParams, "DDRScratch");

    return toBuild;
}

template <typename T>
std::vector<long unsigned int> packToInt64(const std::vector<T>& origData, mv::DType dtype)
{
    unsigned dataSize = origData.size();
    unsigned origDataSize = dtype.getSizeInBits();
    unsigned nElementToPack = 64 / origDataSize;
    unsigned finalLength = dataSize / nElementToPack;

    std::vector<long unsigned int> toReturn(finalLength, 0);

    for(unsigned i = 0; i < finalLength; ++i)
        for(unsigned j = 0; j < nElementToPack; ++j)
            toReturn[i] ^= origData[i*nElementToPack + j] << (j * origDataSize);

    return toReturn;
}

std::unique_ptr<MVCNN::BinaryDataT> mv::RuntimeModel::buildBinaryDataT(ComputationModel&, mv::Element&, mv::Tensor& t)
{
    std::unique_ptr<MVCNN::BinaryDataT> toBuild = std::unique_ptr<MVCNN::BinaryDataT>(new MVCNN::BinaryDataT());
    if (t.isSparse())
    {
        toBuild->data = packToInt64(t.getDataPacked(), t.getDType());
        toBuild->length = t.dataPackedSize();
    }
    else
    {
        toBuild->data = packToInt64(t.getData(), t.getDType());
        toBuild->length = t.getShape().totalSize();
    }

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
    std::vector<std::unique_ptr<MVCNN::TaskListT>> toBuild = std::vector<std::unique_ptr<MVCNN::TaskListT>>(3);
    toBuild[0] = std::unique_ptr<MVCNN::TaskListT>(new MVCNN::TaskListT());
    toBuild[1] = std::unique_ptr<MVCNN::TaskListT>(new MVCNN::TaskListT());
    toBuild[2] = std::unique_ptr<MVCNN::TaskListT>(new MVCNN::TaskListT());

    auto topologicallySortedOps = controlModel.schedulingSort();

    int initialId = 0;

    for(auto vecIt = topologicallySortedOps.begin(); vecIt != topologicallySortedOps.end(); ++vecIt)
    {
        auto opIt = *vecIt;
        std::unique_ptr<MVCNN::TaskListT> * listToUse;
        std::string opType = opIt->getOpType();
        if(opType.find("DPU") != std::string::npos)
            listToUse = &toBuild[0];
        else if(opType.find("DMA") != std::string::npos)
            listToUse = &toBuild[1];
        auto tasks = buildTaskT(cm, compilationDescriptor, opIt, initialId);
        for(auto& task: tasks)
            (*listToUse)->content.push_back(std::move(task));
    }

    //Barrier task list has to be built in the correct order
    auto barrierTasks = om.getOps("BarrierTask");
    std::sort(
        barrierTasks.begin(),
        barrierTasks.end(),
        [](const mv::Data::OpListIterator& a, const mv::Data::OpListIterator& b) -> bool { return a->get<mv::Barrier>("Barrier").getIndex() < b->get<mv::Barrier>("Barrier").getIndex(); }
        );

    unsigned n = barrierTasks.size();
    for(unsigned i = 0; i < n; ++i)
    {
        auto tasks = buildTaskT(cm, compilationDescriptor, controlModel.switchContext(barrierTasks[i]), initialId);
        for(auto& task: tasks)
            toBuild[2]->content.push_back(std::move(task));
    }

    return toBuild;
}

std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildBarrierTaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt)
{
    std::vector<std::unique_ptr<MVCNN::TaskT>> toReturn = std::vector<std::unique_ptr<MVCNN::TaskT>>(1);

    toReturn[0] = std::unique_ptr<MVCNN::TaskT>(new MVCNN::TaskT());
    toReturn[0]->task.type = MVCNN::SpecificTask_ControllerTask;

    auto controllerTask = new MVCNN::ControllerTaskT();
    controllerTask->task.type = MVCNN::ControllerSubTask_BarrierConfigurationTask;

    auto barrierConfigurationTask = new MVCNN::BarrierConfigurationTaskT();
    barrierConfigurationTask->target = buildBarrierT(cm, compilationDescriptor, opIt);

    controllerTask->task.value = barrierConfigurationTask;
    toReturn[0]->task.value = controllerTask;

   return toReturn;
}

std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildSpecificTaskUnion(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt, int& nodeID)
{
    std::vector<std::unique_ptr<MVCNN::TaskT>> toBuild = std::vector<std::unique_ptr<MVCNN::TaskT>>();
    std::string taskType(opIt->getOpType());

    //NOTE: This if conditions of this big switch statements are not definitive and could change in the future
    if(taskType == "MvTensorTask")
        toBuild = buildMvTensorTaskT(cm, compilationDescriptor, opIt);
    else if(taskType == "UPADMATask")
        toBuild = buildUPADMATaskT(cm, compilationDescriptor, opIt);
    else if(taskType == "DMATask")
        toBuild = buildNNDMATaskT(cm, compilationDescriptor, opIt);
    else if(taskType == "NCE1Task")
        toBuild = buildNCE1TaskT(cm, compilationDescriptor, opIt);
    else if(taskType == "DPUTask")
        toBuild = buildNCE2TaskT(cm, compilationDescriptor, opIt);
    else if(taskType == "NNTensorTask")
        toBuild = buildNNTensorTaskT(cm, compilationDescriptor, opIt);
    else if(taskType == "ControllerTask")
        toBuild = buildControllerTaskT(cm, compilationDescriptor, opIt);
    else if(taskType == "BarrierTask")
        toBuild = buildBarrierTaskT(cm, compilationDescriptor, opIt);

    return toBuild;
}

std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildMvTensorTaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt)
{

}

std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildUPADMATaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt)
{
    std::vector<std::unique_ptr<MVCNN::TaskT>> toReturn = std::vector<std::unique_ptr<MVCNN::TaskT>>(1);
    toReturn[0] = std::unique_ptr<MVCNN::TaskT>(new MVCNN::TaskT());
    toReturn[0]->task.type = MVCNN::SpecificTask_UPADMATask;
    auto tmp = new MVCNN::UPADMATaskT();
    tmp->src = buildTensorReferenceT(cm, compilationDescriptor, opIt->getInputTensor(0));
    tmp->dst = buildTensorReferenceT(cm, compilationDescriptor, opIt->getOutputTensor(0));
    toReturn[0]->task.value = tmp;
    return toReturn;
}

std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildNNDMATaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt)
{
    mv::DataModel dm(cm);
    mv::ControlModel controlModel(cm);

    std::vector<std::unique_ptr<MVCNN::TaskT>> toReturn = std::vector<std::unique_ptr<MVCNN::TaskT>>(1);
    toReturn[0] = std::unique_ptr<MVCNN::TaskT>(new MVCNN::TaskT());
    toReturn[0]->task.type = MVCNN::SpecificTask_NNDMATask;
    auto tmp = new MVCNN::NNDMATaskT();
    auto inputTensor = opIt->getInputTensor(0);
    tmp->src = buildTensorReferenceT(cm, compilationDescriptor, inputTensor);
    auto tensorAllocatorName = inputTensor->get<std::set<std::string>>("allocators").begin();
    auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);
    mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, inputTensor);
    mv::Control::StageIterator stg = controlModel.getStage(0);

    if (tensorBufferIt->getMaster() != dm.bufferEnd(*tensorAllocatorName, stg))
    {
        auto masterBuffer = tensorBufferIt->getMaster(); //TODO: or do we need the top one?
        auto masterTensor = (*masterBuffer)->getData();

        auto numericStrides = masterTensor->computeNumericStrides();
        numericStrides.push_back(masterTensor->getDType().getSizeInBits() / 8);

        //Because according to graphfile order is given as NCHW, which is exactly the reverse of our shape assumption WHCN
        std::reverse(numericStrides.begin(), numericStrides.end());
        tmp->src->strides = numericStrides;
    }

    auto outputTensor = opIt->getOutputTensor(0);

    tmp->dst = buildTensorReferenceT(cm, compilationDescriptor, outputTensor);
    tensorAllocatorName = outputTensor->get<std::set<std::string>>("allocators").begin();
    tensorAllocator = dm.getAllocator(*tensorAllocatorName);
    tensorBufferIt = tensorAllocator.getBuffer(0, outputTensor);
    stg = controlModel.getStage(0);

    if (tensorBufferIt->getMaster() != dm.bufferEnd(*tensorAllocatorName, stg))
    {
        auto masterBuffer = tensorBufferIt->getMaster(); //TODO: or do we need the top one?
        auto masterTensor = (*masterBuffer)->getData();
        auto numericStrides = masterTensor->computeNumericStrides();
        numericStrides.push_back(masterTensor->getDType().getSizeInBits() / 8);

        //Because according to graphfile order is given as NCHW, which is exactly the reverse of our shape assumption WHCN
        std::reverse(numericStrides.begin(), numericStrides.end());
        tmp->dst->strides = numericStrides;
    }

    if(opIt->hasAttr("Compression"))
        tmp->compression =  opIt->get<bool>("Compression");
    toReturn[0]->task.value = tmp;
    return toReturn;
}

std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildNCE1TaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt)
{

}

MVCNN::DPULayerType mv::RuntimeModel::convertTaskOp(const std::string& opName)
{
    if (opName == "Add" || opName == "Subtract" || opName == "Multiply")
        return dpuLayerMapping_.at("ElementWise");
    return dpuLayerMapping_.at(opName);
}


std::unique_ptr<MVCNN::PPEFixedFunctionT> mv::RuntimeModel::buildPPEFixedFunctionT(ComputationModel&, mv::Element&, const mv::PPEFixedFunction& ppeFixedFunction)
{
    std::unique_ptr<MVCNN::PPEFixedFunctionT> toBuild = std::unique_ptr<MVCNN::PPEFixedFunctionT>(new MVCNN::PPEFixedFunctionT());

    auto layers = ppeFixedFunction.getLayers();
    unsigned n = layers.size();
    toBuild->Ops = std::vector<MVCNN::PPELayerType>(n);
    for(unsigned i = 0; i < n; ++i)
        toBuild->Ops[i] = convertPPELayerType(layers[i]);
    toBuild->Clamp_Low = ppeFixedFunction.getLowClamp();
    toBuild->Clamp_High = ppeFixedFunction.getHighClamp();

    return toBuild;
}

std::unique_ptr<MVCNN::PPETaskT> mv::RuntimeModel::buildPPETaskT(ComputationModel& cm, mv::Element& compilationDescriptor, const mv::PPETask& ppeTask)
{
    std::unique_ptr<MVCNN::PPETaskT> toBuild = std::unique_ptr<MVCNN::PPETaskT>(new MVCNN::PPETaskT());

    if(ppeTask.hasAttr("scaleData"))
        toBuild->scale_data = buildTensorReferenceT(cm, compilationDescriptor, ppeTask.getScaleData());
    toBuild->fixed_function = buildPPEFixedFunctionT(cm, compilationDescriptor, ppeTask.getFixedFunction());

    return toBuild;
}

std::unique_ptr<MVCNN::PPETaskT> mv::RuntimeModel::buildPPETaskT()
{
    std::unique_ptr<MVCNN::PPETaskT> toBuild = std::unique_ptr<MVCNN::PPETaskT>(new MVCNN::PPETaskT());
    toBuild->fixed_function = std::unique_ptr<MVCNN::PPEFixedFunctionT>(new MVCNN::PPEFixedFunctionT());
    toBuild->fixed_function->Clamp_High = 2147483647;
    toBuild->fixed_function->Clamp_Low = -2147483648;
    toBuild->fixed_function->Ops = std::vector<MVCNN::PPELayerType>();
    toBuild->fixed_function->Ops.reserve(3);

    return toBuild;
}


std::unique_ptr<MVCNN::NCEInvariantFieldsT> mv::RuntimeModel::buildNCEInvariantFieldsT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt)
{
    std::unique_ptr<MVCNN::NCEInvariantFieldsT> toBuild = std::unique_ptr<MVCNN::NCEInvariantFieldsT>(new MVCNN::NCEInvariantFieldsT());

    toBuild->dpu_task_type = convertTaskOp(opIt->get<std::string>("taskOp"));

    if(opIt->hasAttr("PPETask"))
        toBuild->ppe_task = buildPPETaskT(cm, compilationDescriptor, opIt->get<PPETask>("PPETask"));
    else
        toBuild->ppe_task = buildPPETaskT();
    // TODO
    // std::vector<std::unique_ptr<NNTensorTaskT>> nnshv_task;
    // split_over_h: bool = false;

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

    if (opIt->hasAttr("padding"))
    {
        auto kernelPadding = opIt->get<std::array<unsigned short, 4>>("padding");
        toBuild->kernel_padLeft = kernelPadding[0];
        toBuild->kernel_padRight = kernelPadding[1];
        toBuild->kernel_padTop = kernelPadding[2];
        toBuild->kernel_padBottom = kernelPadding[3];
    }
    //input
    mv::DataModel dm(cm);
    mv::ControlModel controlModel(cm);
    auto inputTensor = opIt->getInputTensor(0);
    auto tensorAllocatorName = inputTensor->get<std::set<std::string>>("allocators").begin();
    auto tensorAllocator = dm.getAllocator(*tensorAllocatorName);
    mv::Data::BufferIterator tensorBufferIt = tensorAllocator.getBuffer(0, inputTensor); // 0 is the only stage for now, but this will probably change in the future
    mv::Control::StageIterator stg = controlModel.getStage(0);
    toBuild->input_data = buildTensorReferenceT(cm, compilationDescriptor, inputTensor);

    auto masterBuffer = tensorBufferIt->getMaster();
    if (masterBuffer != dm.bufferEnd(*tensorAllocatorName, stg))
        toBuild->parent_input_tensor = buildTensorReferenceT(cm, compilationDescriptor, (*masterBuffer)->getData());
    else
        toBuild->parent_input_tensor = buildTensorReferenceT(cm, compilationDescriptor, opIt->getInputTensor(0));


    //output
    auto outputTensor = opIt->getOutputTensor(0);
    tensorAllocatorName = outputTensor->get<std::set<std::string>>("allocators").begin();
    tensorAllocator = dm.getAllocator(*tensorAllocatorName);
    tensorBufferIt = tensorAllocator.getBuffer(0, outputTensor); // 0 is the only stage for now, but this will probably change in the future
    toBuild->output_data = buildTensorReferenceT(cm, compilationDescriptor, outputTensor);
    masterBuffer = tensorBufferIt->getMaster();

    if (masterBuffer != dm.bufferEnd(*tensorAllocatorName, stg))
        toBuild->parent_output_tensor = buildTensorReferenceT(cm, compilationDescriptor, (*masterBuffer)->getData());
    else
        toBuild->parent_output_tensor = buildTensorReferenceT(cm, compilationDescriptor, opIt->getOutputTensor(0));

    // NOTE: Remember to ask Eman why this is done
    // toBuild->output_data->data->data_index += toBuild->output_data->leading_offset;

    unsigned num_inputs = opIt->getInputTensor().size();

    //OP inputs == n ->
    // n - 2 activation window (when present)
    // n - 1 weights table
    if(opIt->hasAttr("fakeSparsity"))
    {
        auto activationWindowTensorIterator = opIt->getInputTensor(num_inputs - 2);
        toBuild->activation_window = buildTensorReferenceT(cm, compilationDescriptor, activationWindowTensorIterator);
        toBuild->activation_window_channel_length = activationWindowTensorIterator->get<int>("channelLength");
    }

    if(toBuild->dpu_task_type != MVCNN::DPULayerType_ELTWISE)
    {
        auto weightsTableTensorIterator = opIt->getInputTensor(num_inputs - 1);
        toBuild->weights_table = buildTensorReferenceT(cm, compilationDescriptor, weightsTableTensorIterator);
    }

    switch (toBuild->dpu_task_type)
    {
        case MVCNN::DPULayerType_CONV:
        case MVCNN::DPULayerType_DWCONV:
        case MVCNN::DPULayerType_CMCONV:
        case MVCNN::DPULayerType_FCL:
        case MVCNN::DPULayerType_ELTWISE:
            //std::unique_ptr<TensorReferenceT> parent_weights_tensor;
            toBuild->weights_data = buildTensorReferenceT(cm, compilationDescriptor, opIt->getInputTensor(1));
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

bool mv::RuntimeModel::hardwareBugDepthwise(Control::OpListIterator opIt)
{
    auto splitStrategy = opIt->get<std::string>("splitStrategy");
    auto padding = opIt->get<std::array<unsigned short, 4>>("padding");
    auto kernelSize = opIt->get<std::array<unsigned short, 2>>("kSize");
    return ((splitStrategy == "SplitOverH") &&
        (padding[0] % 2 == 1) &&
        (kernelSize[mv::KERNEL_HEIGHT] > 1));
}

void mv::RuntimeModel::getWorkloadPadding(Control::OpListIterator opIt, Workload &workload)
{
    if (opIt->get<std::string>("taskOp") == "Add" || opIt->get<std::string>("taskOp") == "Subtract" || opIt->get<std::string>("taskOp") == "Multiply")
    {
        workload.padLeft = 0;
        workload.padTop = 0;
        workload.padRight = 0;
        workload.padBottom = 0;
    }
    else
    {
        auto padding = opIt->get<std::array<unsigned short, 4>>("padding");
        auto outputWidth = opIt->getOutputTensor(0)->getShape()[mv::IO_WIDTH_DIMENSION];
        auto outputHeight = opIt->getOutputTensor(0)->getShape()[mv::IO_HEIGHT_DIMENSION];
        if (hardwareBugDepthwise(opIt))
        {
            workload.padLeft = (workload.MinX == 0) ? padding[0] : 0;
            workload.padTop = (workload.MinY == 0) ? padding[2] : 0;
            workload.padRight = ((workload.MaxX + 1) == outputWidth) ? padding[1] : 0;
            workload.padBottom = ((workload.MaxY + 1) == outputHeight) ? padding[3] : 0;
        }
        else
        {
            workload.padLeft = padding[0];
            workload.padTop = padding[2];
            workload.padRight = padding[1];
            workload.padBottom = padding[3];
        }
    }
    return;
}

std::unique_ptr<MVCNN::NCEVariantFieldsT> mv::RuntimeModel::buildNCEVariantFieldsT(ComputationModel& , mv::Element &compilationDescriptor, Control::OpListIterator opIt, Workload workload)
{
    std::unique_ptr<MVCNN::NCEVariantFieldsT> toBuild = std::unique_ptr<MVCNN::NCEVariantFieldsT>(new MVCNN::NCEVariantFieldsT());

    toBuild->mpe_mode = convertMPEMode(workload.MPEMode);
    toBuild->workload_start_X = workload.MinX;
    toBuild->workload_start_Y = workload.MinY;
    toBuild->workload_start_Z = workload.MinZ;
    toBuild->workload_end_X = workload.MaxX;
    toBuild->workload_end_Y = workload.MaxY;
    toBuild->workload_end_Z = workload.MaxZ;
    //Padding should be computed for every cluster
    getWorkloadPadding(opIt, workload);
    toBuild->padLeft = workload.padLeft;
    toBuild->padRight = workload.padRight;
    toBuild->padTop = workload.padTop;
    toBuild->padBottom = workload.padBottom;
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

std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildNCE2TaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt)
{
    std::vector<std::unique_ptr<MVCNN::TaskT>> toReturn = std::vector<std::unique_ptr<MVCNN::TaskT>>(1);

    toReturn[0] = std::unique_ptr<MVCNN::TaskT>(new MVCNN::TaskT());
    toReturn[0]->task.type = MVCNN::SpecificTask_NCE2Task;
    auto toBuild = new MVCNN::NCE2TaskT();
    toBuild->variant = buildNCEVariantFieldsTVector(cm, compilationDescriptor, opIt);
    toBuild->invariant = buildNCEInvariantFieldsT(cm, compilationDescriptor, opIt);

    auto hash = [](const MVCNN::MPE_Mode &g){ return static_cast<std::size_t>(g); };
    auto comp = [](const MVCNN::MPE_Mode &l, const MVCNN::MPE_Mode &r){ return l == r; };

    std::unordered_map<MVCNN::MPE_Mode, unsigned, decltype(hash), decltype(comp)> frequencyCounter(4, hash, comp);
    for(auto& variantField : toBuild->variant)
        ++frequencyCounter[variantField->mpe_mode];

    unsigned maxFrequency = 0;
    for(auto& frequencyCouple : frequencyCounter)
        if(frequencyCouple.second > maxFrequency)
            toBuild->invariant->mpe_frequent_mode = frequencyCouple.first;

    toReturn[0]->task.value = toBuild;
    return toReturn;
}

std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildNNTensorTaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt)
{

}

std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildControllerTaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt)
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

std::unique_ptr<MVCNN::BarrierReferenceT> mv::RuntimeModel::buildBarrierReferenceT()
{
    std::unique_ptr<MVCNN::BarrierReferenceT> toBuild = std::unique_ptr<MVCNN::BarrierReferenceT>(new MVCNN::BarrierReferenceT());
    return toBuild;
}

std::vector<std::unique_ptr<MVCNN::TaskT>> mv::RuntimeModel::buildTaskT(ComputationModel& cm, mv::Element &compilationDescriptor, Control::OpListIterator opIt, int& nodeID)
{

    std::vector<std::unique_ptr<MVCNN::TaskT>> vecToBuild = buildSpecificTaskUnion(cm, compilationDescriptor, opIt, nodeID);

    for(auto& toBuild: vecToBuild)
    {
        toBuild->nodeID = nodeID++;
        toBuild->name = opIt->getName();

        // NOTE: This might change in the future
        if(opIt->hasAttr("opId"))
            toBuild->sourceTaskIDs = {opIt->get<unsigned>("opId")};

        if(opIt->hasAttr("BarrierDeps"))
            toBuild->associated_barriers = buildBarrierReferenceT(cm, compilationDescriptor, opIt->get<mv::BarrierDependencies>("BarrierDeps"));
        else
            toBuild->associated_barriers = buildBarrierReferenceT();
    }

    return vecToBuild;
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
        [](const mv::Data::OpListIterator& a, const mv::Data::OpListIterator& b) -> bool { return a->get<mv::Barrier>("Barrier").getIndex() < b->get<mv::Barrier>("Barrier").getIndex(); }
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

    auto globalConfigurationParameters = cm.getGlobalConfigParams();

    graphFile_.header = buildSummaryHeaderT(cm, compilationDescriptor, std::move(graphFile_.header));

    // TASKS
    graphFile_.task_lists = buildTaskListT(cm, compilationDescriptor);

    // Barrier Table must be build only on dynamic scheduling
    if(globalConfigurationParameters->get<std::string>("barrier_index_assignment") == "Dynamic")
        graphFile_.barrier_table = buildBarrierTable(cm, compilationDescriptor);

    // Binary Data
    graphFile_.binary_data = std::vector<std::unique_ptr<MVCNN::BinaryDataT>>();
    for(auto opIterator = om.opBegin(); opIterator != om.opEnd(); ++opIterator)
    {
        std::string opType = opIterator->getOpType();
        if (opType == "Constant" || opType == "ConstantInt" || opType == "ConstantDataElement" || opType == "WeightsTable" || opType == "SparsityMap")
        {
            auto tIt = opIterator->getOutputTensor(0);
            //std::cout << "Serializing to binary data section " << tensorIt->getName() << std::endl;
            graphFile_.binary_data.push_back(buildBinaryDataT(cm, compilationDescriptor, *tIt));
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

void mv::RuntimeModel::serialize(const std::string& filename)
{
    int bufferSize;
    char * dataBuffer = serialize(bufferSize);
    if (flatbuffers::SaveFile((filename).c_str(), dataBuffer, bufferSize, true))
        Logger::log(mv::Logger::MessageType::Info, "RuntimeModel", "File successfully written to: " + filename);
    else 
        Logger::log(mv::Logger::MessageType::Error, "RuntimeModel", "File was not created. Check configuration");
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
