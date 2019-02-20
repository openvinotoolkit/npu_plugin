#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"

static void GenerateSparsityMapsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void GenerateWeightsTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(GenerateSparsityMaps)
        .setFunc(GenerateSparsityMapsFcn)
        .setDescription(
            "Generates sparsity maps for the Tasks that need them"
        );

        MV_REGISTER_PASS(GenerateWeightsTables)
        .setFunc(GenerateWeightsTablesFcn)
        .setDescription(
            "Generates weights tables for the Tasks that need them"
        );
    }
}

mv::Data::TensorIterator addSparsityMap(mv::OpModel om, mv::Data::OpListIterator dpuTaskOp, const std::string& sparsityMapName, unsigned tensorSize)
{
    // TensorSize is not actually needed. Sparsity map has to be added based either on
    // 1) Pattern generation given by cheat sheet
    // 2) Information given by the user.

    // This method is just a STUB to obtain a proper graph
    std::vector<double> sparsityMapData(tensorSize, 1);
    auto sparsityMap = om.constant(sparsityMapData, {tensorSize}, mv::DType("UInt32"), mv::Order("W"), sparsityMapName);
    om.getSourceOp(sparsityMap)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
    sparsityMap = om.dMATask(sparsityMap, mv::DmaDirectionEnum::DDR2CMX);
    om.getSourceOp(sparsityMap)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
    dpuTaskOp->addInputTensor(sparsityMap);

    // Weight tensor packing should be done here

    return sparsityMap;
}

mv::Data::TensorIterator addWeightsTable(mv::OpModel om, mv::Data::OpListIterator dpuTaskOp, const std::string& kernelWeightsTableName, unsigned outputChannels)
{
    std::vector<double> weightTableData(4 * outputChannels, 0);

    // WeightTableData should be filled here using packing information (and quantization information maybe?)
    for(unsigned i = 0; i < outputChannels; ++i)
    {
        weightTableData[i + 0] = 0; //DATA_PTR
        weightTableData[i + 1] = 0; //SP_PTR
    }
    auto weightTable = om.constant(weightTableData, {outputChannels, 1, 1, 4}, mv::DType("UInt32"), mv::Order("WHCN"), kernelWeightsTableName);
    om.getSourceOp(weightTable)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
    weightTable = om.dMATask(weightTable, mv::DmaDirectionEnum::DDR2CMX);
    om.getSourceOp(weightTable)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
    dpuTaskOp->addInputTensor(weightTable);

    return weightTable;
}


static void GenerateWeightsTablesFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    for(auto dpuTask = om.opBegin(); dpuTask != om.opEnd(); ++dpuTask)
    {
        if(dpuTask->getOpType() == "DPUTask")
        {
            auto opId = dpuTask->get<unsigned>("opId");
            unsigned weightsTableSize;
            std::string opName = dpuTask->getName();

            weightsTableSize = dpuTask->getOutputTensor(0)->getShape()[2];

            std::string kernelWeightsTableName(opName + "WeightsTable");
            std::string kernelWeightsTableDeallocationName("Deallocate"+kernelWeightsTableName);
            auto weightTable = addWeightsTable(om, dpuTask, kernelWeightsTableName, weightsTableSize);

            om.defineFlow(weightTable, dpuTask, dpuTask->inputSlots());
            om.deallocateTask(weightTable, kernelWeightsTableDeallocationName);
            auto dmaKernelWeightsTableFreeOp = om.getOp(kernelWeightsTableDeallocationName);

            dmaKernelWeightsTableFreeOp->set<unsigned>("opId", opId);
            cm.defineFlow(dpuTask, dmaKernelWeightsTableFreeOp);
        }
    }
}

// WARNING: This function is valid only for sparsity map relative to Weights
// (Sparsity maps can also be relative to input)
static void GenerateSparsityMapsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    // NOTE: Extra check needed for the only operation that doesn't need a sparsity map
    for(auto dpuTask = om.opBegin(); dpuTask != om.opEnd(); ++dpuTask)
    {
        if(dpuTask->getOpType() == "DPUTask")
        {
            auto opId = dpuTask->get<unsigned>("opId");
            unsigned sparsityMapSize;
            std::string opName = dpuTask->getName();

            sparsityMapSize = dpuTask->getOutputTensor(0)->getShape().totalSize();

            std::string sparsityMapName(opName + "SparsityMap");
            std::string sparsityMapDeallocationName("Deallocate"+sparsityMapName);
            auto sparsityMap = addSparsityMap(om, dpuTask, sparsityMapName, sparsityMapSize);

            om.defineFlow(sparsityMap, dpuTask, dpuTask->inputSlots());
            om.deallocateTask(sparsityMap, sparsityMapDeallocationName);
            auto dmaKernelWeightsTableFreeOp = om.getOp(sparsityMapDeallocationName);

            dmaKernelWeightsTableFreeOp->set<unsigned>("opId", opId);
            cm.defineFlow(dpuTask, dmaKernelWeightsTableFreeOp);
        }
    }
}
