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

mv::Data::TensorIterator addSparsityMap(mv::OpModel om, mv::Data::OpListIterator dpuTaskOp, const std::string& sparsityMapName, unsigned outputChannels)
{
    std::vector<double> sparsityMapData(4 * outputChannels, 0);
    auto sparsityMap = om.constant(sparsityMapData, {outputChannels, 1, 1, 4}, mv::DType("UInt32"), mv::Order("WHCN"), sparsityMapName);
    om.getSourceOp(sparsityMap)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
    sparsityMap = om.dMATask(sparsityMap, mv::DmaDirectionEnum::DDR2CMX);
    om.getSourceOp(sparsityMap)->set<unsigned>("opId", dpuTaskOp->get<unsigned>("opId"));
    dpuTaskOp->addInputTensor(sparsityMap);

    return sparsityMap;
}

mv::Data::TensorIterator addWeightsTable(mv::OpModel om, mv::Data::OpListIterator dpuTaskOp, const std::string& kernelWeightsTableName, unsigned outputChannels)
{
    std::vector<double> weightTableData(4 * outputChannels, 0);
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

static void GenerateSparsityMapsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    // NOTE: Extra check needed for the only operation that doesn't need a sparsity map Z
    for(auto dpuTask = om.opBegin(); dpuTask != om.opEnd(); ++dpuTask)
    {
        if(dpuTask->getOpType() == "DPUTask")
        {
            auto opId = dpuTask->get<unsigned>("opId");
            unsigned weightsTableSize;  // NOTE: This has to be different, probably
            std::string opName = dpuTask->getName();

            weightsTableSize = dpuTask->getOutputTensor(0)->getShape()[2];

            std::string kernelWeightsTableName(opName + "SparsityMap");
            std::string kernelWeightsTableDeallocationName("Deallocate"+kernelWeightsTableName);
            auto sparsityMap = addSparsityMap(om, dpuTask, kernelWeightsTableName, weightsTableSize);

            om.defineFlow(sparsityMap, dpuTask, dpuTask->inputSlots());
            om.deallocateTask(sparsityMap, kernelWeightsTableDeallocationName);
            auto dmaKernelWeightsTableFreeOp = om.getOp(kernelWeightsTableDeallocationName);

            dmaKernelWeightsTableFreeOp->set<unsigned>("opId", opId);
            cm.defineFlow(dpuTask, dmaKernelWeightsTableFreeOp);
        }
    }
}
