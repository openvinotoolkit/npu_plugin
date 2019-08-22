#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/keembay/ppe_task.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"


static void setDpuTasksMemoryLocationFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void solveDDRSplitStrategiesFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static std::vector<mv::Data::OpListIterator> findSinkLayers(mv::DataModel &dataModel, const mv::Data::TensorIterator &tensor);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(SetDpuTasksMemoryLocation)
            .setFunc(setDpuTasksMemoryLocationFcn)
            .setDescription(
                "Set Dpu Task memory location and adds copy ops if needed");

        MV_REGISTER_PASS(SolveDDRSplitStrategies)
            .setFunc(solveDDRSplitStrategiesFcn)
            .setDescription(
                "Solve Tensor Strategies which are coming back from DDR");
    }
}

void setDpuTasksMemoryLocationFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto opIt = om.getInput();
    while (opIt != om.opEnd())
    {
        std::string opType = opIt->getOpType();

        if (opType == "DPUTask")
        {
            auto taskOp = opIt->get<std::string>("taskOp");
            bool isElementWise = (taskOp == "Add" || taskOp == "Subtract" || taskOp == "Multiply");

            if (taskOp == "ChannelMajorConvolution" ||
                taskOp == "DepthwiseConv"  ||
                taskOp == "MaxPool" || taskOp == "Conv" || isElementWise)
            {
                auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
                auto inputMemoryLocation = opIt->getInputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

                if(outputMemoryLocation != mv::Tensor::MemoryLocation::CMX)
                {
                    auto output = opIt->getOutputTensor(0);
                    auto outputDataFlows = mv::getOutputDataFlow(om, opIt, false);

                    std::vector<mv::Data::FlowListIterator> flows;
                    for(auto outputFlow = opIt.leftmostOutput(); outputFlow != om.flowEnd(); ++outputFlow)
                        flows.push_back(outputFlow);

                    for (auto flow : flows)
                        om.undefineFlow(flow);

                    output->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::CMX);

                    mv::QuantizationParams outputQuantParams = {{},{},{},{}};
                    if (output->hasAttr("quantParams"))
                        outputQuantParams = output->get<mv::QuantizationParams>("quantParams");
                    auto dpuCopyOut = om.dMATask(output, mv::DmaDirectionEnum::CMX2DDR,opIt->getName() + "_copyOut");
                    auto dpuCopyOutOp = om.getSourceOp(dpuCopyOut);
                    dpuCopyOutOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));
                    if (output->hasAttr("quantParams"))
                        dpuCopyOutOp->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams").quantize(outputQuantParams.getShift(), outputQuantParams.getMult());

                    setOutputDataFlow(om, dpuCopyOut, outputDataFlows);
                    dpuCopyOut->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
                }

                if(inputMemoryLocation != mv::Tensor::MemoryLocation::CMX)
                {
                    size_t numInputs = 1;
                    if (isElementWise)
                        numInputs++;
                    for (auto i = 0; i < numInputs; i++)
                    {
                        auto input = opIt->getInputTensor(i);
                        mv::QuantizationParams inputQuantParams = {{},{},{},{}};
                        if(input->hasAttr("quantParams"))
                            inputQuantParams = input->get<mv::QuantizationParams>("quantParams");
                        auto dpuCopyIn = om.dMATask(input, mv::DmaDirectionEnum::DDR2CMX, opIt->getName() + "_copyIn_" + std::to_string(i));
                        auto dpuCopyInOp = om.getSourceOp(dpuCopyIn);

                        if(dpuCopyInOp->getOutputTensor(0)->hasAttr("quantParams"))
                            dpuCopyInOp->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams").quantize(inputQuantParams.getShift(), inputQuantParams.getMult());

                        dpuCopyInOp->set<unsigned>("opId", opIt->get<unsigned>("opId"));

                        auto flows = input->get<std::set<std::string>>("flows");

                        for(auto flowStr: flows)
                        {
                            auto backupFlow = dm.getDataFlow(flowStr);
                            auto idx = backupFlow->get<std::size_t>("sinkInput");
                            if (backupFlow.sink()->getName() == opIt->getName())
                            {
                                auto sink = backupFlow.sink();
                                om.undefineFlow(backupFlow);
                                sink->setInputTensor(dpuCopyIn, idx, false);
                                om.defineFlow(dpuCopyInOp, 0, sink, idx);
                                break;
                            }
                        }

                        dpuCopyIn->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::CMX);
                    }
                }
            }

        }
        ++opIt;
    }
}



void solveDDRSplitStrategiesFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto opIt = om.getInput();
    while (opIt != om.opEnd())
    {
        std::string opType = opIt->getOpType();

        if (opType == "DMATask")
        {
            auto dstTensor = opIt->getOutputTensor(0);
            if (dstTensor->get<mv::Tensor::MemoryLocation>("Location") == mv::Tensor::MemoryLocation::DDR)
            {
                auto nextOp = findSinkLayers(dm, opIt->getOutputTensor(0));
                if (nextOp[0]->getOpType() == "DMATask")
                {
                    if (nextOp[0]->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location") == mv::Tensor::MemoryLocation::CMX)
                    {
                        auto afterNextOp = findSinkLayers(dm, nextOp[0]->getOutputTensor(0));
                        opIt->getOutputTensor(0)->set<std::string>("splitStrategy", afterNextOp[0]->get<std::string>("splitStrategy"));
                        nextOp[0]->getOutputTensor(0)->set<std::string>("splitStrategy", afterNextOp[0]->get<std::string>("splitStrategy"));
                        afterNextOp[0]->getInputTensor(0)->set<std::string>("splitStrategy", afterNextOp[0]->get<std::string>("splitStrategy"));
                    }
                }
            }
        }
        ++opIt;

    }
}

static std::vector<mv::Data::OpListIterator> findSinkLayers(mv::DataModel &dataModel, const mv::Data::TensorIterator &tensor)
{
    std::vector<mv::Data::OpListIterator> sinkOperations;
    auto flowsNames = (tensor)->get<std::set<std::string>>("flows");
    for(auto flowName : flowsNames)
    {
        auto df = dataModel.getDataFlow(flowName);
        sinkOperations.push_back(df.sink());
    }
    return sinkOperations;
}
