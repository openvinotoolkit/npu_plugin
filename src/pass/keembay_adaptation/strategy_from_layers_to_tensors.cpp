#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void strategyLayersToTensors(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(StrategyLayersToTensors)
        .setFunc(strategyLayersToTensors)
        .setDescription(
            "Extend strategies from ops to tensors"
        );
    }
}

void strategyLayersToTensors(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    auto globalParams = model.getGlobalConfigParams();
    unsigned numClusters = globalParams->get<int>("Number_of_Clusters");

    mv::OpModel om(model);
    //One cluster means no need for splitting
    if (numClusters > 1)
    {
        for(auto layer = om.opBegin(); layer != om.opEnd(); ++layer)
        {
            std::string opType = layer->getOpType();
            if (opType == "DPUTask")
            {
                auto opStrategy = layer->get<std::string>("splitStrategy");
                auto outputTensor = layer->getOutputTensor(0);
                outputTensor->set<std::string>("splitStrategy", opStrategy);
                unsigned n = layer->inputSlots();

                // Starting from 1 because input 0 gets the strategy from the output tensor of the
                // Previous operation

                // We need to handle a special case for element wise (as usual)
                // Since it has two inputs

                unsigned startingIndex = 1;
                auto taskOp = layer->get<std::string>("taskOp");

                if(taskOp == "Add" || taskOp == "Multiply" || taskOp == "Subtract")
                    startingIndex = 2;

                for(unsigned i = startingIndex; i < n; ++i)
                {
                    auto inputTensor = layer->getInputTensor(i);
                    inputTensor->set<std::string>("splitStrategy", opStrategy);
                }
            }
        }
        for(auto layer = om.opBegin(); layer != om.opEnd(); ++layer)
        {
            std::string opType = layer->getOpType();
            if (opType == "Input")
            {
                auto opStrategy = layer->get<std::string>("splitStrategy");
                auto outputTensor = layer->getOutputTensor(0);
                outputTensor->set<std::string>("splitStrategy", opStrategy);
            }
        }
        for(auto layer = om.opBegin(); layer != om.opEnd(); ++layer)
        {
            std::string opType = layer->getOpType();
            if (opType == "DMATask")
            {
                auto outputTensor = layer->getOutputTensor(0);
                if (outputTensor->hasAttr("splitStrategy"))
                {
                    auto outputTensorStrategy = outputTensor->get<std::string>("splitStrategy");
                    auto inputTensor = layer->getInputTensor(0);
                    inputTensor->set<std::string>("splitStrategy", outputTensorStrategy);
                }
                //last DMA NNCMX2DDR for Output
                else
                {
                    auto inputTensor = layer->getInputTensor(0);
                    auto inputTensorStrategy = inputTensor->get<std::string>("splitStrategy");
                    outputTensor->set<std::string>("splitStrategy", inputTensorStrategy);
                }
            }
        }
    }
    return;
}
