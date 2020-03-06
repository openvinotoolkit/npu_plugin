#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void strategyLayersToTensors(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

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

void strategyLayersToTensors(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element &)
{
    auto globalParams = model.getGlobalConfigParams();
    mv::OpModel om(model);
    mv::DataModel dm(model);

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

            if(taskOp == "Eltwise")
                startingIndex = 2;

            for(unsigned i = startingIndex; i < n; ++i)
            {
                auto inputTensor = layer->getInputTensor(i);
                inputTensor->set<std::string>("splitStrategy", opStrategy);
                if(inputTensor->isSparse())
                    inputTensor->getSparsityMap()->set<std::string>("splitStrategy", opStrategy);
            }
        }
        else if (opType == "Input" || opType == "Crop" || opType == "UPATask")
        {
            auto opStrategy = layer->get<std::string>("splitStrategy");
            auto outputTensor = layer->getOutputTensor(0);
            outputTensor->set<std::string>("splitStrategy", opStrategy);
        }
    }
    // ASSUMPTION: All the input tensors of a concat share the same
    // splitting strategy
    //NOTE: Concats of concats need to start from the inner nested part
    auto implicitConcatOps = om.getOps("ImplicitConcat");
    for (auto implicitConcat : implicitConcatOps)
    {
        if (implicitConcat->getInputTensor(0)->hasAttr("splitStrategy"))
            implicitConcat->getOutputTensor(0)->set<std::string>("splitStrategy",
                                                    implicitConcat->getInputTensor(0)->get<std::string>("splitStrategy"));
    }
    for(auto layer = om.opBegin(); layer != om.opEnd(); ++layer)
    {
        std::string opType = layer->getOpType();
        if (opType == "ImplicitConcat" || opType == "ImplicitReshape" || opType == "Concat")
        {
            auto opStrategy = layer->getInputTensor(0)->get<std::string>("splitStrategy");
            auto outputTensor = layer->getOutputTensor(0);
            outputTensor->set<std::string>("splitStrategy", opStrategy);
        }
    }

    // Usually Align op takes the strategy from the previous op
    // when the previous op to align has no strategy defined, align takes the strategy from sinkOp of align 
    for(auto layer = om.opBegin(); layer != om.opEnd(); ++layer)
    {
        std::string opType = layer->getOpType();
        if (opType == "Align")
        {
           if (layer->hasAttr("splitStrategy")) 
           {
               auto opStrategy = layer->get<std::string>("splitStrategy");
               auto outputTensor = layer->getOutputTensor(0);
               outputTensor->set<std::string>("splitStrategy", opStrategy);
           }
           else
           {
               auto outputTensor = layer->getOutputTensor(0);
               std::vector<mv::Data::OpListIterator> sinkOperators = findSinkLayers(dm, outputTensor);
               auto opStrategy = sinkOperators[0]->get<std::string>("splitStrategy");
               outputTensor->set<std::string>("splitStrategy", opStrategy);
               layer->set<std::string>("splitStrategy", opStrategy);
           }
        }
    }

    for(auto layer = om.opBegin(); layer != om.opEnd(); ++layer)
    {
        std::string opType = layer->getOpType();
        if (opType == "Slice")
        {
            auto outputTensor = layer->getOutputTensor(0);
            std::vector<mv::Data::OpListIterator> sinkOperators = findSinkLayers(dm, outputTensor);
            auto opStrategy = sinkOperators[0]->get<std::string>("splitStrategy");
            outputTensor->set<std::string>("splitStrategy", opStrategy);
        }
    }

    return;
}
