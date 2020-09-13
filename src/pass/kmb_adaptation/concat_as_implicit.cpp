#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void concatAsImplicitFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void decideConcatLocationFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(ConcatAsImplicit)
            .setFunc(concatAsImplicitFcn)
            .setDescription(
                "Replaces all concats with implicits concats");

        MV_REGISTER_PASS(DecideConcatLocation)
            .setFunc(decideConcatLocationFcn)
            .setDescription(
                "The idea of this pass is the following: the g.o logic currently does not handle explicit concat cases\
                 for splitting/spilling strategies, but there are some concats that MUST not be executed in cmx later from\
                 the scheduler cause of incompatible split strategies, let's mark them(TODO: handle these cases in g.o.\
                 by using and the cmx concat g.o. branch...)");
    }
}

void concatAsImplicitFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    auto concats = om.getOps("Concat");

    for(auto& concat: concats)
    {
        auto inputs = concat->getInputTensor();
        auto axis = concat->get<std::string>("axis");
        auto name = concat->getName();
        mv::QuantizationParams quantParams = {{}, {}, {}, {}};
        std::string splitStrategy;
        bool pipelined = false;
        unsigned pipelineId;
        bool cmxConcatenation = false;
        bool avoidCmxConcatenation = false;
        bool mixedToFloat = false;
        if(concat->hasAttr("splitStrategy"))
            splitStrategy = concat->get<std::string>("splitStrategy");
        if(concat->hasAttr("quantParams"))
            quantParams = concat->get<mv::QuantizationParams>("quantParams");
        if(concat->hasAttr("schedule_for_dpu_dma_overlap"))
        {
            pipelined = true;
            pipelineId = concat->get<unsigned>("schedule_for_dpu_dma_overlap");
        }
        if(concat->hasAttr("cmxConcatenation"))
            cmxConcatenation = concat->get<bool>("cmxConcatenation");
        if(concat->hasAttr("avoid_cmx_concat"))
            avoidCmxConcatenation = concat->get<bool>("avoid_cmx_concat");
        if(concat->hasAttr("mixedToFloat"))
            mixedToFloat = concat->get<bool>("mixedToFloat");

        auto outputLocation = concat->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
        auto opId = concat->get<unsigned>("opId");
        auto outputFlows = mv::getOutputDataFlow(om, concat);
        auto implicitConcat = om.implicitConcat(inputs, axis, quantParams, name);
        om.getSourceOp(implicitConcat)->set<unsigned>("opId", opId);
        implicitConcat->set<mv::Tensor::MemoryLocation>("Location", outputLocation);
        if(!splitStrategy.empty())
            om.getSourceOp(implicitConcat)->set<std::string>("splitStrategy", splitStrategy);
        if(pipelined)
            om.getSourceOp(implicitConcat)->set<unsigned>("schedule_for_dpu_dma_overlap", pipelineId);
        if (cmxConcatenation)
            om.getSourceOp(implicitConcat)->set<bool>("cmxConcatenation", cmxConcatenation);
        if(avoidCmxConcatenation)
            om.getSourceOp(implicitConcat)->set<bool>("avoid_cmx_concat", avoidCmxConcatenation);
        mv::setOutputDataFlow(om, implicitConcat, outputFlows);
        if(mixedToFloat)
        {
            implicitConcat->set<mv::DType>("dType",  mv::DType("Float16"));
            om.getSourceOp(implicitConcat)->set<bool>("mixedToFloat", mixedToFloat);
        }
    }
}

void decideConcatLocationFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);

    auto concats = om.getOps("Concat");

    std::vector<std::pair<std::string, std::string>>incompatibleStrategiesWithOutSpilling =
    {
        {"SplitOverHOverlapped", "Clustering"},
        {"SplitOverHOverlapped", "SplitOverK"},
        {"SplitOverH", "Clustering"},
        {"SplitOverH", "SplitOverK"},
        {"SplitOverK", "SplitOverH"},
        {"SplitOverK", "HKSwitch"},
        {"HKSwitch", "SplitOverH"},
        {"HKSwitch", "HKSwitch"}
    };


    for(auto& concat: concats)
    {

        std::vector<mv::Data::OpListIterator> sinkOperators = findSinkLayers(dm, concat->getOutputTensor()[0]);
        //NOTE: last operation is concat
        if (sinkOperators.empty())
            continue;
        for (auto& inputTensor : concat->getInputTensor())
        {
            auto inputOperation = om.getSourceOp(inputTensor);
            for (auto& sinkOperator : sinkOperators)
            {
                if (inputOperation->hasAttr("splitStrategy") && sinkOperator->hasAttr("splitStrategy"))
                {
                    std::pair<std::string, std::string> possibleCombination(inputOperation->get<std::string>("splitStrategy"),
                                                                            sinkOperator->get<std::string>("splitStrategy"));
                     if (std::find(incompatibleStrategiesWithOutSpilling.begin(), incompatibleStrategiesWithOutSpilling.end(),
                                   possibleCombination) != incompatibleStrategiesWithOutSpilling.end())
                    {
                        concat->set<bool>("avoid_cmx_concat", true);
                        continue;
                    }
                }
            }
        }
    }
}
