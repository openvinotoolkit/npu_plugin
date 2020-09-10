#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include <regex>

static void storeWorkloadStrategyFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(StoreWorkloadStrategy)
        .setFunc(storeWorkloadStrategyFcn)
        .setDescription(
            "This pass applies workload strategies."
        );
    }
}

void storeWorkloadStrategy(mv::Data::OpListIterator& it, int numClusters, std::vector<mv::Element>& strategyList)
{
    for (auto s: strategyList)
    {
        std::string& name_filter = s.get<std::string>("name_filter");
        std::regex exp(name_filter);
        if (std::regex_match(it->getName(), exp))
        {
                if(s.hasAttr("nWorkloads"))
                {
                    it->set<int>("WorkloadStrategy_nWorkloads", s.get<int>("nWorkloads"));
                    it->set<std::string>("WorkloadStrategy_MPE_mode", s.get<std::string>("mpe_mode"));
                }
        }
    }
}

void storeWorkloadStrategyFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    auto globalParams = model.getGlobalConfigParams();

    if (!globalParams->hasAttr("workload_strategy"))
    {
        pass.log(mv::Logger::MessageType::Debug, "No custom workload strategy provided, exiting...");
        return;
    }

    auto strategyList = globalParams->get<std::vector<mv::Element>>("workload_strategy");
    auto numClusters = globalParams->get("Number_of_Clusters");

    mv::OpModel om(model);

    for (auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        auto opType = opIt->getOpType();
        if (!(opType == "Input" || opType == "Output"))
            storeWorkloadStrategy(opIt, numClusters, strategyList);
    }

    pass.log(mv::Logger::MessageType::Warning, "----workload strategies for individual layers----");
    for (auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->hasAttr("workloadStrategy"))
        {
            pass.log(mv::Logger::MessageType::Warning, "op: " + opIt->getName() +
                        " | workload strategy = " + opIt->get<std::string>("workloadStrategy"));
        }
    }
}
