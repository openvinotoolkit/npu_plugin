#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include <regex>

static void applyTensorSplitStrategyFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ApplyTensorSplitStrategy)
        .setFunc(applyTensorSplitStrategyFcn)
        .setDescription(
            "This pass applies tensor splitting strategies."
        );
    }
}

void applyStrategy(mv::Data::TensorIterator& it, int numClusters, std::vector<mv::Element>& strategyList)
{
    for (auto s: strategyList)
    {
        std::string& name_filter = s.get<std::string>("name_filter");
        int cluster_filter = s.get("cluster_filter");
        std::regex exp(name_filter);
        if (std::regex_match(it->getName(), exp))
        {
            if (cluster_filter == 0 || cluster_filter == numClusters)
            {
                std::cout << "tensor: " << it->getName() << " matched filter: " << name_filter << std::endl;
                it->set<std::string>("split_strategy", s.get<std::string>("strategy"));
            }
        }
    }
}

void applyTensorSplitStrategyFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();

    std::vector<mv::Element> strategyList = globalParams->get<std::vector<mv::Element>>("split_strategy");
    int numClusters = globalParams->get("Number_of_Clusters");

    std::cout << "num clusters = " << numClusters << std::endl;

    std::cout << "split strategy = " << std::endl;
    for (auto strategy: strategyList)
        std::cout << strategy.toString() << std::endl;

    mv::OpModel om(model);
    mv::DataModel dm(model);

    for (auto tensorIt = om.tensorBegin(); tensorIt != om.tensorEnd(); ++tensorIt)
    {
        std::cout << tensorIt->getName() << std::endl;
        applyStrategy(tensorIt, numClusters, strategyList);
    }

    for (auto tensorIt = om.tensorBegin(); tensorIt != om.tensorEnd(); ++tensorIt)
    {
        if (tensorIt->hasAttr("split_strategy"))
        {
            std::cout << "tensor: " << tensorIt->getName() << " | strategy = "
                        << tensorIt->get<std::string>("split_strategy") << std::endl;
        }
    }
}
