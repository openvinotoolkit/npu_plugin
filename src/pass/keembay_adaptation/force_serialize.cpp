#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void forceSerializeFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(ForceSerialize)
            .setFunc(forceSerializeFcn)
            .setDescription(
                "Serialize the DPU tasks for graphs with parallel paths");
    }
}


    struct LexicalComparator
    {
        bool operator()(mv::Data::OpListIterator lhs, mv::Data::OpListIterator rhs) const
        {
            return !(lhs->getName() < rhs->getName());
        }
    };
    struct OpItSetComparator
    {
        bool operator()(mv::Data::OpListIterator lhs, mv::Data::OpListIterator rhs) const
        {
            return (lhs->getName() < rhs->getName());
        }
    };
    // NOTE: This graph non member function works only on DAGs
    void visit(mv::Data::OpListIterator root, std::set<mv::Data::OpListIterator, OpItSetComparator>& unmarkedNodes, std::vector<mv::Data::OpListIterator>& toReturn, mv::OpModel& om)
    {
        if(unmarkedNodes.find(root) == unmarkedNodes.end())
            return;

        std::vector<mv::Data::OpListIterator> sortedNbrs;
        for(auto neighbour = root.leftmostChild(); neighbour != om.opEnd(); ++neighbour)
            sortedNbrs.push_back(neighbour);

        std::sort(sortedNbrs.begin(), sortedNbrs.end(), LexicalComparator());

        for (auto nbr: sortedNbrs)
            visit(nbr, unmarkedNodes, toReturn, om);

        unmarkedNodes.erase(root);
        toReturn.push_back(root);
    }
    std::vector<mv::Data::OpListIterator> lexTopologicalSort(mv::OpModel& om)
    {
        std::vector<mv::Data::OpListIterator> toReturn;

        std::set<mv::Data::OpListIterator, OpItSetComparator> unmarkedNodes;
        for(auto node = om.opBegin(); node != om.opEnd(); ++node)
            unmarkedNodes.insert(node);

        while(!unmarkedNodes.empty())
        {
            auto toVisit = unmarkedNodes.begin();
            visit(*toVisit, unmarkedNodes, toReturn, om);
        }

        std::reverse(toReturn.begin(), toReturn.end());
        return toReturn;
    }

void forceSerializeFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto removeNonEssentialOps = [] (std::vector<mv::Data::OpListIterator>& list)
    {
        list.erase(std::remove_if(list.begin(),
                                list.end(),
                                [](mv::Data::OpListIterator it)
                                {
                                    return (it->getOpType() != "DPUTask") && (it->getOpType() != "Input");
                                }),
                                list.end());
    };

    // This *NEEDS* to be based on the order of the Operational model
    auto sortedOps = lexTopologicalSort(om);
    removeNonEssentialOps(sortedOps);

    for (size_t i = 0; i < sortedOps.size() - 1; i++)
    {
        pass.log(mv::Logger::MessageType::Debug, " sortedOps[" +  std::to_string(i) + "] = " + sortedOps[i]->getName() );;
        if (!(cm.pathExists(cm.switchContext(sortedOps[i]), cm.switchContext(sortedOps[i+1]))
            || om.pathExists(sortedOps[i], sortedOps[i+1])
            || cm.pathExists(cm.switchContext(sortedOps[i+1]), cm.switchContext(sortedOps[i]))
            || om.pathExists(sortedOps[i+1], sortedOps[i])))
        {
            pass.log(mv::Logger::MessageType::Debug,
                "FORCE SERIALIZE: adding edge from " + sortedOps[i]->getName() + " to " +
                sortedOps[i+1]->getName());

            cm.defineFlow(sortedOps[i], sortedOps[i+1]);
        }
    }

}
