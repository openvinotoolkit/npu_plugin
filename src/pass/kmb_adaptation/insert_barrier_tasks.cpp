#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/kmb/barrier_definition.hpp"
#include "include/mcm/target/kmb/barrier_deps.hpp"
#include "include/mcm/target/kmb/workloads.hpp"
#include "include/mcm/graph/graph.hpp"
#include "include/mcm/algorithms/path_exists.hpp"
#include "include/mcm/algorithms/edge_exists.hpp"
#include <algorithm>
#include "include/mcm/logger/logger.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/env_loader.hpp"

#define MAX_AVAILABLE_BARRIERS 8

static void insertBarrierTasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(InsertBarrierTasks)
        .setFunc(insertBarrierTasksFcn)
        .setDescription(
            "This pass inserts barrier tasks into the compute graph"
        );

    }

}

static bool opHasBarrier(const std::string& opName , std::vector<mv::Barrier>& barriers)
{
    for (auto b : barriers)
    {
        auto bConsumers = b.getConsumers() ;
        if ( std::find(bConsumers.begin() , bConsumers.end(), opName ) != bConsumers.end() )
        {
            return true;
        }
    }
    return false;
}

static bool emptySetIntersection(const std::set<std::string>& s1, const std::set<std::string>& s2)
{
    auto it1 = s1.begin();
    auto it2 = s2.begin();

    while (it1 != s1.end() && it2 != s2.end())
    {
        if (*it1 == *it2)
            return false;
        else if (*it1 < *it2)
            it1++;
        else
            it2++;
    }

    return true;
}


static void combineRedundantBarriers(mv::ComputationModel& model,
                                    const mv::pass::PassEntry& pass,
                                    std::vector<mv::Barrier>& barriers)
{
    mv::ControlModel cm(model);

    for (auto b = barriers.begin(); b != barriers.end(); b++ )
    {
        for (auto c = std::next(b); c!= barriers.end(); c++ )
        {
            // combine barriers with same producers into 1 barrier
            if ((b->getProducers() == c->getProducers()) && (c->hasConsumers()) && (b->hasConsumers()))
            {
                pass.log(mv::Logger::MessageType::Debug,
                        "combining redundant barriers: " + std::to_string(b->getID())
                        + " and " + std::to_string(c->getID()));
                // move c consumers to b
                for (auto consumer : c->getConsumers())
                {
                    b->addConsumer(consumer);
                    c->removeConsumer(consumer);
                }
            }
            // combine barriers with only one consumer that happen to be the same into 1 barrier
            else if ((b->getNumConsumers() == 1)
                    && (c->getNumConsumers() == 1)
                    && (b->getConsumers() == c->getConsumers()))
            {
                pass.log(mv::Logger::MessageType::Debug,
                        " combining redundant barriers: " + std::to_string(b->getID())
                        + " and " + std::to_string(c->getID())
                        + " : they have have a single consumer and share that consumer");

                // move c's producers to b
                for (auto producer: c->getProducers())
                {
                    b->addProducer(producer);

                    // Clear c so that it can be removed from the graph
                    c->clear();
                }
            }

            // check whether c's producers are a subset of b, if so, move c's consumers to b
            auto prod_b = b->getProducers();
            auto prod_c = c->getProducers();
            auto cons_b = b->getConsumers();
            auto cons_c = c->getConsumers();
            if ((std::includes(prod_b.begin(), prod_b.end(), prod_c.begin(), prod_c.end())
                || std::includes(prod_c.begin(), prod_c.end(), prod_b.begin(), prod_b.end()))
                && b->hasConsumers() && c->hasConsumers())
            {
                if (emptySetIntersection(prod_b, cons_c) && emptySetIntersection(prod_c, cons_b))
                {
                    bool noPath = true;
                    for (auto p1 = prod_b.begin(); p1 != prod_b.end(); ++p1)
                    {
                        for (auto p2 = prod_c.begin(); p2 != prod_c.end(); ++p2)
                        {
                            if (*p1 != *p2 && (cm.pathExists(cm.switchContext(cm.getOp(*p1)), cm.switchContext(cm.getOp(*p2)))
                                            || cm.pathExists(cm.switchContext(cm.getOp(*p2)), cm.switchContext(cm.getOp(*p1)))))
                            {
                                noPath = noPath && false;
                            }
                        }
                    }

                    if (noPath)
                    {
                        pass.log(mv::Logger::MessageType::Debug,
                                "combining redundant barriers: " + std::to_string(b->getID())
                                + " and " + std::to_string(c->getID()));

                        for (auto consumer : c->getConsumers())
                        {
                            b->addConsumer(consumer);
                            c->removeConsumer(consumer);
                        }

                        for (auto producer: c->getProducers())
                        {
                            b->addProducer(producer);
                            c->removeProducer(producer);
                        }

                    }
                }
            }
        }
    }

    auto newEnd = std::remove_if(barriers.begin(), barriers.end(), [](mv::Barrier& x)
        { return !(x.hasConsumers()); } );
    barriers.erase(newEnd, barriers.end());
}

void getBarrierForControlModelOp(mv::ControlModel& cm, mv::Control::OpListIterator& opIt,
                                std::vector<mv::Barrier>& barriers)
{

    auto ctrlOpType = opIt->getOpType();
    if (ctrlOpType != "Input" && ctrlOpType != "Output")
    {
        for (auto parentOp = opIt.leftmostParent(); parentOp != cm.opEnd(); ++parentOp)
        {
            auto parentOpType = parentOp->getOpType();
            if (parentOpType != "Input")
            {
                auto sinkOpName = opIt->getName();
                auto sourceOpName = parentOp->getName();

                if (opHasBarrier(sinkOpName, barriers))
                {
                    for (mv::Barrier& b : barriers)
                    {
                        auto bConsumers = b.getConsumers();
                        auto cons = std::find(bConsumers.begin(), bConsumers.end(), sinkOpName);
                        if (cons != bConsumers.end())
                        {
                            b.addProducer(sourceOpName);
                            //auto updatedList = b.getProducers();
                        }
                    }
                }
                else
                {
                    std::set<std::string> producers;
                    std::set<std::string> consumers;
                    producers.insert(sourceOpName);
                    consumers.insert(sinkOpName);
                    struct mv::Barrier new_barrier(producers, consumers);
                    barriers.push_back(new_barrier);
                }
            }
        }
    }

}

static void addBarriers(mv::ComputationModel& model, std::vector<mv::Barrier>& barriers)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    // NOTE: This topological sort might actually be redundant since
    // barriers ID are set in a separate pass
    auto sortedCtrlOps = cm.topologicalSort();

    // Add control flow barriers, the only one needed now and forever
    for (auto ctrlOp: sortedCtrlOps)
        getBarrierForControlModelOp(cm, ctrlOp, barriers);

}

static void insertBarriersIntoControlFlowGraph(mv::ComputationModel& model, const mv::Element& passDesc, const std::vector<mv::Barrier>& barriers)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto globalConfigurationParameters = model.getGlobalConfigParams();

    std::string indexAssignment = globalConfigurationParameters->get<std::string>("barrier_index_assignment");

    if(passDesc.hasAttr("barrier_index_assignment"))
        indexAssignment = passDesc.get<std::string>("barrier_index_assignment");

    for (auto& barrier: barriers)
    {
        //Following POC convention for the moment, reversable in any moment :)
        std::string barrierName(mv::createBarrierName((*barrier.getConsumers().begin()), barrier.getID()));
        om.barrierTask(barrier, barrierName);

        // Add control flows to insert this barrier to the control flow graph
        auto barrierOp = om.getOp(barrierName);

        // Input flow
        for (auto producer: barrier.getProducers())
        {
            auto sourceOp = om.getOp(producer);
            cm.defineFlow(sourceOp, barrierOp);
        }

        // Output flow
        for (auto consumer: barrier.getConsumers())
        {
            auto destOp = om.getOp(consumer);
            cm.defineFlow(barrierOp, destOp);
        }
    }
}

void resetBarrierIDs(std::vector<mv::Barrier>& barriers)
{
    int id = 0;
    for (auto& barrier: barriers)
    {
        barrier.setID(id);
        id++;
    }
}

void removeExtraProducers(const mv::pass::PassEntry& pass,
                            mv::ComputationModel& model,
                            std::vector<mv::Barrier>& barriers)
{
    // For each barrier, examine whether a given producer is a valid one.
    // A producer is invalid if it is downstream of another producer to
    // this barrier & it has a barrier in front of it.

    mv::OpModel om(model);
    mv::ControlModel cm(model);

    for (auto& barrier: barriers)
    {
        auto producers = barrier.getProducers();
        std::vector<std::string> toRemove;
        for (auto p1: producers)
        {
            for (auto p2: producers)
            {
                if (p1 != p2)
                {
                    if (cm.pathExists(cm.switchContext(om.getOp(p1)), cm.switchContext(om.getOp(p2))))
                    {
                        pass.log(mv::Logger::MessageType::Debug,
                            "path exists between " + p1 + " and " + p2 +
                            "..., removing " + p2 + " from barrier's producer list");
                        toRemove.push_back(p1);
                    }
                }
            }
        }

        for (auto p: toRemove)
            barrier.removeProducer(p);
    }
}

static void insertBarrierTasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element& passDesc, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    std::vector<mv::Barrier> barriers;

    addBarriers(model, barriers);

    combineRedundantBarriers(model, pass, barriers);

    // remove extraneous producers
    // XXX: Q: Do any extraneous consumers need to be removed as well?
    removeExtraProducers(pass, model, barriers);

    insertBarriersIntoControlFlowGraph(model, passDesc, barriers);
}
