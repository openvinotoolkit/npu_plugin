#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/keembay/barrier_definition.hpp"
#include "include/mcm/target/keembay/barrier_deps.hpp"

static void addBarrierRefsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(AddBarrierRefs)
        .setFunc(addBarrierRefsFcn)
        .setDescription(
            "This pass adds barrier dependencies to DxxTasks in the compute graph, essential to scheduling different tasks"
        );
    }
}

void addBarrierRefsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto barrierTasks = om.getOps("BarrierTask");

    for (auto bt: barrierTasks)
    {
        std::cout << "bt: " << bt->getName() << std::endl;
        auto& barrier = bt->get<mv::Barrier>("Barrier");

        for (auto producer: barrier.getProducers())
        {
            std::cout << "prod: " << producer << std::endl;
            auto producerOp = om.getOp(producer);

            if (!producerOp->hasAttr("BarrierDeps"))
            {                
                struct mv::BarrierDependencies bdep;
                producerOp->set<mv::BarrierDependencies>("BarrierDeps", bdep);
            }

            auto& barrierRef = producerOp->get<mv::BarrierDependencies>("BarrierDeps");
            barrierRef.addUpdateBarrier(barrier.getIndex());
        }
        
        for (auto consumer: barrier.getConsumers())
        {
            std::cout << "cons: " << consumer << std::endl;
            auto consumerOp = om.getOp(consumer);
            if (!consumerOp->hasAttr("BarrierDeps"))
            {                
                struct mv::BarrierDependencies bdep;
                consumerOp->set<mv::BarrierDependencies>("BarrierDeps", bdep);
            }

            auto& barrierRef = consumerOp->get<mv::BarrierDependencies>("BarrierDeps");
            // Hmm...this won't work always -- there can be several consumers trying to update a single
            // barrier -- you should wait for the LAST one of them to be done.
            // TODO: Replace with better algorithm for inserting barriers.
            barrierRef.setWaitBarrier(barrier.getIndex());
        }
    }
}