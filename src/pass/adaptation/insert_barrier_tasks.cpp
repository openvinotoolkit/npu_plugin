#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void insertBarrierTasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

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

void insertBarrierTasksFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);

    int numBarriers = 0 ;
    int barrierIndex = 0;
    int barrierGroup = 0;

    for(auto operationIt = om.opBegin(); operationIt != om.opEnd(); ++operationIt)
    {
        int numProducers = 0 ;
        int numConsumers = 0 ; 

        std::cout << "In InsertBarrierTasks pass: group:index= " << barrierGroup << ":" << barrierIndex << std::endl;

        // if op = conv or DMA to DDR       
        // { 
        //     create a barrier task as in example (remove from example or make a new example)
        //     add control flows as in example
 
        numBarriers++ ;
        barrierIndex =numBarriers%8 ;
        barrierGroup =numBarriers/8 ;
        // }
 
//        operationIt->set<unsigned>(currentIdLabel, currentId++);
    }
}
