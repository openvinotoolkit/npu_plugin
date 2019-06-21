#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include <iostream>

static void scheduleHelperPass(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(ScheduleHelper)
        .setFunc(scheduleHelperPass)
        .setDescription(
            "Add specific edges for partial serilization"
        );
    }

}


void scheduleHelperPass(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor& target, mv::Element&, mv::json::Object&)
{
    
    mv::OpModel om(model);
    mv::ControlModel cm(model);
    
    auto globalParams = model.getGlobalConfigParams();
    if (!globalParams->hasAttr("schedule_helper_edges"))
    {
        pass.log(mv::Logger::MessageType::Info, "No schedule helper edges provided");
        return;
    }

    auto edgesList = globalParams->get<std::vector<mv::Element>>("schedule_helper_edges");
    for (auto e : edgesList)
    {
        std::string& source = e.get<std::string>("edge_source");
        std::string& sink = e.get<std::string>("edge_sink");
        pass.log(mv::Logger::MessageType::Debug, "SCHEDULE HELPER adding edge from "+source+" to "+sink);
        auto sourceOp = om.getOp(source);
        auto sinkOp = om.getOp(sink);
        cm.defineFlow(sourceOp, sinkOp);  
    }
}
