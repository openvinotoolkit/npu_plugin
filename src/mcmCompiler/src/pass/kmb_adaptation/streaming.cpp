#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/pass/graphOptimizations/streaming_performace.hpp"

void streamingForPerformanceFnc(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&,
                                mv::Element&, mv::Element&);

namespace mv {
namespace pass {
MV_REGISTER_PASS(StreamingForPerformance).setFunc(streamingForPerformanceFnc).setDescription("");
}
} 

void streamingForPerformanceFnc(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&,
                                mv::Element& passDesc, mv::Element&) {
    mv::OpModel om(model);
    mv::StreamingPerformance streamingPerformance(model, om);

    streamingPerformance.increaseStreamingOverKforPerformance();
}
