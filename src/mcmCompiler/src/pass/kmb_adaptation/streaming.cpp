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

    int maxHStreams = 1000;
    if (passDesc.hasAttr("maxHStreams"))
        maxHStreams = passDesc.get<int>("maxHStreams");
    mv::StreamingPerformance streamingPerformance(om, maxHStreams);

    streamingPerformance.increaseStreamingOverKforPerformance();

    // Note: The idea of this pass is to increase streaming over the height dimension in specific cases
    // to increase performance. Specifically, we consider DPU tasks (convs, dw, maxpool) that have their
    // input tensor in DDR. Performance increase results because smaller DMA of input slices leads to
    // earlier start to compute, and the resulting smaller pieces are often easier for the scheduler
    // to schedule efficiently.
    //
    // Side Note: There are several reasons an input tensor might be in DDR, it could be the network
    // input, or a spilled activation due to tensor size or need to change clustering strategy. In this
    // pass we don't care why the tensor is in DDR, we just use the GO pass' prediction for where the
    // tensor will be located. We skip extra streams in the case that the GO can't predict tensor location
    // such as after an explicit concat (could be CMXed later). For simplicity, we also only consider ops
    // that were already streaming over H, but this pass could be extended to consider non-streaming ops.

    streamingPerformance.increaseStreamingOverHforPerformance(pass);
}
