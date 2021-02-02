#include "include/mcm/base/exception/runtime_error.hpp"
#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/pass_registry.hpp"

static void MemContextForHugeActivations(
    const mv::pass::PassEntry& , mv::ComputationModel&, mv::TargetDescriptor&,
      mv::Element&, mv::Element&);

namespace mv {
  namespace pass {
    MV_REGISTER_PASS(MemContextForHugeActivations)
      .setFunc(MemContextForHugeActivations)
      .setDescription("Create Memory Context For Huge Activations.");
  } // namespace mv //
} // namespace pass //



void MemContextForHugeActivations(const mv::pass::PassEntry&,
    mv::ComputationModel& model, mv::TargetDescriptor& , mv::Element& /*passDesc*/,
    mv::Element&) {
  {
    // Quick memory context implementation //
    // 
    // STEP-1: move resources around:
    // from : (mem_ctx_child)
    // "mobilenet_v1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/batchnorm/add_1
    // size=247808
    //
    // to : (mem_ctx_rep)
    // "mobilenet_v1/MobilenetV1/Conv2d_1_depthwise/BiasAdd/Add"
    //  size=495616 
    //
    // STEP-2: add control edges between the rep and children for mem_ctx_child
    //
    //
    mv::OpModel omodel(model);
    try {

      mv::Data::OpListIterator mem_ctx_rep_op = omodel.getOp(
          "mobilenet_v1/MobilenetV1/Conv2d_0/BatchNorm/batchnorm/add_1");
      mv::Data::OpListIterator mem_ctx_left_valley_op = omodel.getOp(
          "mobilenet_v1/MobilenetV1/Conv2d_1_depthwise/BiasAdd/Add");
      mv::Data::OpListIterator mem_ctx_peak_op = omodel.getOp(
         "mobilenet_v1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/batchnorm/add_1"
         );
      mv::Data::OpListIterator mem_ctx_right_valley_op = omodel.getOp(
          "mobilenet_v1/MobilenetV1/Conv2d_2_depthwise/BiasAdd/Add");

      // Precondition //
      size_t peak_size =
        mem_ctx_peak_op->getOutputTensor(0UL)->getClusterSize();
      size_t left_valley_size =
        mem_ctx_left_valley_op->getOutputTensor(0UL)->getClusterSize();
      size_t right_valley_size =
        mem_ctx_right_valley_op->getOutputTensor(0UL)->getClusterSize();
      size_t rep_orig_size =
        mem_ctx_rep_op->getOutputTensor(0UL)->getClusterSize();

      if ((left_valley_size >= right_valley_size) &&
          (left_valley_size == rep_orig_size)) {

        printf("[PeakValleyCondition] : passed\n");
        printf("[peak = %lu left_valley=%lu right_valley=%lu\n", peak_size,
            left_valley_size, right_valley_size);
        printf("[mem_context_utility=%lu]\n", (peak_size + left_valley_size));

        mem_ctx_rep_op->set<size_t>("memory_context_utility",
            peak_size + left_valley_size);
        mem_ctx_rep_op->set<size_t>("memory_context_offset", left_valley_size);
        mem_ctx_rep_op->set<std::string>("memory_context_rep",
              mem_ctx_rep_op->getName());
        mem_ctx_rep_op->set<size_t>("actual_size_inside_memory_context",
              rep_orig_size);

        mem_ctx_left_valley_op->set<size_t>("memory_context_utility", 0UL);
        mem_ctx_left_valley_op->set<size_t>("memory_context_offset", 0);
        mem_ctx_left_valley_op->set<std::string>("memory_context_rep",
              mem_ctx_rep_op->getName());
        mem_ctx_left_valley_op->set<size_t>("actual_size_inside_memory_context",
              left_valley_size);

        mem_ctx_peak_op->set<size_t>("memory_context_utility", 0UL);
        mem_ctx_peak_op->set<size_t>("memory_context_offset", left_valley_size);
        mem_ctx_peak_op->set<std::string>("memory_context_rep",
              mem_ctx_rep_op->getName());
        mem_ctx_peak_op->set<size_t>("actual_size_inside_memory_context",
              peak_size);

        // add resource control edges //
        // rep to peak and right_valley//
        mv::Data::FlowListIterator fitr1 = omodel.defineFlow(
            mem_ctx_rep_op->getOutputTensor(0UL), mem_ctx_right_valley_op, 0UL);
        fitr1->set<bool>("resource_control_flow", true);

        mv::Data::FlowListIterator fitr2 = omodel.defineFlow(
            mem_ctx_rep_op->getOutputTensor(0UL), mem_ctx_peak_op, 0UL);
        fitr2->set<bool>("resource_control_flow", true);

        printf("[ResourceControl] (%s,%s)\n", mem_ctx_rep_op->getName().c_str(),
              mem_ctx_right_valley_op->getName().c_str());

      } else {
        printf("[PeakValleyCondition]: failed\n");
      }
    } catch (mv::ArgumentError err) {
      // ignore argument error //
    }
  }
}
