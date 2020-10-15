#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/model/iterator/tensor.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/pass_registry.hpp"



static void RemoveRedundantDMAsFromConcat(
    const mv::pass::PassEntry& , mv::ComputationModel&, mv::TargetDescriptor&,
    mv::Element&, mv::Element&);

namespace mv {
  namespace pass {

    MV_REGISTER_PASS(RemoveRedundantDMAsFromConcat)
      .setFunc(RemoveRedundantDMAsFromConcat)
      .setDescription("Remove Redundant DMAs from Concat");

  } // namespace mv //
} // namespace pass //

static void RemoveRedundantDMAsFromConcat(
    const mv::pass::PassEntry& , mv::ComputationModel& model,
    mv::TargetDescriptor&, mv::Element&, mv::Element&) {

  mv::OpModel om(model);
  for (auto op_itr=om.opBegin(); op_itr!=om.opEnd(); ++op_itr) {
    if (op_itr->getOpType() == "ImplicitConcat") {
      std::vector<std::string> dma_reads;
      for (auto citr=op_itr.leftmostChild(); citr!=om.opEnd(); ++citr) {
        if (citr->getOpType() == "DMATask") {
          dma_reads.push_back(citr->getName());
        }
      }

      if (dma_reads.size() > 1UL) {
        mv::Data::OpListIterator parent_op_itr = om.getOp(dma_reads[0]);
        size_t source_output_idx = 0UL;
        mv::Data::TensorIterator parent_output_tensor =
            parent_op_itr->getOutputTensor(source_output_idx);

        for (size_t i=1; i<dma_reads.size(); i++) {
          mv::Data::OpListIterator dma_read_itr = om.getOp(dma_reads[i]);
          for (auto child_edge_itr=dma_read_itr.leftmostOutput();
                child_edge_itr!=om.flowEnd(); ++child_edge_itr) {
            mv::Data::OpListIterator child_op_itr = child_edge_itr.sink();
            size_t sink_input_idx = child_edge_itr->get<size_t>("sinkInput");
            child_op_itr->setInputTensor(parent_output_tensor, sink_input_idx,
                  false);
            om.defineFlow(parent_op_itr, source_output_idx,
                child_op_itr, sink_input_idx);
          }
        }

        for (size_t i=1; i<dma_reads.size(); i++) {
          mv::Data::OpListIterator dma_read_itr = om.getOp(dma_reads[i]);
          om.removeOp(dma_read_itr);
        }
      }
    }
  }
}


