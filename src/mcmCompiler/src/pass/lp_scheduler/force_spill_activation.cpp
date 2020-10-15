#include "include/mcm/computation/flow/data_flow.hpp"
#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/logger/logger.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/tensor/tensor.hpp"
#include "src/pass/lp_scheduler/operation_precedence_dag.hpp"


namespace mv {
namespace lp_scheduler {

class Force_Spill_Activation {
  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef mv::model_traits<mv::OpModel> omtraits;
    typedef mv::Op* operation_t;
    ////////////////////////////////////////////////////////////////////////////


    Force_Spill_Activation(mv::OpModel& model) : model_(model) {}
  
    bool spill_at(const std::string& op_name) {
      mv::OpModel &om = model_;
      mv::Data::OpListIterator  spilled_op_itr = om.getOp(op_name);
      if (!is_op_spillable(spilled_op_itr)) { return false; }

      operation_t spilled_op = &(*spilled_op_itr);
      std::list<operation_t> spilled_op_children;
      //////////////////////////////////////////////////////////////////////////
      // STEP-0: for all the outgoing ops connected to this op determine the
      // input tensor indexes
      std::unordered_map<operation_t, size_t> input_tensor_index_map;
      {
        std::vector<mv::Data::FlowListIterator> flows;
        for(auto outputFlow = spilled_op_itr.leftmostOutput();
          outputFlow != om.flowEnd(); ++outputFlow) {
          operation_t sink_op = &(*(outputFlow.sink()));
          spilled_op_children.push_back(sink_op);
          flows.push_back(outputFlow);
          size_t idx = outputFlow->get<size_t>("sinkInput");
          input_tensor_index_map[sink_op] = idx;
        }

        for (auto flow : flows) om.undefineFlow(flow);
      }
      //////////////////////////////////////////////////////////////////////////
      // STEP-1: create one DMA write op //
      std::string dma_op_name = spilled_op->getName() + "_forceSpilledWrite";
      mv::DmaDirection write_dma_direction(std::string("NNCMX2DDR"));
      mv::Data::TensorIterator spilled_op_output_tensor_itr =
          spilled_op->getOutputTensor(0UL);
      mv::Data::TensorIterator spill_write_tensor_itr = om.dMATask(
          spilled_op_output_tensor_itr, write_dma_direction, 0, dma_op_name);
      Data::OpListIterator write_op_itr =
          om.getSourceOp(spill_write_tensor_itr);
      // set a dummy flows attribute //
      {
        std::set<std::string> toSet;
        spill_write_tensor_itr->set<std::set<std::string>>("flows", toSet);
      }
      write_op_itr->setInputTensor(spilled_op_output_tensor_itr, 0UL, false);

      //////////////////////////////////////////////////////////////////////////
      // STEP-4: create a new spill read ops by connecting spill_write tensor
      // to each of them as inputs.
      size_t read_index = 0UL;
      mv::DmaDirection read_dma_direction(std::string("DDR2NNCMX"));
      dma_op_name =
        spilled_op->getName() + "_forceSpilledRead" + std::to_string(read_index++);
      mv::Data::TensorIterator spill_read_tensor_itr =
        om.dMATask(spill_write_tensor_itr, read_dma_direction, 0, dma_op_name);
      Data::OpListIterator read_op_itr =
          om.getSourceOp(spill_read_tensor_itr);
      read_op_itr->setInputTensor(spill_write_tensor_itr, 0UL, false);

      // now connect output of this read all ops in this subtree //
      for (auto child=spilled_op_children.begin();
            child!=spilled_op_children.end(); ++child) {
        operation_t child_op = *child;
        Data::OpListIterator child_op_itr = om.getOp(child_op->getName());
        assert(child_op_itr != om.opEnd());

        // find the input index in the original spilled op //
        auto idx_itr = input_tensor_index_map.find(child_op);
        assert(idx_itr != input_tensor_index_map.end());
        size_t idx = idx_itr->second;
        child_op_itr->setInputTensor(spill_read_tensor_itr, idx, false);
        om.defineFlow(spill_read_tensor_itr, child_op_itr, idx);
      }

      return true;
    }

  private:

    // the spilled op can only be connected to other DPUTasks //
    bool is_op_spillable(mv::Data::OpListIterator op_itr) const {
      if (op_itr == model_.opEnd()) { return false; }
      if (!is_dpu_task(op_itr)) { return false; }
      for (typename omtraits::const_child_operation_iterator_t
            citr=omtraits::begin_child_operations(op_itr);
              citr!=omtraits::end_operations(model_); ++citr) {
        if (!is_dpu_task(citr)) { return false; }
      }
      return true;
    }

    template<typename Iterator>
    bool is_dpu_task(Iterator itr) const {
      return itr->getOpType() == "DPUTask";
    }



    mv::OpModel &model_;
}; // class Force_Spill_Activation //

} // namespace lp_scheduler 
} // namespace mv



static void ForceSpillActivationPass(
    const mv::pass::PassEntry& , mv::ComputationModel&, mv::TargetDescriptor&,
    mv::Element&, mv::Element&);

namespace mv {
  namespace pass {
    MV_REGISTER_PASS(ForceSpillActivationPass)
      .setFunc(ForceSpillActivationPass)
      .defineArg(json::JSONType::String, "op_name")
      .setDescription("Spill activation at the specified node");
  } // namespace mv //
} // namespace pass //

void ForceSpillActivationPass(const mv::pass::PassEntry&,
    mv::ComputationModel& model, mv::TargetDescriptor& target,
    mv::Element& passDesc, mv::Element&) {

  mv::OpModel om(model);
  std::string spill_op_name = passDesc.get<std::string>("op_name");

  mv::lp_scheduler::Force_Spill_Activation spiller(om);

  bool spilled = spiller.spill_at(spill_op_name);
  if (!spilled) {
    throw "[ForceSpillActivationPass]: failed to spill at specified op";
  }
}


// Force address setter //
static void ForceAddressSetterPass(
    const mv::pass::PassEntry& , mv::ComputationModel&, mv::TargetDescriptor&,
    mv::Element&, mv::Element&);

namespace mv {
  namespace pass {
    MV_REGISTER_PASS(ForceAddressSetterPass)
      .setFunc(ForceAddressSetterPass)
      .defineArg(json::JSONType::String, "op_name")
      .defineArg(json::JSONType::NumberInteger, "address")
      .setDescription("Force address of the tensor");
  } // namespace mv //
} // namespace pass //

void ForceAddressSetterPass(const mv::pass::PassEntry&,
    mv::ComputationModel& model, mv::TargetDescriptor& target,
    mv::Element& passDesc, mv::Element&) {

  std::string op_name = passDesc.get<std::string>("op_name");
  size_t address = passDesc.get<int>("address");

  printf("Setting address = %lu\n", address);
  mv::OpModel om(model);
  mv::Data::OpListIterator op_itr = om.getOp(op_name);
  mv::Data::TensorIterator op_output_tensor_itr = op_itr->getOutputTensor(0UL);

  op_output_tensor_itr->setAddress(address);
}

// Force sparse output//
static void ForceSparseOutputPass(
    const mv::pass::PassEntry& , mv::ComputationModel&, mv::TargetDescriptor&,
    mv::Element&, mv::Element&);

namespace mv {
  namespace pass {
    MV_REGISTER_PASS(ForceSparseOutputPass)
      .setFunc(ForceSparseOutputPass)
      .defineArg(json::JSONType::String, "op_name")
      .setDescription("Force sparse output");
  } // namespace mv //
} // namespace pass //

void ForceSparseOutputPass(const mv::pass::PassEntry&,
    mv::ComputationModel& model, mv::TargetDescriptor& target,
    mv::Element& passDesc, mv::Element&) {

  std::string op_name = passDesc.get<std::string>("op_name");

  mv::OpModel om(model);
  mv::Data::OpListIterator op_itr = om.getOp(op_name);
  mv::Data::TensorIterator op_output_tensor_itr = op_itr->getOutputTensor(0UL);

  op_output_tensor_itr->setSparse();
}
