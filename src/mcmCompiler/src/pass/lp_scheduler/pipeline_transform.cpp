#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/computation_model.hpp"
#include "pass/lp_scheduler/pipeline_transform.hpp"
#include "pass/lp_scheduler/pipeline_chains_transform.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"

static void LocatePipeLinedOps(
    const mv::pass::PassEntry& , mv::ComputationModel&, mv::TargetDescriptor&,
      mv::Element&, mv::Element&);

namespace mv {
  namespace pass {
    MV_REGISTER_PASS(LocatePipeLinedOps)
      .setFunc(LocatePipeLinedOps)
      .setDescription("Locate Pipelined Layer Ops");
  } // namespace mv //
} // namespace pass //



template<typename T>
static bool does_this_op_generate_sparse_output(mv::OpModel& model, T op) {
  auto op_itr = model.getOp(op->getName());
  mv::Data::TensorIterator output_tensor_itr =
      op_itr->getOutputTensor(0UL);
  return output_tensor_itr->isSparse();
}
void LocatePipeLinedOps(const mv::pass::PassEntry&,
    mv::ComputationModel& model, mv::TargetDescriptor& , mv::Element& passDesc,
    mv::Element&) {
  mv::OpModel om(model);
  typedef mv::scheduler::Pipelining_Transform pipeliner_t;
  typedef typename pipeliner_t::pipeline_subgraph_t subgraph_t;

  std::list<subgraph_t> subgraphs;
  pipeliner_t pipeliner(om);

  pipeliner.locate_pipeline_subgraphs(std::back_inserter(subgraphs));

  for (auto itr=subgraphs.begin(); itr!=subgraphs.end(); ++itr) {
    itr->normalize(om);
    if (itr->is_pipelineable(917504)) {
      printf("root = %s pipelinable=yes\n", (itr->name()).c_str());
    }
  }
}

struct noop_back_insert_iterator_t {
  const noop_back_insert_iterator_t& operator++() const { return *this; }

  template<typename T>
  void operator=(const T&) { }
  noop_back_insert_iterator_t& operator*() { return *this; }
}; // noop_back_insert_iterator_t //

void ChainPipeliningTransform(const mv::pass::PassEntry&,
    mv::ComputationModel& model, mv::TargetDescriptor& , mv::Element& passDesc,
    mv::Element&) {
  mv::OpModel om(model);
  typedef mv::scheduler::Pipeline_Chains pipeline_chains_t;
  typedef typename pipeline_chains_t::chain_subgraph_t subgraph_t;

  pipeline_chains_t pipeliner(om);

  size_t pipeline_stages = 0UL;

  if (passDesc.hasAttr("select_stages")) {
    pipeline_stages = (size_t) passDesc.get<int>("select_stages");
  }

  // for large inplace eltwises reduce the stages to 0 //
  {
    for (auto op_itr = om.opBegin(); op_itr != om.opEnd(); ++op_itr) {
      if (op_itr->hasAttr("inplace_eltwise_rep")) {
        pipeline_stages = 0UL;
      }
    }
  }

  //mv::GenerateDotFromModel(om, "OpModel",
   //     "before_pipeline_chain_transform.dot");
  FILE *pipeline_report_fptr = fopen("chain_pipeline_report.txt", "w");
  if (!pipeline_report_fptr)
    throw mv::RuntimeError("ChainPipeliningTransform", "Cannot open chain_pipeline_report.txt for write");
  pipeliner.transform_op_model(pipeline_report_fptr, pipeline_stages);
  fclose(pipeline_report_fptr);
  //mv::GenerateDotFromModel(om, "OpModel",
    // after_pipeline_chain_transform.dot");
}

void ChainPipeliningInverseTransform(const mv::pass::PassEntry&,
    mv::ComputationModel& model, mv::TargetDescriptor& , mv::Element& passDesc,
    mv::Element&) {
  mv::OpModel om(model);
  typedef mv::scheduler::Pipeline_Chains pipeline_chains_t;
  typedef typename pipeline_chains_t::chain_subgraph_t subgraph_t;

  pipeline_chains_t pipeliner(om);

  mv::GenerateDotFromModel(om, "OpModel",
            "before_pipeline_chain_inverse_transform.dot");
  FILE *pipeline_report_fptr = fopen("chain_pipeline_report.txt", "w");
  if (!pipeline_report_fptr)
    throw mv::RuntimeError("ChainPipeliningInverseTransform", "Cannot open chain_pipeline_report.txt for write");

  std::list<mv::Data::OpListIterator> ops_to_remove;
  for (mv::Data::OpListIterator oitr=om.opBegin(); oitr!=om.opEnd();
        ++oitr) {
    if (oitr->getOpType() == "PseudoOp") {
      ops_to_remove.push_back(oitr);
    }
  }

  for (mv::Data::OpListIterator oitr : ops_to_remove) {
    om.removeOp(oitr);
  }

  mv::GenerateDotFromModel(om, "OpModel",
      "after_pipeline_chain_inverse_transform.dot");
  fclose(pipeline_report_fptr);
}

namespace mv {
  namespace pass {
    MV_REGISTER_PASS(ChainPipeliningTransform)
      .setFunc(ChainPipeliningTransform)
      .setDescription("ChainPipeliningTransform");
  } // namespace mv //
} // namespace pass //

namespace mv {
  namespace pass {
    MV_REGISTER_PASS(ChainPipeliningInverseTransform)
      .setFunc(ChainPipeliningInverseTransform)
      .setDescription("ChainPipeliningInverseTransform");
  } // namespace mv //
} // namespace pass //

void AddPseudoDependency(const mv::pass::PassEntry&,
    mv::ComputationModel& model, mv::TargetDescriptor& , mv::Element& passDesc,
    mv::Element&) {

  bool has_source = passDesc.hasAttr("source");
  bool has_sink = passDesc.hasAttr("sink");
  bool add_flow_source_attribute =
      passDesc.hasAttr("add_flow_attribute_source");
  bool add_flow_sink_attribute = passDesc.hasAttr("add_flow_attribute_sink");

  mv::OpModel om(model);

  mv::Data::OpListIterator src_itr =
      om.getOp(passDesc.get<std::string>("source"));
  mv::Data::OpListIterator sink_itr =
      om.getOp(passDesc.get<std::string>("sink"));


  mv::Data::TensorIterator src_tensor_itr = src_itr->getOutputTensor(0UL);

 // if (!om.pathExists(src_itr, sink_itr))
  {
    mv::Data::FlowListIterator flow_itr =
        om.defineFlow(src_tensor_itr, sink_itr, 0UL);
    flow_itr->set<bool>("pseudo_data_flow", true);

    if (add_flow_source_attribute) {
      src_itr->set<bool>("pipeline_flow_control", true);
    }

    if (add_flow_sink_attribute) {
      sink_itr->set<bool>("pipeline_flow_control", true);
    }
  }
}

namespace mv {
  namespace pass {
    MV_REGISTER_PASS(AddPseudoDependency)
      .setFunc(AddPseudoDependency)
      .setDescription("PesudoDependencyToForceScheduling");
  } // namespace mv //
} // namespace pass //


void LocateInplaceEltwiseOps(const mv::pass::PassEntry&,
    mv::ComputationModel& cmodel, mv::TargetDescriptor& , mv::Element& passDesc,
    mv::Element&) {
  typedef mv::Op const * operation_t;
  size_t eltwise_threshold = 100000UL;

  if (passDesc.hasAttr("threshold")) {
    eltwise_threshold = (size_t) passDesc.get<int>("threshold");
  }

  mv::OpModel model(cmodel);

  for (auto op_itr = model.opBegin(); op_itr != model.opEnd(); ++op_itr) {
    if (!(op_itr->hasAttr("taskOp") &&
            (op_itr->get<std::string>("taskOp") == "Eltwise"))) {continue;}

    ////////////////////////////////////////////////////////////////////////
    // TODO(vamsikku): currently we ignore eltwise's which generate sparse 
    // if we were to overwrite input of we need to also carefully overwrite
    // sparsity maps and storage element pointers.
    if (does_this_op_generate_sparse_output(model, op_itr)) {
      continue;
    }
    ////////////////////////////////////////////////////////////////////////


    operation_t parent_op = NULL;
    // prefer a ELTWISE Input if possible //
    {
      for (auto pitr=op_itr.leftmostParent(); pitr!=model.opEnd(); ++pitr) {
        if ((pitr->hasAttr("taskOp") &&
            (pitr->get<std::string>("taskOp") == "Eltwise"))) {
          if (does_this_op_generate_sparse_output(model, pitr)) {
            // TODO(vamsikku): see above TODO we currently dont support
            // sparse //
            continue;
          }
          parent_op = &(*pitr);
          break;
        }
      }
    }

    if (!parent_op) {
      for (size_t i=0; i<op_itr->inputSlots(); ++i) {
        mv::Data::TensorIterator input_tensor = op_itr->getInputTensor(0UL);
        mv::Data::OpListIterator parent_op_itr =
            model.getSourceOp(input_tensor);

        if (does_this_op_generate_sparse_output(model, parent_op_itr) ||
            (parent_op_itr->getOpType() != "DPUTask") ) {
          continue;
        }
        parent_op = &(*parent_op_itr);
        break;
      }
    }

    if (!parent_op) {
      continue;
    }

    operation_t eltwise_op = &(*op_itr);
    {
      // if parent and this eltwise don't have the same strategy we cannot
      // overwrite parent input with eltwise output
      auto parent_op_itr = model.getOp(parent_op->getName());
      std::string parent_strategy =
          parent_op_itr->get<std::string>("splitStrategy");
      std::string eltwise_strategy =
          op_itr->get<std::string>("splitStrategy");

      bool is_valid = (parent_strategy == eltwise_strategy)
        ||
      (parent_strategy == "Clustering" && eltwise_strategy == "SplitOverK")
        ||
      (parent_strategy == "SplitOverK" && eltwise_strategy == "Clustering");


      if (!is_valid) {
        continue;
      }
    }

    auto eltwise_output_tensor = op_itr->getOutputTensor(0UL);
    if (eltwise_output_tensor->getClusterSize() < eltwise_threshold) {
      continue;
    }


    op_itr->set<std::string>("inplace_eltwise_rep", parent_op->getName());
  }
}


namespace mv {
  namespace pass {
    MV_REGISTER_PASS(LocateInplaceEltwiseOps)
      .setFunc(LocateInplaceEltwiseOps)
      .setDescription("LocateInpalceEltwiseOps");
  } // namespace mv //
} // namespace pass //













