#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/computation_model.hpp"
#include "pass/lp_scheduler/pipeline_transform.hpp"
#include "pass/lp_scheduler/pipeline_chains_transform.hpp"

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

  mv::GenerateDotFromModel(om, "OpModel",
        "before_pipeline_chain_transform.dot");
  FILE *pipeline_report_fptr = fopen("chain_pipeline_report.txt", "w");
  pipeliner.transform_op_model(pipeline_report_fptr, pipeline_stages);
  fclose(pipeline_report_fptr);
  mv::GenerateDotFromModel(om, "OpModel","after_pipeline_chain_transform.dot");
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
