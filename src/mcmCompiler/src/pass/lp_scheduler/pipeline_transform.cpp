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

void LocatePipeLineChains(const mv::pass::PassEntry&,
    mv::ComputationModel& model, mv::TargetDescriptor& , mv::Element& passDesc,
    mv::Element&) {
  mv::OpModel om(model);
  typedef mv::scheduler::Pipeline_Chains pipeline_chains_t;
  typedef typename pipeline_chains_t::chain_subgraph_t subgraph_t;

  std::list<subgraph_t> subgraphs;
  pipeline_chains_t pipeliner(om);

  pipeliner.locate_chains(std::back_inserter(subgraphs));

  for (auto itr=subgraphs.begin(); itr!=subgraphs.end(); ++itr) {
    itr->print();
  }
}

namespace mv {
  namespace pass {
    MV_REGISTER_PASS(LocatePipeLineChains)
      .setFunc(LocatePipeLineChains)
      .setDescription("Locate Pipeline Chains");
  } // namespace mv //
} // namespace pass //
