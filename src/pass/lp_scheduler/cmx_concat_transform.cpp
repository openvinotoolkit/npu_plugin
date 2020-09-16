#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/computation_model.hpp"
#include "pass/lp_scheduler/cmx_concat_transform.hpp"

static void LocateCMXConcateableOps(
    const mv::pass::PassEntry& , mv::ComputationModel&, mv::TargetDescriptor&,
    mv::Element&, mv::Element&);

namespace mv {
  namespace pass {

    MV_REGISTER_PASS(LocateCMXConcateableOps)
      .setFunc(LocateCMXConcateableOps)
      .setDescription("Locate CMX Concateable Ops");

  } // namespace mv //
} // namespace pass //



void LocateCMXConcateableOps(const mv::pass::PassEntry&,
    mv::ComputationModel& model, mv::TargetDescriptor& , mv::Element& passDesc,
    mv::Element&) {
  mv::OpModel om(model);
  typedef mv::scheduler::CMX_Concatenation locator_t;
  typedef typename locator_t::concat_subgraph_t concat_subgraph_t; 
  auto params = model.getGlobalConfigParams();
  size_t cmx_size = params->get<unsigned>("totalCmx");

  locator_t locator(om);
  std::list<concat_subgraph_t> concats;

  locator.locate_concat_subgraphs(std::back_inserter(concats));

  const std::string output_file = passDesc.get<std::string>("output");
  FILE *fptr = fopen(output_file.c_str(), "w");
  if (!fptr) {
    throw std::string("[ERROR]: unable to open file for writing\n");
  }

  for (auto itr=concats.begin(); itr!=concats.end(); ++itr) {
    itr->dump(fptr);
    if ((*itr).is_cmx_concateable(cmx_size)) {
      locator.transform(*itr, cmx_size);
    }
  }
  fclose(fptr);
}
