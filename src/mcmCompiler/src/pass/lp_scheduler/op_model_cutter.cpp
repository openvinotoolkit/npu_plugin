#include <unordered_set>
#include <list>

#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/logger/logger.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/tensor/tensor.hpp"

static void OpModelCutter(const mv::pass::PassEntry&, mv::ComputationModel&,
    mv::TargetDescriptor&, mv::Element&, mv::Element&);


namespace mv {
namespace pass {

MV_REGISTER_PASS(OpModelCutter)
  .setFunc(OpModelCutter)
  .defineArg(json::JSONType::String, "op_name")
  .setDescription("Name of the op at which the mv::OpModel will be cut");

} // namespace pass // 
} // namespace mv //

//TODO(vamsikku): ideally only need const mv::OpModel& unfortunately mv::OpModel
//is not const correct.
template<typename BackInsertIterator>
void FindAllOpsWhichConnectToThisOp(mv::OpModel& omodel,
      const std::string& op_name, BackInsertIterator output) {

  std::list<mv::Op *> bfs_list;
  mv::Data::OpListIterator oitr = omodel.getOp(op_name);

  assert(oitr != omodel.opEnd());

  bfs_list.push_back(&(*oitr));
  std::unordered_set<mv::Op *> explored_nodes;
  explored_nodes.insert( &(*oitr) );

  while (!bfs_list.empty()) {
    mv::Op *curr_op = bfs_list.front();
    output = curr_op;
    mv::Data::OpListIterator curr_itr = omodel.getOp(curr_op->getName());
    assert(curr_itr != omodel.opEnd());

    for (mv::Data::OpParentIterator pitr = curr_itr.leftmostParent();
          pitr != omodel.opEnd(); ++pitr) {
      if (explored_nodes.find( &(*pitr) ) == explored_nodes.end()) {
        bfs_list.push_back(&(*pitr));
        explored_nodes.insert( &(*pitr) );
      }
    }
    bfs_list.pop_front();
  }
} 

void OpModelCutter(const mv::pass::PassEntry& , mv::ComputationModel& model,
    mv::TargetDescriptor& , mv::Element& passDesc, mv::Element& ) {

  mv::OpModel omodel(model);
  std::string cut_op_name = passDesc.get<std::string>("op_name");

  {
    std::ostringstream log_stream;
    log_stream << "[OpModelCutter] cut_op_name=" << cut_op_name << std::endl;
    passDesc.log(mv::Logger::MessageType::Info, log_stream.str());
  }

  mv::Data::OpListIterator output_op_itr;

  bool multipleOutputs = false;
  if (omodel.getNumNetworkOutputs() > 1)
      multipleOutputs = true;

  //STEP-0: first find the output node and erase it //
  for (mv::Data::OpListIterator itr = omodel.opBegin();
        itr!=omodel.opEnd(); ++itr) {

    if (itr->getOpType() != "Output") { continue; }

    if (itr->getName() == cut_op_name) {
      {
        std::ostringstream log_stream;
        log_stream << "[OpModelCutter] mv::OpModel already ends at = %s" <<
          cut_op_name << std::endl;
        passDesc.log(mv::Logger::MessageType::Info, log_stream.str());
      }
      return;
    }

    output_op_itr = itr;
    break;
  }

  assert(output_op_itr != omodel.opEnd());

  //STEP-1: remove all children for the op which wants to be the output //
  mv::Data::OpListIterator new_oitr = omodel.getOp(cut_op_name);
  if (new_oitr == omodel.opEnd()) {
    {
      std::ostringstream log_stream;
      log_stream << "[OpModelCutter] op " << cut_op_name <<
          "does not exist in mv::OpModel" << std::endl;
      passDesc.log(mv::Logger::MessageType::Info, log_stream.str());
    }
    return;
  }

  {
    std::ostringstream log_stream;
    std::list<std::string> ops_to_remove;
    for (mv::Data::OpChildIterator citr=new_oitr.leftmostChild();
          citr!=omodel.opEnd(); ++citr) {
      { // log //
        log_stream << "[removedOp] op="<< citr->getName() << std::endl;;
      }
      ops_to_remove.push_back(citr->getName());
    }
    for (auto nitr=ops_to_remove.begin(); nitr!=ops_to_remove.end(); ++nitr) {
      mv::Data::OpListIterator opitr = omodel.getOp(*nitr);
      omodel.removeOp(opitr);
    }

    passDesc.log(mv::Logger::MessageType::Info, log_stream.str());
  }

  //STEP-2: compute connected components
  std::unordered_set<mv::Op *> connected_ops;
  {
    std::list<mv::Op *> connected_op_list;
    FindAllOpsWhichConnectToThisOp(omodel, cut_op_name,
          std::back_inserter(connected_op_list));
    for (auto citr=connected_op_list.begin(); citr!=connected_op_list.end();
        ++citr) {
      connected_ops.insert(*citr);
    }
  }

  //STEP-3: remove all the ops which are not connected to the cut_op //
  std::list<mv::Data::OpListIterator> disconnected_ops;
  for (mv::Data::OpListIterator itr = omodel.opBegin();
        itr!=omodel.opEnd(); ++itr) {
    if (connected_ops.find(&(*itr)) == connected_ops.end()) {
      disconnected_ops.push_back(itr);
    }
  }
  for (auto ditr=disconnected_ops.begin(); ditr!=disconnected_ops.end();
        ++ditr) {
    if ((*ditr)->getOpType() == "Output") { continue; }
    omodel.removeOp(*ditr);
  }

  //STEP-4: make the cut_op connect to new output //
  mv::Data::TensorIterator cut_op_tensor_itr = new_oitr->getOutputTensor(0UL);
  output_op_itr->setInputTensor(cut_op_tensor_itr, 0UL, true);
  //NOTE: Most of the times the precision type needs to be fp16 as we compare against cpu, so leaving it...
  output_op_itr->set("precision", mv::DType("Float16"));
  omodel.defineFlow(cut_op_tensor_itr, output_op_itr, 0UL);
  if (multipleOutputs)
  {
      omodel.setNumNetworkOutputs(1);
      omodel.setOutputNode(output_op_itr);
  }
}
