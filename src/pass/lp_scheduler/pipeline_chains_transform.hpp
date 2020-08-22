#ifndef PIPELINE_CHAIN_TRANSFORM_HPP
#define PIPELINE_CHAIN_TRANSFORM_HPP

#include <unordered_set>
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/model/iterator/tensor.hpp"
#include "include/mcm/op_model.hpp"

namespace mv {
namespace scheduler {

class Pipeline_Chains {

  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef const mv::Op* operation_t;
    typedef mv::Op* operation_non_const_t;
    typedef std::list<operation_t> op_list_t;

    struct chain_subgraph_t {
      op_list_t dpu_chain_;
      std::list<op_list_t> weight_reads_;

      void print() const {
        printf("\n===========================\n");
        auto read_itr = weight_reads_.begin();

        for (operation_t dpu_op : dpu_chain_) {
          printf("%s : ", (dpu_op->getName()).c_str());
          for (operation_t read_op : *read_itr) {
            printf(" %s ", (read_op->getName()).c_str());
          }
          printf("\n");
          ++read_itr;
        }
        printf("\n===========================\n");
      }
    }; // struct chain_subgraph_t //

    struct control_edge_t {
      control_edge_t(mv::Data::OpListIterator src,
            mv::Data::OpListIterator sink)
          : source_itr_(src) , sink_itr_(sink) { } 

      control_edge_t(const control_edge_t& o) : source_itr_(o.source_itr_), 
        sink_itr_(o.sink_itr_) {}

      control_edge_t& operator=(const control_edge_t& o) {
        source_itr_ = o.source_itr_;
        sink_itr_ = o.sink_itr_;
      }
      mv::Data::OpListIterator source_itr_;
      mv::Data::OpListIterator sink_itr_;
    }; // struct control_edge_t //
    ////////////////////////////////////////////////////////////////////////////


    Pipeline_Chains(mv::OpModel& omodel) : omodel_(omodel) {}

    template<typename T>
    bool is_weight_read(T op) const {
      mv::Data::OpListIterator oitr = omodel_.getOp(op->getName());
      if (oitr->getOpType() != "DMATask") { return false; }
      // indegree must be 1 and 
      auto pitr = oitr.leftmostParent();
      auto pitr_next = pitr;
      
      ++pitr_next;
      if (pitr_next != omodel_.opEnd()) { return false; }

      return (pitr->getOpType() == "ConstantDataElement") ||
        (pitr->getOpType() == "ConstantInt");
    }

    // If op has multiple inputs this returns NULL //
    template<typename T>
    operation_t get_single_non_weight_input(T dpu_op) const {
      mv::Data::OpListIterator oitr = omodel_.getOp(dpu_op->getName());
      operation_t single_input_op = NULL;

      for (auto pitr=oitr.leftmostParent(); pitr!=omodel_.opEnd(); ++pitr) {
        if (is_weight_read(pitr)) { continue; }
        if (single_input_op) { return NULL; }
        single_input_op = &(*pitr);
      }
      return single_input_op;
    }

    template<typename T, typename OutputIterator>
    void get_weight_read_inputs(T dpu_op, OutputIterator output) const {
      mv::Data::OpListIterator oitr = omodel_.getOp(dpu_op->getName());
      operation_t single_input_op = NULL;

      for (auto pitr=oitr.leftmostParent(); pitr!=omodel_.opEnd(); ++pitr) {
        if (is_weight_read(pitr)) {
          output = &(*pitr);
        }
      }
    }

    // If op has multiple outputs this returns NULL //
    template<typename T>
    operation_t get_single_output(T dpu_op) const {
      mv::Data::OpListIterator oitr = omodel_.getOp(dpu_op->getName());
      operation_t single_output_op = NULL;

      auto citr = oitr.leftmostChild();
      auto citr_next = citr;

      if (citr == omodel_.opEnd()) { return NULL; }
      ++citr_next;
      if (citr_next != omodel_.opEnd()) { return NULL; }

      single_output_op = &(*citr);
      return single_output_op;
    }

    template<typename OutputIterator>
    size_t locate_chains(OutputIterator output) {
      std::unordered_set<operation_t> already_in_some_chain;
      for (mv::Data::OpListIterator oitr=omodel_.opBegin();
            oitr!=omodel_.opEnd(); ++oitr) {

        if (oitr->getOpType() != "DPUTask") { continue; }
        operation_t dop = &(*oitr);

        if (already_in_some_chain.find(dop) != already_in_some_chain.end()) {
          continue;
        }

        chain_subgraph_t chain_subgraph;

        ////////////////////////////////////////////////////////////////////////
        // STEP-0: find the chain //
        op_list_t &chain = chain_subgraph.dpu_chain_;
        std::list<op_list_t> &input_reads = chain_subgraph.weight_reads_;
        operation_t input_op = dop;

        while ((input_op = get_single_non_weight_input(input_op)) &&
                (input_op->getOpType() == "DPUTask")) {
          chain.push_front(input_op);
          op_list_t weight_reads;
          get_weight_read_inputs(input_op, std::back_inserter(weight_reads));
          input_reads.push_front(weight_reads);
        }

        chain.push_back(dop);
        {
          op_list_t weight_reads;
          get_weight_read_inputs(dop, std::back_inserter(weight_reads));
          input_reads.push_back(weight_reads);
        }

        operation_t output_op = dop;
        while ( (output_op = get_single_output(output_op)) &&
                (output_op->getOpType() == "DPUTask") ) {
          chain.push_back(output_op);
          op_list_t weight_reads;
          get_weight_read_inputs(output_op, std::back_inserter(weight_reads));
          input_reads.push_back(weight_reads);
        }
        ////////////////////////////////////////////////////////////////////////

        for (operation_t chain_op : chain) {
          already_in_some_chain.insert(chain_op);
        }

        output = chain_subgraph;
      }
    }


    template<typename ControlEdgeOutput, typename SubGraphContainer>
    void transform_op_model(ControlEdgeOutput output,
        SubGraphContainer& chain_subgraphs) {

      static_assert( std::is_same<chain_subgraph_t,
            typename SubGraphContainer::value_type>::value,
              "Invalid container for chain subgraphs");

      mv::OpModel &om = omodel_;
      chain_subgraphs.clear();
      locate_chains(std::back_inserter(chain_subgraphs));

      for (chain_subgraph_t chain_subgraph : chain_subgraphs) {
        const std::list<op_list_t>& weight_reads = chain_subgraph.weight_reads_;
        const op_list_t& dpu_chain = chain_subgraph.dpu_chain_;

        auto curr_dpu_itr = dpu_chain.begin();
        auto pprev_dpu_itr = curr_dpu_itr;

        auto curr_itr = weight_reads.begin();
        auto pprev_itr = curr_itr;
        if (curr_itr == weight_reads.end()) { continue; }
        ++curr_itr;
        ++curr_dpu_itr;

        while (curr_itr != weight_reads.end()) {
          // add control edge between first read of pprev and all reads of
          // curr //
          const op_list_t & pprev_read_list = *pprev_itr;
          const op_list_t & curr_read_list = *curr_itr;

          if (!pprev_read_list.empty()) {
            operation_t pprev_read_op = pprev_read_list.front();
            mv::Data::OpListIterator src_itr =
                omodel_.getOp((*pprev_dpu_itr)->getName());

            for (operation_t curr_read_op : curr_read_list ){
              mv::Data::OpListIterator sink_itr =
                  omodel_.getOp(curr_read_op->getName());
              output = control_edge_t(src_itr, sink_itr);
            }
          }

          ++curr_itr;
          ++pprev_itr;
          ++curr_dpu_itr;
          ++pprev_dpu_itr;
        }
      }
    }


  private:
    mv::OpModel& omodel_;
}; // class Pipeline_Chains //

} // namespace scheduler //
} // namespace mv//








#endif

