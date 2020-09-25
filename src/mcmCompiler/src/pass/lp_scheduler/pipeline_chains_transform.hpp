#ifndef PIPELINE_CHAIN_TRANSFORM_HPP
#define PIPELINE_CHAIN_TRANSFORM_HPP

#include <cstdio>
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

      void print(FILE *fptr=stdout) const {
        fprintf(fptr, "\n===========================\n");
        auto read_itr = weight_reads_.begin();

        size_t prev_read_size = 0UL;
        size_t prev_output_size = 0UL;
        for (operation_t dpu_op : dpu_chain_) {
          fprintf(fptr, "%s : ", (dpu_op->getName()).c_str());
          size_t total_read_size = 0UL;
          for (operation_t read_op : *read_itr) {
            fprintf(fptr, " %s ", (read_op->getName()).c_str());
            total_read_size +=
              ((const_cast<mv::Op *>(read_op))->getOutputTensor(0UL))->
                getClusterSize();
          }
          fprintf(fptr, "\n");
          ++read_itr;
        }
        fprintf(fptr, "\n===========================\n");
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

    static const std::string pipeline_chain_control_edge_attribute() {
      return "pipeline_chain_control_edge";
    }

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

    template<typename T>
    size_t get_total_read_weight(T dpu_op) const {
      mv::Data::OpListIterator oitr = omodel_.getOp(dpu_op->getName());
      if (oitr->getOpType() != "DPUTask") { return 0UL; }

      size_t return_value = 0UL;
      for (auto pitr=oitr.leftmostParent(); pitr!=omodel_.opEnd(); ++pitr) {
        operation_t pop = &(*pitr);
        if (is_weight_read(pop)) {
          return_value += pitr->getOutputTensor(0UL)->getClusterSize();
        }
      }
      return return_value;
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

    template<typename T>
    operation_t get_non_weight_input(T dpu_op) const {
      mv::Data::OpListIterator oitr = omodel_.getOp(dpu_op->getName());
      operation_t single_input_op = NULL;

      for (auto pitr=oitr.leftmostParent(); pitr!=omodel_.opEnd(); ++pitr) {
        if (is_weight_read(pitr)) { continue; }
        single_input_op = &(*pitr);
        break;
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
    void locate_chains(OutputIterator output) {
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
          if (already_in_some_chain.find(input_op) !=
                already_in_some_chain.end()) {
            break;
          }
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
          if (already_in_some_chain.find(output_op) !=
                already_in_some_chain.end()) {
            break;
          }
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

    void transform_op_model(FILE *fptr=stdout, size_t select_stages=0UL) {
      std::list<control_edge_t> control_edges;
      std::list<chain_subgraph_t> subgraphs;
      transform_op_model(std::back_inserter(control_edges), subgraphs,
          select_stages, fptr);
    }

    template<typename ControlEdgeOutput, typename SubGraphContainer>
    void transform_op_model_old(ControlEdgeOutput output,
        SubGraphContainer& chain_subgraphs, FILE *fptr=stdout) {

      static_assert( std::is_same<chain_subgraph_t,
            typename SubGraphContainer::value_type>::value,
              "Invalid container for chain subgraphs");

      mv::OpModel &om = omodel_;
      chain_subgraphs.clear();
      locate_chains(std::back_inserter(chain_subgraphs));


      char buf[4096];
      static size_t pseudo_op_id = 0UL;
      for (chain_subgraph_t chain_subgraph : chain_subgraphs) {

        const std::list<op_list_t>& weight_reads = chain_subgraph.weight_reads_;
        const op_list_t& dpu_chain = chain_subgraph.dpu_chain_;
        if (dpu_chain.size() <= 3UL) { continue; }

        chain_subgraph.print(fptr);
        /*
         *  Input: G_sub
         *
         *           [read1]  [read2]         [read-N]
         *             |        |               |
         *             v        v               v
         *  (head)-->[DPU1]-->[DPU2]-->.....->[DPU-N]
         *
         *    transform with a pseduo op chain
         *
         *  Output: G_sub U G_sub_pseduo
         *
         *  (head)-->[PSEUDO-1]-->[read-2]
         *              |
         *              v
         *           [PSEDUO-2]-->[read-3]
         *              |
         *              v
         *           [PSEDUO-3]-->[read-4]
         *              .
         *              .
         *              .
         *           [PSEUDO-N-1]-->[read-N]
         */
        auto curr_dpu_itr = dpu_chain.begin();
        auto curr_weights_itr = weight_reads.begin();

        operation_t chain_head = get_non_weight_input(*curr_dpu_itr);

        if (chain_head->getOpType() == "DMATask") {
          auto chain_head_itr = om.getOp(chain_head->getName());
          chain_head = NULL;
          for (auto pitr=chain_head_itr.leftmostParent(); pitr!=om.opEnd();
                ++pitr) {
            if (pitr->getOpType() != "DMATask") {
              chain_head = &(*pitr);
              break;
            }
          }
        }

        if (!chain_head) {
          fprintf(fptr, "chain_head invalid for %s\n",
              (*curr_dpu_itr)->getName());
          continue;
        }

        {
          const op_list_t& curr_weight_reads = (*curr_weights_itr);
          for (operation_t weight_read : curr_weight_reads) {
            mv::Data::OpListIterator weight_read_itr =
                om.getOp(weight_read->getName());
            weight_read_itr->set<bool>("pipeline_data_start", true);
            auto chain_head_itr = om.getOp(chain_head->getName());
            //om.defineFlow(chain_head_itr->getOutputTensor(0UL),
             //     weight_read_itr, 0UL);
          }
        }

        auto prev_dpu_itr = curr_dpu_itr;
        operation_t pseudo_tail = chain_head, curr_pseudo_op;
        for (++curr_dpu_itr, ++curr_weights_itr; curr_dpu_itr!=dpu_chain.end();
             ++curr_dpu_itr, ++curr_weights_itr, ++prev_dpu_itr) {
          mv::Data::TensorIterator tail_output_tensor_itr =
              (om.getOp(pseudo_tail->getName()))->getOutputTensor(0UL);

          std::vector<mv::Data::TensorIterator> inputs;
          inputs.push_back(tail_output_tensor_itr);

          sprintf(buf, "PseduoOp-%lu", ++pseudo_op_id);
          mv::Data::TensorIterator curr_pseudo_op_tensor_itr =
              om.pseudoOp(inputs, buf);
          mv::Data::OpListIterator curr_pseudo_op_itr =
              om.getSourceOp(curr_pseudo_op_tensor_itr);

          bool is_activation_too_big = false;
          /*
            ((om.getOp((*prev_dpu_itr)->getName()))->getOutputTensor(0UL))
                 ->getClusterSize() > 100000UL;
           */

          if (!is_activation_too_big) {
            auto net_dpu_itr = prev_dpu_itr;

            if (net_dpu_itr != dpu_chain.begin()) {
              --net_dpu_itr;
            }

            {
              auto prev_weights_itr = curr_weights_itr;
              if (prev_weights_itr != weight_reads.begin()) {
                  --prev_weights_itr;
                const op_list_t& prev_weight_reads = (*prev_weights_itr);
                for (operation_t weight_read : prev_weight_reads) {
                  mv::Data::OpListIterator weight_read_itr =
                      om.getOp(weight_read->getName());
                  om.defineFlow(weight_read_itr->getOutputTensor(0UL),
                      curr_pseudo_op_itr, 0UL);
                }
              }
            }

            omodel_.defineFlow(curr_pseudo_op_tensor_itr,
              om.getOp((*prev_dpu_itr)->getName()), 0UL);
            // add pseudo data flows between curr_pseudo_op and all the reads of
            // curr_dpu_itr //
            const op_list_t& reads_of_this_dpu = *curr_weights_itr;
            for (operation_t weight_read : reads_of_this_dpu) {
              mv::Data::OpListIterator weight_read_itr =
                  om.getOp(weight_read->getName());
              omodel_.defineFlow(curr_pseudo_op_tensor_itr, weight_read_itr,
                    0UL);
              weight_read_itr->set<std::string>(
                  pipeline_chain_control_edge_attribute(),
                  (*net_dpu_itr)->getName());
            }
          }

          pseudo_tail = &(*curr_pseudo_op_itr);
        }

      } // foreach chain subgraph //
    }

    template<typename ControlEdgeOutput, typename SubGraphContainer>
    void transform_op_model(ControlEdgeOutput output,
        SubGraphContainer& chain_subgraphs, size_t select_stages=0UL,
        FILE *fptr=stdout) {

      static_assert( std::is_same<chain_subgraph_t,
            typename SubGraphContainer::value_type>::value,
              "Invalid container for chain subgraphs");

      mv::OpModel &om = omodel_;
      chain_subgraphs.clear();
      locate_chains(std::back_inserter(chain_subgraphs));


      char buf[4096];
      static size_t pseudo_op_id = 0UL;
      for (chain_subgraph_t chain_subgraph : chain_subgraphs) {
        const std::list<op_list_t>& weight_reads = chain_subgraph.weight_reads_;
        const op_list_t& dpu_chain = chain_subgraph.dpu_chain_;

        chain_subgraph.print(fptr);
        auto curr_dpu_itr = dpu_chain.begin();
        auto curr_itr = weight_reads.begin();


        auto pprev_dpu_itr = curr_dpu_itr;

        auto pprev_itr = curr_itr;

        const op_list_t & weight_reads_start_list = weight_reads.front();


        if (curr_itr == weight_reads.end()) { continue; }
        ++curr_itr;
        ++curr_dpu_itr;

        if (curr_itr == weight_reads.end()) { continue; }
        ++curr_itr;
        ++curr_dpu_itr;

        size_t curr_dpu_index = 2UL;
        std::unordered_map<operation_t, size_t> stage_memory;
        while (curr_dpu_itr != dpu_chain.end()) {
          const op_list_t & curr_read_list = *curr_itr;
          const op_list_t & pprev_read_list = *pprev_itr;
          if (!pprev_read_list.empty())
          {

            //if ((select_stages + 2UL) < curr_dpu_index) { continue; }

            auto net_dpu_itr = pprev_dpu_itr;
            for (size_t sl=0; (sl<select_stages) &&
                  (net_dpu_itr != dpu_chain.begin()); ++sl) { --net_dpu_itr; }
            mv::Data::OpListIterator src_itr =
                omodel_.getOp((*net_dpu_itr)->getName());

            size_t demand = 0UL;
            for (operation_t curr_read_op : curr_read_list ){
              mv::Data::OpListIterator curr_read_op_itr =
                  omodel_.getOp(curr_read_op->getName());
              demand +=
                  (curr_read_op_itr->getOutputTensor(0UL))->getClusterSize();
            }

            if (stage_memory.find((*net_dpu_itr)) == stage_memory.end()) {
              auto net_dpu_op_itr = omodel_.getOp((*net_dpu_itr)->getName());
              stage_memory[*net_dpu_itr] =
                (net_dpu_op_itr->getOutputTensor(0UL))->getClusterSize();
              auto next_dpu_itr = net_dpu_itr;
              ++next_dpu_itr;
              if (next_dpu_itr != dpu_chain.end()) {
                auto next_dpu_op_itr =
                    omodel_.getOp((*next_dpu_itr)->getName());
                stage_memory[*net_dpu_itr] +=
                  (next_dpu_op_itr->getOutputTensor(0UL))->getClusterSize();
              }
            }
            //TODO(vamsikku): parameterize this based on CMX availablity //
            if ( (stage_memory[*net_dpu_itr] + demand) > 700000) {
              goto MOVE_TO_NEXT_SUBGRAPH;
            }

            stage_memory[*net_dpu_itr] += demand;

            for (operation_t curr_read_op : curr_read_list ){
              mv::Data::OpListIterator sink_itr =
                  omodel_.getOp(curr_read_op->getName());
              sink_itr->set<bool>("pipeline_flow_control", true);
              mv::Data::TensorIterator src_tensor_itr
                  = src_itr->getOutputTensor(0UL);
              {
                mv::Data::FlowListIterator flow_itr =
                    omodel_.defineFlow(src_tensor_itr, sink_itr, 0UL);
                flow_itr->set<bool>("pseudo_data_flow", true);
              }
            }
          }

MOVE_TO_NEXT_SUBGRAPH:
          ++curr_itr;
          ++pprev_itr;
          ++curr_dpu_itr;
          ++pprev_dpu_itr;
          ++curr_dpu_index;
        }

#if 0
        printf("\n\n");
        printf("======================\n");
        for (auto itr=stage_memory.begin(); itr!=stage_memory.end(); ++itr) {
          printf("[stage_memory] op=%s demand=%lu\n",
              (itr->first->getName()).c_str(), itr->second);
        }
        printf("======================\n");
#endif


      }


    }


  private:
    mv::OpModel& omodel_;
}; // class Pipeline_Chains //

} // namespace scheduler //
} // namespace mv//








#endif
