#ifndef LOCATE_CMX_CONCATEABLE_OPS_HPP
#define LOCATE_CMX_CONCATEABLE_OPS_HPP

#include <unordered_set>

#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/model/iterator/tensor.hpp"

namespace mv {
namespace scheduler {

class CMX_Concatenation {
  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef const mv::Op* operation_t;
    typedef mv::Op* non_const_operation_t;
    typedef std::list<operation_t> op_list_t;


    class exception_t : std::string {
      public:
        exception_t(const std::string& msg) : std::string(msg) {}
        exception_t(const char *msg) : std::string(msg) {}
        const std::string& getMessage() const { return  *this; }
    }; // class exception_t //


    struct op_selector_t {

      op_selector_t(bool select_reads=false)
        : select_dma_reads_(select_reads) {}

      bool operator()(mv::Data::OpListIterator op_itr) const {

        if (op_itr->isImplicit()) { return true; }
        std::string op_type = op_itr->getOpType();

        if (op_type != "DMATask") { return false;}

        mv::DmaDirectionEnum dma_dir =
            op_itr->get< mv::DmaDirection >("direction");

        return select_dma_reads_ ?
          (dma_dir == mv::DmaDirectionEnum::DDR2NNCMX) :
          (dma_dir ==  mv::DmaDirectionEnum::NNCMX2DDR);
      }

      bool select_dma_reads_;
    }; // struct op_selector_t //

    struct control_edge_t {
      control_edge_t(mv::Data::OpListIterator src,
            mv::Data::OpListIterator sink)
          : source_itr_(src) , sink_itr_(sink) { } 

      mv::Data::OpListIterator source_itr_;
      mv::Data::OpListIterator sink_itr_;
    };

    typedef std::list<control_edge_t> control_edge_list_t;

    struct noop_back_inserter_t {
      template<typename T>
      void operator=(const T&) {}
      noop_back_inserter_t& operator*() { return *this; }
    }; // struct noop_back_inserter_t //

    // D_in = input DPU tasks which write into the concat
    // D_out = DPU tasks which read from the concat
    // W = writes into the concat.
    // R = reads from the concat.
    struct concat_subgraph_t {

      concat_subgraph_t()
        : dpu_in_(), dpu_out_(), reads_(), writes_(),
          concat_root_(operation_t(NULL)),
          representative_dpu_(operation_t(NULL)),
          representative_dpu_depth_() {}

      bool is_valid() const {
        return !( (concat_root_ == operation_t(NULL)) || dpu_in_.empty() ||
            dpu_out_.empty() || reads_.empty() || writes_.empty() );
      }

      // Precondition: is_valid() for all the methods below//
      // Returns concat buffer size in bytes //
      size_t master_buffer_size() const {
        size_t output_size =0UL;
        for (auto itr=dpu_in_.begin(); itr!=dpu_in_.end(); ++itr) {
          output_size +=
           (const_cast<mv::Op *>(*itr))->getOutputTensor(0UL)->getClusterSize();
        }
        size_t concat_output_size = 
            (const_cast<mv::Op *>(concat_root_)->getOutputTensor(0UL))->
              getClusterSize();
        return std::max(output_size, concat_output_size);
      }

      template<typename OpIterator>
      void dump_op_list(OpIterator obegin, OpIterator oend,
            FILE *fptr=stdout, const char *list_name="") const {
        fprintf(fptr, "%s={ ", list_name);
        for (auto itr=obegin; itr!=oend; ++itr) {
          if (itr == obegin) {
            fprintf(fptr, " %s", (*itr)->getName().c_str());
          } else {
            fprintf(fptr, ", %s", (*itr)->getName().c_str());
          }
        }
        fprintf(fptr, " }\n");
      }


      size_t get_max_populated_input_size_for_dpus_driven_by_concat() const {
        size_t max_input_size =0UL;
        for (auto itr=dpu_out_.begin(); itr!=dpu_out_.end(); ++itr) {
          max_input_size =
              std::max(max_input_size,
                  total_populated_input_cmx_memory(*itr));
        }
        return max_input_size;
      }



      size_t get_total_output_size_for_dpus_driven_by_concat() const {
        size_t output_size =0UL;
        for (auto itr=dpu_out_.begin(); itr!=dpu_out_.end(); ++itr) {
          output_size +=
           (const_cast<mv::Op *>(*itr))->getOutputTensor(0UL)->getClusterSize();
        }
        return output_size;
      }

      bool is_cmx_concateable(size_t cmx_size=917504UL) const {
        size_t max_input_size = get_max_input_size_for_dpus_driving_concat();
        return ((max_input_size + master_buffer_size()) < cmx_size);
      }

      size_t get_max_input_size_for_dpus_driving_concat() const {

        size_t max_input_size =0UL;
        for (auto itr=dpu_in_.begin(); itr!=dpu_in_.end(); ++itr) {
          max_input_size =
              std::max(max_input_size, total_input_cmx_memory(*itr));
        }
        return max_input_size;
      }

      void dump(FILE *fptr=stdout) const {
        if (!is_valid()) { return; }

        size_t mbuffer_size = master_buffer_size();
        fprintf(fptr, "[root_concat]: %s mbuffer_size=%lu\n",
              concat_root_->getName().c_str(), mbuffer_size); 

        size_t total_cmx_memory = 0UL;
        for (auto itr=dpu_in_.begin(); itr!=dpu_in_.end(); ++itr) {
          fprintf(fptr, "[dpu]: %s inputs=%lu\n", ((*itr)->getName()).c_str(),
              total_input_cmx_memory(*itr));
          total_cmx_memory += total_input_cmx_memory(*itr);
        }
        size_t max_input_size_across_all_dpus = max_input_size_across_streams();
        fprintf(fptr, "[is_cmx_concatable]:%s max_input_size=%lu "
              " mbuffer_size=%lu\n",
              is_concatable_in_cmx(max_input_size_across_all_dpus,
                  mbuffer_size) ? "YES" : "NO",
              max_input_size_across_all_dpus, mbuffer_size);
        fprintf(fptr, "[is_fully_cmx_concatable]:%s total_input_size=%lu"
            " mbuffer_size=%lu\n",
            is_concatable_in_cmx(total_cmx_memory, mbuffer_size)
              ? "YES" : "NO", total_cmx_memory, mbuffer_size);

        dump_op_list(dpu_in_.begin(), dpu_in_.end(), fptr, "D_in");
        dump_op_list(dpu_out_.begin(), dpu_out_.end(), fptr, "D_out");
        dump_op_list(reads_.begin(), reads_.end(), fptr, "R");
        dump_op_list(writes_.begin(), writes_.end(), fptr, "W");
        fprintf(fptr, "[DPU_IN_REP] dpu_rep=%s depth=%lu",
            representative_dpu_->getName().c_str(),
            representative_dpu_depth_);
        fprintf(fptr, "\n\n\n");
      }

      size_t total_input_cmx_memory(operation_t op) const {
        size_t total_size = 0UL;
        for (size_t i=0; i<op->inputSlots(); ++i) {
          mv::Data::TensorIterator tensor_itr =
            const_cast<non_const_operation_t>(op)->getInputTensor(i);

          if (tensor_itr->get<mv::Tensor::MemoryLocation>("Location") ==
                mv::Tensor::MemoryLocation::NNCMX) {
            total_size += tensor_itr->getClusterSize();
          }
        }
        return total_size;
      }

      size_t total_populated_input_cmx_memory(operation_t op) const {
        size_t total_size = 0UL;
        for (size_t i=0; i<op->inputSlots(); ++i) {
          mv::Data::TensorIterator tensor_itr =
            const_cast<non_const_operation_t>(op)->getInputTensor(i);

          if (!(tensor_itr->isPopulated())) { continue; }
          if (tensor_itr->get<mv::Tensor::MemoryLocation>("Location") ==
                mv::Tensor::MemoryLocation::NNCMX) {
            total_size += tensor_itr->getClusterSize();
          }
        }
        return total_size;
      }

      size_t max_input_size_across_streams() const {
        size_t max_input_size = 0UL;
        for (auto itr=dpu_in_.begin(); itr!=dpu_in_.end(); ++itr) {
          max_input_size = std::max(max_input_size,
              total_input_cmx_memory(*itr));
        }
        return max_input_size;
      }

      // MAX_INPUT_SIZE (across all streaming DPUs) + BUFFER_SIZE < CMX //
      bool is_concatable_in_cmx(
          size_t max_input_size, size_t master_buffer_size,
          size_t cmx_size=917504) const {

        return (max_input_size + master_buffer_size) < cmx_size;
      }

      template<typename T>
      bool is_dpu_op(T op_itr) const {
        return (op_itr->getOpType() == "DPUTask");
      }

      size_t compute_representative_dpu_task_using_max_input_size() {
        representative_dpu_ = NULL;
        size_t rep_input_size = 0UL;

        for (auto itr=dpu_in_.begin(); itr!=dpu_in_.end(); ++itr) {
          size_t curr_input_size = total_input_cmx_memory(*itr);
          if (curr_input_size > rep_input_size) {
            rep_input_size = curr_input_size;
            representative_dpu_ = *itr;
          }
        }
        return 0UL;
      }

      // Precondition:  model must be a DAG //
      size_t compute_representative_dpu_task_using_depth(mv::OpModel& model) {
        if (!is_valid()){
          throw exception_t("Invalid concat subgraph");
        }

        std::list<operation_t> zero_in_degree_nodes[2UL];
        std::unordered_map<operation_t, size_t> in_degree_map;
        std::set<operation_t> dpu_ops_which_need_depth;
        size_t curr_depth = 0;

        // STEP-0: compute the in-degree's of all nodes //
        for (auto op_itr = model.opBegin(); op_itr != model.opEnd(); ++op_itr) {
          size_t in_degree = 0;
          for (auto pitr=op_itr.leftmostParent(); pitr!=model.opEnd(); ++pitr) {
            ++in_degree;
          }
          operation_t op = &(*op_itr);
          in_degree_map[ op ] = in_degree;
          if (!in_degree) {
            zero_in_degree_nodes[0].push_back(op);
          }
        }

        // STEP-1: initialize dpu_ops_which_need_depth //
        for (auto itr=dpu_in_.begin(); itr!=dpu_in_.end(); ++itr) { 
          dpu_ops_which_need_depth.insert( *itr );
        }

        representative_dpu_ = NULL;
        representative_dpu_depth_ = std::numeric_limits<size_t>::max();

        while (!zero_in_degree_nodes[curr_depth%2UL].empty() &&
               !representative_dpu_) {

          bool parity = ((curr_depth%2UL) == 1UL);
          for (auto zitr=zero_in_degree_nodes[parity].begin();
                zitr!=zero_in_degree_nodes[parity].end(); ++zitr) {

            // update the in-degree //
            mv::Data::OpListIterator zop_itr = model.getOp((*zitr)->getName());
            for (auto citr=zop_itr.leftmostChild();
                  citr!=model.opEnd(); ++citr) {
              operation_t cop = &(*citr);
              auto ditr = in_degree_map.find(cop);
              if ( (ditr == in_degree_map.end()) || (ditr->second == 0UL) ) {
                throw exception_t("Missing entry in the in-degree map (or)"
                    " invalid in-degree for op= " + cop->getName() );
              }
              --(ditr->second);
              if (!(ditr->second)) {
                zero_in_degree_nodes[!parity].push_back(cop);
                auto dpu_op_itr = dpu_ops_which_need_depth.find(cop);
                if (dpu_op_itr != dpu_ops_which_need_depth.end()) {
                  representative_dpu_ = cop;
                  representative_dpu_depth_ = curr_depth;
                  break;
                }
              }
            }

            if (representative_dpu_) { break; }
          }

          zero_in_degree_nodes[parity].clear();
          curr_depth++;
        }

        if (!representative_dpu_) {
          throw exception_t("Unable identify a canonical DPU task.");
        }

        return representative_dpu_depth_;
      }

      op_list_t dpu_in_;
      op_list_t dpu_out_;
      op_list_t reads_;
      op_list_t writes_;
      operation_t concat_root_;
      operation_t representative_dpu_; // representative DPU task from dpu_in_
      size_t representative_dpu_depth_; 
    }; // struct concat_subgraph_t //

    ////////////////////////////////////////////////////////////////////////////

    CMX_Concatenation(mv::OpModel& model, std::string ignore_list="")
      : omodel_(model), ignore_these_concats_() {
      populate_ignore_list(ignore_list);
      compute_dpu_depth_map();
    }

    static const std::string cmx_concat_control_edge_attribute() {
      return "cmx_concat_control_edges";
    }

    static const std::string cmx_concat_attribute() {
      return "cmx_concatable";
    }

    static const std::string cmx_concat_reader_attribute() {
      return "cmx_concat_reader";
    }

    static const std::string cmx_concat_writer_attribute() {
      return "cmx_concat_writer";
    }

    static const std::string cmx_concat_buffer_attribute() {
      return "cmx_concat_buffer";
    }

    template<typename OutputIterator>
    size_t locate_concat_subgraphs(OutputIterator output) {
      size_t total_sub_graphs = 0UL;
      for (mv::Data::OpListIterator oitr=omodel_.opBegin();
            oitr!=omodel_.opEnd(); ++oitr) {
        if (is_root_concat(oitr)) {
          concat_subgraph_t subgraph;
          subgraph.concat_root_ = &(*oitr);

          locate_dpu_in_and_write_tasks(oitr,
              std::back_inserter(subgraph.dpu_in_),
              std::back_inserter(subgraph.writes_));

          locate_dpu_out_and_read_tasks(oitr,
              std::back_inserter(subgraph.dpu_out_),
              std::back_inserter(subgraph.reads_));

          // compute the representative dpu task //
          if (subgraph.is_valid()) {
            //subgraph.compute_representative_dpu_task_using_depth(omodel_);
            subgraph.compute_representative_dpu_task_using_max_input_size();
            *output = subgraph;
            ++total_sub_graphs;
          }
        }
      }
      return total_sub_graphs;
    }

    template<typename OutputIterator>
    size_t locate_all_concat_subgraphs(OutputIterator output) {
      size_t total_sub_graphs = 0UL;
      for (mv::Data::OpListIterator oitr=omodel_.opBegin();
            oitr!=omodel_.opEnd(); ++oitr) {
        if (is_root_concat(oitr)) {
          concat_subgraph_t subgraph;
          subgraph.concat_root_ = &(*oitr);

          locate_dpu_in_and_write_tasks(oitr,
              std::back_inserter(subgraph.dpu_in_),
              std::back_inserter(subgraph.writes_));

          locate_dpu_out_and_read_tasks(oitr,
              std::back_inserter(subgraph.dpu_out_),
              std::back_inserter(subgraph.reads_));
          *output = subgraph;
          ++total_sub_graphs;
        }
      }
      return total_sub_graphs;
    }

    template<typename ControlEdgeOutput>
    void transform_op_model(ControlEdgeOutput output,
        size_t cmx_size=917504UL) {
      std::list<concat_subgraph_t> concat_subgraphs;
      transform_op_model(output, concat_subgraphs, cmx_size);
    }

    template<typename ControlEdgeOutput, typename SubGraphContainer>
    void transform_op_model(ControlEdgeOutput output,
          SubGraphContainer& concat_subgraphs, size_t cmx_size=917504UL) {

      static_assert( std::is_same<concat_subgraph_t,
            typename SubGraphContainer::value_type>::value,
              "Invalid container for concat subgraphs");

      FILE *fptr = fopen("cmx_concat_report.txt", "w");

      locate_concat_subgraphs(std::back_inserter(concat_subgraphs));
      for (auto sitr=concat_subgraphs.begin(); sitr!=concat_subgraphs.end();
            ++sitr) {
        concat_subgraph_t& subgraph = *sitr;
        bool can_transform =
            is_cmx_concateable_in_current_opmodel(subgraph, cmx_size);
        if (can_transform) {
          transform_and_get_control_edges(subgraph, output);
        }


        if (fptr) {
          fprintf(fptr, "concat = %s transformed_to_cmx = %s\n",
              subgraph.concat_root_->getName().c_str(),
              can_transform ? "YES" : "NO");
        }
      }

      if (fptr) {
        fclose(fptr);
      }
    }

    bool has_no_cmx_concat_flag(const concat_subgraph_t& subgraph) const {
      mv::OpModel &om = omodel_;
      mv::Data::OpListIterator concat_op =
          omodel_.getOp((subgraph.concat_root_)->getName());
      return concat_op->hasAttr("avoid_cmx_concat") && 
          concat_op->get("avoid_cmx_concat");
    }

    bool does_this_concat_have_any_crops(const concat_subgraph_t& subgraph)
        const {
      return does_this_concat_have_any_parents_or_children_of_this_op_type(
            subgraph, "Crop");
    }
    bool is_this_a_complex_concat(const concat_subgraph_t& subgraph) const {
      return does_this_concat_have_any_parents_or_children_of_this_op_type(
            subgraph, "ImplicitConcat");
    }

    bool does_this_concat_childs_service_compiler_provided_sparsity(
      const concat_subgraph_t& subgraph) const {
      // Eltwise compiler provided sparsity requires address information
      // to build correctly it's SE tables
      // see ticket VPUNND-3529 detailing issue and provide proper address
      // propagation logic in the future
      for (auto itr=subgraph.dpu_out_.begin(); itr!=subgraph.dpu_out_.end(); ++itr)
        if((*itr)->hasAttr("activationSparsityCompilerSolving") &&
          (*itr)->get<bool>("activationSparsityCompilerSolving") &&
          (*itr)->get<std::string>("taskOp") == "Eltwise")
          return true;
      return false;
    }

    //TODO(vamsikku): temporary work around partially written concat buffer //
    bool does_rep_dpu_has_lower_depth_than_others(
        const concat_subgraph_t& subgraph) const {
      operation_t dpu_rep = subgraph.representative_dpu_;
      auto itr = dpu_depth_map_.find(dpu_rep);
      if (itr == dpu_depth_map_.end()) {
        throw exception_t("DPU missing in depth map");
      }
      size_t dpu_rep_depth = itr->second;

      for (operation_t dpu : subgraph.dpu_in_) {
        auto itr = dpu_depth_map_.find(dpu);
        if (itr->second > dpu_rep_depth) { return true; }
      }
      return false;
    }

    bool is_this_an_unsupported_concat(const concat_subgraph_t& subgraph) const{
      return has_no_cmx_concat_flag(subgraph) ||
        does_this_concat_have_any_crops(subgraph) ||
        is_this_a_complex_concat(subgraph) ||
        does_rep_dpu_has_lower_depth_than_others(subgraph) ||
        does_this_concat_childs_service_compiler_provided_sparsity(subgraph);
    }

    bool does_this_concat_have_any_parents_or_children_of_this_op_type(
        const concat_subgraph_t& subgraph, const std::string& op_type) const {

      mv::OpModel &om = omodel_;
      mv::Data::OpListIterator concat_op =
          om.getOp((subgraph.concat_root_)->getName());
      for (auto pitr=concat_op.leftmostParent(); pitr!=om.opEnd(); ++pitr) {
        if (pitr->getOpType() == op_type) { return true; }
      }

      for (auto ditr=subgraph.dpu_in_.begin(); ditr!=subgraph.dpu_in_.end();
            ++ditr) {
        mv::Data::OpListIterator dop_itr = omodel_.getOp((*ditr)->getName());
        for (auto citr = dop_itr.leftmostChild(); citr!=omodel_.opEnd(); ++citr) {
          if (citr->getOpType() == op_type) { return true; }
        }
      }
      return false;
    }

    template<typename T>
    bool is_implicit_concat_in_cmx(T op) const {
      auto op_itr = omodel_.getOp(op->getName());
      if (op_itr->getOpType() != "ImplicitConcat") { return false; }
      if (!is_the_output_of_this_op_in_cmx(op)) { return false; }
      return op_itr->getOutputTensor(0UL)->getClusterSize();
    }

    size_t total_input_cmx_memory(operation_t op) const {
      size_t total_size = 0UL;
      for (size_t i=0; i<op->inputSlots(); ++i) {
        mv::Data::TensorIterator tensor_itr =
          const_cast<non_const_operation_t>(op)->getInputTensor(i);
        mv::Data::OpListIterator pop = omodel_.getSourceOp(tensor_itr);
        if (!pop->isImplicit()) {
          size_t curr_size = 0UL;
          if (is_the_output_of_this_op_in_cmx(pop)) {
              curr_size = pop->getOutputTensor(0UL)->getClusterSize();
          } 
          total_size += curr_size;
        } else {
          total_size += total_input_cmx_memory_of_implicit_op(&(*pop));
        }
      }
      return total_size;
    }
    

    template<typename T>
    bool is_implicit_concat(T op) const {
      return op->getOpType() == "ImplicitConcat";
    }

    template<typename T>
    size_t op_output_size(T op) const {
      auto op_itr = omodel_.getOp(op->getName());
      if (!(op_itr->outputSlots())) { return 0UL; }
      auto tensor_itr = op_itr->getOutputTensor(0UL);
      return tensor_itr->getClusterSize();
    }


    size_t total_input_cmx_memory_of_implicit_op(operation_t op) const {
      size_t memory_size;
      mv::Data::OpListIterator curr_op_itr = omodel_.getOp(op->getName());

      while (!is_the_output_of_this_op_in_cmx(curr_op_itr) &&
            !is_implicit_concat(curr_op_itr)) {
        curr_op_itr = omodel_.getOp((curr_op_itr.leftmostParent())->getName());
      }

      if (is_implicit_concat(curr_op_itr)) {
        memory_size = total_input_cmx_memory_for_cmx_concat(&(*curr_op_itr));
      } else {
        mv::Data::TensorIterator tensor_itr = curr_op_itr->getOutputTensor(0UL);
        memory_size = tensor_itr->getClusterSize();
      }
      return memory_size;
    }


    template<typename T>
    bool is_the_output_of_this_op_in_cmx(T op) const {
      if (op->outputSlots() != 1UL) { return false; }
      auto op_itr = omodel_.getOp(op->getName());
      mv::Data::TensorIterator tensor_itr = op_itr->getOutputTensor(0UL);
      return tensor_itr->get<mv::Tensor::MemoryLocation>("Location") ==
        mv::Tensor::MemoryLocation::NNCMX;
    }

    // Assuming that that the root concat in this subgraph is CMXed now compute
    // the max input-memory required across the dpus driven by this concat //
    size_t max_input_memory_for_dpus_driven_by_this_concat_assuming_its_cmxed(
        const concat_subgraph_t& subgraph) const {

      // 1. Get all the real ops connected to each of the dpus //
      std::unordered_set<operation_t> concat_read_map;
      for (auto ritr=subgraph.reads_.begin(); ritr!=subgraph.reads_.end();
            ++ritr) {
        concat_read_map.insert(*ritr);
      }

      size_t max_input_size = 0;
      // for each of the DPU ops driven by this concat //
      for (auto ditr=subgraph.dpu_out_.begin(); ditr!=subgraph.dpu_out_.end();
            ++ditr) {
        std::list<operation_t> real_input_ops;
        locate_real_ops_or_cmx_concats_connected_to_this_op(*ditr,
              std::back_inserter(real_input_ops));
        size_t curr_input_size = 0UL;
        // Determine the total input memory form these real ops //
        for (auto real_input_itr=real_input_ops.begin();
              real_input_itr!=real_input_ops.end(); ++real_input_itr) {
          operation_t rop = *real_input_itr;
          size_t input_size = 0UL;

          if (is_the_output_of_this_op_in_cmx(rop) &&
                (concat_read_map.find(rop) == concat_read_map.end()) ) {
            input_size = op_output_size(rop);
            curr_input_size += input_size;
          }
        }
        max_input_size = std::max(curr_input_size, max_input_size);
      }
      return max_input_size;
    }

    size_t total_input_cmx_memory_for_cmx_concat(operation_t op) const {
      if (!(op->getOpType() == "ImplicitConcat")) {
        throw exception_t("The input operation must be an ImplicitConcat");
      }


      std::list<operation_t> driving_dpus;
      mv::Data::OpListIterator concat_op_itr = omodel_.getOp(op->getName());
      locate_concating_dpu_tasks(concat_op_itr,
            std::back_inserter(driving_dpus));

      size_t total_size = 0UL;
      for (auto ditr=driving_dpus.begin(); ditr!=driving_dpus.end(); ++ditr) {
        operation_t dpu_op = *ditr;
        mv::Data::TensorIterator tensor_itr =
            const_cast<non_const_operation_t>(op)->getOutputTensor(0UL);
          total_size += tensor_itr->getClusterSize();
      }
      return total_size;
    }

    size_t get_max_input_size_for_dpus_driving_concat(
          const concat_subgraph_t& subgraph) const {

      size_t max_input_size =0UL;
      for (auto itr=subgraph.dpu_in_.begin(); itr!=subgraph.dpu_in_.end();
            ++itr) {
        max_input_size =
            std::max(max_input_size, total_input_cmx_memory(*itr));
      }
      return max_input_size;
    }

    size_t get_max_input_size_for_dpus_driven_by_concat(
          const concat_subgraph_t& subgraph) const {

      size_t max_input_size =0UL;
      for (auto itr=subgraph.dpu_out_.begin(); itr!=subgraph.dpu_out_.end();
            ++itr) {
        max_input_size =
            std::max(max_input_size, total_input_cmx_memory(*itr));
      }
      return max_input_size;
    }

    template<typename T>
    bool is_this_dpu_op_writing_into_cmx_concat(T op) const {
      mv::Data::OpListIterator op_itr = omodel_.getOp(op->getName());
      if (op_itr->getOpType() != "DPUTask") { return false; }

      auto cop_itr = op_itr.leftmostChild();
      return is_implicit_concat_in_cmx(cop_itr);
    }


    //Precondition: is_this_dpu_op_writing_into_cmx_concat //
    template<typename T>
    size_t get_cmx_concat_buffer_size_driven_by_this_dpu(T op) const {
      if (!is_this_dpu_op_writing_into_cmx_concat(op)) {
        throw exception_t("Invalid DPU op for this function");
      }

      mv::Data::OpListIterator op_itr = omodel_.getOp(op->getName());
      auto cop_itr = op_itr.leftmostChild();
      return (cop_itr->getOutputTensor(0UL))->getClusterSize();
    }

    size_t get_max_output_size_for_dpus_driven_by_concat(
        const concat_subgraph_t& subgraph) const {
      size_t max_output_size =0UL;
      for (auto itr=subgraph.dpu_out_.begin(); itr!=subgraph.dpu_out_.end();
            ++itr) {
        // NOTE if the DPU in the current opmodel is connected to a CMX concat
        // then the output size is the CMX concat buffer size //
        max_output_size = 
          std::max(max_output_size,
              is_this_dpu_op_writing_into_cmx_concat(*itr) ?
              get_cmx_concat_buffer_size_driven_by_this_dpu(*itr) :
              (const_cast<mv::Op *>(*itr))->
                getOutputTensor(0UL)->getClusterSize()
            );
      }
      return max_output_size;
    }


    bool is_cmx_concateable_in_current_opmodel(
        const concat_subgraph_t& subgraph, size_t cmx_size,
        FILE *fptr=NULL) const {

      if (is_this_an_unsupported_concat(subgraph)) { return false; }

      size_t max_input_size =
          get_max_input_size_for_dpus_driving_concat(subgraph);
      size_t master_buffer_size = subgraph.master_buffer_size();
      size_t max_input_size_dpu_reads =
        max_input_memory_for_dpus_driven_by_this_concat_assuming_its_cmxed(
              subgraph);

      size_t max_output_size_dpu_reads =
          get_max_output_size_for_dpus_driven_by_concat(subgraph);

      size_t resource_utility_estimate = (
          master_buffer_size +
          std::max( max_input_size,
                    (max_output_size_dpu_reads + max_input_size_dpu_reads)
                  )
          );
      double cmx_fullness = double(resource_utility_estimate)/double(cmx_size);

#if 0
      printf("Concat = %s utility=%lu percentage=%0.3lf "
          " max_input_size_dpu_reads=%lu max_output_size_dpu_reads=%lu "
          " master_buffer_size=%lu max_input_size=%lu\n",
          subgraph.concat_root_->getName().c_str(), resource_utility_estimate,
          cmx_fullness, max_input_size_dpu_reads, max_output_size_dpu_reads,
          master_buffer_size, max_input_size);
#endif

      auto concat_itr =
          ignore_these_concats_.find(subgraph.concat_root_->getName());
      if ( concat_itr != ignore_these_concats_.end()) {
        printf("IgnoredConcat: %s \n", concat_itr->c_str());
        return false;
      }

      return (cmx_fullness < double(1.0));
    }

    //Control edges will be sent to output iterator and the resource shuffling
    //will be done adding an attribute "cmx_concateable: resource_size"
    template<typename ControlEdgeOutputIterator>
    size_t transform_and_get_control_edges(concat_subgraph_t& subgraph,
        ControlEdgeOutputIterator output) {
      // TYPE1: dpu_rep -> dpu_non_rep
      // TYPE2: dpu_rep->d , d \in dpu_out

      mv::OpModel &om = omodel_;
      transform(subgraph);
      operation_t dpu_rep = subgraph.representative_dpu_;
      auto dpu_rep_itr = om.getOp(dpu_rep->getName());
      size_t total_control_edges = 0UL;
      size_t master_buffer_size = subgraph.master_buffer_size();

      dpu_rep_itr->set<size_t>(cmx_concat_attribute(), master_buffer_size);
      dpu_rep_itr->set<bool>(cmx_concat_writer_attribute(), true);
      for (auto ditr=subgraph.dpu_in_.begin(); ditr!=subgraph.dpu_in_.end();
            ++ditr) {
        if (*ditr == dpu_rep) { continue; }

        auto dpu_itr = om.getOp((*ditr)->getName());
        output = control_edge_t(dpu_rep_itr, dpu_itr);
        ++total_control_edges;

        dpu_itr->set<size_t>(cmx_concat_attribute(), 0UL);
        dpu_itr->set<bool>(cmx_concat_writer_attribute(), true);
      }

      for (auto ditr=subgraph.dpu_out_.begin(); ditr!=subgraph.dpu_out_.end();
          ++ditr) {
        auto dpu_itr = om.getOp((*ditr)->getName());
        output = control_edge_t(dpu_rep_itr, dpu_itr);

        // update the resource increase delta of dpu_itr //
        operation_t dout_op = *ditr;
        auto ritr = resource_increase_delta_.find(dout_op);
        if (ritr == resource_increase_delta_.end()) {
          ritr = resource_increase_delta_.insert(
                std::make_pair(dout_op, 0UL)).first;
        }
        ritr->second += master_buffer_size;
        ++total_control_edges;
      }
      return total_control_edges;
    }


    //Precondition: must be valid with representative DPU task //
    void transform(concat_subgraph_t& subgraph, size_t cmx_size=917504UL) {

      if (!is_cmx_concateable_in_current_opmodel(subgraph, cmx_size)) {
        throw exception_t("Precondition violation: ");
      }

      mv::OpModel &om = omodel_;
      operation_t d_star = subgraph.representative_dpu_;
      size_t cmx_concat_buffer_size = subgraph.master_buffer_size();

      op_list_t& reads = subgraph.reads_;
      op_list_t& writes = subgraph.writes_;

      // STEP-0: remove the reads and writes //
      for (auto ritr=reads.begin(); ritr!=reads.end(); ++ritr) {
        short_circuit_read_or_write(*ritr);
      }
      for (auto witr=writes.begin(); witr!=writes.end(); ++witr) {
        short_circuit_read_or_write(*witr);
      }
      for (auto ritr=reads.begin(); ritr!=reads.end(); ++ritr) {
        om.removeOp(om.getOp( (*ritr)->getName() ));
      }
      for (auto witr=writes.begin(); witr!=writes.end(); ++witr) {
        om.removeOp(om.getOp( (*witr)->getName() ));
      }

      // STEP-1: change the memory location concat from DDR to CMX //
      mv::Tensor::MemoryLocation cmx_location("NNCMX");
      mv::Data::OpListIterator root_concat_itr =
          om.getOp(subgraph.concat_root_->getName());

      root_concat_itr->getOutputTensor(0UL)->set<mv::Tensor::MemoryLocation>(
          "Location", cmx_location);
      if (d_star->hasAttr("splitStrategy") &&
            (d_star->get<std::string>("splitStrategy") == "Clustering") ) {
        root_concat_itr->getOutputTensor(0UL)->set<bool>(
            cmx_concat_buffer_attribute(), true);
        for (auto ditr=subgraph.dpu_out_.begin(); ditr!=subgraph.dpu_out_.end();
              ++ditr) {
          auto dpu_itr = om.getOp((*ditr)->getName());
          dpu_itr->set<bool>(cmx_concat_reader_attribute(), true);
        }
      }

      //short_circuit_crops_into_cmx_concat(subgraph);
    }

  private:


    void trim_and_add_to_ignored_concats(size_t search_ptr, size_t comma_idx,
        const std::string& ignore_list) {
      if (!comma_idx) { return; }

      size_t start_idx = search_ptr, end_idx = comma_idx-1UL;

      while (start_idx <= end_idx) {
        // trim any white spaces begining or at the end //
        if (std::isspace( ignore_list[start_idx] )) {
          ++start_idx;
        } else if (std::isspace( ignore_list[end_idx] )) {
          --end_idx;
        } else {
          break;
        }
      }

      if (start_idx <= end_idx) {
        // trimmed string between [start_idx, end_idx] //
        std::string key =
            ignore_list.substr(start_idx, (end_idx - start_idx) + 1UL);
        ignore_these_concats_.insert(key);
      }
    }

    void populate_ignore_list(const std::string& ignore_list) {
      size_t search_ptr  = 0UL, comma_idx = ignore_list.length();

      while ((comma_idx = ignore_list.find_first_of(',', search_ptr))
            != std::string::npos) {
        trim_and_add_to_ignored_concats(search_ptr, comma_idx, ignore_list);
        search_ptr = (comma_idx + 1UL);
      }

      if (search_ptr < ignore_list.length()) {
        trim_and_add_to_ignored_concats(search_ptr, comma_idx, ignore_list);
      }
    }

    bool has_unit_indegree_and_unit_outdegree(operation_t op) const {
      mv::Data::OpListIterator op_itr = omodel_.getOp(op->getName());
      auto citr = op_itr.leftmostChild();
      bool unit_in_degree = (citr != omodel_.opEnd()) &&
          ((++citr) == omodel_.opEnd());
      if (!unit_in_degree) { return false; }
      auto pitr = op_itr.leftmostParent();
      return (pitr != omodel_.opEnd()) && ((++pitr) == omodel_.opEnd());
    }

    bool has_unit_indegree(operation_t op) const {
      auto op_itr = omodel_.getOp(op->getName());
      auto pitr = op_itr.leftmostParent();
      return (pitr != omodel_.opEnd()) && ((++pitr) == omodel_.opEnd());
    }

    void short_circuit_crops_into_cmx_concat(
        const concat_subgraph_t& subgraph) const {
      mv::OpModel &om = omodel_;
      mv::Data::OpListIterator concat_itr =
          om.getOp(subgraph.concat_root_->getName());
      std::list<operation_t> crops_to_be_removed;
      for (auto pitr=concat_itr.leftmostParent(); pitr!=om.opEnd(); ++pitr) {
        if (pitr->getOpType() == "Crop") {
          short_circuit_read_or_write( &(*pitr) );
          crops_to_be_removed.push_back(&(*pitr));
        }
      }

      // now remove all the crops from the opmodel //
      for (auto crop_itr=crops_to_be_removed.begin();
          crop_itr!=crops_to_be_removed.end(); ++crop_itr) {
        om.removeOp(om.getOp((*crop_itr)->getName()));
      }

      // recompute the shape of the concat output tensor //
      mv::Data::OpListIterator concat_op_itr =
          om.getOp((subgraph.concat_root_)->getName());
      std::string concat_axis = concat_op_itr->get("axis");
      mv::Shape old_shape = (concat_op_itr->getOutputTensor(0UL))->getShape();

      size_t concat_dim = 0UL;
      for (auto ditr=subgraph.dpu_in_.begin(); ditr!=subgraph.dpu_in_.end();
            ++ditr) {
        operation_t op = *ditr;
        mv::Data::TensorIterator tensor_itr =
            (const_cast<non_const_operation_t>(op))->getInputTensor(0UL);
        mv::Shape shape = tensor_itr->getShape();
        concat_dim += shape[shape.getAxis(concat_axis)];
      }

      mv::Shape new_shape = old_shape;
      new_shape[new_shape.getAxis(concat_axis)] = concat_dim;

      (concat_op_itr->getOutputTensor(0UL))->setShape(new_shape);
    }

    void short_circuit_read_or_write(operation_t op) const {

      if (!has_unit_indegree(op)) {
        printf("short_circuit_read_or_write(%s) must be unit-indegree\n",
              op->getName().c_str());
        fflush(stdout);
        throw exception_t("Read/Write must have unit "
              "indegree and unit outdegree");
      }

      // outputIdx: parent->op
      // inputIdx: op->child
      mv::OpModel &om = omodel_;
      mv::Data::OpListIterator op_itr = om.getOp(op->getName());
      auto parent_edge = op_itr.leftmostInput();
      size_t source_output_idx = parent_edge->get<size_t>("sourceOutput");
      mv::Data::OpListIterator parent_op_itr =
          om.getOp(parent_edge->get<std::string>("sourceOp"));
      mv::Data::TensorIterator parent_output_tensor =
          parent_op_itr->getOutputTensor(source_output_idx);

      for (auto child_edge_itr=op_itr.leftmostOutput();
            child_edge_itr!=om.flowEnd(); ++child_edge_itr) {
        mv::Data::OpListIterator child_op_itr = child_edge_itr.sink();
        size_t sink_input_idx = child_edge_itr->get<size_t>("sinkInput");
        child_op_itr->setInputTensor(parent_output_tensor, sink_input_idx,
              false);
        om.defineFlow(parent_op_itr, source_output_idx,
            child_op_itr, sink_input_idx);
      }
    }

    bool is_root_concat(mv::Data::OpListIterator op_itr) const {
      if (!is_implicit_concat(op_itr)) { return false; }
      // no children should be concats //
      for (auto itr=op_itr.leftmostChild(); itr!=omodel_.opEnd(); ++itr) {
        if (is_implicit_concat(itr)) { return false; }
      }
      return true;
    }
   
    template<typename OpIterator>
    bool is_dpu_op(OpIterator op) const {
      return op->getOpType() == "DPUTask";
    }

    template<typename OpIterator>
    bool is_dma_op(OpIterator op) const {
      return op->getOpType() == "DMATask";
    }

    template<typename DpuOutputIterator>
    size_t locate_concating_dpu_tasks(mv::Data::OpListIterator concat_op_itr,
        DpuOutputIterator dpu_output) const {
      noop_back_inserter_t noop_output;
      return locate_dpu_in_and_write_tasks(concat_op_itr, dpu_output,
          noop_output);
    }

    template<typename DpuOutputIterator, typename WriteOutputIterator>
    size_t compute_depth_of_dpu_in_tasks(mv::Data::OpListIterator concat_op_itr,
          DpuOutputIterator dpu_output, WriteOutputIterator dma_output) {
      if (!is_root_concat(concat_op_itr)) { return false; }

      std::list<mv::Data::OpListIterator> bfs_list;
      std::unordered_set<operation_t> marked_nodes;

      bfs_list.push_back(concat_op_itr);
      op_selector_t op_selector;
      size_t total_dpu_concats = 0UL, total_dma_writes = 0UL;

      do {
        mv::Data::OpListIterator curr_op = bfs_list.front();
        bfs_list.pop_front();
        for (mv::Data::OpParentIterator pitr=curr_op.leftmostParent();
              pitr!=omodel_.opEnd(); ++pitr) {
          mv::Data::OpListIterator oitr = omodel_.getOp(pitr->getName());

          if (is_dpu_op(oitr)) {
            *dpu_output = &(*oitr);
            ++total_dpu_concats;
          } else if (op_selector(oitr) &&
              (marked_nodes.find(&(*oitr)) == marked_nodes.end())) {
            if (is_dma_op(oitr)) {
              *dma_output = &(*oitr);
              ++total_dma_writes;
            }
            bfs_list.push_back(oitr);
            marked_nodes.insert(&(*oitr));
          }
        }
      } while (!bfs_list.empty());

      return (total_dpu_concats + total_dma_writes);
    }


    template<typename RealOpOutputIterator>
    size_t locate_real_ops_or_cmx_concats_connected_to_this_op(operation_t op,
          RealOpOutputIterator output) const {

      mv::OpModel &om = omodel_;
      mv::Data::OpListIterator op_itr = om.getOp(op->getName());
        
      std::list<mv::Data::OpListIterator> bfs_list;
      std::unordered_set<operation_t> marked_nodes;

      bfs_list.push_back(op_itr);
      size_t total_real_ops = 0UL;
      do {
        mv::Data::OpListIterator curr_op = bfs_list.front();
        bfs_list.pop_front();
        for (mv::Data::OpParentIterator pitr=curr_op.leftmostParent();
              pitr!=omodel_.opEnd(); ++pitr) {
          mv::Data::OpListIterator oitr = om.getOp(pitr->getName());

          if (marked_nodes.find(&(*oitr)) != marked_nodes.end()) { continue; }

          if ((!oitr->isImplicit()) || is_implicit_concat_in_cmx(oitr)) {
            *output = &(*oitr);
            ++total_real_ops;
          } else {
            bfs_list.push_back(oitr);
          }
          marked_nodes.insert(&(*oitr));
        }
      } while (!bfs_list.empty());

      return total_real_ops;
    }


    //BFS up //
    template<typename DpuOutputIterator, typename WriteOutputIterator>
    size_t locate_dpu_in_and_write_tasks(mv::Data::OpListIterator concat_op_itr,
          DpuOutputIterator dpu_output, WriteOutputIterator dma_output) const {
      if (!is_root_concat(concat_op_itr)) { return false; }

      std::list<mv::Data::OpListIterator> bfs_list;
      std::unordered_set<operation_t> marked_nodes;

      bfs_list.push_back(concat_op_itr);
      op_selector_t op_selector;
      size_t total_dpu_concats = 0UL, total_dma_writes = 0UL;

      do {
        mv::Data::OpListIterator curr_op = bfs_list.front();
        bfs_list.pop_front();
        for (mv::Data::OpParentIterator pitr=curr_op.leftmostParent();
              pitr!=omodel_.opEnd(); ++pitr) {
          mv::Data::OpListIterator oitr = omodel_.getOp(pitr->getName());

          if (is_dpu_op(oitr)) {
            *dpu_output = &(*oitr);
            ++total_dpu_concats;
          } else if (op_selector(oitr) &&
              (marked_nodes.find(&(*oitr)) == marked_nodes.end())) {
            if (is_dma_op(oitr)) {
              *dma_output = &(*oitr);
              ++total_dma_writes;
            }
            bfs_list.push_back(oitr);
            marked_nodes.insert(&(*oitr));
          }
        }
      } while (!bfs_list.empty());

      return (total_dpu_concats + total_dma_writes);
    }

    //BFS down //
    template<typename DpuOutputIterator, typename ReadOutputIterator>
    size_t locate_dpu_out_and_read_tasks(mv::Data::OpListIterator concat_op_itr,
          DpuOutputIterator dpu_output, ReadOutputIterator dma_output) {
      if (!is_root_concat(concat_op_itr)) { return false; }

      std::list<mv::Data::OpListIterator> bfs_list;
      std::unordered_set<operation_t> marked_nodes;

      bfs_list.push_back(concat_op_itr);
      op_selector_t op_selector(true);
      size_t total_dpu_concats = 0UL, total_dma_writes = 0UL;

      do {
        mv::Data::OpListIterator curr_op = bfs_list.front();
        bfs_list.pop_front();
        for (mv::Data::OpChildIterator citr=curr_op.leftmostChild();
              citr!=omodel_.opEnd(); ++citr) {
          mv::Data::OpListIterator oitr = omodel_.getOp(citr->getName());

          if (is_dpu_op(oitr)) {
            *dpu_output = &(*oitr);
            ++total_dpu_concats;
          } else if (op_selector(oitr) &&
              (marked_nodes.find(&(*oitr)) == marked_nodes.end())) {
            if (is_dma_op(oitr)) {
              *dma_output = &(*oitr);
              ++total_dma_writes;
            }
            bfs_list.push_back(oitr);
            marked_nodes.insert(&(*oitr));
          }
        }
      } while (!bfs_list.empty());
      return (total_dpu_concats + total_dma_writes);
    }

    void compute_dpu_depth_map() {
      dpu_depth_map_.clear();
      mv::OpModel &model = omodel_;
      //////////////////////////////////////////////////////////////////////////
      std::list<operation_t> zero_in_degree_nodes[2UL];
      std::unordered_map<operation_t, size_t> in_degree_map;
      size_t curr_depth = 0;

      // STEP-0: compute the in-degree's of all nodes //
      for (auto op_itr = model.opBegin(); op_itr != model.opEnd(); ++op_itr) {
        size_t in_degree = 0;
        for (auto pitr=op_itr.leftmostParent(); pitr!=model.opEnd(); ++pitr) {
          ++in_degree;
        }
        operation_t op = &(*op_itr);
        in_degree_map[ op ] = in_degree;
        if (!in_degree) {
          zero_in_degree_nodes[0].push_back(op);
        }
      }

      while (!zero_in_degree_nodes[curr_depth%2UL].empty()) {
        bool parity = ((curr_depth%2UL) == 1UL);
        for (auto zitr=zero_in_degree_nodes[parity].begin();
              zitr!=zero_in_degree_nodes[parity].end(); ++zitr) {

          // update the in-degree //
          mv::Data::OpListIterator zop_itr = model.getOp((*zitr)->getName());
          for (auto citr=zop_itr.leftmostChild(); citr!=model.opEnd(); ++citr) {
            operation_t cop = &(*citr);
            auto ditr = in_degree_map.find(cop);
            if ( (ditr == in_degree_map.end()) || (ditr->second == 0UL) ) {
              throw "Missing entry in the in-degree map (or)"
                  " invalid in-degree for op= " + cop->getName();
            }
            --(ditr->second);
            if (!(ditr->second)) {
              zero_in_degree_nodes[!parity].push_back(cop);
              if (cop->getOpType() == "DPUTask") {
                dpu_depth_map_[cop] = curr_depth;
              }
            }
          }
        }
        zero_in_degree_nodes[parity].clear();
        curr_depth++;
      }
      //////////////////////////////////////////////////////////////////////////

    }




    // When a one concat is transformed from DDR concat to CMX concat it uses
    // CMX space which needs to locked until all DPUs which read from the
    // concat are finished. So we need charge this in the input space
    // requirement of the DPU op which reads from the concat.
    std::unordered_map<operation_t, size_t> resource_increase_delta_;
    mv::OpModel& omodel_;
    std::unordered_set<std::string> ignore_these_concats_;
    std::unordered_map<operation_t, size_t> dpu_depth_map_;
}; // class CMX_Concatenation //


} // namespace scheduler
} // namespace mv //
#endif
