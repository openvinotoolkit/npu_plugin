#ifndef LP_SCHEDULER_PASS_H
#define LP_SCHEDULER_PASS_H

#include <cerrno>
#include <cstdio>
#include <set>
#include <cstring>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <iterator>


#include "include/mcm/computation/model/base_op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/iterator/tensor.hpp"
#include "include/mcm/logger/logger.hpp"
#include "lp_scheduler/control_edge_generator.hpp"
#include "lp_scheduler/operation_precedence_dag.hpp"
#include "scheduler/dag_address_generator.hpp"
#include "include/mcm/utils/warning_manager.hpp"

namespace mv {
namespace lp_scheduler {

struct Control_Edge {
  Control_Edge(mv::Op const *source, mv::Op const *sink) : source_(source),
    sink_(sink) {}

  bool operator<(const Control_Edge& o) const {
    return (source_ != o.source_) ? (source_ < o.source_) : (sink_ < o.sink_);
  }

  const char* source_name() const { return (source_->getName()).c_str(); }
  const char* sink_name() const { return (sink_->getName()).c_str(); }

  mv::Op const * source_;
  mv::Op const * sink_;
}; // struct Control_Edge //

inline void dump_exception(const char *str) {
  FILE *fptr = fopen("scheduler_exceptions.txt", "a");
  assert(fptr);
  fprintf(fptr, "%s\n", str);
  fclose(fptr);
}


template<typename SchedulerType>
struct Tensor_Address_Assignment {
  typedef SchedulerType scheduler_t;
  typedef typename scheduler_t::scheduled_op_info_t scheduled_op_info_t;

  Tensor_Address_Assignment(mv::ComputationModel& model) : model_(model) {}

  void operator()(const scheduled_op_info_t& op_info) {

    if (!op_info.has_active_resource()) { return; }

    // get its output tensor and set the address //
    mv::Op * const op_ptr = const_cast<mv::Op *>(op_info.op_);
    assert(op_ptr);

    if (op_ptr->getName().find("_spilledWrite") != std::string::npos) {return;}

    mv::Data::TensorIterator tensor_itr = op_ptr->getOutputTensor(0UL);
    assert(op_info.begin_resource() >=1 );
    size_t address = op_info.begin_resource() - 1UL;
    tensor_itr->setAddress( address );
    tensor_itr->set<bool>("lp_scheduler_cmx_address", true);
  }

  void operator()(const Scheduled_Op& op_info) {

    if (!op_info.has_valid_address()) { return; }

    // get its output tensor and set the address //
    mv::Op * const op_ptr = const_cast<mv::Op *>(op_info.op_);
    assert(op_ptr);

    if (op_ptr->getName().find("_spilledWrite") != std::string::npos) {return;}

    mv::Data::TensorIterator tensor_itr = op_ptr->getOutputTensor(0UL);
    assert(op_info.cmx_address_start_ >=1 );
    size_t address = op_info.cmx_address_start_ - 1UL;
    tensor_itr->setAddress( address );
    tensor_itr->set<bool>("lp_scheduler_cmx_address", true);
  }

  mv::ComputationModel &model_;
}; // struct Tensor_Address_Assignment //

struct Tensor_Allocator_Assignment {

  Tensor_Allocator_Assignment(mv::ComputationModel& model)
    : model_(model), address_attributes_() {
      address_attributes_[0UL] = "lp_scheduler_cmx_address";
      address_attributes_[1UL] = "lp_scheduler_ddr_address";
  }

  inline bool has_lp_scheduler_address_attribute(
      mv::Data::TensorIterator tensor_itr, const std::string& attr_name) const {
    return (tensor_itr->hasAttr(attr_name) && tensor_itr->get<bool>(attr_name));
  }

  inline bool has_any_lp_scheduler_address_attributes(
      mv::Data::TensorIterator tensor_itr) const {
    return has_lp_scheduler_address_attribute(
        tensor_itr, address_attributes_[0UL]) ||
      has_lp_scheduler_address_attribute(
          tensor_itr, address_attributes_[1UL]);
  }

  void operator()(mv::Data::TensorIterator tensor_itr) const {
    if(!has_any_lp_scheduler_address_attributes(tensor_itr)) { return; }

    mv::DataModel dm(model_);
    auto & tensor_allocators = tensor_itr->get<std::set<std::string>>("allocators");
    if (tensor_allocators.empty())
      throw mv::ArgumentError("Tensor_Allocator_Assignment", "",  "Tensor Allocators empty", "");
    auto tensor_alloc_name = tensor_allocators.begin();
    auto tensor_alloc= dm.getAllocator(*tensor_alloc_name);
    mv::Data::BufferIterator tensor_buffer_itr =
        tensor_alloc.getBuffer(0, tensor_itr);
    mv::Data::BufferIterator master_tensor_buffer_itr =
        tensor_alloc.getTopMasterBuffer(tensor_buffer_itr);
    master_tensor_buffer_itr->setOffset((tensor_itr->get<size_t>("address")));
  }

  mv::ComputationModel &model_;
  std::string address_attributes_[2UL];
}; // struct Tensor_Allocator_Assignment //

template<>
struct interval_traits<Scheduled_Op> {
  typedef size_t unit_t;
  typedef Scheduled_Op interval_t;

  static unit_t interval_begin(const interval_t& interval) {
    return interval.cmx_address_start_;
  }

  static unit_t interval_end(const interval_t& interval) {
    return interval.cmx_address_end_;
  }
}; // struct interval_traits<Scheduled_Op> //

template<typename OpDag>
class ImplicitConcat_Connected_Component {
  public:
  //////////////////////////////////////////////////////////////////////////////
    typedef OpDag dag_t;
    typedef typename dag_t::operation_t operation_t;
    typedef typename dag_t::const_operation_iterator_t
        const_operation_iterator_t;
    typedef std::unordered_map<operation_t, operation_t> union_find_array_t;
    typedef std::list<operation_t> read_list_t;
    typedef std::unordered_map<operation_t, read_list_t > slave_concat_reads_t;
  //////////////////////////////////////////////////////////////////////////////

    ImplicitConcat_Connected_Component(const dag_t& dag)
      : dag_(dag), union_find_array_(), slave_concat_reads_() { build(); }

    operation_t implicit_concat_root(const operation_t& op) const {
      auto itr = union_find_array_.find(op);
      if (itr == union_find_array_.end()) { 
        dump_exception("[ImplicitConcatRoot]\n");
        throw "[ImplicitConcatRoot] missing operation: " + op->getName();
      }
      return itr->second;
    }

    const read_list_t& slave_reads_of_master_concat(
        const operation_t& op) const {
      typename slave_concat_reads_t::const_iterator itr =
          slave_concat_reads_.find(op);
      if (itr == slave_concat_reads_.end()) {
        dump_exception("[ImplicitConcatInvariant]: invalid master concat\n");
        throw "[ImplicitConcatInvariant]: invalid master concat ";
      }
      return itr->second;
    }
    
  private:

    void build() {
      for (const_operation_iterator_t itr=dag_.begin_nodes();
            itr!=dag_.end_nodes(); ++itr) {
        operation_t op = *itr;
        collapsing_find(op);
      }
      build_slave_concat_reads();
    }

    void collapsing_find(const operation_t& op) {
      if (!(dag_.is_implicit_op(op))) { return; }
      std::list<operation_t> path_to_root;

      if (union_find_array_.find(op) != union_find_array_.end() ) {
        // path already collapsed //
        return;
      }

      operation_t curr_op = op;
      do {
        path_to_root.push_back(curr_op);
      } while ((curr_op = implicit_concat_parent(curr_op)) != NULL);

      if (!path_to_root.empty()) {
        operation_t root = path_to_root.back();
        for (auto itr=path_to_root.begin(); itr!=path_to_root.end(); ++itr) {
          union_find_array_[*itr] = root;
        }
      }
    }

    void build_slave_concat_reads() {
      slave_concat_reads_.clear();
      for (const_operation_iterator_t itr=dag_.begin_nodes();
          itr!=dag_.end_nodes(); ++itr) {

        operation_t op = *itr;
        if (!is_implicit_concat(op))  { continue; }
        typename union_find_array_t::const_iterator uitr =
            union_find_array_.find(op);

        if (uitr == union_find_array_.end()) {
          dump_exception("build_slave_concat_reads\n");
          throw "[build_slave_concat_reads] : "
              "Invalid concat union-find state\n";
        }

        // add all reads from this concat to the slave_read table of the
        // root concat.
        operation_t concat_master = uitr->second;
        typename slave_concat_reads_t::iterator sitr =
            slave_concat_reads_.find(concat_master);
        if (sitr == slave_concat_reads_.end()) {
            sitr = slave_concat_reads_.insert(
                std::make_pair(concat_master, read_list_t())).first;
        }
        read_list_t &slave_read_list = sitr->second;

        for (const_operation_iterator_t citr=dag_.begin_nodes(op);
              citr!=dag_.end_nodes(op); ++citr) {
          operation_t cop = *citr;
          if (dag_.is_output_op(cop)) { continue; }

          bool is_read_or_upa =
            (dag_.is_dma_op_moving_data_from_ddr_to_cmx(cop) || 
              dag_.is_upa_op(cop));

          if (!(is_implicit_concat(cop) || is_read_or_upa) ) {
            FILE *fptr = fopen("non_reads.txt", "a");

            for (const_operation_iterator_t kitr=dag_.begin_nodes(op); 
                kitr != dag_.end_nodes(op); ++kitr) {
              fprintf(fptr, "%s -> %s\n", op->getName().c_str(), 
                  (*kitr)->getName().c_str());
            }
            fclose(fptr);

            dump_exception("[ImplictConcatInvariant]: implict concat reads\n");
            throw "[ImplicitConcatInvariant] : implict concat should only have"
              " dataflow to either concats or reads\n";
          }

          if (is_read_or_upa) {
            slave_read_list.push_back(cop);
          }
        }
      }
    }

   
    operation_t implicit_concat_parent(const operation_t& op) const {
      for (const_operation_iterator_t citr=dag_.begin_nodes(op);
            citr!=dag_.end_nodes(op); ++citr) {
        operation_t cop = *citr;
        if (is_implicit_concat(cop)) {
          return cop;
        }
      }
      return NULL;
    }

    bool is_implicit_concat(const operation_t& op) const {
      return op->getOpType() == "ImplicitConcat";
    }


    const dag_t &dag_;
    union_find_array_t union_find_array_;
    slave_concat_reads_t slave_concat_reads_;
}; // class ImplicitConcat_Connected_Component //



class Control_Edge_Set {
  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef mv::Op const * operation_t;
    typedef mv::model_traits<mv::ControlModel> mtraits;
    typedef mv::model_traits<mv::OpModel> mtraits_op;
    typedef typename mtraits::const_operation_iterator_t op_iterator_t;
    typedef typename mtraits::const_child_operation_iterator_t child_op_itr_t;
    typedef Control_Edge control_edge_t;
    typedef Scheduled_Op scheduled_op_t;
    typedef std::set< control_edge_t > edge_set_t;
    typedef std::unordered_map<operation_t, op_iterator_t> iterator_lookup_t;
    typedef std::unordered_map<operation_t, operation_t> relocating_dma_map_t;
    typedef std::unordered_map<operation_t, size_t> control_in_degree_map_t;
    typedef typename edge_set_t::const_iterator const_edge_iterator_t;
    typedef size_t schedule_time_t;

    template<typename dag_t>
    struct real_op_selector_t {
      bool operator()(const dag_t& in, const operation_t& op) const {
        return !in.is_implicit_op(op) && !in.is_output_op(op);
      }
    }; // struct real_op_selector_t //

    ////////////////////////////////////////////////////////////////////////////

    Control_Edge_Set(mv::ControlModel& cmodel, bool clear_control_edges=true)
      : control_edge_set_(), iterator_lookup_(), relocating_dma_map_(),
      in_degree_(), zero_indegree_temporal_control_(false) {
        init(cmodel, clear_control_edges);
    }

    void operator()(const scheduled_op_t& a, const scheduled_op_t& b) {
      control_edge_set_.insert( control_edge_t(a.op_, b.op_) );
    }

    const_edge_iterator_t begin() const { return control_edge_set_.begin(); }
    const_edge_iterator_t end() const { return control_edge_set_.end(); }

    void set_zero_indegree_temporal_control(bool flag) {
      zero_indegree_temporal_control_ = flag;
    }

    template<typename ControlEdgeIterator>
    void add_control_edges(mv::ComputationModel& model,
        ControlEdgeIterator cbegin, ControlEdgeIterator cend) {

      for (; cbegin != cend; ++cbegin) {
        const control_edge_t& cedge = *cbegin;
        add_control_edge(cedge.source_, cedge.sink_, model);
      }

    }

    template<typename ControlEdgeIterator>
    void add_control_edges_op_iterator(mv::ComputationModel& model,
        ControlEdgeIterator cbegin, ControlEdgeIterator cend) {

      mv::OpModel om(model);
      mv::ControlModel cm(model);

      for (; cbegin != cend; ++cbegin) {
        typename ControlEdgeIterator::value_type cedge = *cbegin;

        if (cedge.source_itr_ == om.opEnd()) {
          printf("source_itr = om.opEnd()\n");
        } else {
          printf("source_itr = %s\n", (cedge.source_itr_)->getName().c_str());
        }

        if (cedge.sink_itr_ == om.opEnd()) {
          printf("sink_itr = om.opEnd()\n");
        } else {
          printf("sink_itr = %s\n", (cedge.sink_itr_)->getName().c_str());
        }
        fflush(stdout);

        assert( (cedge.source_itr_ != om.opEnd()) &&
              (cedge.sink_itr_ != om.opEnd()) );

        operation_t source = &(*(cedge.source_itr_));
        operation_t sink = &(*(cedge.sink_itr_));
        add_control_edge(source, sink, cm);
      }

    }

    template<typename OpDag, typename ScheduledOpIterator>
    void add_cmx_memory_control_edges(
        const OpDag& dag, mv::ComputationModel& model,
        ScheduledOpIterator sbegin, ScheduledOpIterator send,
        bool generate_temporal_edges=true) {

      mv::ControlModel cm(model);
      mv::OpModel om(model);

      add_control_edges_between_compute_ops_and_relocating_dmas(dag, model);
      add_control_edges_between_inputs_and_compute_ops(dag, model);

      if (!generate_temporal_edges) {
        add_control_edges_between_compute_ops_and_writes(dag, model);
        add_memory_control_edges(dag, model, sbegin, send);
        add_control_edges_for_implicit_concats_general(dag, model);
        add_control_edges_for_upa_tasks(dag, model);
      } else {
        add_temporal_control_edges(dag, sbegin, send, model,
            zero_indegree_temporal_control_);
      }
    }

    template<typename OpDag, typename ScheduledOpIterator>
    void add_ddr_memory_control_edges(
        const OpDag& dag, mv::ComputationModel& model,
        ScheduledOpIterator sbegin, ScheduledOpIterator send, FILE *fptr=NULL) {
      add_control_edges_between_writes_and_reads(dag, model);
      add_memory_control_edges(dag, model, sbegin, send, fptr);
    }

  private:


    template<typename dag_t> 
    void add_control_edges_between_writes_and_reads(const dag_t& dag,
        mv::ComputationModel& model) {

      for (typename dag_t::const_operation_iterator_t itr=dag.begin_nodes();
          itr!=dag.end_nodes(); ++itr) {
        operation_t op = *itr;
        if (!dag.is_dma_op_moving_data_from_cmx_to_ddr(op)) { continue; }

        for (typename dag_t::const_operation_iterator_t
              citr=dag.begin_nodes(op); citr!=dag.end_nodes(op); ++citr) {
          operation_t cop = *citr;
          if (dag.is_dma_op_moving_data_from_ddr_to_cmx(cop) ||
              dag.is_upa_op(cop)) {
            add_control_edge(op, cop, model);
          }
        }
      }
    }

    template<typename OpDag, typename ScheduledOpIterator>
    void add_memory_control_edges(const OpDag& input_dag,
        mv::ComputationModel& model,
        ScheduledOpIterator sbegin, ScheduledOpIterator send, FILE *fptr=NULL) {
      typedef real_op_selector_t<OpDag> op_selector_t;
      typedef Memory_Control_Edge_Generator<OpDag, op_selector_t> generator_t;
      typedef typename generator_t::memory_control_edge_t memory_control_edge_t;

      std::list<memory_control_edge_t> final_control_edges;
      generator_t generator;

      generator.generate(input_dag, sbegin, send,
          std::back_inserter(final_control_edges));

      for (auto eitr=final_control_edges.begin();
            eitr!=final_control_edges.end(); ++eitr) {
        add_control_edge(eitr->source_, eitr->sink_, model);
      }

      if (fptr) {
        for (auto eitr=final_control_edges.begin();
              eitr!=final_control_edges.end(); ++eitr) {
          fprintf(fptr, "[MemoryControlEdge] %s -> %s \n",
              (eitr->source_)->getName().c_str(),
              (eitr->sink_)->getName().c_str());
        }
      }
    }


    template<typename OpDag, typename ScheduledOpIterator>
    void add_memory_control_edges_old(
        const OpDag& input_dag, mv::ComputationModel& model,
        ScheduledOpIterator sbegin, ScheduledOpIterator send) {

      typedef OpDag dag_t;
      typedef std::unordered_map<operation_t, scheduled_op_t>
          original_schedule_map_t;
      typedef mv::lp_scheduler::Control_Edge_Generator<scheduled_op_t>
          control_edge_generation_algo_t;

      control_edge_generation_algo_t algo;

      control_edge_set_.clear();
      algo.generate_control_edges(sbegin, send, *this);


      original_schedule_map_t original_schedule;

      // STEP-1: save the schedule //
      for (; sbegin != send; ++sbegin) {
        const scheduled_op_t& sop = *sbegin;
        typename original_schedule_map_t::iterator os_itr =
            original_schedule.find(sop.op_);
        assert(os_itr == original_schedule.end());
        original_schedule.insert(std::make_pair(sop.op_, sop));
      }



      // STEP-2: for each control edge generated (u, v) do the following
      //  (a) call add_control_edge(u, v)
      //  (b) let t_u and t_v be the schedule times of u and v then 
      //      for all the nodes X = { w | (u, w) \in E and t_u <= t_w < t_v }
      //      call add_control_edge(x, v) , x \in X.

      for (const_edge_iterator_t eitr=begin(); eitr!=end(); ++eitr) {
        // control model node iterators //
        operation_t source_op = eitr->source_;
        operation_t sink_op = eitr->sink_;

        if (input_dag.is_implicit_op(sink_op) ||
            input_dag.is_output_op(sink_op)) { continue; }

        add_control_edge(source_op, sink_op, model); // (a) //
        schedule_time_t source_time, sink_time;

        // source time //
        typename original_schedule_map_t::const_iterator itr;
        itr = original_schedule.find(source_op);
        assert(itr != original_schedule.end());
        source_time = (itr->second).schedule_time_;

        // sink time //
        itr = original_schedule.find(sink_op);
        assert(itr != original_schedule.end());
        sink_time = (itr->second).schedule_time_;

        for (typename dag_t::const_operation_iterator_t
            cop_itr =  input_dag.begin_nodes(source_op);
            cop_itr != input_dag.end_nodes(source_op); ++cop_itr) {
          const operation_t& child_op = *(cop_itr);
          itr = original_schedule.find(child_op);
          schedule_time_t child_time = (itr->second).schedule_time_;

          if ( !( (child_time > source_time) && 
                  (child_time < sink_time) ) ) { continue; }
          if (input_dag.is_implicit_op(child_op) ||
              input_dag.is_output_op(sink_op)) { continue; }
          // add control edge //
          add_control_edge(child_op, sink_op, model); // (a) //
        }
      }
    }

    template<typename OpDag, typename ScheduledOpIterator>
    inline size_t add_temporal_control_edges_only_to_output(
        const OpDag& input_dag,
        ScheduledOpIterator sbegin, ScheduledOpIterator send,
        mv::ComputationModel& model, bool zero_indegree_temporal_edges=false) {
      return add_temporal_control_edges(input_dag, sbegin, send, model,
          zero_indegree_temporal_edges, true);
    }

    // The scheduled ops are ordered by their start time //
    template<typename OpDag, typename ScheduledOpIterator>
    size_t add_temporal_control_edges(const OpDag& input_dag,
        ScheduledOpIterator sbegin, ScheduledOpIterator send,
        mv::ComputationModel& model, bool zero_indegree_temporal_edges=false,
        bool add_temporal_edges_only_for_output=false) {

      static_assert(std::is_same<typename ScheduledOpIterator::value_type,
          scheduled_op_t>::value, "Invalid ScheduledOpIterator");

      if (sbegin == send) { return 0UL; }

      std::list<scheduled_op_t> curr_scheduled_real_ops,
          prev_scheduled_real_ops;
      size_t curr_time = (*sbegin).schedule_time_;
      size_t total_temporal_control_edges = 0UL;

      for (; sbegin != send; ++sbegin) {
        const scheduled_op_t& curr_op = *sbegin;

        if ((curr_op.schedule_time_) != curr_time) {
          assert(curr_op.schedule_time_ > curr_time); // monotonic //

          if (!curr_scheduled_real_ops.empty()) {
            prev_scheduled_real_ops = curr_scheduled_real_ops;
          }
          curr_time = curr_op.schedule_time_;
          curr_scheduled_real_ops.clear();
        }


        if ( !input_dag.is_implicit_op(curr_op.op_)) {
          curr_scheduled_real_ops.push_back(curr_op);

          if (!zero_indegree_temporal_edges ||
              (in_degree_.find(curr_op.op_) == in_degree_.end()) ) {
            // add control edges between prev scheduled real ops and current
            // real op.
            for (auto oitr=prev_scheduled_real_ops.begin();
                  oitr!=prev_scheduled_real_ops.end(); ++oitr ) {
              if (!add_temporal_edges_only_for_output || 
                    ((curr_op.op_)->getOpType() == "Output")) {
                add_control_edge(oitr->op_, curr_op.op_, model);
                ++total_temporal_control_edges;
              }
            } 
          }
        }
      }

      return total_temporal_control_edges;
    }

    bool add_control_edge(const operation_t& source, const operation_t& sink,
        mv::ComputationModel& model) {
      mv::ControlModel cmodel(model);
      iterator_lookup_t::const_iterator source_itr =
        iterator_lookup_.find(source);
      iterator_lookup_t::const_iterator sink_itr =
        iterator_lookup_.find(sink);

      assert((source_itr != iterator_lookup_.end()) &&
            (sink_itr != iterator_lookup_.end()) );

      op_iterator_t oitr_source=source_itr->second, oitr_sink=sink_itr->second;
      auto flow_itr = cmodel.checkControlFlow(oitr_source, oitr_sink);

      bool edge_added = false;
      //TODO(vamsikku): there is no-need for check
      // !comodel.pathExists(oitr_sink, oitr_source), however the calling the
      // calling this (CosumerControl) need to check avoiding edges between
      // the sibiling and then this check can be removed.
#if 0
      if ( (flow_itr == cmodel.flowEnd()) &&
          !(cmodel.pathExists(oitr_source, oitr_sink)) ) {
        if (cmodel.pathExists(oitr_sink, oitr_source)) {
          printfInfo("LpScheduler:",
              "[cycle : edge (sink<-source) = (%s <- %s)]\n",
              sink->getName().c_str(), source->getName().c_str());
          throw "[LpScheduler] unexpected cycle in the control DAG ";
        }
        cmodel.defineFlow(oitr_source, oitr_sink);
        edge_added = true;
      }
#endif

      if (flow_itr == cmodel.flowEnd()) {
        cmodel.defineFlow(oitr_source, oitr_sink);
        edge_added = true;
      }

      if (edge_added) {
        typename control_in_degree_map_t::iterator in_degree_itr =
            in_degree_.find(sink);

        if (in_degree_itr == in_degree_.end()) {
          in_degree_itr = (in_degree_.insert(std::make_pair(sink, 0UL))).first;
        }
        (in_degree_itr->second)++;
      }

      return edge_added;
    }

    void clear_all_edges_in_control_model(mv::ComputationModel& model) {
      in_degree_.clear();
      mv::ControlModel cm(model);
      mv::Control::FlowListIterator fitr, fitr_next;
      for (fitr=cm.flowBegin(); fitr!=cm.flowEnd();) {
        fitr_next = fitr; ++fitr_next;
        cm.undefineFlow(fitr);
        fitr = fitr_next;
      }
    }

    template<typename OpDag>
    void add_control_edges_between_inputs_and_compute_ops(const OpDag& dag,
        mv::ComputationModel& model) {
      typedef OpDag dag_t;

      for (typename dag_t::const_operation_iterator_t itr=dag.begin_nodes();
          itr!=dag.end_nodes(); ++itr) {
        operation_t op = *itr;
        if (!dag.resource_utility(op)) { continue; }

        // add control edges from inputs to this compute op //
        for (typename dag_t::const_operation_iterator_t
            citr=dag.begin_nodes(op); citr != dag.end_nodes(op); ++citr) {
          operation_t child_op = *citr;
          if (!dag.is_dpu_op(child_op)) { continue; }

          printfInfo("LpScheduler:",
              "[AddInputEdges(%s -> %s)]\n", (op->getName()).c_str(),
              (child_op->getName()).c_str());
          add_control_edge(op, child_op, model);
        }
      }
    }

    bool is_valid_task_in_blob(operation_t op) const {
      bool val = (op->getOpType() == "UPATask") || (op->getOpType() == "DMATask") ||
        (op->getOpType() == "DPUTask");
      return val;
    }

    template<typename OpDag>
    void add_control_edges_for_upa_tasks(const OpDag& dag,
        mv::ComputationModel& model) {
      typedef OpDag dag_t;

      for (typename dag_t::const_operation_iterator_t itr=dag.begin_nodes();
          itr!=dag.end_nodes(); ++itr) {
        operation_t op = *itr;
        if (!(op->getOpType() == "UPATask")) { continue; }

        for (typename dag_t::const_operation_iterator_t
            citr=dag.begin_nodes(op); citr != dag.end_nodes(op); ++citr) {
          operation_t child_op = *citr;
          if (!is_valid_task_in_blob(child_op)) { continue; }
          add_control_edge(op, child_op, model);
        }

        for (typename dag_t::const_operation_iterator_t
            pitr=dag.begin_parent_nodes(op); pitr != dag.end_parent_nodes(op);
              ++pitr) {
          operation_t parent_op = *pitr;
          if (!is_valid_task_in_blob(parent_op)) { continue; }
          add_control_edge(parent_op, op, model);
        }
      }
    }

   
    // This case works if the structure of concats is a Tree// 
    template<typename OpDag>
    void add_control_edges_for_implicit_concats_tree(const OpDag& dag,
        mv::ComputationModel& model) {
      typedef OpDag dag_t;
      typedef ImplicitConcat_Connected_Component<dag_t> connected_comp_algo_t;
      typedef typename connected_comp_algo_t::read_list_t read_list_t;

      connected_comp_algo_t concat_connected_comp(dag);

      for (typename dag_t::const_operation_iterator_t itr=dag.begin_nodes();
          itr!=dag.end_nodes(); ++itr) {
        operation_t op = *itr;
        if (!(op->getOpType() == "ImplicitConcat")) { continue; }

        operation_t master_op =
            concat_connected_comp.implicit_concat_root(op);
        const read_list_t& master_children =
            concat_connected_comp.slave_reads_of_master_concat(master_op);

        // Add control edges between parents and reads of master concat//
        for (typename dag_t::const_operation_iterator_t
            pitr=dag.begin_parent_nodes(op); pitr != dag.end_parent_nodes(op);
              ++pitr) {
          operation_t parent_op = *pitr;
          if (dag.is_implicit_op(parent_op)) { continue; }

          // this list also includes slave reads //
          for (auto citr=master_children.begin(); citr!=master_children.end();
                ++citr) { 
            add_control_edge(parent_op, *citr, model);
          }
        }

      }
    }


    // This case works if the structure of concats is a DAG //
    struct in_out_adjacency_list_t {
      std::set<operation_t> in_coming_;
      std::set<operation_t> out_going_;
      void clear() { in_coming_.clear(); out_going_.clear(); }
    }; // struct in_out_adjacency_list_t //

    template<typename OpDag>
    void add_control_edges_for_implicit_concats_general(const OpDag& dag,
        mv::ComputationModel& model) {
      typedef OpDag dag_t;
      typedef std::unordered_map<operation_t, in_out_adjacency_list_t>
          concat_sub_graph_t;
      concat_sub_graph_t concat_sub_graph;

      //STEP-0: Build a subgraph with ImplicitConcats and nodes adjacent on
      //them//
      for (typename dag_t::const_operation_iterator_t itr=dag.begin_nodes();
          itr!=dag.end_nodes(); ++itr) {
        operation_t op = *itr;
        if (!(op->getOpType() == "ImplicitConcat")) { continue; }

        in_out_adjacency_list_t &in_out_adj_list = concat_sub_graph[op]; 

        for (typename dag_t::const_operation_iterator_t
            pitr=dag.begin_parent_nodes(op); pitr != dag.end_parent_nodes(op);
              ++pitr) {
          operation_t parent_op = *pitr;
          in_out_adj_list.in_coming_.insert(parent_op);
          concat_sub_graph[parent_op].out_going_.insert(op);
        }

        for (typename dag_t::const_operation_iterator_t
            citr=dag.begin_nodes(op); citr != dag.end_nodes(op); ++citr) {
          operation_t child_op = *citr;
          in_out_adj_list.out_going_.insert(child_op);
          concat_sub_graph[child_op].in_coming_.insert(op);
        }
      }

      //STEP-1: eliminate all the implicit concats from the subgraph but 
      //still retaining the dependencies.
      for (concat_sub_graph_t::iterator nitr=concat_sub_graph.begin();
            nitr != concat_sub_graph.end(); ++nitr) {
        operation_t op = nitr->first;
        if (!(op->getOpType() == "ImplicitConcat")) { continue; }

        // eliminate concat from the subgraph //
        in_out_adjacency_list_t &in_out_adj_list = nitr->second;
        std::set<operation_t>& in_coming = in_out_adj_list.in_coming_;
        std::set<operation_t>& out_going = in_out_adj_list.out_going_;

        for (auto pitr=in_coming.begin(); pitr!=in_coming.end(); ++pitr) {
          operation_t parent_op = *pitr;
          (concat_sub_graph[parent_op]).out_going_.erase(op);
        }

        for (auto citr=out_going.begin(); citr!=out_going.end(); ++citr) {
          operation_t child_op = *citr;
          (concat_sub_graph[child_op]).in_coming_.erase(op);
        }

        // explode control edges due o removal of ImplicitConcat //
        for (auto pitr=in_coming.begin(); pitr!=in_coming.end(); ++pitr) {
          for (auto citr=out_going.begin(); citr!=out_going.end(); ++citr) {
            // add an edge (pitr->citr) //
            operation_t src = *pitr;
            operation_t sink = *citr;
            (concat_sub_graph[sink]).in_coming_.insert(src);
            (concat_sub_graph[src]).out_going_.insert(sink);

            if (!(src->getOpType() == "ImplicitConcat") && 
                  !(sink->getOpType() == "ImplicitConcat")) {
              add_control_edge(src, sink, model);
            }
          }
        }
        in_out_adj_list.clear();
      } // foreach ImplicitConcat //
    }

    template<typename OpDag>
    void add_control_edges_between_compute_ops_and_writes(
        const OpDag& dag, mv::ComputationModel& model) {
      typedef OpDag dag_t;

      for (typename dag_t::const_operation_iterator_t itr=dag.begin_nodes();
          itr!=dag.end_nodes(); ++itr) {
        operation_t op = *itr;
        if (!dag.is_dpu_op(op)) { continue; }

        for (typename dag_t::const_operation_iterator_t 
              citr=dag.begin_nodes(op); citr!=dag.end_nodes(op); ++citr) {
          operation_t cop = *citr;
          if (dag.is_dma_op_moving_data_from_cmx_to_ddr(cop)) {
            add_control_edge(op, cop, model);
          }
        }
      }
    }

    // Since the DMATasks which copy data from CMX2DDR does not use any 
    // resource the control edges will be missing. So we detect this case
    // and add control edges.
    template<typename OpDag>
    void add_control_edges_between_compute_ops_and_relocating_dmas(
        const OpDag& dag, mv::ComputationModel& model) {
      typedef OpDag dag_t;


      for (typename dag_t::const_operation_iterator_t itr=dag.begin_nodes();
          itr!=dag.end_nodes(); ++itr) {

        operation_t op = *itr;
        if (!dag.is_output_of_this_compute_op_relocated(op)) { continue; }

        operation_t cop = dag.get_output_relocating_dma_op(op);

        // add a control edge between op and cop //
        add_control_edge(op, cop, model);

        assert(relocating_dma_map_.find(op) == relocating_dma_map_.end());

        relocating_dma_map_.insert(std::make_pair(op, cop));

        // update iterator so that it redirects to cop //
        printfInfo("LpScheduler:",
            "[redirecting %s to %s]\n", (op->getName()).c_str(),
              (cop->getName()).c_str());
      }
    }

    operation_t get_redirected_source(operation_t source) const {
      relocating_dma_map_t::const_iterator map_itr =
          relocating_dma_map_.find(source);
      return (map_itr == relocating_dma_map_.end()) ? source :
          map_itr->second;
    }

    void init(mv::ControlModel& cmodel, bool clear_control_edges) {
      iterator_lookup_.clear();
      for (op_iterator_t itr=mtraits::begin_operations(cmodel);
          itr!=mtraits::end_operations(cmodel); ++itr) {
        operation_t op = &(*itr);
        assert(iterator_lookup_.find(op) == iterator_lookup_.end());
        iterator_lookup_.insert(std::make_pair(op, itr));
      }
      if (clear_control_edges) {
        clear_all_edges_in_control_model(cmodel);
      }
    }


    std::set< control_edge_t > control_edge_set_;
    iterator_lookup_t iterator_lookup_;
    relocating_dma_map_t relocating_dma_map_;
    control_in_degree_map_t in_degree_;
    bool zero_indegree_temporal_control_;
}; //  class Control_Edge_Set //


// Given a sequence (schedule) of scheduled ops (Scheduled_Op) with some //
template<typename OpDag>
class Dynamic_Spill_Node_Inserter {
  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef OpDag dag_t;
    typedef Scheduled_Op scheduled_op_t;
    typedef typename dag_t::operation_t operation_t;
    typedef typename dag_t::const_operation_iterator_t
        const_operation_iterator_t;
    typedef std::list<operation_t> op_list_t;
    typedef typename op_list_t::iterator op_list_iterator_t;

    struct implicit_sub_structure_t {
      implicit_sub_structure_t(operation_t head=NULL, operation_t tail=NULL)
        : head_(head), tail_(tail) {} 

      bool is_valid() const { return (head_!=NULL) && (tail_!=NULL); }
      bool operator<(const implicit_sub_structure_t& o) const {
        return (head_ != o.head_) ? (head_ < o.head_) : (tail_ < o.tail_);
      }
      operation_t head_;
      operation_t tail_;
    };

    typedef std::unordered_map<operation_t, operation_t> effective_child_map_t;
    typedef std::set<operation_t> effective_children_t;
    struct spilled_read_subtree_t {
      spilled_read_subtree_t() : read_op_(), consumer_list_(),
        implicit_sub_structure_() {}

      spilled_read_subtree_t(operation_t read_op)
        : read_op_(read_op), consumer_list_(), implicit_sub_structure_() {}

      spilled_read_subtree_t(const implicit_sub_structure_t& substr)
        : read_op_(), consumer_list_(), implicit_sub_structure_(substr) {}

      spilled_read_subtree_t(const spilled_read_subtree_t& o)
        : read_op_(o.read_op_), consumer_list_(o.consumer_list_), 
          effective_children_(o.effective_children_), 
          effective_child_map_(o.effective_child_map_) {}

      spilled_read_subtree_t& operator=(const spilled_read_subtree_t& o) {
        read_op_ = o.read_op_; consumer_list_ = o.consumer_list_;
        effective_children_ = o.effective_children_;
        effective_child_map_ = o.effective_child_map_;
      }

      void add_spill_read_consumer(operation_t consumer) {
        consumer_list_.push_back(consumer);
      }

      bool has_implicit_sub_structure() const {
        return implicit_sub_structure_.is_valid();
      }

      bool is_refined() const { return has_implicit_sub_structure(); }

      void print() const {
        printfInfo("LpScheduler:",
            "[read_op=%s]->{ ", read_op_->getName().c_str());
        for (auto itr=consumer_list_.begin(); itr!=consumer_list_.end();
              ++itr){
          printfInfo("LpScheduler:", " %s ", (*itr)->getName().c_str());
        }
        printfInfo("LpScheduler:", " }\n");
      }

      // Precondition: effective_child_map_ must be populated //
      void compute_effective_children() {
        effective_children_.clear();
        for (auto citr=consumer_list_.begin(); citr!=consumer_list_.end();
              ++citr) {
          auto eitr = effective_child_map_.find(*citr);
          if (eitr == effective_child_map_.end()) {
            effective_children_.insert(*citr);
          } else {
            effective_children_.insert(eitr->second);
          }
        }
      }

      operation_t read_op_;
      op_list_t consumer_list_;
      implicit_sub_structure_t implicit_sub_structure_;
      effective_child_map_t effective_child_map_;
      effective_children_t effective_children_;
    }; // struct spilled_read_subtree_t //
    typedef std::list<spilled_read_subtree_t> spilled_read_subtrees_t;
    typedef typename spilled_read_subtrees_t::iterator
        spilled_read_subtrees_iterator_t;

    struct spilled_subtree_t {

      spilled_subtree_t() : spilled_write_op_(), read_subtrees_() {}
      spilled_subtree_t(operation_t spilled_write_op)
        : spilled_write_op_(spilled_write_op), read_subtrees_() {}

      void add_spilled_read(operation_t read_op) {
        read_subtrees_.push_back(spilled_read_subtree_t(read_op));
      }

      // Adds a new consumer to the last read subtree //
      void add_spill_read_consumer(operation_t consumer) {
        assert(!read_subtrees_.empty());
        (read_subtrees_.back()).add_spill_read_consumer(consumer);
        children_.insert(consumer);
      }

      bool has_child(operation_t op) const {
        return (children_.find(op) != children_.end());
      }

      //Precondition: has_valid_write() //
      void print() const {
        printfInfo("LpScheduler:",
            "[write_op=%s]\n", spilled_write_op_->getName().c_str());
        for (auto ritr=read_subtrees_.begin(); ritr!=read_subtrees_.end();
              ++ritr) {
          ritr->print();
        }
        printfInfo("LpScheduler:", "\n");
      }

      bool has_valid_write() const { return spilled_write_op_ != NULL; }

      operation_t spilled_write_op_;
      spilled_read_subtrees_t read_subtrees_;
      std::unordered_set<operation_t> children_;
    }; // struct spilled_subtree_t //

    typedef std::unordered_map<operation_t, spilled_subtree_t> spilled_op_map_t;
    typedef typename spilled_op_map_t::const_iterator
        spilled_op_map_const_iterator_t;
    typedef std::unordered_map<operation_t, std::string> redundant_spill_map_t;
    ////////////////////////////////////////////////////////////////////////////

    Dynamic_Spill_Node_Inserter(
        const dag_t& input_dag, mv::ComputationModel& model)
      : model_(model), spilled_op_map_(), input_dag_(input_dag),
        redundant_spill_map_() {}

    template<typename ScheduledOpIterator>
    void add_spill_read_write_ops(
        ScheduledOpIterator sbegin, ScheduledOpIterator send) {

      compute_spill_subtrees(sbegin, send);

      // now for each subtree structure update the model //
      for (typename spilled_op_map_t::iterator itr = spilled_op_map_.begin();
            itr != spilled_op_map_.end(); ++itr) {
        const operation_t& spilled_op = itr->first;

        if (input_dag_.is_op_reading_weights(spilled_op)) {
          create_spill_read_forest_structure_in_model(spilled_op, itr->second);
        } else {
          // this is for compute ops //
          create_spill_subtree_structure_in_model(spilled_op, itr->second);
        }
      }
    }

    //NOTE: this will update the scheduled_op_t entries in the schedule. For
    //non-redundant spilled reads and writes a new mv::Op is created and for the
    //reundant ops in the schedule the corresponding corresponding op_ is set to
    //NULL. The client can use this to purge redundant ops in the schedule.//
    template<typename ScheduledOpIterator, typename ControlEdgeBackInserter>
    void update_scheduled_ops_with_new_read_write_ops(
        ScheduledOpIterator sbegin, ScheduledOpIterator send,
        ControlEdgeBackInserter coutput) {
      typedef ScheduledOpIterator schedule_iterator_t;
      typedef Control_Edge control_edge_t;
      typedef std::unordered_map<operation_t, scheduled_op_t>
          spilled_op_schedule_t;

      static_assert(std::is_same<typename ScheduledOpIterator::value_type,
          scheduled_op_t>::value, "Invalid ScheduledOpIterator");

      spilled_op_schedule_t spilled_op_schedule;

      for (schedule_iterator_t sitr=sbegin; sitr!= send; ++sitr) {
        scheduled_op_t &sop = *sitr;
        if (!has_its_output_spilled(sop.op_)) { continue; }

        // CASE-1: Handle the original spilled op in the schedule.//
        if (sop.is_original_op()){
          if (!is_redundant_spill(sop.op_)) {
            // save the schedule info to lookup later while handling
            // corresponding spilled read or spilled write.
            spilled_op_schedule[sop.op_] = sop;
          } else {
            sop.op_ = NULL; // redundant ops are erased //
          }
          continue;
        }

        // CASE-2: Handle a spilled read and write op in the schedule.//
        typename spilled_op_map_t::iterator itr = spilled_op_map_.find(sop.op_);
        if (itr == spilled_op_map_.end())
          throw RuntimeError("LpScheduler", "Subtree not found.");
        spilled_subtree_t &subtree = itr->second;
        bool is_original_spilled_op_redundant = !(subtree.spilled_write_op_);

        if (sop.is_spilled_write()) {
          // spilled write //

          // CASE-2.1: If this spilled write corresponds to a redundant orignal
          // op then this write becomes redundant as well.
          if (is_original_spilled_op_redundant) {
            assert(subtree.spilled_write_op_ == NULL);
            sop.op_ = NULL;
            continue;
          }

          // CASE-2.2: Typical case of non-redundant spilled write. //

          // Transfer its CMX addresess to generate control edges //
          const scheduled_op_t &sinfo = spilled_op_schedule[sop.op_];
          sop.cmx_address_start_ = sinfo.cmx_address_start_;
          sop.cmx_address_end_ = sinfo.cmx_address_end_;
          coutput = control_edge_t(sop.op_, subtree.spilled_write_op_);
          sop.op_ = subtree.spilled_write_op_;
        } else {
          // spilled read //
          assert(!((subtree.read_subtrees_).empty()));
          operation_t new_read_op = ((subtree.read_subtrees_).front()).read_op_;

          // write edge from write to read op//
          if (!is_original_spilled_op_redundant) {
            assert(subtree.spilled_write_op_);
            coutput = control_edge_t(subtree.spilled_write_op_, new_read_op);
          }

          // add edges from read ops to consumer ops //
          const op_list_t& consumers =
              (subtree.read_subtrees_).front().consumer_list_;
          for (auto consumer_itr=consumers.begin();
                consumer_itr != consumers.end(); ++consumer_itr ) {
            coutput = control_edge_t(new_read_op, *consumer_itr);
          }
          sop.op_ = new_read_op;
          (subtree.read_subtrees_).pop_front(); // subtrees are ordered //
        }

        sop.op_type_ = op_type_e::ORIGINAL_OP;
      }

    } // update_scheduled_ops_with_new_read_write_ops //

    void print() const {
      for (auto itr=spilled_op_map_.begin(); itr!=spilled_op_map_.end();
            ++itr) {
        printf("========================\n");
        if (!has_redundant_spilled_write(itr)) {
          printfInfo("LpSchedulerPass", "[spilled_op=%s]\n",
              (itr->first)->getName().c_str());
          (itr->second).print();
        } 
        printfInfo("LpSchedulerPass", "========================\n");
      }
    }

    void print_redundant_spilled_ops(FILE *out_fptr=stdout) const {
      typename redundant_spill_map_t::const_iterator itr;
      for (itr=redundant_spill_map_.begin(); itr!=redundant_spill_map_.end();
            ++itr) {
        fprintf(out_fptr, "[Redundant Spill] op=%s\n", (itr->second).c_str());
      }
    }

    template<typename ControlEdgeOutput>
    size_t generate_control_edges_for_spilled_cmx_concat_ops(
        ControlEdgeOutput output) {
      typedef typename dag_t::cmx_concat_subgraph_t cmx_concat_subgraph_t;
      typedef mv::lp_scheduler::Control_Edge control_edge_t;

      size_t control_edge_count = 0UL;
      for (auto sitr=spilled_op_map_.begin(); sitr!=spilled_op_map_.end();  
            ++sitr) {
        cmx_concat_subgraph_t const *subgraph_ptr =
          input_dag_.does_this_dpu_have_cmx_concat_subgraph(sitr->first);
        if (subgraph_ptr) {
          operation_t spilled_op = sitr->first;
          operation_t spilled_write_op = (sitr->second).spilled_write_op_;

          if (!spilled_write_op) {
            throw "SpilledWriteOp cannot be NULL";
          }

          // add control edges from all the dpu_in_.ops() to the write op in 
          // the spilled subtree //
          for (auto dpu_itr=(subgraph_ptr->dpu_in_).begin();
              dpu_itr != (subgraph_ptr->dpu_in_).end(); ++dpu_itr) {
            operation_t dop = *dpu_itr;
            control_edge_t cedge(dop, spilled_write_op);

            *output = cedge;
            ++control_edge_count;
          }

        }
      }
      return control_edge_count;
    }

  private:

    bool has_redundant_spilled_write(spilled_op_map_const_iterator_t itr) const
    {
      assert(itr != spilled_op_map_.end());
      return (itr->second).has_valid_write();
    }

    // Takes a schedule and computes the following structure for each op
    // which got spilled:
    //
    // Original DataFlow:
    //
    //
    //   (A)---+------>(c1)
    //         |
    //         +------>(c2)
    //         |                         (F)----|
    //         |                                -> index 0
    //         +----------------------------------->(c4) index 1
    //         |
    //         .
    //         .
    //         +------>(cn)
    //
    // New DataFlow:
    //
    //  {c1,c2}, SPILL_WRITE, SPILL_READ, {c4}, SPILL_READ, {c5,c6} ...
    //
    //  (A)---+------>(c1)
    //        |
    //        +------>(c2)
    //        |
    //        +------>[spill_write]--+----->[spill_read]-+---->{c4} index 1
    //                               |
    //                               +----->[spill_read]-+---->{c5}
    //                                                   |
    //                                                   +---->{c6}
    template<typename ScheduledOpIterator>
    void compute_spill_subtrees(ScheduledOpIterator sbegin,
          ScheduledOpIterator send) {
      typedef typename OpDag::const_operation_iterator_t
          const_operation_iterator_t;

      static_assert(std::is_same<typename ScheduledOpIterator::value_type,
            scheduled_op_t>::value, "Invalid ScheduledOpIterator");

      for (; sbegin != send; ++sbegin) {
        scheduled_op_t &sched_op = *sbegin;
        operation_t op = sched_op.op_;

        if (sched_op.is_original_op()){

          if (!input_dag_.is_dpu_op(op)) { continue; }

          // We have DPU Task we need to check if any of its inputs are spilled
          for (const_operation_iterator_t
                pitr=input_dag_.begin_parent_nodes(op);
                pitr != input_dag_.end_parent_nodes(op); ++pitr) {

            operation_t pop = *pitr;
            if (!has_its_output_spilled(pop)) { continue; }

            typename spilled_op_map_t::iterator itr = spilled_op_map_.find(pop);
            assert(itr != spilled_op_map_.end());

            //now add this op to the spill tree structure //
            (itr->second).add_spill_read_consumer(op);
          }
        } else if (sched_op.is_spilled_read()) {
          typename spilled_op_map_t::iterator itr = spilled_op_map_.find(op);

          // since spilled write must come before this //
          assert(itr != spilled_op_map_.end());
          (itr->second).add_spilled_read(op);
        } else {
          // spilled write op //

          typename spilled_op_map_t::iterator itr = spilled_op_map_.find(op);
          // since the activation data is not changing we don't need to
          // write it back to DDR .
          if (itr != spilled_op_map_.end()) { continue; }

          spilled_op_map_.insert(std::make_pair(op, spilled_subtree_t(op)));
        }

      } //foreach scheduled op //
    }

    bool refine_subtrees_using_path_splitting_uniq_paths(mv::OpModel& om,
        operation_t spilled_op, spilled_read_subtrees_t &read_subtrees) {

      bool atleast_one_subtree_refined = false;
      const dag_t& dag = input_dag_;
      std::unordered_set<operation_t> direct_children;

      //STEP-0: determine directly connected children //
      mv::Data::OpListIterator spilled_op_itr = om.getOp(spilled_op->getName());
      for (auto citr=spilled_op_itr.leftmostChild(); citr!=om.opEnd(); ++citr) {
        operation_t op = &(*citr);
        direct_children.insert(op);
      }


      //STEP-1: Refine each subtree if necessary. //
      spilled_read_subtrees_iterator_t subtree_itr = read_subtrees.begin();

      while (subtree_itr!=read_subtrees.end()) {
        // We have an unrefined spill read structure //
        // 
        // Refinement Procedure:
        // 
        // STEP-1.1: filter the in direct children out leaving only the indirect
        //           children. Note that short-circuiting of implicit ops can
        //           create indirect children.
        //
        // STEP-1.2: om.pathSplit(spill_op_itr, I.begin(), I.end())
        // 
        // STEP-1.3: determine additional reads based on divergence in the paths
        //           since the structure is a tree there is exactly one path
        //           between spilled_op and op in I //
        //
        // STEP-1.4: insert back the refined trees back into the main list of
        // spill read sub trees.
        spilled_read_subtree_t &curr_subtree = *subtree_itr;
        op_list_t& orig_child_list = curr_subtree.consumer_list_;

        // STEP-1.1 //
        std::list< op_list_iterator_t > indirect_children_itrs;
        for (op_list_iterator_t child_itr=orig_child_list.begin();
              child_itr!=orig_child_list.end(); ++child_itr) {
          operation_t child_op = *child_itr;
          if (direct_children.find(child_op) == direct_children.end()) {
            indirect_children_itrs.push_back(child_itr);
          }
        } // foreach children //

       
        // NOTE: the current subtree subtree_itr is now updated and has only
        // direct children //
        std::list< mv::Data::OpListIterator  > indirect_children;
        for (auto iitr=indirect_children_itrs.begin();
              iitr!=indirect_children_itrs.end(); ++iitr) {
          operation_t op = *(*iitr);
          indirect_children.push_back(om.getOp(op->getName()));
          orig_child_list.erase(*iitr);
        }

        if (!indirect_children.empty()) {
          // STEP-1.2 //
          om.pathSplitImplicit(spilled_op_itr,
                indirect_children.begin(), indirect_children.end());

          // STEP-1.3 : we have to create a new read for each unique path from
          // the spilled op to one of these children. The purpose of this loop
          // is to determine these unique paths. 
          //
          //
          // NOTE: since the structure is a tree there is exactly one path from
          // the spilled_op to each of these children and we use the end points
          // to determine which ones are unique.//
          std::map<implicit_sub_structure_t, op_list_t> unique_paths;
          for (auto ichild=indirect_children.begin();
                ichild!=indirect_children.end(); ++ichild) {

            std::list<mv::Data::OpListIterator> implicit_path;
            om.getImplicitPath(spilled_op_itr, *ichild, implicit_path);

            if (implicit_path.empty()){
              throw "Implicit path between " + spilled_op_itr->getName() +
                  " and " + (*ichild)->getName() + " cannot be empty";
            }

            operation_t child = &(*(*ichild));
            operation_t head = &(*(*implicit_path.begin()));
            operation_t tail = &(*(*implicit_path.rbegin()));

            implicit_sub_structure_t key(head, tail);
            unique_paths[ key ].push_back(child); 
            atleast_one_subtree_refined = true;
          }

          // STEP-1.4: //
          spilled_read_subtrees_iterator_t
              next_unrefined_subtree_itr = subtree_itr;
          ++next_unrefined_subtree_itr;
         
          // We insert all the refined subtrees before the next unrefined
          // subtree in the list of subtrees corresponding to spill.
          for (auto uniq_path_itr=unique_paths.begin();
              // uniqu_path_itr->first = implicit_sub_structure_t //
                uniq_path_itr!=unique_paths.end(); ++uniq_path_itr) {
            spilled_read_subtrees_iterator_t refined_subtree_itr =
                read_subtrees.insert(next_unrefined_subtree_itr,
                      spilled_read_subtree_t(uniq_path_itr->first) );
            spilled_read_subtree_t &refined_subtree = *refined_subtree_itr;
            refined_subtree.consumer_list_ = uniq_path_itr->second;
          }

          spilled_read_subtrees_iterator_t curr_subtree_itr = subtree_itr;
          if (orig_child_list.empty()) {
            read_subtrees.erase(curr_subtree_itr);
          }
          subtree_itr = next_unrefined_subtree_itr;
        } else {
          // if subtree has no indirect children then no need of refinement //
          ++subtree_itr;
        }
      } // while (subtree_itr != read_subtrees.end()) //

      return atleast_one_subtree_refined;
    }

    bool refine_subtrees_using_path_splitting(mv::OpModel& om,
        operation_t spilled_op, spilled_read_subtrees_t &read_subtrees) {

      bool atleast_one_subtree_refined = false;
      const dag_t& dag = input_dag_;
      std::unordered_set<operation_t> direct_children;

      //STEP-0: determine directly connected children //
      mv::Data::OpListIterator spilled_op_itr = om.getOp(spilled_op->getName());
      for (auto citr=spilled_op_itr.leftmostChild(); citr!=om.opEnd(); ++citr) {
        operation_t op = &(*citr);
        direct_children.insert(op);
      }


      //STEP-1: Refine each subtree if necessary. //
      spilled_read_subtrees_iterator_t subtree_itr = read_subtrees.begin();

      for (;subtree_itr!=read_subtrees.end(); ++subtree_itr) {
        // We have an unrefined spill read structure //
        // 
        // Refinement Procedure:
        // 
        // STEP-1.1: filter the in direct children out leaving only the indirect
        //           children. Note that short-circuiting of implicit ops can
        //           create indirect children.
        //
        // STEP-1.2: om.pathSplit(spill_op_itr, I.begin(), I.end())
        // 
        // STEP-1.3: for each indirect child determine effective child of the 
        //           spilled op.
        spilled_read_subtree_t &curr_subtree = *subtree_itr;
        op_list_t& orig_child_list = curr_subtree.consumer_list_;

        // STEP-1.1 //
        std::list< mv::Data::OpListIterator  > indirect_children;
        for (op_list_iterator_t child_itr=orig_child_list.begin();
              child_itr!=orig_child_list.end(); ++child_itr) {
          operation_t child_op = *child_itr;
          if (direct_children.find(child_op) == direct_children.end()) {
            indirect_children.push_back(om.getOp(child_op->getName()));
          }
        } // foreach children //

        if (!indirect_children.empty()) {
          // STEP-1.2 //
          om.pathSplitImplicit(spilled_op_itr,
                indirect_children.begin(), indirect_children.end());

          effective_child_map_t &effective_child_map =
              subtree_itr->effective_child_map_;
          for (auto ichild=indirect_children.begin();
                ichild!=indirect_children.end(); ++ichild) {

            std::list<mv::Data::OpListIterator> implicit_path;
            om.getImplicitPath(spilled_op_itr, *ichild, implicit_path);

            if (implicit_path.empty()){
              throw "Implicit path between " + spilled_op_itr->getName() +
                  " and " + (*ichild)->getName() + " cannot be empty";
            }

            operation_t child = &(*(*ichild));
            operation_t head = &(*(*implicit_path.begin()));
            effective_child_map.insert(std::make_pair(child, head));
            atleast_one_subtree_refined = true;
          }
        }
        subtree_itr->compute_effective_children();
      } // foreach read subtree //
      return atleast_one_subtree_refined;
    }

    typedef std::unordered_map<operation_t, size_t> input_tensor_index_map_t;
    typedef typename spilled_read_subtrees_t::iterator
        spilled_read_subtree_iterator_t;

    // Creates a spilled read substructure where there are spilled op is
    // connected to consumers of its output via an implicit op substructure.
    //
    // Precondition:
    // spilled_read_subtree_itr->has_implicit_sub_structure() = true
    void create_spilled_read_substructure_in_model_with_implicit_ops(
        mv::OpModel& om, operation_t spilled_op,
        mv::Data::TensorIterator spill_write_tensor_itr,
        spilled_read_subtree_iterator_t spilled_read_subtree_itr,
        size_t& read_index) {
      /*
      // Once we get to this point the spill_read_subtree_t is already refined
      // and has implicit_sub_structure.
      //
      // Incoming OpModel:
      // 
      //   spilled_op---+-------+
      //                |       |
      //                |       v
      //                |      _spilledWrite(DMA WRITE)
      //                |          
      //                v 
      //           head(implicit) 
      //                |
      //                |
      //   ..complex path of implict ops..
      //               |
      //               v
      //           tail(implict)
      //             / | \
      //            /  |  \
      //           /   |   \
      //          c1    c2  c3
      //
      //
      // Resultant OpModel:
      //
      //
      //   spilled_op---+
      //                |
      //                v
      //              _spilledWrite (DMA WRITE)
      //                |\
      //                v 
      //           head(implicit) 
      //                |
      //                |
      //   ..complex path of implict ops..
      //                |
      //                |
      //                v
      //           tail(implict)
      //                |
      //                v
      //              _spilledRead (new DMA READ)
      //              / | \
      //             /  |  \
      //            /   |   \
      //           c1    c2  c3
      // STEP-1: get the sinkInput index of head
      // STEP-2: remove the data flow between spilled_op and head.
      // STEP-3: create a data flow between _spilledWrite and head 
      // STEP-4: create a lookup table for sinkInput index of c1,c2...cn by
      //         looking at the flows between tail and {c1, c2.... }
      // STEP-5: clear all the flows between tail and {c1, c2, ... }
      // STEP-6: create a new DMA READ and update the read_op value in the
      //         subtree
      // STEP-7: add data flows between _spilledRead and {c1,c2....} use the
      //         lookup table to find out the index.
      */

      mv::Data::OpListIterator spilled_op_itr = om.getOp(spilled_op->getName());

      input_tensor_index_map_t input_tensor_index_map;
      std::vector<mv::Data::FlowListIterator> data_flows_to_erase;
      const implicit_sub_structure_t &implicit_sub_structure=
          spilled_read_subtree_itr->implicit_sub_structure_;
      const op_list_t &children = spilled_read_subtree_itr->consumer_list_;
      operation_t head = implicit_sub_structure.head_;
      operation_t tail = implicit_sub_structure.tail_;

      // locate sinkInput for head //
      for (auto data_flow_itr=spilled_op_itr.leftmostOutput();
            data_flow_itr!=om.flowEnd(); ++data_flow_itr) {
        operation_t sink_op = &(*(data_flow_itr.sink()));
        if (sink_op == head) {
          input_tensor_index_map[head] =
              data_flow_itr->get<size_t>("sinkInput");
          data_flows_to_erase.push_back(data_flow_itr);
          break;
        }
      }

      if (input_tensor_index_map.find(head) == input_tensor_index_map.end()) {
        throw std::string("[ImplicitSubtree] invalid implicit substructure");
      }


      // STEPS-1,2,4,5  are combined.//
      // locate sinkInputs between tail and consumers //
      {
        std::set<operation_t> children_set;
        for (auto citr=children.begin(); citr!=children.end(); ++citr) {
          children_set.insert(*citr);
        }

        mv::Data::OpListIterator tail_itr = om.getOp(tail->getName());
        for (auto data_flow_itr=tail_itr.leftmostOutput();
              data_flow_itr!=om.flowEnd(); ++data_flow_itr) {
          operation_t sink_op = &(*(data_flow_itr.sink()));
          if (children_set.find(sink_op) == children_set.end()) { continue;}
          input_tensor_index_map[sink_op] =
              data_flow_itr->get<size_t>("sinkInput");
          data_flows_to_erase.push_back(data_flow_itr);
        }
      }

      if (input_tensor_index_map.size() != (children.size() + 1UL) ) {
        throw std::string("[ImplicitSubstructure] unable to locate dataflows"
              " forall childrent");
      }
      
      for (auto flow : data_flows_to_erase) om.undefineFlow(flow);
      // STEP-1,2,4,5 are combined above //


      //STEP-3//
      size_t head_sink_input = input_tensor_index_map[head];
      mv::Data::OpListIterator head_itr = om.getOp(head->getName());
      head_itr->setInputTensor(spill_write_tensor_itr, head_sink_input, false);
      om.defineFlow(spill_write_tensor_itr, head_itr, head_sink_input);

      //STEP-6://
      mv::Data::OpListIterator tail_itr = om.getOp(tail->getName());
      mv::Data::TensorIterator tail_tensor_itr = tail_itr->getOutputTensor(0UL);

      mv::DmaDirection read_dma_direction(std::string("DDR2NNCMX"));
      std::string dma_read_op_name = spilled_op->getName() + "-" +
        head->getName() + "_implicitSpilledRead" + std::to_string(read_index++);
      mv::Data::TensorIterator spill_read_tensor_itr =
          om.dMATask(tail_tensor_itr, read_dma_direction, 0, dma_read_op_name);
      mv::Data::OpListIterator read_op_itr =
          om.getSourceOp(spill_read_tensor_itr);
      read_op_itr->setInputTensor(tail_tensor_itr, 0UL, false);

      // save the read op into the structure//
      spilled_read_subtree_itr->read_op_ = &(*read_op_itr);

      // STEP-7 //
      for (auto citr=children.begin(); citr!=children.end(); ++citr) {
        operation_t child_op = *citr;

        if (input_tensor_index_map.find(child_op) ==
              input_tensor_index_map.end()) {
          throw std::string("[ImplicitSpillStructure]: cannot find  " +
                child_op->getName() + " in the input tensor index map");
        }


        size_t idx = input_tensor_index_map[child_op];

        // define flow //
        mv::Data::OpListIterator child_op_itr = om.getOp(child_op->getName());
        child_op_itr->setInputTensor(spill_read_tensor_itr, idx, false);
        om.defineFlow(spill_read_tensor_itr, child_op_itr, idx );
      }

    }

    void create_spilled_read_substructure_in_model(mv::OpModel& om,
        operation_t spilled_op, mv::Data::TensorIterator spill_write_tensor_itr,
        const input_tensor_index_map_t& input_tensor_index_map,
        spilled_read_subtree_iterator_t spilled_read_subtree_itr,
        size_t& read_index) {

      // STEP-0: create a read DMA //
      mv::DmaDirection read_dma_direction(std::string("DDR2NNCMX"));
      std::string dma_read_op_name = spilled_op->getName() + "_spilledRead" +
          std::to_string(read_index++);
      mv::Data::TensorIterator spill_read_tensor_itr =
          om.dMATask(spill_write_tensor_itr, read_dma_direction, 0,
              dma_read_op_name);
      mv::Data::OpListIterator read_op_itr =
          om.getSourceOp(spill_read_tensor_itr);
      read_op_itr->setInputTensor(spill_write_tensor_itr, 0UL, false);
      
      // save the read op into the structure//
      spilled_read_subtree_itr->read_op_ = &(*read_op_itr);
    
      // Use effective children instead of original children:
      const effective_children_t &children =
          spilled_read_subtree_itr->effective_children_;

      // Erase all flows from spilled_op to effective children //
      //////////////////////////////////////////////////////////////////////////
      // STEP-3: erase all outgoing flows from the spilled op and direct 
      // children in spill sub tree
      {
        mv::Data::OpListIterator spilled_op_itr =
            om.getOp(spilled_op->getName());
        std::vector<mv::Data::FlowListIterator> flows;

        for(auto outputFlow = spilled_op_itr.leftmostOutput();
          outputFlow != om.flowEnd(); ++outputFlow) {
          operation_t sink_op = &(*(outputFlow.sink()));
          if (children.find(sink_op) != children.end()) {
            flows.push_back(outputFlow);
          }
        }

        for (auto flow : flows) om.undefineFlow(flow);
      }



      for (auto child=children.begin(); child!=children.end(); ++child) {
        operation_t child_op = *child;
        mv::Data::OpListIterator child_op_itr = om.getOp(child_op->getName());
        // find the input index in the original spilled op //
        auto idx_itr = input_tensor_index_map.find(child_op);
        if (idx_itr == input_tensor_index_map.end()) {
          throw std::string("[RefinementInvariant]: failed effective children "
              " must have a input_tensor_index_map entry");
        }
        size_t idx = idx_itr->second;
        child_op_itr->setInputTensor(spill_read_tensor_itr, idx, false);
        om.defineFlow(spill_read_tensor_itr, child_op_itr, idx);
      }
    }


    // Creates a spill subtree structure under the given op whose output
    // got spilled. Additionally the new write op addresses are added into
    // the substructure
    void create_spill_subtree_structure_in_model(operation_t spilled_op_in,
          spilled_subtree_t& spilled_sub_tree) {
      mv::Op *spilled_op = const_cast<mv::Op *>(spilled_op_in);
      mv::OpModel om(model_);
      spilled_read_subtrees_t &read_subtrees = spilled_sub_tree.read_subtrees_;

      //STEP-(minus 0): if the spilled_op should be the concat //
      typename dag_t::cmx_concat_subgraph_t const *concat_subgraph = NULL;
      if ( (concat_subgraph =
              input_dag_.does_this_dpu_have_cmx_concat_subgraph(spilled_op)) ) {
        spilled_op = const_cast<mv::Op *>(concat_subgraph->concat_root_);
      }

      //STEP-0: refine read subtrees before introducing the spill structure.
      //This will mark the read subtrees into two categories:
      //  a.) subtrees with implicit substructure
      //  b.) subtrees with no implicit substructue
      refine_subtrees_using_path_splitting(om, spilled_op, read_subtrees);

      //////////////////////////////////////////////////////////////////////////
      // STEP-1: create one DMA write op //
      mv::DataModel dm(model_);
      std::string dma_op_name = spilled_op->getName() + "_spilledWrite";
      mv::DmaDirection write_dma_direction(std::string("NNCMX2DDR"));
      mv::Data::TensorIterator spilled_op_output_tensor_itr =
          spilled_op->getOutputTensor(0UL);

      mv::Data::TensorIterator spill_write_tensor_itr = om.dMATask(
          spilled_op_output_tensor_itr, write_dma_direction, 0, dma_op_name);
      Data::OpListIterator write_op_itr =
          om.getSourceOp(spill_write_tensor_itr);
      write_op_itr->setInputTensor(spilled_op_output_tensor_itr, 0UL, false);
      // set a dummy flows attribute //
      {
        std::set<std::string> toSet;
        spill_write_tensor_itr->set<std::set<std::string>>("flows", toSet);
        // clear any address attributes on this tensor //
        if (spill_write_tensor_itr->hasAttr("address")) {
          spill_write_tensor_itr->erase("address");
        }

        if (spill_write_tensor_itr->hasAttr("lp_scheduler_cmx_address")) {
          spill_write_tensor_itr->erase("lp_scheduler_cmx_address");
        }
      }
      // save the write op into subtree structure //
      spilled_sub_tree.spilled_write_op_ = &(*write_op_itr);


      //////////////////////////////////////////////////////////////////////////
      // STEP-2: for all the outgoing ops connected to this op determine the
      // input tensor indexes
      std::unordered_map<operation_t, size_t> input_tensor_index_map;
      {
        mv::Data::OpListIterator spilled_op_itr =
          om.getOp(spilled_op->getName());
        for(auto outputFlow = spilled_op_itr.leftmostOutput();
          outputFlow != om.flowEnd(); ++outputFlow) {
          size_t idx = outputFlow->get<size_t>("sinkInput");
          operation_t sink_op = &(*(outputFlow.sink()));
          input_tensor_index_map[sink_op] = idx;
        }
      }
      //STEP-3: erasing of old flows is now moved to
      //create_spilled_read_substructure

      //////////////////////////////////////////////////////////////////////////
      // STEP-4: create a new spill read ops by connecting spill_write tensor
      // to each of them as inputs.
      size_t read_index = 0UL;

      for (typename spilled_read_subtrees_t::iterator
            spill_read_itr=read_subtrees.begin();
              spill_read_itr!=read_subtrees.end(); ++spill_read_itr) {
        create_spilled_read_substructure_in_model(om, spilled_op_in,
            spill_write_tensor_itr, input_tensor_index_map, spill_read_itr,
              read_index);
      }
    }

    // Creates a read-spill forest without a spilled write op to the DDR //
    //
    //
    //   (read)----->consumer1
    //          |
    //          |
    //          ---->consumer2
    //
    //   (read2)----->consumer3
    //          |
    //          ....
    //          |
    //          +----->consumer-n
    //
    //
    void create_spill_read_forest_structure_in_model(operation_t spilled_op_in,
        spilled_subtree_t& spilled_sub_tree) {

      mv::Op *spilled_op = const_cast<mv::Op *>(spilled_op_in);
      mv::OpModel om(model_);
      mv::DataModel dm(model_);

      // Precondition: this must be a data op. //
      assert(spilled_op->inputSlots() == 1UL);


      //////////////////////////////////////////////////////////////////////////
      // STEP-1: get the input tensor corresponding to the original read-op //
      mv::Data::TensorIterator spilled_op_input_tensor_itr =
          spilled_op->getInputTensor(0UL);

      // A NULL spilled_write_op_ means the write op is redundant//
      spilled_sub_tree.spilled_write_op_ = NULL;


      //////////////////////////////////////////////////////////////////////////
      // STEP-2: for all the outgoing ops connected to this op determine the
      // input tensor indexes
      //
      // TODO(vamsikku): this is a common operations between forest and subtree
      // code it should be a function on its own.
      std::unordered_map<operation_t, size_t> input_tensor_index_map;
      {
        mv::Data::OpListIterator spilled_op_itr =
          om.getOp(spilled_op->getName());
        for(auto outputFlow = spilled_op_itr.leftmostOutput();
          outputFlow != om.flowEnd(); ++outputFlow) {
          size_t idx = outputFlow->get<size_t>("sinkInput");
          operation_t sink_op = &(*(outputFlow.sink()));
          input_tensor_index_map[sink_op] = idx;
        }
      }


      bool is_spilled_read_redundant = false;
      //////////////////////////////////////////////////////////////////////////
      // STEP-3: erase all outgoing flows from the spilled op and children in
      // spill sub tree
      {
        mv::Data::OpListIterator spilled_op_itr =
            om.getOp(spilled_op->getName());
        std::vector<mv::Data::FlowListIterator> undef_flows;
        size_t spilled_op_out_degree = 0UL;

        for (auto outputFlow = spilled_op_itr.leftmostOutput();
          outputFlow != om.flowEnd(); ++outputFlow, ++spilled_op_out_degree) {
          operation_t sink_op = &(*(outputFlow.sink()));
          if (spilled_sub_tree.has_child(sink_op)) {
            undef_flows.push_back(outputFlow);
          }
        }

        is_spilled_read_redundant =
            (undef_flows.size() == spilled_op_out_degree);

        for (auto flow : undef_flows) om.undefineFlow(flow);
      }

      //////////////////////////////////////////////////////////////////////////
      // STEP-3.1: postpone erase the of redundant spilled_op to avoid automatic
      // ref-count base removal of the underlying source tensor.


      //////////////////////////////////////////////////////////////////////////
      // STEP-4: create a new spill read subtrees ops using the input_tensor of
      // the original spilled op //
      //
      //
      // TODO(vamsikku): the only change from the subtree structure is the input
      // tensor for each of the reads so create a function to reuse the same
      // function across both methods.
      size_t read_index = 0UL;
      mv::DmaDirection read_dma_direction(std::string("DDR2NNCMX"));
      spilled_read_subtrees_t &read_subtrees = spilled_sub_tree.read_subtrees_;
      std::string dma_op_name;
      mv::Data::TensorIterator spill_read_tensor_itr;

      for (typename spilled_read_subtrees_t::iterator
            spill_read_itr=read_subtrees.begin();
              spill_read_itr!=read_subtrees.end(); ++spill_read_itr) {

        dma_op_name = spilled_op->getName() + "_spilledReadForest" +
            std::to_string(read_index++);
        spill_read_tensor_itr = om.dMATask(
            spilled_op_input_tensor_itr, read_dma_direction, 0, dma_op_name);
        if(spill_read_tensor_itr->isSparse())
          spill_read_tensor_itr->set<bool>("allocateSparsityMap", false); //TODO(Add a flag to dMATask() which can set allocateSparsityMap to false) 
        Data::OpListIterator read_op_itr =
            om.getSourceOp(spill_read_tensor_itr);
        read_op_itr->setInputTensor(spilled_op_input_tensor_itr, 0UL, false);
        spill_read_tensor_itr->set<std::size_t>("address",
              std::numeric_limits<size_t>::max());

        // save the read op //
        spill_read_itr->read_op_ = &(*read_op_itr);

        // now connect output of this read all ops in this subtree //
        const op_list_t &children = spill_read_itr->consumer_list_;
        for (auto child=children.begin(); child!=children.end(); ++child) {
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
      }

      if (is_spilled_read_redundant) {
        Data::OpListIterator spilled_op_itr = om.getOp(spilled_op->getName());
        assert(spilled_op_itr != om.opEnd());
        assert(redundant_spill_map_.find(spilled_op) ==
              redundant_spill_map_.end());
        redundant_spill_map_.insert(
            std::make_pair(spilled_op, spilled_op->getName()));
        om.removeOp(spilled_op_itr);
        spill_read_tensor_itr->set<bool>("allocateSparsityMap", true); // If the original DMA Op is considered redundant and removed
      }                                                                // then we want the 'new' spilled DMA to allocate the sparsity map
    }

    bool has_its_output_spilled(operation_t op) const {
      return (spilled_op_map_.find(op) != spilled_op_map_.end());
    }

    bool is_redundant_spill(const operation_t& op) const {
      return (redundant_spill_map_.find(op) != redundant_spill_map_.end());
    }

    mv::ComputationModel& model_;
    spilled_op_map_t spilled_op_map_;
    dag_t input_dag_;
    redundant_spill_map_t redundant_spill_map_;
}; // class Dynamic_Spill_Node_Inserter //


struct Remove_Redundant_Spill_Writes {
  typedef mv::Op const * operation_t;
  typedef Scheduled_Op scheduled_op_t;

  template<typename InputScheduledOpIterator,
    typename BackInsertScheduledOpIterator>
  static void remove(
      InputScheduledOpIterator begin, InputScheduledOpIterator end,
      BackInsertScheduledOpIterator output) {
    std::unordered_map<operation_t, scheduled_op_t> spilled_writes;
    for (; begin != end; ++begin) {
      const scheduled_op_t& sop = *begin;
      const operation_t& op = sop.op_;

      if (!sop.is_spilled_write() ||
          (spilled_writes.find(op) == spilled_writes.end())) {
        if (sop.is_spilled_write()) {
          spilled_writes.insert(std::make_pair(op, sop));
        }
        output = sop;
      }
    }
  }

}; // class Remove_Redundant_Spill_Writes //


//NOTE: to use this class the tensors should have "allocators" attribute. //
template<typename OpDag>
class Master_Slave_Buffer_Relations {
  public:

    ////////////////////////////////////////////////////////////////////////////
    typedef OpDag dag_t;
    typedef scheduler_traits<dag_t> traits;
    typedef typename dag_t::operation_t operation_t;
    typedef typename dag_t::const_operation_iterator_t
        const_operation_iterator_t;
    typedef std::unordered_map<operation_t, operation_t> master_map_t;
    typedef std::unordered_set<operation_t> slave_map_t;
    typedef typename slave_map_t::const_iterator slave_map_iterator_t;
    typedef typename master_map_t::iterator master_map_iterator_t;
    typedef typename master_map_t::const_iterator const_master_map_iterator_t;
    ////////////////////////////////////////////////////////////////////////////

    Master_Slave_Buffer_Relations(const dag_t& in, mv::ComputationModel& cmodel)
      : dag_(in), master_map_(), has_slaves_(), cmodel_ptr_(&cmodel) {
        compute_slave_master_relations();
    }

    void print(FILE *fptr=stderr) const {
      for (const_master_map_iterator_t itr=master_map_.cbegin();
            itr!=master_map_.cend(); ++itr) {
        operation_t slave = itr->first;
        operation_t master = itr->second;
        fprintf(fptr, "[Master_Slave_Relations] slave=%s master=%s [%c]\n",
            (slave->getName()).c_str(), (master->getName()).c_str(),
            (master == slave) ? '*' : ' ');
      }
    }

    operation_t master_tensor_op(const operation_t& op) const {
      const_master_map_iterator_t itr = master_map_.find(op);
      return (itr == master_map_.end()) ? operation_t(NULL) : itr->second;
    }

    bool has_slaves(const operation_t& op) const {
      slave_map_iterator_t itr = has_slaves_.find(op);
      return !(itr == has_slaves_.end());
    }

    bool does_this_op_use_ddr_resources(const operation_t& op) const {
      if (!op->outputSlots()) { return false; }
      mv::Op * op_ptr = const_cast<mv::Op *>(op);
      mv::Data::TensorIterator tensor_itr = op_ptr->getOutputTensor(0UL);
      return (!tensor_itr->isPopulated()) &&
        (tensor_itr->get<mv::Tensor::MemoryLocation>("Location") ==
          mv::Tensor::MemoryLocation::DDR);
    }

  private:

    void compute_slave_master_relations() {
      master_map_.clear();
      const_operation_iterator_t itr=traits::operations_begin(dag_),
                                 itr_end = traits::operations_end(dag_);
      for (; itr != itr_end; ++itr) {
        operation_t op = *itr;
        if (does_this_op_use_ddr_resources(op)) {
          operation_t mop = get_op_associated_with_master_buffer(op);
          assert(master_map_.find(op) == master_map_.end());
          master_map_.insert(std::make_pair(op, mop));
          if (op != mop) {
            has_slaves_.insert(mop);
          }
        }
      }
    }


    //Preconditon: does_this_op_use_ddr_resources(op) //
    operation_t get_op_associated_with_master_buffer(
          const operation_t& op) const {
      mv::Op * op_ptr = const_cast<mv::Op *>(op);
      mv::Data::TensorIterator tensor_itr = op_ptr->getOutputTensor(0UL);

      // check if master buffer is the same //
      mv::DataModel dm(*cmodel_ptr_);
      auto & tallocs = tensor_itr->get<std::set<std::string>>("allocators");
      if (tallocs.empty())
        throw mv::ArgumentError("get_op_associated_with_master_buffer", "",  "Tensor Allocators empty", "");
      auto talloc_name = tallocs.begin();
      auto talloc = dm.getAllocator(*talloc_name);
      mv::Data::BufferIterator tensor_buffer_itr = talloc.getBuffer(0UL,
            tensor_itr);
      mv::Data::BufferIterator master_buffer_itr =
          talloc.getTopMasterBuffer(tensor_buffer_itr);
      mv::Data::TensorIterator master_tensor_itr =
          master_buffer_itr->getData();

      mv::OpModel om(*cmodel_ptr_);
      mv::Data::OpListIterator master_op_itr =
          om.getSourceOp(master_tensor_itr);

      // Unable to find an op corresopding to tensor associated with
      // master buffer //
      assert(master_op_itr != om.opEnd());

      return &(*master_op_itr);
    }

    const dag_t& dag_;
    master_map_t master_map_;
    slave_map_t has_slaves_;
    mv::ComputationModel *cmodel_ptr_;
}; // class Master_Slave_Buffer_Relations //


template<typename OpDag>
class DDR_Address_Generator {
  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef OpDag dag_t;
    typedef typename dag_t::scheduled_op_t scheduled_op_t;
    typedef typename dag_t::operation_t operation_t;
    typedef typename dag_t::resource_utility_map_t resource_utility_map_t;
    typedef typename dag_t::const_operation_iterator_t
        const_operation_iterator_t;

    struct ddr_op_selector_t {
      ddr_op_selector_t(dag_t const *dag_ptr=NULL) : input_dag_ptr_(dag_ptr) {}

      bool operator()(const operation_t& op) const {
        return input_dag_ptr_->does_this_op_use_any_resources(op);
      }
      dag_t const * input_dag_ptr_;
    }; // struct ddr_op_selector_t //

    struct ddr_op_utility_t {
      ddr_op_utility_t(dag_t const *dag_ptr=NULL) : input_dag_ptr_(dag_ptr) {}

      size_t operator()(const operation_t& op) const {
        return input_dag_ptr_->resource_utility(op);
      }
      dag_t const * input_dag_ptr_;
    }; // struct ddr_op_utility_t//

    typedef DAG_Address_Generator<dag_t, size_t, ddr_op_selector_t,
              ddr_op_utility_t> address_generator_t;
    typedef typename address_generator_t::address_info_t address_info_t;
    typedef std::unordered_map<operation_t, address_info_t> ddr_address_table_t;
    typedef Master_Slave_Buffer_Relations<dag_t> master_slave_relations_t;

    struct ddr_address_setter_t {
      ddr_address_setter_t(FILE *fptr=NULL,
          ddr_address_table_t *address_table_ptr=NULL,
          const master_slave_relations_t *msrelations_ptr=NULL)
        : address_table_ptr_(address_table_ptr), fptr_(fptr),
          msrelations_ptr_(msrelations_ptr) {}

      bool operator=(const address_info_t& address_info) const {
        mv::Op * const op_ptr = const_cast<mv::Op *>(
            msrelations_ptr_->master_tensor_op(address_info.op_) );
        mv::Data::TensorIterator tensor_itr = op_ptr->getOutputTensor(0UL);
        assert(address_info.address_begin_ >= 1UL);
        size_t address = address_info.address_begin_ - 1UL;
        tensor_itr->setAddress( address );
        tensor_itr->set<bool>("lp_scheduler_ddr_address", true);
        if (fptr_) {
          fprintf(fptr_, " op=%s ddr=[%lu %lu]\n", (op_ptr->getName()).c_str(),
                address_info.address_begin_, address_info.address_end_);
        }

        if (address_table_ptr_) {
          operation_t key = address_info.op_;
          assert(address_table_ptr_->find(key) == address_table_ptr_->end());
          (*address_table_ptr_)[key] = address_info;
        }
        return true;
      }

      ddr_address_table_t *address_table_ptr_;
      FILE *fptr_;
      const master_slave_relations_t *msrelations_ptr_;
    }; // struct ddr_address_setter_t //

    struct sched_op_t{
      operation_t op_;
      size_t time_;

      operator operation_t() const { return op_; }
      operator size_t() const { return time_; }

      sched_op_t(operation_t op, size_t time) : op_(op), time_(time) {}
    }; // struct sched_op_t //


    template<typename dag_t>
    struct real_op_selector_t {
      bool operator()(const dag_t& in, const operation_t& op) const {
        return !in.is_implicit_op(op) && !in.is_output_op(op);
      }
    }; // struct real_op_selector_t //

    template<typename OrigScheduledOpIterator>
    struct address_annotated_scheduled_op_iterator_t {
      typedef OrigScheduledOpIterator orig_sched_op_iterator_t;
      typedef scheduled_op_t value_type;

      address_annotated_scheduled_op_iterator_t()
        : begin_(), end_(), table_ptr_(NULL), scheduled_op_() {}

      address_annotated_scheduled_op_iterator_t(orig_sched_op_iterator_t beg,
          orig_sched_op_iterator_t end, const ddr_address_table_t& table)
        : begin_(), end_(), table_ptr_(&table), scheduled_op_() {
          begin_ = beg; end_ = end;
      }

      // only invalid iterators are equivalent //
      bool operator==(
          const address_annotated_scheduled_op_iterator_t& o) const {
        return !is_valid() && !o.is_valid();
      }

      bool operator!=(
          const address_annotated_scheduled_op_iterator_t& o) const {
        return !(*this == o);
      }

      //Precondition: is_valid() //
      void operator++() { ++begin_; }

      bool is_valid() const { return !(begin_ == end_); }

      const scheduled_op_t& operator*() const {
        scheduled_op_.op_ = begin_->op_;
        scheduled_op_.schedule_time_ = begin_->time_;
        auto itr = table_ptr_->find(scheduled_op_.op_);
        if (itr != table_ptr_->end()) {
          scheduled_op_.set_start_address((itr->second).address_begin_);
          scheduled_op_.set_end_address((itr->second).address_end_);
        } else {
          scheduled_op_.invalidate_address();
        }
        return scheduled_op_;
      }

      orig_sched_op_iterator_t begin_;
      orig_sched_op_iterator_t end_;
      const ddr_address_table_t *table_ptr_;
      mutable scheduled_op_t scheduled_op_;
    };
    ////////////////////////////////////////////////////////////////////////////

    DDR_Address_Generator(mv::ComputationModel& model, dag_t& dag,
          size_t upper_bound=16777216UL) : input_dag_(dag), model_(model),
    high_watermark_(), upper_bound_(upper_bound) {}


    void print(FILE *fptr=stdout) const {
      const_operation_iterator_t citr=input_dag_.begin_nodes(),
                               citr_end=input_dag_.end_nodes();
      for (; citr != citr_end; ++citr) {
        operation_t op = *citr;
        if (!input_dag_.does_this_op_use_any_resources(op)) { continue; }

        fprintf(fptr, "[DDR_Address_Generator] op=%s resource=%lu\n",
            op->getName().c_str(), input_dag_.resource_utility(op));
        const_operation_iterator_t child_itr=input_dag_.begin_nodes(op),
                                 child_itr_end=input_dag_.end_nodes(op);
        fprintf(fptr, "============<children>===============\n");
        for (; child_itr != child_itr_end; ++child_itr) {
          operation_t child_op = *child_itr;
          fprintf(fptr, "%s\n", (child_op->getName()).c_str());
        }
        fprintf(fptr, "============</children>===============\n");
      }

    }

    template<typename ScheduleIterator, typename BackInsertIterator>
    void read_schedule(ScheduleIterator sbegin, ScheduleIterator send,
        BackInsertIterator output) {
      for (; sbegin != send; ++sbegin) {
        output = sched_op_t((operation_t)(*sbegin), (size_t)(*sbegin) );
      }
    }

    template<typename OperationIterator>
    void add_scratch_info_into_model(
        const master_slave_relations_t& msrelations, OperationIterator obegin,
        OperationIterator oend) {

      mv::BufferMap& buffer_map = model_.bufferMap();
      mv::DataModel dm(model_);
      uint32_t scratchHighWatermark = 0;
      for (; obegin!=oend; ++obegin) {
        mv::Op *master_op =
          const_cast<mv::Op *>(msrelations.master_tensor_op(*obegin));
        mv::Data::TensorIterator tensor_itr = master_op->getOutputTensor(0UL);
        if (scratchHighWatermark >= std::numeric_limits<uint32_t>::max() - tensor_itr->getShape().totalSize())
          throw mv::RuntimeError("Scheduler", "Scratch buffer exceeds 32-bit address space");
        scratchHighWatermark += tensor_itr->getShape().totalSize();
      }
      buffer_map.addScratch(
        "Scratch",
        mv::Order("W"),
        {scratchHighWatermark},
        mv::DType("Default")
      );
    }

    template<typename ScheduleIterator>
    bool generate_tensor_addresses(
        ScheduleIterator sched_begin, ScheduleIterator sched_end,
        const char *file_name=NULL) { 
      return generate_tensor_addresses(sched_begin, sched_end, file_name, false);
    }

    // Takes a schedule and generates address for tensors which reside in DDR.
    // The address will //
    template<typename ScheduleIterator>
    bool generate_tensor_addresses(
        ScheduleIterator sched_begin, ScheduleIterator sched_end,
        const char *file_name, bool add_ddr_control_edges) {

      // STEP-0: read the schedule into memory //
      std::list<sched_op_t> schedule;
      read_schedule(sched_begin, sched_end, std::back_inserter(schedule));

      // STEP-1: compute the master slave relationships //
      master_slave_relations_t msrelations(input_dag_, model_);

      // STEP-2: update the dag with edges between the first slave and consumers
      // of the master buffer.
      update_dag_with_master_slave_relations(schedule.begin(), schedule.end(),
          msrelations);

      ddr_address_table_t ddr_address_table;
      FILE *fptr = file_name ?  fopen(file_name, "w") : NULL;
      ddr_address_setter_t address_setter(fptr, &ddr_address_table,
            &msrelations);


      // STEP-3: generate addresses using new DAG and resource model //
      address_generator_t address_generator(input_dag_, upper_bound_,
            ddr_op_utility_t(&input_dag_), ddr_op_selector_t(&input_dag_) );
      bool status = address_generator.generate(
          schedule.begin(), schedule.end(), address_setter);

      high_watermark_ = address_generator.get_high_watermark();
      if (status) {
        auto params = model_.getGlobalConfigParams();
        params->set<int>("DDRScratch", (int)(high_watermark_));
        if (high_watermark_)
          model_.bufferMap().addScratch("Scratch", mv::Order("W"),
              {high_watermark_}, mv::DType("Default"));
      }


      if (add_ddr_control_edges) {
        generate_ddr_memory_control_edges(ddr_address_table,
          schedule.cbegin(), schedule.cend(), fptr);
      }


      if (fptr) {
        fprintf(fptr, "[DDR_Address_Generator] high_watermark=%lu\n",
              high_watermark_);
        fclose(fptr);
      }

      return status;
    }

  private:

    template<typename ScheduledOpIterator>
    void generate_ddr_memory_control_edges(
        const ddr_address_table_t& ddr_address_table,
        ScheduledOpIterator sbegin, ScheduledOpIterator send, FILE *fptr=NULL) {
     typedef address_annotated_scheduled_op_iterator_t<ScheduledOpIterator>
        scheduled_op_iterator_t;

     mv::ControlModel cm(model_);
     Control_Edge_Set ddr_control_edge_set(cm, false /*dont clear CMX edges*/);
     scheduled_op_iterator_t begin(sbegin, send, ddr_address_table), end;

     ddr_control_edge_set.add_ddr_memory_control_edges(input_dag_, cm,
         begin, end, fptr);
    }

    //Following changes are made to the DAG [ G(V,E) ]:
    //
    // Let {u_1,u_2,...u_t..} be the slaves associated with the master 'm'.
    // and u = arg_min{schedule_time(u_i)} (i.e first slave in the schedule)
    //
    // Then change as follows:
    // (1) resource[u] = resource[m]
    //     resource[m] = 0, for other slaves resource[u_i] = 0UL
    //
    // (2) W = { w | (m,w) \in E }
    //     add edges { (u,w) | w \in W } to the set E
    template<typename ScheduleIterator>
    void update_dag_with_master_slave_relations(
          ScheduleIterator sbegin, ScheduleIterator send,
          const master_slave_relations_t& msrelations) {

      resource_utility_map_t &resource_map = input_dag_.resource_utility_map_;
      std::unordered_set<operation_t> master_op_ref;

      resource_map.clear();

      // locate the first slave operations in the schedule. //
      for (; sbegin != send; ++sbegin) {
        operation_t op = (*sbegin).op_;
        if (!msrelations.does_this_op_use_ddr_resources(op)) { continue; }

        operation_t mop = msrelations.master_tensor_op(op);
        if (mop == NULL) { continue; }

        if ((op != mop) && (master_op_ref.find(mop) == master_op_ref.end())) {
          // this is the first slave into the master tensor in DDR//
          resource_map[op] = dag_t::output_tensor_size(mop);

          // assert that mop is not in the resource table //
          assert(resource_map.find(mop) == resource_map.end());
          const_operation_iterator_t citr = input_dag_.begin_nodes(mop),
                                     citr_end = input_dag_.end_nodes(mop);
          // add outgoing edges from the master buffer //
          for (; citr != citr_end; ++citr) {
            input_dag_.add_directed_edge(op, *citr);
          }
          master_op_ref.insert(mop);
        } else if ((op == mop) && !msrelations.has_slaves(op)) {
          // this op does not have slaves so it owns the full tensor in DDR //
          resource_map[op] = dag_t::output_tensor_size(mop);
        }
      }
    }

    dag_t& input_dag_;
    mv::ComputationModel& model_;
    size_t high_watermark_;
    size_t upper_bound_;
}; // class DDR_Address_Generator //




// Simple Schedule Record Encoding Format: <op_name, size_t>
template<typename OpDAG>
struct Schedule_Reader_Writer {

  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef OpDAG dag_t;
    typedef mv::lp_scheduler::scheduler_traits<dag_t> traits;
    typedef typename traits::operation_t operation_t;

    struct scheduled_op_t {
      operation_t op_;
      size_t schedule_time_;

      operator size_t() const { return schedule_time_; }
      operator operation_t() const { return op_; }
    }; // struct scheduled_op_t //


    class Schedule_Read_Iterator {
      public:

        Schedule_Read_Iterator(std::istringstream &input, mv::OpModel& model)
          : input_it_(input), op_model_ptr_(&model) {
            if (is_valid()) { next_valid_record(); }
        }

        Schedule_Read_Iterator() : op_model_ptr_(nullptr) {
        }

        Schedule_Read_Iterator(const Schedule_Read_Iterator&) = default;

        const scheduled_op_t& operator*() const { return current_record_; }

        bool operator==(const Schedule_Read_Iterator& o) const {
          return (!is_valid()) && !(o.is_valid());
        }
        bool operator!=(const Schedule_Read_Iterator& o) const {
          return !(*this == o);
        }

        void operator++() {
          if (!is_valid()) { return;}
          next_valid_record();
        }

      private:
        bool is_valid() const { return op_model_ptr_ != nullptr; }

        void next_valid_record() {
          assert(is_valid());

          const std::istream_iterator<std::string> end;
          if (input_it_ == end) {
              invalidate();
              return;
          }

          std::string opname, schtime;

          opname  = *input_it_++;
          schtime = *input_it_++;

          mv::Data::OpListIterator op_itr = op_model_ptr_->getOp(opname);
          if (op_itr == op_model_ptr_->opEnd())
            throw RuntimeError("LpScheduler", "[Schedule_Reader_Writer] unable to locate "
                               + opname + "OpModel");

          current_record_.op_ = &(*op_itr);
          current_record_.schedule_time_ = std::stoul(schtime);
        }

        void invalidate() {
          op_model_ptr_ = nullptr;
        }

        std::istream_iterator<std::string> input_it_;
        mv::OpModel *op_model_ptr_;
        scheduled_op_t current_record_;
    };  // class Schedule_Read_Iterator //
    typedef Schedule_Read_Iterator schedule_read_iterator_t;
    ////////////////////////////////////////////////////////////////////////////


    static const char* ddr_address_attribute() {
      return "lp_scheduler_state_for_ddr_address_generation";
    }

    template<typename ScheduleIterator>
    static bool write_to_stringstream(std::ostringstream& stream,
          ScheduleIterator begin, ScheduleIterator end) {
      while (begin != end) {
        operation_t op = traits::scheduled_op(*begin);
        size_t time = traits::scheduled_op_time(*begin);
        stream << op->getName() << " " << time << "\n";
        ++begin;
      }

      return true;
    }

    static schedule_read_iterator_t begin_read(std::istringstream& input,
        mv::OpModel& op_model) {
      return schedule_read_iterator_t(input, op_model);
    }

    static schedule_read_iterator_t end_read() {
      return schedule_read_iterator_t();
    }

}; // struct Schedule_Reader_Writer //

// Repack input DMAs (zero-indegree) for the scheduled compute ops to improve
// the CMX utility:
//
// For each zero-indegree DMA task 'x' starting at 't=i' and having CMX
// address [a, b] identify all other tasks:
//
// Y = { y | schedule_time(y) < 'i' and (y,x) is a memory control edge between
//           tasks 'y' and 'x' }
// NOTE: the CMX address interval for all tasks in Y overlaps [a, b]
//
//
// Repack this DMA task to new time t_repack defined below:
//
// t_repack = max { schedule_time(z) | (y,z) is an edge in between tasks 'y' and
//             'z' in the DAG (opmodel) such that schedule_time(z) < 'i' }
//
// NOTE: for a successfully repacked op 't_repack < i'. Additionally with this
// repacking we keep all the future cmx addresses intact and there is no change.
template<typename DagType, typename Traits=scheduler_traits<DagType> >
class Repack_Input_DMA_Tasks {
  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef Traits traits;
    typedef typename traits::dag_t dag_t;
    typedef typename traits::operation_t operation_t;
    typedef typename traits::scheduled_op_t scheduled_op_t;
    typedef typename traits::unit_t unit_t;
    typedef typename traits::schedule_time_t schedule_time_t;
    typedef typename traits::data_op_selector_t data_op_selector_t;
    typedef typename traits::const_operation_iterator_t
        const_operation_iterator_t;
    typedef Control_Edge_Generator<scheduled_op_t>
        memory_control_edge_generator_t;

    struct repack_info_t {
      typedef std::list<scheduled_op_t> overlap_op_list_t;

      repack_info_t() : original_op_(), limiting_op_(), overlap_op_list_() {}

      repack_info_t(const scheduled_op_t& sched_op) : limiting_op_(sched_op),
        original_op_(sched_op), overlap_op_list_() {}

      void update(const scheduled_op_t& op) {
        if (traits::scheduled_time(op) > traits::scheduled_time(limiting_op_)) {
          limiting_op_ = op;
        }
      }

      void add_to_overlap_list(const scheduled_op_t& op) {
        overlap_op_list_.push_back(op);
      }

      bool is_repackable() const {
        return (traits::scheduled_time(original_op_) -
                traits::scheduled_time(limiting_op_) ) > schedule_time_t(1UL);
      }

      operation_t original_op(void) const {
        return traits::scheduled_operation(original_op_);
      }

      operation_t limiting_op(void) const {
        return traits::scheduled_operation(limiting_op_);
      }

      void print() const {

        operation_t curr_op = traits::scheduled_operation(original_op_);
        operation_t limit_op = traits::scheduled_operation(limiting_op_);
        schedule_time_t curr_time = traits::scheduled_time(original_op_);
        schedule_time_t repack_time = traits::scheduled_time(limiting_op_);

        std::cout << "op=" << traits::operation_name(curr_op) <<
          " curr_time=" << curr_time << " repack_time=" << repack_time  <<
          " limiting_op=" << traits::operation_name(limit_op) << std::endl;
        fflush(stdout);
        assert(repack_time != curr_time);
      }

      scheduled_op_t limiting_op_;
      scheduled_op_t original_op_;
      overlap_op_list_t overlap_op_list_;
    }; // struct repack_info_t //

    typedef std::set<schedule_time_t> repack_time_slots_t;
    typedef std::unordered_map<operation_t, repack_info_t> repack_map_t;
    typedef typename repack_map_t::iterator repack_map_iterator_t;
    typedef std::unordered_map<operation_t, scheduled_op_t>
        original_schedule_info_t;

    struct update_repack_map_t {
      update_repack_map_t() : repack_ptr_(NULL), data_op_selector_ptr_(NULL){}
      update_repack_map_t(repack_map_t& map, const data_op_selector_t& selector)
        : repack_ptr_(&map), data_op_selector_ptr_(&selector) {}

      update_repack_map_t(const update_repack_map_t& o)
        : repack_ptr_(o.repack_ptr_),
        data_op_selector_ptr_(o.data_op_selector_ptr_) {}

      void operator() (const scheduled_op_t& a, const scheduled_op_t& b) const {
        assert(repack_ptr_);
        const operation_t& bop = traits::scheduled_operation(b);
        if (!((*data_op_selector_ptr_)(bop))) { return; }
        if (traits::scheduled_time(b) == schedule_time_t(2UL)) { return; }

        repack_map_iterator_t itr = repack_ptr_->find(bop);
        if (itr == repack_ptr_->end()) {
          itr = (repack_ptr_->insert(std::make_pair(bop, b))).first;
          (itr->second).limiting_op_ = a;
        } else {
          (itr->second).update(a);
        }
        (itr->second).add_to_overlap_list(a);
      }

      repack_map_t *repack_ptr_;
      const data_op_selector_t *data_op_selector_ptr_;
    }; // struct update_repack_map_t //

    struct schedule_time_ordering_t {
      bool operator()(const scheduled_op_t& a, const scheduled_op_t& b) {
        return (traits::scheduled_time(a) < traits::scheduled_time(b));
      }

    };

    ////////////////////////////////////////////////////////////////////////////

    Repack_Input_DMA_Tasks(const dag_t &dag,
        const data_op_selector_t& op_selector)
      : input_dag_ptr_(&dag), repack_map_(), data_op_selector_(op_selector),
      original_schedule_info_(), repack_time_slots_() {}

    Repack_Input_DMA_Tasks(
        const data_op_selector_t& op_selector=data_op_selector_t())
      : input_dag_ptr_(NULL), repack_map_(), data_op_selector_(op_selector),
      original_schedule_info_(), repack_time_slots_() {}

    //Precondition: input scheduled ops are sorted on increasing time. //
    //Precondition: should be iteratable multiple times. //
    //returns a new sequence of scheduled ops //
    template<typename ScheduledOpIterator, typename BackInsertIterator>
    void repack(ScheduledOpIterator sched_begin,
        ScheduledOpIterator sched_end, BackInsertIterator output) {

      // STEP-0: for each input DMA to an op find the limiting op w.r.t to
      // active address range (e.g. CMX address range).
      {
        update_repack_map_t repack_updater(repack_map_, data_op_selector_);
        memory_control_edge_generator_t generator;
        generator.generate_control_edges(sched_begin, sched_end,
              repack_updater);
      }

      // STEP-1: save the original schedule for updating the repack times //

      {
        original_schedule_info_.clear();
        repack_time_slots_.clear();
        for (ScheduledOpIterator sitr=sched_begin; sitr != sched_end; ++sitr) {
          const scheduled_op_t& sched_op = *sitr;

          bool new_insert = (original_schedule_info_.insert(std::make_pair(
                  traits::scheduled_operation(sched_op), sched_op))).second;
          UNUSED(new_insert);
          assert(new_insert);

          if (traits::is_valid_scheduled_op(sched_op)) {
            repack_time_slots_.insert(traits::scheduled_time(sched_op));
          }
        }
      }

      // STEP-2: now adjust limiting ops based on consumers of the limiting op//
      adjust_repack_times();
      remove_non_repackable_ops();

      //STEP-3: generate a new schedule //
      //TODO(vamsikku): this could be done efficiently if we keep the partially
      //sorted non-repacked ops in a different list //
      std::vector<scheduled_op_t> new_schedule;
      new_schedule.reserve(original_schedule_info_.size());
      typename repack_map_t::const_iterator ritr;

      total_data_ops_ = 0UL;
      repacked_data_ops_ = 0UL;
      average_repack_level_ = double(0.0);

      for (auto oitr=original_schedule_info_.begin();
            oitr!=original_schedule_info_.end(); ++oitr) {

        if (data_op_selector_(oitr->first)) { ++total_data_ops_; }


        if ((ritr = repack_map_.find(oitr->first)) != repack_map_.end()) {
          const repack_info_t& rinfo = ritr->second;
          scheduled_op_t updated_sched_op = rinfo.original_op_;
          if (rinfo.is_repackable()) {
            ++repacked_data_ops_;
            schedule_time_t new_time =
              traits::scheduled_time(rinfo.limiting_op_) + schedule_time_t(1UL);
            // find a time slot for this op to move the op if not there will
            // a new time slot will be created and will cause increase in the
            // makespan since time is ignored by the final serializer. //
            schedule_time_t repack_time_slot = get_repack_time_slot(new_time);

            average_repack_level_ += double(
                traits::scheduled_time(updated_sched_op) - repack_time_slot);
            traits::set_new_schedule_time(updated_sched_op, repack_time_slot);
          }
          new_schedule.push_back(updated_sched_op);
        } else {
          new_schedule.push_back(oitr->second);
        }
      }

      std::sort(new_schedule.begin(), new_schedule.end(),
            schedule_time_ordering_t());

      for (auto nitr=new_schedule.begin(); nitr!=new_schedule.end(); ++nitr) {
        *output = *nitr;
      }
    }

    void print() const {
      std::cout << "===============<Repacked Ops>====================="
          <<std::endl;
      typename repack_map_t::const_iterator itr;
      for (itr=repack_map_.begin(); itr!=repack_map_.end(); ++itr) {
        (itr->second).print();
      }
      std::cout << "===============</Repacked Ops>====================="
          <<std::endl;
    }

    double average_repack_level() const {
      // (\Sum repack_level(v)) / total_data_ops
      //  if not repacked repack_level(v) = 1 //
      //  if its repacked then the repack() function updates
      //  average_repack_level_
      return total_data_ops_ ?
        (double(average_repack_level_ +
                 double(total_data_ops_-repacked_data_ops_))/
          double(total_data_ops_)) : double(0.0);
    }

  private:

    void adjust_repack_times() {
      assert(input_dag_ptr_);
      typedef typename repack_info_t::overlap_op_list_t limiting_op_list_t;

      for (typename repack_map_t::iterator ritr=repack_map_.begin();
            ritr!=repack_map_.end(); ++ritr) {

        repack_info_t& repack_info = ritr->second;
        if (!repack_info.is_repackable()) { continue; }

        const limiting_op_list_t &limiting_ops = repack_info.overlap_op_list_;
        operation_t repack_op = repack_info.original_op();

        assert(repack_op == ritr->first);

        // Determine the smallest time which this op can be repacked by looking
        // at all the overlapping ops in the active address space.
        for (auto limiting_sched_op=limiting_ops.begin();
              limiting_sched_op!=limiting_ops.end(); ++limiting_sched_op) {
          schedule_time_t curr_time =
              traits::scheduled_time(repack_info.original_op_);
          schedule_time_t repack_time =
              traits::scheduled_time(*limiting_sched_op);

          operation_t limiting_op = traits::scheduled_op(*limiting_sched_op);

          assert(repack_time < curr_time);

          // find the largest schedule time for consumers of the limiting op in
          // the range [repack_time, curr_time] //
          const_operation_iterator_t citr, citr_end;
          citr = traits::outgoing_operations_begin(*input_dag_ptr_, limiting_op);
          citr_end = traits::outgoing_operations_end(*input_dag_ptr_,
                limiting_op);

          schedule_time_t new_repack_time = repack_time;
          operation_t new_limiting_op = limiting_op;

          for (; citr != citr_end; ++citr) { // foreach outgoing edge //
            const operation_t& cop = *citr;
            schedule_time_t t = get_original_schedule_time(cop);

            // skip the consumers not in the time range //
            if (!( (t > repack_time) && (t < curr_time))) { continue; }

            if (t > new_repack_time) {
              new_repack_time = t;
              new_limiting_op = cop;
            }
          } // foreach outdoing edge //


          // updated scheduled op //
          if (new_repack_time != repack_time) {
            repack_info.update(get_original_scheduled_op(new_limiting_op));
          }
        }

      }

    } // adjust_repack_time() //

    schedule_time_t get_original_schedule_time(const operation_t& op) const {
      typename original_schedule_info_t::const_iterator itr;

      itr = original_schedule_info_.find(op);
      assert(itr != original_schedule_info_.end());

      const scheduled_op_t& sched_op = itr->second;
      return traits::scheduled_time(sched_op);
    }

    scheduled_op_t get_original_scheduled_op(const operation_t& op) const {
      typename original_schedule_info_t::const_iterator itr;

      itr = original_schedule_info_.find(op);
      assert(itr != original_schedule_info_.end());
      return itr->second;
    }

    // If the time difference between limiting op and the repackable op is 1.
    // Then we cannot repack this task //
    void remove_non_repackable_ops() {
      typename repack_map_t::iterator itr, itr_next;
      itr = repack_map_.begin();

      while ( itr != repack_map_.end() ) {
        const repack_info_t& repack_info = itr->second;
        if (!repack_info.is_repackable()) {
          itr_next = itr; ++itr_next;
          repack_map_.erase(itr);
          itr = itr_next;
        } else {
          ++itr;
        }
      }
    }

    schedule_time_t get_repack_time_slot(schedule_time_t new_time) const {
      assert(!(repack_time_slots_.empty()));
      typename repack_time_slots_t::const_iterator itr =
          repack_time_slots_.lower_bound(new_time);

      return (itr == repack_time_slots_.end()) ?
          *(repack_time_slots_.rbegin()) : *itr;
    }

    const dag_t *input_dag_ptr_;
    repack_map_t repack_map_;
    const data_op_selector_t &data_op_selector_;
    original_schedule_info_t original_schedule_info_;
    repack_time_slots_t repack_time_slots_;
    size_t total_data_ops_ = 0UL;
    size_t repacked_data_ops_ = 0UL;
    double average_repack_level_ = 0.0;
}; // class Repack_Input_DMA_Tasks //



} // namespace lp_scheduler//
} // namespace mv //

#endif
