#ifndef LP_SCHEDULER_PASS_H
#define LP_SCHEDULER_PASS_H

#include <cstdio>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_set>

#include "include/mcm/computation/model/base_op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/iterator/tensor.hpp"
#include "lp_scheduler/operation_precedence_dag.hpp"

namespace mv {
namespace lp_scheduler {

enum class op_type_e {ORIGINAL_OP=0, SPILLED_WRITE_OP=1, SPILLED_READ_OP=2};

struct Scheduled_Op {

  Scheduled_Op() : op_(NULL),
    schedule_time_(std::numeric_limits<size_t>::max()), schedule_end_time_(),
  cmx_address_start_(), cmx_address_end_(), op_type_() {}

  Scheduled_Op(mv::Op const *op, size_t t, size_t start, size_t end,
      op_type_e op_type=op_type_e::ORIGINAL_OP) : op_(op), schedule_time_(t),
  schedule_end_time_(), cmx_address_start_(start), cmx_address_end_(end),
  op_type_(op_type) {}

  bool operator==(const Scheduled_Op& o) const {
    return (op_ == o.op_) && (schedule_time_ == o.schedule_time_) &&
      (cmx_address_start_ == o.cmx_address_start_) &&
      (cmx_address_end_ == o.cmx_address_end_);
  }

  bool is_spilled_read() const {
    return (op_type_ == op_type_e::SPILLED_READ_OP);
  }
  bool is_spilled_write() const {
    return (op_type_ == op_type_e::SPILLED_WRITE_OP);
  }
  bool is_original_op() const { return (op_type_ == op_type_e::ORIGINAL_OP); }
  const char *op_type_name() const {
    if (op_type_ == op_type_e::SPILLED_READ_OP) { return "SPILLED READ"; }
    if (op_type_ == op_type_e::SPILLED_WRITE_OP) { return "SPILLED WRITE"; }
    return "ORIGINAL";
  }

  bool has_valid_address() const { 
    return (cmx_address_start_ <= cmx_address_end_);
  }

  bool has_active_resource() const { return has_valid_address(); }

  mv::Op const * op_;
  size_t schedule_time_;
  size_t schedule_end_time_;
  size_t cmx_address_start_;
  size_t cmx_address_end_;
  op_type_e op_type_;
}; // struct Scheduled_Op //


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

  Tensor_Allocator_Assignment(mv::ComputationModel& model) : model_(model) {}

  void operator()(mv::Data::TensorIterator tensor_itr) const {
    if (!tensor_itr->hasAttr("lp_scheduler_cmx_address") || 
          !tensor_itr->get<bool>("lp_scheduler_cmx_address")) { return; }

    if (tensor_itr->getName().find("_spilledWrite")
          != std::string::npos) {return;}

    mv::DataModel dm(model_);
    auto tensor_alloc_name=
        tensor_itr->get<std::set<std::string>>("allocators").begin();
    auto tensor_alloc= dm.getAllocator(*tensor_alloc_name);
    mv::Data::BufferIterator tensor_buffer_itr =
        tensor_alloc.getBuffer(0, tensor_itr);
    mv::Data::BufferIterator master_tensor_buffer_itr =
        tensor_alloc.getTopMasterBuffer(tensor_buffer_itr);
    master_tensor_buffer_itr->setOffset((tensor_itr->get<size_t>("address")));
  }

  mv::ComputationModel &model_;
}; // struct Tensor_Allocator_Assignment //


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
    ////////////////////////////////////////////////////////////////////////////

    Control_Edge_Set(mv::ControlModel& cmodel)
      : control_edge_set_(), iterator_lookup_(), relocating_dma_map_(),
      in_degree_(), zero_indegree_temporal_control_(false) { init(cmodel); }

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

    // TODO(vamsikku): 
    // Given two ops (op1, op2) such that op1 (t1) is scheduled before op2 (t2) 
    // and their CMX addresses overlap we need to add control edges between
    // all consumers of op1 scheduled between [t1, t2] and op2 //
    template<typename OpDag, typename ScheduledOpIterator>
    void add_consumer_control_edges(ScheduledOpIterator sbegin,
        ScheduledOpIterator send, operation_t source, operation_t sink) {
      std::unordered_map<operation_t, size_t> op_schedule_info;


    }

    template<typename OpDag, typename ScheduledOpIterator>
    void add_edges_to_fresh_control_model(
        const OpDag& dag, mv::ComputationModel& model,
        ScheduledOpIterator sbegin, ScheduledOpIterator send) {
      typedef OpDag dag_t;

      mv::ControlModel cm(model);
      mv::OpModel om(model);

      add_control_edges_between_compute_ops_and_relocating_dmas(dag, model);

      for (const_edge_iterator_t eitr=begin(); eitr!=end(); ++eitr) {
        // control model node iterators //
        operation_t source_op = get_redirected_source(eitr->source_);
        operation_t sink_op = eitr->sink_;

        add_control_edge(source_op, sink_op, model);
#if 0
        //TODO(vamsikku): re-enable consumer control edges to reduce the 
        //number of temporal edges.
        if (dag.is_input_op(source_op) ||
            dag.has_edge_between_ops(source_op, sink_op)) { continue; }

        // now for all the ops (non-empty resource) which consume 
        for (typename dag_t::const_operation_iterator_t
              dop_itr=dag.begin_nodes(source_op);
              dop_itr != dag.end_nodes(source_op); ++dop_itr) {
          operation_t consumer_op = *dop_itr;
          if (!dag.resource_utility(consumer_op)) { continue; }


          operation_t redirected_consumer_op =
              get_redirected_source(consumer_op);

          bool consumer_control_edge =
              add_control_edge(redirected_consumer_op, sink_op, model);

          if (consumer_control_edge) {
            printf("[ConsumerControl: (%s) -> (%s)]\n",
                redirected_consumer_op->getName().c_str(),
                sink_op->getName().c_str());
          } else {

          }
        }
#endif

      }
      add_control_edges_between_inputs_and_compute_ops(dag, model);
      add_temporal_control_edges(dag, sbegin, send, model,
            zero_indegree_temporal_control_);
    }


  private:

    // The scheduled ops are ordered by their start time //
    template<typename OpDag, typename ScheduledOpIterator>
    size_t add_temporal_control_edges(const OpDag& input_dag,
        ScheduledOpIterator sbegin, ScheduledOpIterator send,
        mv::ComputationModel& model, bool zero_indegree_temporal_edges=false) {

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
              add_control_edge(oitr->op_, curr_op.op_, model);
              ++total_temporal_control_edges;
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
      if ( (flow_itr == cmodel.flowEnd()) &&
          !(cmodel.pathExists(oitr_source, oitr_sink)) &&
          !(cmodel.pathExists(oitr_sink, oitr_source)) ) {
        if (cmodel.pathExists(oitr_sink, oitr_source)) {
          printf("[cycle : edge (sink<-source) = (%s <- %s)]\n",
              sink->getName().c_str(), source->getName().c_str());
          fflush(stdout);
        }
        assert(!cmodel.pathExists(oitr_sink, oitr_source));
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

          printf("[AddInputEdges(%s -> %s)]\n", (op->getName()).c_str(),
              (child_op->getName()).c_str());
          add_control_edge(op, child_op, model);
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

        relocating_dma_map_t::const_iterator map_itr =
            relocating_dma_map_.find(op);

        assert(map_itr == relocating_dma_map_.end());

        relocating_dma_map_.insert(std::make_pair(op, cop));

        // update iterator so that it redirects to cop //
        printf("[redirecting %s to %s]\n", (op->getName()).c_str(),
              (cop->getName()).c_str());
      }
    }

    operation_t get_redirected_source(operation_t source) const {
      relocating_dma_map_t::const_iterator map_itr =
          relocating_dma_map_.find(source);
      return (map_itr == relocating_dma_map_.end()) ? source :
          map_itr->second;
    }

    void init(mv::ControlModel& cmodel) {
      iterator_lookup_.clear();
      for (op_iterator_t itr=mtraits::begin_operations(cmodel);
          itr!=mtraits::end_operations(cmodel); ++itr) {
        operation_t op = &(*itr);
        assert(iterator_lookup_.find(op) == iterator_lookup_.end());
        iterator_lookup_.insert(std::make_pair(op, itr));
      }
      clear_all_edges_in_control_model(cmodel);
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

    struct spilled_read_subtree_t {
      spilled_read_subtree_t() : read_op_(), consumer_list_() {}

      spilled_read_subtree_t(operation_t read_op)
        : read_op_(read_op), consumer_list_() {}

      void add_spill_read_consumer(operation_t consumer) {
        consumer_list_.push_back(consumer);
      }

      void print() const {
        printf("[read_op=%s]->{ ", read_op_->getName().c_str());
        for (auto itr=consumer_list_.begin(); itr!=consumer_list_.end();
              ++itr){
          printf(" %s ", (*itr)->getName().c_str());
        }
        printf(" }\n");
      }

      operation_t read_op_;
      op_list_t consumer_list_;
    }; // struct spilled_read_subtree_t //
    typedef std::list<spilled_read_subtree_t> spilled_read_subtrees_t;

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
        printf("[write_op=%s]\n", spilled_write_op_->getName().c_str());
        for (auto ritr=read_subtrees_.begin(); ritr!=read_subtrees_.end();
              ++ritr) {
          ritr->print();
        }
        printf("\n");
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
    size_t add_spill_read_write_ops(
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
        spilled_subtree_t &subtree = itr->second;
        bool is_original_spilled_op_redundant = is_redundant_spill(sop.op_);
        
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
          printf("[spilled_op=%s]\n", (itr->first)->getName().c_str());
          (itr->second).print();
        } 
        printf("========================\n");
      }
    }

    void print_redundant_spilled_ops(FILE *out_fptr=stdout) const {
      typename redundant_spill_map_t::const_iterator itr;
      for (itr=redundant_spill_map_.begin(); itr!=redundant_spill_map_.end();
            ++itr) {
        fprintf(out_fptr, "[Redundant Spill] op=%s\n", (itr->second).c_str());
      }
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

    // Creates a spill subtree structure under the given op whose output 
    // got spilled. Additionally the new write op addresses are added into
    // the substructure
    void create_spill_subtree_structure_in_model(operation_t spilled_op_in,
          spilled_subtree_t& spilled_sub_tree) {
      mv::Op *spilled_op = const_cast<mv::Op *>(spilled_op_in);
      mv::OpModel om(model_);
      mv::DataModel dm(model_);
      std::string dma_op_name = spilled_op->getName() + "_spilledWrite";

      //////////////////////////////////////////////////////////////////////////
      // STEP-1: create one DMA write op //
      mv::DmaDirection write_dma_direction(std::string("NNCMX2DDR"));
      mv::Data::TensorIterator spilled_op_output_tensor_itr =
          spilled_op->getOutputTensor(0UL);
      mv::Data::TensorIterator spill_write_tensor_itr = om.dMATask(
          spilled_op_output_tensor_itr, write_dma_direction, dma_op_name);
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


      //////////////////////////////////////////////////////////////////////////
      // STEP-3: erase all outgoing flows from the spilled op and children in 
      // spill sub tree
      {
        mv::Data::OpListIterator spilled_op_itr = om.getOp(spilled_op->getName());
        std::vector<mv::Data::FlowListIterator> flows;
        for(auto outputFlow = spilled_op_itr.leftmostOutput();
          outputFlow != om.flowEnd(); ++outputFlow) {
          operation_t sink_op = &(*(outputFlow.sink()));
          if (spilled_sub_tree.has_child(sink_op)) {
            flows.push_back(outputFlow);
          }
        }

        for (auto flow : flows) om.undefineFlow(flow);
      }

      //////////////////////////////////////////////////////////////////////////
      // STEP-4: create a new spill read ops by connecting spill_write tensor
      // to each of them as inputs.
      size_t read_index = 0UL;
      mv::DmaDirection read_dma_direction(std::string("DDR2NNCMX"));
      spilled_read_subtrees_t &read_subtrees = spilled_sub_tree.read_subtrees_;

      for (typename spilled_read_subtrees_t::iterator
            spill_read_itr=read_subtrees.begin();
              spill_read_itr!=read_subtrees.end(); ++spill_read_itr) {
        dma_op_name =
          spilled_op->getName() + "_spilledRead" + std::to_string(read_index++);
        mv::Data::TensorIterator spill_read_tensor_itr =
          om.dMATask(spill_write_tensor_itr, read_dma_direction, dma_op_name);
        Data::OpListIterator read_op_itr =
            om.getSourceOp(spill_read_tensor_itr);
        read_op_itr->setInputTensor(spill_write_tensor_itr, 0UL, false);

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

      for (typename spilled_read_subtrees_t::iterator
            spill_read_itr=read_subtrees.begin();
              spill_read_itr!=read_subtrees.end(); ++spill_read_itr) {

        dma_op_name = spilled_op->getName() + "_spilledReadForest" +
            std::to_string(read_index++);
        mv::Data::TensorIterator spill_read_tensor_itr = om.dMATask(
            spilled_op_input_tensor_itr, read_dma_direction, dma_op_name);
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
      }
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






} // namespace lp_schdeduler //
} // namespace mv //

#endif
