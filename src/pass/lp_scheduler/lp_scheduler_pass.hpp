#ifndef LP_SCHEDULER_PASS_H
#define LP_SCHEDULER_PASS_H

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


  Scheduled_Op(mv::Op const *op, size_t t, size_t start, size_t end,
      op_type_e op_type=op_type_e::ORIGINAL_OP) : op_(op), schedule_time_(t),
  schedule_end_time_(), cmx_address_start_(start), cmx_address_end_(end),
  op_type_(op_type) {}

  bool operator==(const Scheduled_Op& o) const {
    return (op_ == o.op_) && (schedule_time_ == o.schedule_time_) &&
      (cmx_address_start_ == o.cmx_address_start_) &&
      (cmx_address_end_ == o.cmx_address_end_);
  }

  bool is_spilled_read() { return (op_type_ == op_type_e::SPILLED_READ_OP); }
  bool is_spilled_write() { return (op_type_ == op_type_e::SPILLED_WRITE_OP);}
  bool is_original_op() { return (op_type_ == op_type_e::ORIGINAL_OP); }

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
    mv::Data::TensorIterator tensor_itr = op_ptr->getOutputTensor(0UL);
    assert(op_info.begin_resource() >=1 );
    size_t address = op_info.begin_resource() - 1UL;
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

    mv::DataModel dm(model_);
    auto tensor_alloc_name=
        tensor_itr->get<std::set<std::string>>("allocators").begin();
    auto tensor_alloc= dm.getAllocator(*tensor_alloc_name);
    mv::Data::BufferIterator tensor_buffer_itr =
        tensor_alloc.getBuffer(0, tensor_itr);
    tensor_buffer_itr->setOffset(tensor_itr->get<size_t>("address"));
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


    template<typename OpDag, typename ScheduledOpIterator>
    void add_edges_to_fresh_control_model(
        const OpDag& dag, mv::ComputationModel& model,
        ScheduledOpIterator sbegin, ScheduledOpIterator send) {
      typedef OpDag dag_t;

      mv::ControlModel cm(model);
      mv::OpModel om(model);

      clear_all_edges_in_control_model(model);
      add_control_edges_between_compute_ops_and_relocating_dmas(dag, model);

      for (const_edge_iterator_t eitr=begin(); eitr!=end(); ++eitr) {
        // control model node iterators //
        operation_t source_op = get_redirected_source(eitr->source_);
        operation_t sink_op = eitr->sink_;

        add_control_edge(source_op, sink_op, model);
#if 0 
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


        if (!input_dag.is_implicit_op(curr_op.op_)) {
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
        if (dag.is_dpu_op(op) || !dag.resource_utility(op)) { continue; }

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
    }

   
    std::set< control_edge_t > control_edge_set_;
    iterator_lookup_t iterator_lookup_;
    relocating_dma_map_t relocating_dma_map_;
    control_in_degree_map_t in_degree_;
    bool zero_indegree_temporal_control_;
}; //  class Control_Edge_Set //

// Given a sequence (schedule) of scheduled ops (Scheduled_Op) with some //
class Dynamic_Spill_Node_Inserter {
  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef Scheduled_Op scheduled_op_t;
    typedef mv::Op const * operation_t;
    typedef std::list<operation_t> op_list_t;

    struct spilled_read_subtree_t {
      spilled_read_subtree_t(operation_t read_op)
        : read_op_(read_op), consumer_list_() {}

      void add_spill_read_consumer(operation_t consumer) {
        consumer_list_.push_back(consumer);
      }

      operation_t read_op_;
      op_list_t consumer_list_;
    }; // struct spilled_read_subtree_t //
    typedef std::list<spilled_read_subtree_t> spilled_read_subtrees_t;

    struct spilled_subtree_t {
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

      operation_t spilled_write_op_;
      spilled_read_subtrees_t read_subtrees_;
      std::unordered_set<operation_t> children_;
    }; // struct spilled_subtree_t //

    typedef std::unordered_map<operation_t, spilled_subtree_t> spilled_op_map_t;
    ////////////////////////////////////////////////////////////////////////////

    Dynamic_Spill_Node_Inserter(mv::ComputationModel& model)
      : model_(model), spilled_op_map_() {}

    //NOTE: this will update the scheduled_op_t 
    template<typename OpDag, typename ScheduledOpIterator>
    size_t add_spill_read_write_nodes(const OpDag& dag,
        ScheduledOpIterator sbegin, ScheduledOpIterator send) {
      compute_spill_subtrees(dag, sbegin, send);

      // now for each subtree structure update the model //
      for (spilled_op_map_t::const_iterator itr = spilled_op_map_.begin();
          itr != spilled_op_map_.end(); ++itr) {
        create_spill_subtree_structure_in_model(itr->first, itr->second);
      }
    }

  private:

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
    template<typename OpDag, typename ScheduledOpIterator>
    void compute_spill_subtrees(const OpDag& dag,
        ScheduledOpIterator sbegin, ScheduledOpIterator send) {
      typedef typename OpDag::const_operation_iterator_t
          const_operation_iterator_t;

      static_assert(std::is_same<typename ScheduledOpIterator::value_type,
            scheduled_op_t>::value, "Invalid ScheduledOpIterator");

      for (; sbegin != send; ++sbegin) {
        scheduled_op_t &sched_op = *sbegin;
        operation_t op = sched_op.op_;

        if (sched_op.is_original_op()){

          if (!dag.is_dpu_op(op)) { continue; }

          // We have DPU Task we need to check if any of its inputs are spilled
          for (const_operation_iterator_t pitr=dag.begin_parent_nodes(op);
                pitr != dag.end_parent_nodes(op); ++pitr) {

            operation_t pop = *pitr;
            if (!dag.is_dpu_op(pop) ||
                  !has_activation_spilled(pop)) { continue; }

            spilled_op_map_t::iterator itr = spilled_op_map_.find(pop);
            assert(itr != spilled_op_map_.end());

            //now add this op to the spill tree structure //
            (itr->second).add_spill_read_consumer(op);
          }
        } else if (sched_op.is_spilled_read()) {
          spilled_op_map_t::iterator itr = spilled_op_map_.find(op);

          // since spilled write must come before this //
          assert(itr != spilled_op_map_.end()); 
          (itr->second).add_spilled_read(op);
        } else {
          // spilled write op //

          spilled_op_map_t::iterator itr = spilled_op_map_.find(op);
          // since the activation data is not changing we don't need to 
          // write it back to DDR .
          if (itr != spilled_op_map_.end()) { continue; }

          spilled_op_map_.insert(std::make_pair(op, spilled_subtree_t(op)));
        }
      } //foreach scheduled op //
    }

    // creates a spill subtree structure under the given op whose activation 
    // got spilled.
    bool create_spill_subtree_structure_in_model(operation_t spilled_op_in,
        const spilled_subtree_t& spilled_sub_tree) {
      typedef std::set<std::string> outgoing_edge_names_t;

      mv::Op *spilled_op = const_cast<mv::Op *>(spilled_op_in);
      mv::OpModel om(model_);
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
      // set a dummy flows attribute //
      {
        std::set<std::string> toSet;
        spill_write_tensor_itr->set<std::set<std::string>>("flows", toSet);
      }


      //////////////////////////////////////////////////////////////////////////
      // STEP-2: for all the outgoing ops connected to this op determine the 
      // input tensor indexes 
      std::unordered_map<operation_t, size_t> input_tensor_index_map;
      mv::Data::OpListIterator spilled_op_itr = om.getOp(spilled_op->getName());
      {
        auto outgoing_edges_itr = spilled_op_itr.leftmostOutput();
        for (; outgoing_edges_itr != om.flowEnd(); ++outgoing_edges_itr) {
          size_t idx = outgoing_edges_itr->get<size_t>("sinkInput");
          operation_t sink_op = &(*(outgoing_edges_itr.sink()));
          input_tensor_index_map[sink_op] =
              outgoing_edges_itr->get<std::size_t>("sinkInput");
        }
      }

      //////////////////////////////////////////////////////////////////////////
      // STEP-3: erase all outgoing flows from the spilled op and children in 
      // spill sub tree
      {
        mv::Data::FlowListIterator outgoing_edges_itr =
          spilled_op_itr.leftmostOutput(), next_edge_itr;
        while (outgoing_edges_itr != om.flowEnd()) {
          operation_t sink_op = &(*(outgoing_edges_itr.sink()));
          if (spilled_sub_tree.has_child(sink_op)) {
            // erase this edge //
            next_edge_itr = outgoing_edges_itr;
            ++next_edge_itr;
            om.undefineFlow(outgoing_edges_itr);
            outgoing_edges_itr = next_edge_itr;
          } else {
            ++outgoing_edges_itr;
          }
        }
      }

      //////////////////////////////////////////////////////////////////////////
      // STEP-4: create a new spill read ops by connecting spill_write tensor
      // to each of them as inputs.
      size_t read_index = 0UL;
      mv::DmaDirection read_dma_direction(std::string("DDR2NNCMX"));
      const spilled_read_subtrees_t &read_subtrees =
          spilled_sub_tree.read_subtrees_;

      for (auto spill_read_itr=read_subtrees.begin();
            spill_read_itr!=read_subtrees.end(); ++spill_read_itr) {
        dma_op_name =
          spilled_op->getName() + "_spilledRead" + std::to_string(read_index++);
        mv::Data::TensorIterator spill_read_tensor_itr =
          om.dMATask(spill_write_tensor_itr, read_dma_direction, dma_op_name);

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
          om.defineFlow(spill_read_tensor_itr, child_op_itr, idx);
        }
      }


    }

    bool has_activation_spilled(operation_t dpu_op) const {
      return (spilled_op_map_.find(dpu_op) != spilled_op_map_.end());
    }

    mv::ComputationModel& model_;
    spilled_op_map_t spilled_op_map_;
}; // class Dynamic_Spill_Node_Inserter //


} // namespace lp_schdeduler //
} // namespace mv //

#endif
