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

struct Scheduled_Op {
  Scheduled_Op(mv::Op const *op, size_t t, size_t start, size_t end) 
    : op_(op), schedule_time_(t), schedule_end_time_(),
      cmx_address_start_(start), cmx_address_end_(end) {}

  bool operator==(const Scheduled_Op& o) const {
    return (op_ == o.op_) && (schedule_time_ == o.schedule_time_) &&
      (cmx_address_start_ == o.cmx_address_start_) &&
      (cmx_address_end_ == o.cmx_address_end_);
  }

  mv::Op const * op_;
  size_t schedule_time_;
  size_t schedule_end_time_;
  size_t cmx_address_start_;
  size_t cmx_address_end_;
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
    tensor_itr->setAddress( op_info.begin_resource() );
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
      in_degree_() { init(cmodel); }

    void operator()(const scheduled_op_t& a, const scheduled_op_t& b) {
      control_edge_set_.insert( control_edge_t(a.op_, b.op_) );
    }

    const_edge_iterator_t begin() const { return control_edge_set_.begin(); }
    const_edge_iterator_t end() const { return control_edge_set_.end(); }

    // Adds the control edges in this set to the control model //
    template<typename OpDag>
    void add_edges_to_control_model(const OpDag& dag,
          mv::ComputationModel& model) {

      mv::ControlModel cm(model);
      op_iterator_t oitr_source, oitr_sink;
      for (const_edge_iterator_t eitr=begin(); eitr!=end(); ++eitr) {
        oitr_source = iterator_lookup_[eitr->source_];
        oitr_sink = iterator_lookup_[eitr->sink_];
        auto flowIt = cm.checkControlFlow(oitr_source, oitr_sink);
        if ( (flowIt == cm.flowEnd()) &&
              !(cm.pathExists(oitr_source, oitr_sink))) {
          assert(!cm.pathExists(oitr_sink, oitr_source));

          mv::Control::FlowListIterator psedge =
              cm.defineFlow(oitr_source, oitr_sink);
          psedge->set<bool>("PartialSerialisationEdge", true);
        } 
      }
      add_control_edges_from_input_dma_tasks(dag, model);
    }


    template<typename OpDag, typename ScheduledOpIterator>
    void add_edges_to_fresh_control_model(
        const OpDag& dag, mv::ComputationModel& model,
        ScheduledOpIterator sbegin, ScheduledOpIterator send) {
      typedef OpDag dag_t;

      mv::ControlModel cm(model);
      mv::OpModel om(model);

      clear_all_edges_in_control_model(model);
      add_control_edges_between_inputs_and_compute_ops(dag, model);
      add_control_edges_between_compute_ops_and_relocating_dmas(dag, model);


      for (const_edge_iterator_t eitr=begin(); eitr!=end(); ++eitr) {
        // control model node iterators //
        operation_t source_op = get_redirected_source(eitr->source_);
        operation_t sink_op = eitr->sink_;

        add_control_edge(source_op, sink_op, model);
        if (dag.is_input_op(source_op)) { continue; }

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


      }

      add_temporal_control_edges(dag, sbegin, send, model);
      //add_control_edges_from_input_dma_tasks(dag, model);
    }

    template<typename OpDag>
    struct implicit_op_color_functor_t {
      bool operator()(const OpDag& dag, const operation_t& op) const {
        return dag.is_implicit_op(op);
      }
    }; // struct implicit_op_color_functor_t //

    template<typename OpDag>
    void add_implicit_op_closure_control_edges(const OpDag& dag,
        mv::ComputationModel& model) {

      mv::ControlModel cmodel(model);
      typedef typename OpDag::const_operation_iterator_t
          const_operation_iterator_t;
      typedef mv::lp_scheduler::Color_Connected_Vertices<OpDag>
          color_closure_t;

      color_closure_t color_closure_algo(dag);
      for (const_operation_iterator_t itr=dag.begin_nodes();
            itr!=dag.end_nodes(); ++itr) {

        // compute the color-closure of DMATask or DPUTask //
        operation_t pop = *itr;
        if (!((pop->getOpType() == "DMATask") || (pop->getOpType() == "DPUTask")
              || (pop->getOpType() == "Input") )) { continue; }

        std::list<operation_t> color_closure;
        color_closure_algo.compute_connected_vertices(pop,
              std::back_inserter(color_closure),
              implicit_op_color_functor_t<OpDag>() );

        if (!color_closure.empty()) {
          for (auto citr=color_closure.begin(); citr!=color_closure.end();
                ++citr) {
            const operation_t& cop = *citr;
            // add a control edge between (pop, cop) //
            add_control_edge(pop, cop, model);
          }
        }

      }
    }

  private:

    // The scheduled ops are ordered by their start time //
    template<typename OpDag, typename ScheduledOpIterator>
    size_t add_temporal_control_edges(const OpDag& input_dag,
        ScheduledOpIterator sbegin, ScheduledOpIterator send,
        mv::ComputationModel& model) {

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

          if (in_degree_.find(curr_op.op_) == in_degree_.end()) {
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
      if ( (flow_itr == cmodel.flowEnd()) &&
          !(cmodel.pathExists(oitr_source, oitr_sink)) ) {
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
        if (!dag.is_dpu_op(op)) { continue; }

        // add control edges from inputs to this compute op //

        for (typename dag_t::const_operation_iterator_t
            pitr=dag.begin_parent_nodes(op); pitr != dag.end_parent_nodes(op);
              ++pitr) {
          operation_t parent_op = *pitr;
          if (!dag.resource_utility(parent_op)) { continue; }
          add_control_edge(parent_op, op, model);
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

    void add_edges_from_op_model(mv::ComputationModel& model) {
      mv::ControlModel cm(model);
      mv::OpModel dm(model);

      op_iterator_t oitr_source, oitr_sink;

      for (auto itr = mtraits_op::begin_operations(dm);
            itr != mtraits_op::end_operations(dm); ++itr) {
        operation_t parent_op = &(*itr);
        oitr_source = iterator_lookup_[parent_op];

        for (auto citr = itr.leftmostChild(); citr != dm.opEnd(); ++citr) {
          operation_t child_op = &(*citr);
          oitr_sink = iterator_lookup_[child_op];
          auto flowIt = cm.checkControlFlow(oitr_source, oitr_sink);
          if (flowIt == cm.flowEnd()) {
            mv::Control::FlowListIterator psedge =
                cm.defineFlow(oitr_source, oitr_sink);
          }
        }
      }
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

    
    template<typename OpDag>
    void add_control_edges_from_input_dma_tasks(const OpDag& dag,
          mv::ComputationModel& model) {

      typedef typename OpDag::operation_t operation_t;
      typedef typename OpDag::const_operation_iterator_t node_iterator_t;
      typedef typename std::unordered_set< operation_t > zero_in_t;
      typedef typename zero_in_t::iterator zero_in_itr_t;
      typedef typename zero_in_t::const_iterator const_zero_in_itr_t;


      mv::ControlModel cm(model);

      zero_in_t zero_in_degree_dmas;

      // find all all DMA tasks with zero in-degree //
      for (node_iterator_t itr=dag.begin_nodes(), itr_end=dag.end_nodes();
            itr != itr_end; ++itr) {
        operation_t op = *itr;
        if (dag.is_dma_op(op) && !dag.operation_in_degree(op)) {
          // this DMA task has zero indegree before adding new control edges//
          zero_in_degree_dmas.insert(op);
        }
      }

      // the new control edges may have created a incoming control edge so
      // eliminate them.
      for (const_edge_iterator_t eitr=begin(); eitr!=end(); ++eitr) {
        zero_in_itr_t itr = zero_in_degree_dmas.find(eitr->sink_);
        if (itr == zero_in_degree_dmas.end()) {
          zero_in_degree_dmas.erase(eitr->sink_);
        }
      }

      operation_t input_op = dag.get_input_op();
      assert(input_op);
      op_iterator_t op_itr_source = iterator_lookup_[input_op], op_itr_sink; 
      assert(op_itr_source != op_itr_sink);

      // now for all the dmas in the set add control edges from input to the
      // dmas //
      for (const_zero_in_itr_t itr=zero_in_degree_dmas.begin();
            itr!=zero_in_degree_dmas.end(); ++itr) {
        op_itr_sink = iterator_lookup_[*itr];
        auto flowIt = cm.checkControlFlow(op_itr_source, op_itr_sink);
        if ( flowIt == cm.flowEnd() ) {
          mv::Control::FlowListIterator psedge =
              cm.defineFlow(op_itr_source, op_itr_sink);
          psedge->set<bool>("PartialSerialisationEdge", true);
        } 
      }
    }

    std::set< control_edge_t > control_edge_set_;
    iterator_lookup_t iterator_lookup_;
    relocating_dma_map_t relocating_dma_map_;
    control_in_degree_map_t in_degree_;
}; //  class Control_Edge_Set //


} // namespace lp_schdeduler //
} // namespace mv //

#endif
