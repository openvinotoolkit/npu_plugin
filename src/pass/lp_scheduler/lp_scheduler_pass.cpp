#include <set>
#include <unordered_set>

#include "include/mcm/computation/model/base_op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "lp_scheduler/operation_precedence_dag.hpp"
#include "pass/lp_scheduler/control_edge_generator.hpp"
#include "scheduler/feasible_scheduler.hpp"


static void LpSchedulerPass(const mv::pass::PassEntry& , mv::ComputationModel&,
    mv::TargetDescriptor& , mv::Element&, mv::Element&);

namespace mv {
namespace pass {

MV_REGISTER_PASS(LpScheduler)
  .setFunc(LpSchedulerPass)
  .defineArg(json::JSONType::String, "output")
  .setDescription("Run Feasible Scheduler Algorithm.");

} // namespace mv //
} // namespace pass //

typedef mv::scheduler::Operation_Dag<mv::ControlModel> control_dag_t;
typedef mv::scheduler::Operation_Dag<mv::OpModel> dag_t;

struct Scheduled_Op {

  Scheduled_Op(mv::Op const *op, size_t t, size_t start, size_t end) 
    : op_(op), schedule_time_(t), cmx_address_start_(start),
      cmx_address_end_(end) {}

  bool operator==(const Scheduled_Op& o) const {
    return (op_ == o.op_) && (schedule_time_ == o.schedule_time_) &&
      (cmx_address_start_ == o.cmx_address_start_) &&
      (cmx_address_end_ == o.cmx_address_end_);
  }

  mv::Op const * op_;
  size_t schedule_time_;
  size_t cmx_address_start_;
  size_t cmx_address_end_;
}; // struct Scheduled_Op //
typedef Scheduled_Op scheduled_op_t;

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
typedef Control_Edge control_edge_t;

class Control_Edge_Set {
  public:
    typedef mv::Op const * operation_t;
    typedef mv::model_traits<mv::ControlModel> mtraits;
    typedef mv::model_traits<mv::OpModel> mtraits_op;
    typedef typename mtraits::const_operation_iterator_t op_iterator_t;
    typedef typename mtraits::const_child_operation_iterator_t child_op_itr_t;
    typedef std::set< control_edge_t > edge_set_t;
    typedef std::unordered_map<operation_t, op_iterator_t> iterator_lookup_t;
    typedef typename edge_set_t::const_iterator const_edge_iterator_t;

    Control_Edge_Set(mv::ControlModel& cmodel)
      : control_edge_set_(), iterator_lookup_() { init(cmodel); }

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

          if (!cm.pathExists(oitr_sink, oitr_source)) {
            // adding this edge to the control model creates a cycle //
            continue;
          }

          mv::Control::FlowListIterator psedge =
              cm.defineFlow(oitr_source, oitr_sink);
          psedge->set<bool>("PartialSerialisationEdge", true);
        } 
      }
      add_control_edges_from_input_dma_tasks(dag, model);
    }

    template<typename OpDag>
    void add_edges_to_fresh_control_model(const OpDag& dag,
          mv::ComputationModel& model) {

      mv::ControlModel cm(model);
      clear_all_edges_in_control_model(model);
      add_edges_from_op_model(model);

      op_iterator_t oitr_source, oitr_sink;
      for (const_edge_iterator_t eitr=begin(); eitr!=end(); ++eitr) {
        oitr_source = iterator_lookup_[eitr->source_];
        oitr_sink = iterator_lookup_[eitr->sink_];
        auto flowIt = cm.checkControlFlow(oitr_source, oitr_sink);
        if ( (flowIt == cm.flowEnd()) &&
              !(cm.pathExists(oitr_source, oitr_sink))) {

          assert(!cm.pathExists(oitr_sink, oitr_source));

          //if (cm.pathExists(oitr_sink, oitr_source)) { continue; }

          mv::Control::FlowListIterator psedge =
              cm.defineFlow(oitr_source, oitr_sink);
          psedge->set<bool>("PartialSerialisationEdge", true);
        } 
      }
      add_control_edges_from_input_dma_tasks(dag, model);
    }

  private:

    void clear_all_edges_in_control_model(mv::ComputationModel& model) const {
      mv::ControlModel cm(model);
      mv::Control::FlowListIterator fitr, fitr_next;
      for (fitr=cm.flowBegin(); fitr!=cm.flowEnd();) {
        fitr_next = fitr; ++fitr_next;
        cm.undefineFlow(fitr);
        fitr = fitr_next;
      }
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
            psedge->set<bool>("PartialSerialisationEdge", true);
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
}; //  class Control_Edge_Set //


typedef Control_Edge_Set control_edge_set_t;
typedef mv::pass::Control_Edge_Generator<scheduled_op_t>
  control_edge_generator_t;

namespace mv {
namespace lp_scheduler {

template<>
struct interval_traits<scheduled_op_t> {
  typedef size_t unit_t;
  typedef scheduled_op_t interval_t;

  static unit_t interval_begin(const interval_t& interval) {
    return interval.cmx_address_start_;
  }

  static unit_t interval_end(const interval_t& interval) {
    return interval.cmx_address_end_;
  }

}; // struct interval_traits<Scheduled_Op> //

} // namespace lp_scheduler //
} // namespace mv //


void LpSchedulerPass(const mv::pass::PassEntry& pass,
    mv::ComputationModel& model, mv::TargetDescriptor& target,
    mv::Element& passDesc, mv::Element& compOutput) {
  typedef mv::lp_scheduler::mv_memory_scheduler_with_spilling_t scheduler_t;
  typedef typename scheduler_t::scheduled_op_info_t scheduled_op_info_t;

  mv::OpModel cm(model);
  dag_t input_dag(cm);
  typedef mv::lp_scheduler::scheduler_traits<dag_t> traits_t;
  auto params = model.getGlobalConfigParams();

  dag_t::resource_t upper_bound = params->get<unsigned>("cmx");
  std::string output_file = passDesc.get<std::string>("output");
  FILE *fptr = fopen(output_file.c_str(), "w");
  assert(fptr);

  scheduler_t scheduler(input_dag, upper_bound), scheduler_end;
  std::list<scheduled_op_t> scheduled_ops;
  bool has_any_implicit_ops = false;

  while (scheduler != scheduler_end) {
    const scheduled_op_info_t &scheduled_op = *scheduler;

    if (scheduled_op.op_type_name() != std::string("ORIGINAL")) {
      has_any_implicit_ops = true;
    }

    mv::Op const *op = scheduled_op.op_;
    size_t rbegin = scheduled_op.begin_resource();
    size_t rend = scheduled_op.end_resource();

    scheduled_ops.push_back(scheduled_op_t(op, scheduled_op.time_,
          rbegin, rend));

    fprintf(fptr, "op = %-20s  type = %-15s  time = %lu ",
        (scheduled_op.op_)->getName().c_str(), scheduled_op.op_type_name(),
          scheduled_op.time_);

    if (scheduled_op.has_active_resource()) {
      fprintf(fptr, " resource=[%lu %lu]\n", scheduled_op.begin_resource(),
          scheduled_op.end_resource());
    } else {
      fprintf(fptr, " resource=<none>\n");
    }
    ++scheduler;
  }

  mv::ControlModel cmodel(model);
  control_edge_set_t control_edges(cmodel);
  control_edge_generator_t algo;

  if (!has_any_implicit_ops) {
    algo.generate_control_edges(scheduled_ops.begin(), scheduled_ops.end(),
        control_edges);
  } else {
    fprintf(fptr, "WARNING: control edge generation with implicit ops not yet"
        " implemented\n");
  }

  std::unordered_set<std::string> scheduled_ops_set;
  for (auto itr=scheduled_ops.begin(); itr != scheduled_ops.end(); ++itr) {
    const scheduled_op_t& op = *itr;
    scheduled_ops_set.insert( (op.op_)->getName() );
  }

  fprintf(fptr, "\n\n");
  for (auto itr=control_edges.begin(); itr != control_edges.end(); ++itr) {
    fprintf(fptr, "control_edge: %s -> %s \n",
          (*itr).source_name(), (*itr).sink_name());
  }


  if (!has_any_implicit_ops) {
    control_edges.add_edges_to_control_model(input_dag, model);
  } else {
    fprintf(fptr, "WARNING: control edge generation with implicit ops not yet"
        " implemented\n");
  }


  {
    mv::ControlModel cmodel(model);
    fprintf(fptr, "[DAG Invariant: %s]\n",
          cmodel.isDag() ? "PASSED" : "FAILED");
  }

  for (auto itr=traits_t::operations_begin(input_dag);
        itr != traits_t::operations_end(input_dag); ++itr) {
    if (scheduled_ops_set.find((*itr)->getName()) == scheduled_ops_set.end()) {
      fprintf(fptr, "[unscheduled_op]: op=%s\n", (*itr)->getName().c_str());
    }
  }
  fclose(fptr);
}
