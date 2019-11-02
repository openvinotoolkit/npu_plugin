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
    typedef std::set< control_edge_t > edge_set_t;
    typedef typename edge_set_t::const_iterator const_edge_iterator_t;

    Control_Edge_Set() : control_edge_set_() {}

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
      typename OpDag::op_itr_t oitr_source, oitr_sink;
      for (const_edge_iterator_t eitr=begin(); eitr!=end(); ++eitr) {
        oitr_source = dag.get_op_iterator(eitr->source_);
        oitr_sink = dag.get_op_iterator(eitr->sink_);
        auto flowIt = cm.checkControlFlow(oitr_source, oitr_sink);
        if ( (flowIt == cm.flowEnd()) &&
              !(cm.pathExists(oitr_source, oitr_sink)) ) {
          mv::Control::FlowListIterator psedge =
              cm.defineFlow(oitr_source, oitr_sink);
          psedge->set<bool>("PartialSerialisationEdge", true);
        } 
      }
      add_control_edges_from_input_dma_tasks(dag, model);
      printf("[DAG Invariant: %s]\n", cm.isDag() ? "PASSED" : "FAILED");
    }

  private:

    
    template<typename OpDag>
    void add_control_edges_from_input_dma_tasks(const OpDag& dag,
          mv::ComputationModel& model) {

      typedef typename OpDag::op_itr_t mv_op_itr_t;
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
      mv_op_itr_t op_itr_source = dag.get_op_iterator(input_op), op_itr_sink;
      assert(op_itr_source != op_itr_sink);

      // now for all the dmas in the set add control edges from input to the
      // dmas //
      for (const_zero_in_itr_t itr=zero_in_degree_dmas.begin();
            itr!=zero_in_degree_dmas.end(); ++itr) {
        op_itr_sink = dag.get_op_iterator(*itr);
        auto flowIt = cm.checkControlFlow(op_itr_source, op_itr_sink);
        if ( flowIt == cm.flowEnd() ) {
          mv::Control::FlowListIterator psedge =
              cm.defineFlow(op_itr_source, op_itr_sink);
          psedge->set<bool>("PartialSerialisationEdge", true);
        } 
      }
    }

    std::set< control_edge_t > control_edge_set_;
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

  mv::ControlModel cm(model);
  control_dag_t input_dag(cm);
  typedef mv::lp_scheduler::scheduler_traits<control_dag_t> traits_t;
  auto params = model.getGlobalConfigParams();

  control_dag_t::resource_t upper_bound = params->get<unsigned>("cmx");
  std::string output_file = passDesc.get<std::string>("output");
  FILE *fptr = fopen(output_file.c_str(), "w");
  assert(fptr);

  mv::scheduler::mv_control_lp_scheduler_t scheduler(input_dag, upper_bound),
      end;
  std::list<scheduled_op_t> scheduled_ops;

  while (scheduler != end) {
    mv::Op const *op = *scheduler;
    auto rstate = scheduler.resource_state();
    auto rinfo = rstate.get_resource_usage_info(op);
    scheduled_ops.push_back(scheduled_op_t(op, scheduler.current_time(),
          rinfo.begin_, rinfo.end_));
    ++scheduler;
  }

  control_edge_set_t control_edges;
  control_edge_generator_t algo;

  algo.generate_control_edges(scheduled_ops.begin(), scheduled_ops.end(),
        control_edges);

  std::unordered_set<std::string> scheduled_ops_set;
  for (auto itr=scheduled_ops.begin(); itr != scheduled_ops.end(); ++itr) {
    const scheduled_op_t& op = *itr;
    fprintf(fptr, "scheduled_op: %s (type=%s) time=%lu cmx=[%lu %lu] mem=%lu\n",
        (op.op_)->getName().c_str(), ((op.op_)->getOpType()).c_str(),
         op.schedule_time_, op.cmx_address_start_, op.cmx_address_end_,
         (op.cmx_address_end_ - op.cmx_address_start_));
    scheduled_ops_set.insert( (op.op_)->getName() );
  }

  fprintf(fptr, "\n\n");
  for (auto itr=control_edges.begin(); itr != control_edges.end(); ++itr) {
    fprintf(fptr, "control_edge: %s -> %s \n",
          (*itr).source_name(), (*itr).sink_name());
  }

  control_edges.add_edges_to_control_model(input_dag, model);

  for (auto itr=traits_t::operations_begin(input_dag);
        itr != traits_t::operations_end(input_dag); ++itr) {
    if (scheduled_ops_set.find((*itr)->getName()) == scheduled_ops_set.end()) {
      fprintf(fptr, "[unscheduled_op]: op=%s\n", (*itr)->getName().c_str());
    }
  }
  fclose(fptr);
}
