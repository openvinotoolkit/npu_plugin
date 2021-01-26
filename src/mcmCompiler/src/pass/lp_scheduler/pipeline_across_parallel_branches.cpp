#include <time.h>
#include "include/mcm/base/exception/runtime_error.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/resource/memory_allocator.hpp"
#include "include/mcm/logger/logger.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/tensor/tensor.hpp"
#include "lp_scheduler/lp_scheduler_pass.hpp"
#include "pass/lp_scheduler/control_edge_generator.hpp"
#include "scheduler/feasible_scheduler.hpp"
#include "include/mcm/utils/helpers.hpp"

static void PipeLineAcrossParallelBranches(const mv::pass::PassEntry&,
    mv::ComputationModel&, mv::TargetDescriptor& , mv::Element&, mv::Element&);

namespace mv {
namespace pass {

MV_REGISTER_PASS(PipeLineAcrossParallelBranches)
  .setFunc(PipeLineAcrossParallelBranches)
  .setDescription("Add Pseudo Edges For Pipelining Across Parallel Branches");
} // namespace mv //
} // namespace pass //



typedef mv::scheduler::Operation_Dag<mv::OpModel> dag_t;
typedef mv::lp_scheduler::mv_memory_scheduler_with_spilling_t scheduler_t;
typedef typename scheduler_t::scheduled_op_info_t scheduled_op_info_t;
typedef mv::lp_scheduler::scheduler_traits<dag_t> traits_t;
typedef typename dag_t::operation_t operation_t;
typedef typename mv::lp_scheduler::Producer_Consumer_Contiguous_Resource<
    size_t, operation_t> resource_state_t;
typedef typename resource_state_t::free_interval_iterator_t
  free_interval_iterator_t;
typedef typename dag_t::const_operation_iterator_t const_operation_iterator_t;
typedef std::list<scheduled_op_info_t> scheduled_op_info_list_t;


class non_pseudo_edge_iterator_t {
  public:
  non_pseudo_edge_iterator_t() : begin_(), end_(), op_(), input_dag_ptr_() {}

  non_pseudo_edge_iterator_t(const dag_t& dag, operation_t op)
    : begin_(), end_(), op_(op), input_dag_ptr_(&dag)  {
    begin_ = dag.begin_nodes(op);
    end_ = dag.end_nodes(op);
    if (!reached_end() && input_dag_ptr_->is_pseudo_edge(op_, *begin_)) {
      move_to_next_non_pseudo_edge();
    }
  }

  operation_t operator*() const { return *begin_; }

  bool operator==(const non_pseudo_edge_iterator_t& o) const {
    return reached_end() && o.reached_end();
  }

  bool operator!=(const non_pseudo_edge_iterator_t& o) const {
    return !(*this == o);
  }
  //Precondition: reached_end() = false //
  non_pseudo_edge_iterator_t& operator++() {
    move_to_next_non_pseudo_edge();
    return *this;
  }

  non_pseudo_edge_iterator_t& operator=(const non_pseudo_edge_iterator_t& o) {
    begin_ = o.begin_;
    end_ = o.end_;
    op_ = o.op_;
    input_dag_ptr_ = o.input_dag_ptr_;
    return *this;
  }

  private:

  const_operation_iterator_t begin_;
  const_operation_iterator_t end_;
  operation_t op_;
  dag_t const *input_dag_ptr_;

  void move_to_next_non_pseudo_edge() {
    do {
      ++begin_;
    } while (!reached_end() && (input_dag_ptr_->is_pseudo_edge(op_, *begin_)));
  }

  bool reached_end() const { return begin_ == end_; }

}; // class non_pseudo_edge_iterator_t //

struct local_pass_util {

  template<typename T>
  static bool is_this_dpu_in_chain_pipeline(mv::OpModel& omodel, T op) {
    mv::Data::OpListIterator itr = omodel.getOp(op->getName());
    return itr->hasAttr("chain_pipelined_dpu") &&
        itr->get<bool>("chain_pipelined_dpu");
  }

  template<typename T>
  static bool is_weight_read(mv::OpModel& omodel, T op) {
    mv::Data::OpListIterator oitr = omodel.getOp(op->getName());
    if (oitr->getOpType() != "DMATask") { return false; }
    // indegree must be 1 and
    auto pitr = oitr.leftmostParent();
    auto pitr_next = pitr;

    ++pitr_next;
    if (pitr_next != omodel.opEnd()) { return false; }

    return (pitr->getOpType() == "ConstantDataElement") ||
      (pitr->getOpType() == "ConstantInt");
  }


  template<typename T, typename OutputIterator>
  static void get_weight_read_inputs(mv::OpModel& omodel, T dpu_op,
        OutputIterator output) {
    mv::Data::OpListIterator oitr = omodel.getOp(dpu_op->getName());

    for (auto pitr=oitr.leftmostParent(); pitr!=omodel.opEnd(); ++pitr) {
      if (is_weight_read(omodel, pitr)) {
        output = &(*pitr);
      }
    }
  }


  template<typename T>
  static size_t get_total_read_weight(mv::OpModel& omodel, T dpu_op) {
    mv::Data::OpListIterator oitr = omodel.getOp(dpu_op->getName());
    if (oitr->getOpType() != "DPUTask") { return 0UL; }

    size_t return_value = 0UL;
    for (auto pitr=oitr.leftmostParent(); pitr!=omodel.opEnd(); ++pitr) {
      operation_t pop = &(*pitr);
      if (is_weight_read(omodel, pop)) {
        return_value += pitr->getOutputTensor(0UL)->getClusterSize();
      }
    }
    return return_value;
  }

  template<typename T>
  static void add_pseudo_edge_dpu_read(mv::OpModel& om, T src_op, T sink_op) {
    mv::Data::OpListIterator src_itr = om.getOp(src_op->getName());
    mv::Data::OpListIterator sink_itr = om.getOp(sink_op->getName());
    mv::Data::TensorIterator src_tensor_itr = src_itr->getOutputTensor(0UL);

    if (!om.pathExists(src_itr, sink_itr))
    {
      mv::Data::FlowListIterator flow_itr =
          om.defineFlow(src_tensor_itr, sink_itr, 0UL);
      flow_itr->set<bool>("pseudo_data_flow", true);
      sink_itr->set<bool>("pipeline_flow_control", true);
    }
  }

}; // struct local_pass_util //

////////////////////////////////////////////////////////////////////////////////
///////////////////////// PASS STATIC PARAMETERS ///////////////////////////////
static size_t WEIGHT_LEVEL0_THRESHOLD = 60000UL;
static size_t WEIGHT_LEVEL1_THRESHOLD = 100000UL;
static size_t WEIGHT_LEVEL2_THRESHOLD = 200000UL;
static size_t MINIMUM_MAKE_SPAN_THRESHOLD = 30UL;
static size_t DEFAULT_STAGES = 6UL;
static size_t LEVEL2_STAGES = 12UL;
static size_t LEVEL1_STAGES = 10UL;
static size_t LEVEL0_STAGES = 8UL;
static size_t MAX_CHAIN_PIPELINED_DPU_THRESHOLD = 10UL;
////////////////////////////////////////////////////////////////////////////////

void PipeLineAcrossParallelBranches(const mv::pass::PassEntry& ,
    mv::ComputationModel& model, mv::TargetDescriptor& , mv::Element& ,
    mv::Element&) {

  mv::OpModel om(model);
  dag_t input_dag;
  auto params = model.getGlobalConfigParams();
  dag_t::resource_t upper_bound = params->get<unsigned>("totalCmx");
  { // build the dependency DAG //
    //input_dag.enable_cmx_concat_transforms(om, upper_bound);
    input_dag.reset(om);
  }

  std::unique_ptr<FILE, mv::utils::RaiiWrapper<FILE, mv::utils::releaseFile>> fptr(nullptr);
  if (mv::isDebugFilesEnabled()) {
    fptr.reset(fopen("./pipe_line_across_parallel_branches_report.txt", "w"));
    if (!fptr) {
      throw mv::RuntimeError("PipeLineAcrossParallelBranches",
            "Unable to open file");
    }
  }

  scheduler_t scheduler(input_dag, upper_bound), scheduler_end;
  typedef typename std::unordered_map<size_t, scheduled_op_info_list_t>
      schedule_time_map_t;
  schedule_time_map_t schedule_time_map;
  std::unordered_map<operation_t, scheduled_op_info_t> scheduled_op_map;
  std::unordered_map<size_t, size_t> free_space_map;
  std::list<operation_t> dpu_op_list;
  std::string scheduled_op_type;
  resource_state_t rstate;
  size_t prev_time = 0UL, make_span = 0UL;
  size_t total_chain_pipelined_dpus = 0UL;
  rstate.initialize_resource_upper_bound(upper_bound);

  std::unordered_set<operation_t> unassigned_ops;

  while (scheduler != scheduler_end) {
    const scheduled_op_info_t &scheduled_op_info = *scheduler;
    operation_t sched_op  = scheduled_op_info.op_;
    size_t curr_time = scheduled_op_info.time_;

    schedule_time_map[ curr_time ].push_back(scheduled_op_info);
    scheduled_op_map[sched_op] = scheduled_op_info;
    std::string opName = scheduled_op_info.op_type_name();

    if (opName != "ORIGINAL") { break; }

    typename schedule_time_map_t::const_iterator mitr;
    if ((prev_time != curr_time) &&
        ((mitr = schedule_time_map.find(prev_time))!=schedule_time_map.end()) ){
      mitr = schedule_time_map.find(prev_time);
      const scheduled_op_info_list_t &sched_ops_info = mitr->second;
      for (const scheduled_op_info_t &sched_op_info : sched_ops_info) {
        {
          bool unassign = rstate.unassign_resources(sched_op_info.op_);
          unassigned_ops.insert(sched_op_info.op_);
          if (!unassign) {
            throw mv::RuntimeError("PipeLineParallelBranches",
                  "Unable to unassign resources");
          }
        }
      }
    }

    size_t free_space = 0UL;
    if (!rstate.empty()) {
      free_interval_iterator_t fitr, fitr_end;
      fitr = rstate.begin_free_intervals();
      fitr_end = rstate.end_free_intervals();
      for (; fitr != fitr_end; ++fitr) {
        size_t a = fitr.interval_begin();
        size_t b = std::min(upper_bound, fitr.interval_end());
        if ((b-a) > 1) {
          free_space += (b-a);
        }
      }
    } else {
      free_space = upper_bound;
    }

    if (scheduled_op_info.has_active_resource()) {
      size_t rbegin = scheduled_op_info.begin_resource();
      size_t rend = scheduled_op_info.end_resource();

      non_pseudo_edge_iterator_t nitr(input_dag, sched_op), nitr_end;
      bool assign = rstate.assign_resources(sched_op, ((rend - rbegin)+1UL),
            nitr, nitr_end);
      if (!assign) {
        throw mv::RuntimeError("PipeLineParallelBranches",
              "Unable to assign resources");
      }
    }

    if (sched_op->getOpType() == "DPUTask") {
      dpu_op_list.push_back(sched_op);
      if (local_pass_util::is_this_dpu_in_chain_pipeline(om, sched_op)) {
        total_chain_pipelined_dpus++;
      }
    }


    {
      if (fptr.get()) {
        fprintf(fptr.get(), "op=%s type=%s time=%lu ", (sched_op->getName()).c_str(),
            scheduled_op_info.op_type_name(), curr_time);
        if (scheduled_op_info.has_active_resource()) {
          size_t rbegin = scheduled_op_info.begin_resource();
          size_t rend = scheduled_op_info.end_resource();
          fprintf(fptr.get(), " resource=[%lu,%lu] size=%lu", rbegin, rend,
                (rend-rbegin)+1UL);
        } else {
          fprintf(fptr.get(), " resource=<none> ");
        }
        fprintf(fptr.get(), " free_space=%lu\n", free_space);
      }
    }


    if (free_space_map.find(curr_time) == free_space_map.end()) {
      free_space_map[curr_time] = free_space;
    }
    free_space_map[curr_time] = std::max(free_space, free_space_map[curr_time]);

    if (prev_time != curr_time) {
      prev_time = curr_time;
      make_span = curr_time;
    }
    ++scheduler;
  }

  //For networks with small make span or long chains avoid this transformation//
  if ((make_span < MINIMUM_MAKE_SPAN_THRESHOLD) ||
      (dpu_op_list.size() < MINIMUM_MAKE_SPAN_THRESHOLD) ||
      (total_chain_pipelined_dpus >= MAX_CHAIN_PIPELINED_DPU_THRESHOLD) ) {
      return;
  }

  // STEP-1: find all DPU tasks for which weights are not pipelined //
  // TODO(vamsikku): handle spilling //
  std::list<operation_t> non_pipelined_dpus;
  std::list<operation_t> pipelined_dpus;

  for (operation_t dpu_op : dpu_op_list) {
    std::list<operation_t> reads;
    local_pass_util::get_weight_read_inputs(om, dpu_op,
          std::back_inserter(reads));

    ////////////////////////////////////////////////////////////////////////////
    // TODO(vamsikku): If the network has any DPU ops which needsODUoffset its
    // very flaky with any pipelining of weights and there is a JIRA to figure
    // out why this happens.
    {
      auto dpu_op_itr = om.getOp(dpu_op->getName());
      if (dpu_op_itr->hasAttr("needsODUoffset")) { return; }
    }
    ////////////////////////////////////////////////////////////////////////////

    const scheduled_op_info_t &info = scheduled_op_map[dpu_op];
    size_t dpu_time = info.time_;

    // find max read time //
    size_t max_read_time = 0UL;
    for (operation_t read : reads) {
      const scheduled_op_info_t &read_info = scheduled_op_map[read];
      max_read_time = std::max(max_read_time, read_info.time_);
    }

    if (dpu_time == (max_read_time + 1UL)) {
      non_pipelined_dpus.push_back(dpu_op);
    } else {
      pipelined_dpus.push_back(dpu_op);
    }
  }

  for (operation_t dpu_op : non_pipelined_dpus) {

    ////////////////////////////////////////////////////////////////////////////
    // ignore already chain pipelined dpus //
    auto itr = om.getOp(dpu_op->getName());
    if ( itr->hasAttr("chain_pipelined_dpu") &&
         (itr->get<bool>("chain_pipelined_dpu")) ) { continue; }
    ////////////////////////////////////////////////////////////////////////////


    size_t total_weight_size =
        local_pass_util::get_total_read_weight(om, dpu_op);
    const scheduled_op_info_t &info = scheduled_op_map[dpu_op];
    size_t dpu_time = info.time_;
    size_t stages = DEFAULT_STAGES; // TODO(vamsikku): prameterize stages //

    if (total_weight_size >= WEIGHT_LEVEL2_THRESHOLD) {
      stages = LEVEL2_STAGES;
    }else if (total_weight_size >= WEIGHT_LEVEL1_THRESHOLD) {
      stages = LEVEL1_STAGES;
    } else if (total_weight_size >= WEIGHT_LEVEL0_THRESHOLD) {
      stages = LEVEL0_STAGES;
    }

    if (dpu_time < stages) { continue; }

    bool can_add_pseudo_edges = true;
    for (size_t i=dpu_time-1UL; i>=dpu_time-stages; i--) {
     if (free_space_map[i] < total_weight_size ) {
       can_add_pseudo_edges = false;
       break;
     }
    }

    if (can_add_pseudo_edges) {
      // find a DPU op at time dpu_time-4UL and add pseudo edges between that
      // DPU and all the reads //
      typename schedule_time_map_t::const_iterator mitr;
      size_t curr_level = dpu_time-stages;

      operation_t src_dpu_op = NULL;
      while ((curr_level > 1) && !src_dpu_op) {
        mitr = schedule_time_map.find(curr_level);
        if (mitr == schedule_time_map.end()) { continue; }
        for (const scheduled_op_info_t &sched_op_info : mitr->second) {
          if (sched_op_info.op_->getOpType() == "DPUTask") {
            src_dpu_op = sched_op_info.op_;
            break;
          }
        }
        if (!src_dpu_op) { curr_level--; }
      }

      if (src_dpu_op == NULL) { continue; }
      {
        auto src_dpu_op_itr = om.getOp(src_dpu_op->getName());
        if ( src_dpu_op_itr->hasAttr("chain_pipelined_dpu") &&
             (src_dpu_op_itr->get<bool>("chain_pipelined_dpu")) ) { continue; }
        auto sink_dpu_op_itr = om.getOp(dpu_op->getName());
        if (!om.pathExists(src_dpu_op_itr, sink_dpu_op_itr)) { continue; }
      }

      size_t final_level = curr_level;
      // now reduce the free space between [final_level, dpu_time-1] //
      for (size_t i=dpu_time-1UL; i>=final_level; i--) {
        free_space_map[i] -= total_weight_size;
      }

      std::list<operation_t> reads;
      local_pass_util::get_weight_read_inputs(om, dpu_op,
            std::back_inserter(reads));

      // Add pseudo edges //
      for (operation_t read : reads) {
        if (fptr) {
          fprintf(fptr.get(), "[AddPseudoDepenency(%s,%s)\n",
              src_dpu_op->getName().c_str(), read->getName().c_str());
        }
        local_pass_util::add_pseudo_edge_dpu_read(om, src_dpu_op ,read);
      }
    }
  }
}
