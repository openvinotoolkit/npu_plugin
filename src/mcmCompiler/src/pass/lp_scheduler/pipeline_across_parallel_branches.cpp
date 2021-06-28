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

static void UPATaskChainScheduleHandling(const mv::pass::PassEntry&,
    mv::ComputationModel&, mv::TargetDescriptor& , mv::Element&, mv::Element&);

namespace mv {
namespace pass {

MV_REGISTER_PASS(PipeLineAcrossParallelBranches)
  .setFunc(PipeLineAcrossParallelBranches)
  .setDescription("Add Pseudo Edges For Pipelining Across Parallel Branches");

MV_REGISTER_PASS(UPATaskChainScheduleHandling)
  .setFunc(UPATaskChainScheduleHandling)
  .setDescription("Add Resource Control Edges For UPA Chains on Parallel Branches");
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

  non_pseudo_edge_iterator_t(const non_pseudo_edge_iterator_t& o)
    : begin_(o.begin_), end_(o.end_), op_(o.op_), input_dag_ptr_(o.input_dag_ptr_) { }

  ~non_pseudo_edge_iterator_t() = default;

  // Are move constructor and assignment operator needed?

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

    if (!om.pathExists(src_itr, sink_itr) && !om.pathExists(sink_itr, src_itr))
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
  dag_t::resource_t upper_bound = params->get<int>("cmx");
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

      const non_pseudo_edge_iterator_t nitr(input_dag, sched_op), nitr_end;
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

bool containsOnlyUPATasks(std::set<std::string>& levelOps, std::unordered_map<std::string, bool>& edgesInserted, mv::OpModel& om)
{
    for (auto opName : levelOps)
    {
        auto parallelOp = om.getOp(opName);
        if (parallelOp->isUPA() || parallelOp->getOpType() == "UPATask") { continue; }
        if ((edgesInserted.find(parallelOp->getName()) != edgesInserted.end())) { continue; }
        return false;
    }
    return true;
}

void createUPAChain(mv::Data::OpListIterator upaTask, std::vector<std::string>& upaChains, mv::OpModel& om)
{
    if (upaTask->isUPA() || upaTask->getOpType() == "UPATask")
    {
        upaChains.push_back(upaTask->getName());
        for (auto child = upaTask.leftmostChild(); child != om.opEnd(); ++child)
            createUPAChain(child, upaChains, om);
    }
    return;
}

std::vector<std::vector<std::string>> locateUPAChains(mv::OpModel& om, std::map<size_t, std::set<std::string>>& level_ops)
{
    std::vector<std::vector<std::string>> upaChains;
    for (auto itr = level_ops.begin(); itr != level_ops.end(); itr++)
    {
      for (auto op_name : (*itr).second)
      {
        auto op = om.getOp(op_name);
        if (op->getOpType() == "UPATask")
        {
          bool headCandidate = true;
          for (auto parent = op.leftmostParent(); parent != om.opEnd(); ++parent)
          {
            if (parent->isUPA() || parent->getOpType() == "UPATask")
                headCandidate = false;
          }
          if (!headCandidate) { continue; }
          for (auto child = op.leftmostChild(); child != om.opEnd(); ++child)
          {
            if (child->isUPA() || child->getOpType() == "UPATask")
            {
                std::vector<std::string> temp;
                createUPAChain(op, temp, om);
                upaChains.push_back(temp);
            }
          }
        }
      }
    }
    return upaChains;
}

bool trailingOperation(mv::Data::OpListIterator& op)
{
  if (op.childrenSize() == 0) { return true; }
  // follow the op to output
  mv::Data::OpListIterator child_op = op.leftmostChild();
  while (child_op.childrenSize() != 0 || child_op->isImplicit() ||
         child_op->getOpType() == "UPATask")
  {
    if (child_op->getOpType() == "ImplicitConcat") { return false; }
    child_op = child_op.leftmostChild();
  }
  return (child_op.childrenSize() == 0);
}

void addTrailingUPAflagToChains(std::vector<std::vector<std::string>>& upaChains, mv::OpModel& om)
{
  for (auto upaChain : upaChains)
  {
    // obtain the last chain UPA
    auto tailOp = om.getOp(upaChain.back());
    // if not followed by any concrete operations - consider trailing UPA
    if (trailingOperation(tailOp))
      tailOp->set<bool>("trailingUPA", true);
  }
}

std::list<operation_t> retrieve_concrete_child_ops(mv::Data::OpListIterator currOp, mv::OpModel& model)
{
  std::list<operation_t> concrete_child_ops;
  for (auto child = currOp.leftmostChild(); child != model.opEnd(); ++child)
  {
    if (child->isImplicit())
    {
      std::list<operation_t> temp = retrieve_concrete_child_ops(child, model);
      concrete_child_ops.merge(temp);
    }
    else
    {
      operation_t cop = &(*child);
      concrete_child_ops.push_back(cop);
    }
  }
  return concrete_child_ops;
}

void populateOpLevelMaps(mv::OpModel& om, std::unordered_map<std::string, size_t>& task_level, 
      std::map<size_t, std::set<std::string>>& level_ops)
{
    std::list<std::string> zero_in_degree_nodes[2UL];
    std::unordered_map<std::string, bool> propagated_ops;
    std::unordered_map<std::string, size_t> in_degree_map;
    size_t curr_depth = 0;
    // STEP-0: compute the in-degree's of all nodes //
    //NOTE: in_degree means the number of inputs of an op, and the pseudo data flows
    //if an op is zero_in_degree goes to zero_in_degree_nodes, like constants
    for (auto op_itr = om.opBegin(); op_itr != om.opEnd(); ++op_itr)
    {
        in_degree_map[ op_itr->getName() ] = op_itr->getInputTensor().size();
        if (op_itr->getInputTensor().size() == 0)
            zero_in_degree_nodes[0].push_back(op_itr->getName());
    }

    // NOTE: Topological sort according to zero_in_degree algorithm,
    // link: https://www.geeksforgeeks.org/topological-sorting-indegree-based-solution/
    // STEP-1: populate the dpu-levels map, pretty much
    // takes the opmodel as a dag and provides the ops that are on which level
    // e.g. A->B->C , A->D then (2, {B,D} )
    while (!zero_in_degree_nodes[curr_depth%2UL].empty())
    {
        bool parity = ((curr_depth%2UL) == 1UL);
        for (auto zitr=zero_in_degree_nodes[parity].begin();
              zitr!=zero_in_degree_nodes[parity].end(); ++zitr)
        {
          // update the in-degree //
          mv::Data::OpListIterator zop_itr = om.getOp((*zitr));
          std::list<operation_t> child_ops = retrieve_concrete_child_ops(zop_itr, om);
          for (auto& cop : child_ops)
          {
            if (propagated_ops.find(cop->getName()) != propagated_ops.end())
              continue;
            std::string cop_name = cop->getName();
            auto ditr = in_degree_map.find(cop_name);
            if (ditr->second == 0UL) { continue; }
            if ( (ditr == in_degree_map.end()) )
            {
                throw mv::RuntimeError("Chain Pipelining Pass", "Missing entry in the in-degree map (or)"
                  " invalid in-degree for op= " + cop_name);
            }
            --(ditr->second);
            if (!(ditr->second))
            {
                propagated_ops.insert({cop->getName(), true});
                zero_in_degree_nodes[!parity].push_back(cop_name);
                task_level[cop_name] = (curr_depth);
                if(level_ops.find(curr_depth) != level_ops.end())
                {
                  level_ops[curr_depth].insert(cop_name);
                }
                else
                {
                  level_ops[curr_depth] = {cop_name};
                }
            }
          }
        }
        zero_in_degree_nodes[parity].clear();
        curr_depth++;
    }
}

void UPATaskChainScheduleHandling(const mv::pass::PassEntry&, mv::ComputationModel& model, 
      mv::TargetDescriptor&, mv::Element&, mv::Element&) 
{
  mv::OpModel om(model);

  // td 0003 parallel outputs
  // auto t_40_weights = om.getOp("model/conv2d_40/Conv2D/Transpose21780_const7916723/quantized/to_f16:quantized_DDR2CMX");
  // auto t_36_concat = om.getOp("model/segm_logits/add");
  // auto flow_itr = om.defineFlow(t_36_concat->getOutputTensor(mv::IO_TENSOR_OUTPUT), t_40_weights, 0UL);

  // create op level map
  std::unordered_map<std::string, size_t> task_level;
  std::map<size_t, std::set<std::string>> level_ops;
  populateOpLevelMaps(om, task_level, level_ops);

  // find UPA chains, ordered by level
  std::vector<std::vector<std::string>> upaChains = locateUPAChains(om, level_ops);

  // trailing UPA Tasks
  addTrailingUPAflagToChains(upaChains, om);

  // add edges from UPA chains to parallel ops
  for (auto upaChain : upaChains)
  {
    // obtain chain head (first UPA with lowest level) and tail (last UPA with highest level)
    auto chainHead = upaChain.front();
    auto chainTail = upaChain.back();
    auto headOp = om.getOp(chainHead);
    auto tailOp = om.getOp(chainTail);
    bool trailingUPACase = tailOp->hasAttr("trailingUPA") && tailOp->get<bool>("trailingUPA");
    // add to control map to avoid cycles
    std::unordered_map<std::string, bool> edgesInserted;
    edgesInserted.insert({chainHead, true});
    // if level 0, skip as all constants would need edges
    if (task_level[chainHead] == 0) { continue; }
    // ### TWO POSSIBLE SCENARIOS ###
    // 1. TRAILINING UPA CHAINS - want to schedule all UPAs in the end
    if (trailingUPACase)
    {
      // interested in scenario where all tensors are in DDR, this should happen 
      // by default as UPA Tasks are not prefetched, if issues seen edges need 
      // to be added fromm the last DPU/DMA Tasks to the trailing UPA Tasks
      continue;
    }
    // 2. NON-TRAILING UPA CHAINS - want to schedule all UPAs in the beginning
    else
    {
      // use head and tail levels as range in search for DMA Tasks
      size_t headLevel = task_level[chainHead];
      // obtain all possible parallel operations
      for (auto sameLevelOp : level_ops[headLevel])
      {
        size_t parallelOpLevel = task_level[sameLevelOp];
        // retrieve the operation
        auto parallelOp = om.getOp(sameLevelOp);
        // ops already have edges inserted
        if (edgesInserted.find(parallelOp->getName()) != edgesInserted.end()) { continue; }
        // case of UPA Tasks (not in CMX) - fast forward
        while (parallelOp->getOpType() == "UPATask")
          parallelOp = parallelOp.leftmostChild();
        // case of possible CMX concat
        if (parallelOpLevel == headLevel && parallelOp->getOpType() == "DMATask" && 
            parallelOp->get<mv::DmaDirection>("direction") == mv::DDR2NNCMX && 
            parallelOp.leftmostParent()->getOpType() == "ImplicitConcat")
          while ((parallelOp->isImplicit() || (parallelOp->getOpType() == "DMATask"
                && parallelOp->get<mv::DmaDirection>("direction") == mv::DDR2NNCMX))
                && parallelOp->getOpType() != "ImplicitConcat")
            parallelOp = parallelOp.leftmostChild();
        // all other cases
        else
        {
          // DMAs, Implicit ops can be removed by CMX concat pass, retrieve concrete top operation
          while ((parallelOp->isImplicit() || parallelOp->getOpType() == "DMATask")
                && parallelOp->getOpType() != "ImplicitConcat")
            parallelOp = parallelOp.leftmostParent();
        }
        // add a control flow only if no flows exist
        if (!om.pathExists(parallelOp, tailOp) && !om.pathExists(tailOp, parallelOp))
        {
          size_t inputs = parallelOp.parentsSize();
          auto flow_itr = om.defineFlow(tailOp->getOutputTensor(mv::IO_TENSOR_OUTPUT), parallelOp, inputs + 1);
          flow_itr->set<bool>("resource_control_flow", true);
          edgesInserted.insert({parallelOp->getName(), true});
        }
      }
    }
  }
}
