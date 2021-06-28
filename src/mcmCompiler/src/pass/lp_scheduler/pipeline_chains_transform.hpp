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
      typedef std::unordered_map<operation_t, size_t> read_size_map_t;
      op_list_t dpu_chain_;
      std::list<op_list_t> weight_reads_;
      std::list<op_list_t> activation_reads_;
      read_size_map_t total_read_size_map_;

      size_t cmx_size(operation_t op) const {
        return ((const_cast<mv::Op *>(op))->getOutputTensor(0UL))->
            getClusterSize();
      }

      void compute_total_read_sizes() {
        if (weight_reads_.empty() && activation_reads_.empty())
          throw RuntimeError("LpScheduler", "compute_total_read_sizes(): weight_reads_ and activation_reads_ is empty");
        std::list<op_list_t>::iterator read_itr;
        if (activation_reads_.empty())
          read_itr = weight_reads_.begin();
        else
          read_itr = activation_reads_.begin();
        for (operation_t dpu_op : dpu_chain_) {
          size_t total_read_size = 0UL;
          for (operation_t read_op : *read_itr) {
            total_read_size +=
              ((const_cast<mv::Op *>(read_op))->getOutputTensor(0UL))->
                getClusterSize();
          }
          total_read_size_map_.insert(std::make_pair(dpu_op, total_read_size));
          ++read_itr;
        }
      }

      void set_chain_pipeline_attribute(mv::OpModel& om) const {
        for (operation_t dpu_op : dpu_chain_) {
          mv::Data::OpListIterator dpu_op_itr = om.getOp(dpu_op->getName());
          dpu_op_itr->set<bool>("chain_pipelined_dpu", true);
        }
      }

      void set_chain_pipeline_attribute(mv::OpModel& om,
          operation_t dpu_op) const {
        mv::Data::OpListIterator dpu_op_itr = om.getOp(dpu_op->getName());
        dpu_op_itr->set<bool>("chain_pipelined_dpu", true);
      }

      void print(FILE *fptr=stdout) {
        if (dpu_chain_.size() < 2UL) { return; }

        fprintf(fptr, "\n===========================\n");
        compute_total_read_sizes();

        size_t max_read_size = std::numeric_limits<size_t>::min();
        size_t max_output_size = std::numeric_limits<size_t>::min();

        size_t index = 0UL;

        auto dpu_op_itr = dpu_chain_.begin();

        for (;dpu_op_itr!= dpu_chain_.end(); ++dpu_op_itr) {
          operation_t dpu_op = *dpu_op_itr;
          fprintf(fptr, "%s :  reads=%zu output=%zu",
              (dpu_op->getName()).c_str(),
              total_read_size_map_[dpu_op], cmx_size(dpu_op));
          fprintf(fptr, "\n");

          {
            if (max_read_size < total_read_size_map_[dpu_op]) {
              max_read_size = total_read_size_map_[dpu_op];
            }
            if (max_output_size < cmx_size(dpu_op)) {
              max_output_size = cmx_size(dpu_op);
            }
          }
          ++index;
        }
        fprintf(fptr, "max_read_size=%zu max_output_size=%zu total=%zu\n",
            max_read_size, max_output_size, max_read_size+max_output_size);
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
        return *this;
      }

      bool operator==(const control_edge_t& o) {
        bool sourceQ = (source_itr_->getName() == o.source_itr_->getName());
        bool sinkQ = (sink_itr_->getName() == o.sink_itr_->getName());
        return (sourceQ && sinkQ);
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

      for (auto pitr=oitr.leftmostParent(); pitr!=omodel_.opEnd(); ++pitr) {
        if (is_weight_read(pitr)) {
          output = &(*pitr);
        }
      }
    }

    template<typename T, typename OutputIterator>
    void get_non_shared_read_inputs(T dpu_op, OutputIterator output) const {
      mv::Data::OpListIterator oitr = omodel_.getOp(dpu_op->getName());
      output = &(*oitr.leftmostParent()); // activation
      bool has_shared_weights = is_sharing_weights_operation(dpu_op);
      for (auto pitr=oitr.leftmostParent(); pitr!=omodel_.opEnd(); ++pitr) {
        if (!is_weight_read(pitr)) { continue; }
        if (has_shared_weights) {
          has_shared_weights = false;
          continue; // skip shared weights
        }
        output = &(*pitr);
      }
    }

    template<typename T, typename OutputIterator>
    void get_vf_read_inputs(T dpu_op, OutputIterator output) const {
      mv::Data::OpListIterator oitr = omodel_.getOp(dpu_op->getName());
      // don't skip shared weights as they are always spilled
      for (auto pitr=oitr.leftmostParent(); pitr!=omodel_.opEnd(); ++pitr) {
        output = &(*pitr);
      }
    }

    template<typename T, typename OutputIterator>
    void add_vertical_subgraph(T dpu_op, OutputIterator output, std::unordered_map<std::string, bool>& control_map) const {
      mv::Data::OpListIterator oitr = omodel_.getOp(dpu_op->getName());
      if (oitr->hasAttr("verticalFusion") && (control_map.find(oitr->getName()) == control_map.end()))
      {
        control_map[oitr->getName()] = true;
        output = dpu_op; // add current op
      }
      // recurse BFS to the tail
      for (auto citr = oitr.leftmostChild(); citr != omodel_.opEnd(); ++citr)
      {
        if (citr->hasAttr("verticalFusionSubgraphTail") && citr->get<bool>("verticalFusionSubgraphTail"))
        {
          if (control_map.find(citr->getName()) == control_map.end())
          {
            control_map[citr->getName()] = true;
            output = &(*citr);
          }
        }
      }
      for (auto citr = oitr.leftmostChild(); citr != omodel_.opEnd(); ++citr)
      {
        if (!(citr->hasAttr("verticalFusionSubgraphTail") && citr->get<bool>("verticalFusionSubgraphTail")))
          add_vertical_subgraph(&(*citr), output, control_map);
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

    template<typename OpIterator>
    size_t op_memory_demand(OpIterator op) const {
      auto op_itr = omodel_.getOp(op->getName());
      return (op_itr->getOutputTensor(0UL))->getClusterSize();
    }

    template<typename OpIterator>
    bool op_has_this_attribute(OpIterator op, const std::string& attr_name)
      const {
      auto op_itr = omodel_.getOp(op->getName());
      return op_itr->hasAttr(attr_name);
    }

    bool is_eltwise_with_soh_runtime_sparsity(operation_t dpu_op) const {
      auto op_itr = omodel_.getOp(dpu_op->getName());
      bool is_eltwise_with_soh_runtime_sparsity_flag = false;
      if (op_itr->getOpType() == "DPUTask")
      {
        if (op_itr->get<std::string>("taskOp") == "Eltwise" &&
          op_itr->get<std::string>("splitStrategy") == "SplitOverH" &&
          op_itr->get<bool>("inputActivationSparsity"))
          is_eltwise_with_soh_runtime_sparsity_flag = true;

      }
      return is_eltwise_with_soh_runtime_sparsity_flag;
    }

    bool is_last_dpu_on_output_branch(operation_t dpu_op)
    {
      // if no more DPU ops follow this op then it is considered to be output branch
      auto op_itr = omodel_.getOp(dpu_op->getName());
      if (op_itr.childrenSize() == 0) { return true; }
      for (auto child = op_itr.leftmostChild(); child != omodel_.opEnd(); ++child)
      {
        if (child->getOpType() == "DPUTask")
          return false;
        else
          return is_last_dpu_on_output_branch(&(*child));
      }
      return dpu_op->getOpType() != "DPUTask";
    }

    void postProcessImplicitOperationsForDAG(std::map<size_t, op_list_t> &dpu_levels, operation_t opIt, std::size_t depth)
    {
      mv::OpModel &model = omodel_;
      auto cop = *opIt;
      auto previousActivationOperation = model.getSourceOp(cop.getInputTensor()[0]);
      //NOTE: in the level we meet slice operations we need to move their following
      // streaming operations in the level of slice
      if (opIt->getOpType() == "DPUTask" && previousActivationOperation->getOpType() == "Slice")
      {
        dpu_levels[depth].remove(opIt);
        dpu_levels[depth - 1].push_back(opIt);

      }
    }

    void clearEmptyDepth(std::map<size_t, op_list_t> &dpu_levels)
    {
      //NOTE: after moving streming operations to slice level we might come with empty list
      for (auto leveling_dag = dpu_levels.cbegin(); leveling_dag != dpu_levels.cend();)
      {
        if (leveling_dag->second.size() == 0)
          dpu_levels.erase(leveling_dag++);
        else
          ++leveling_dag;
      }
    }

    template<typename LevelItr>
    bool can_this_level_be_appended_to_chain(LevelItr itr)
    {
      return ( (itr->second.size() == 1UL) || comeFromTheSameParentStream(itr->second) );
    }

    bool is_sharing_weights_operation(operation_t dpu_op) const {
      auto op_itr = omodel_.getOp(dpu_op->getName());
      return (op_itr->hasAttr("shareWeights") && op_itr->get<bool>("shareWeights"));
    }

    bool share_the_same_weights(op_list_t operationLevelStream) const
    {
      std::set<std::string> weightOpNames;
      weightOpNames.clear();

      for (auto& i : operationLevelStream)
      {
          auto op_itr = omodel_.getOp(i->getName());
          weightOpNames.insert(omodel_.getSourceOp(op_itr->getInputTensor(mv::IO_TENSOR_WEIGHTS_SET))->getName());
      }

      return (weightOpNames.size() == 1);
    }

    bool activation_input_in_cmx(size_t cmx_size, operation_t op_itr) const
    {
      // if the input to vertical fusion subgraphs or stream over h ops is in CMX do not pipeline
      auto head = omodel_.getOp(op_itr->getName());
      auto opType = head->getOpType();
      if (opType == "ImplicitConcat")
      {
        // if concat has a flag not to be cmx-ed
        if (head->hasAttr("avoid_cmx_concat") && head->get<bool>("avoid_cmx_concat"))
          return false;
        // if concat will not fit in cmx
        auto concatSize = head->getOutputTensor(mv::IO_TENSOR_OUTPUT)->computeTotalSize();
        if (head->get<std::string>("splitStrategy") != "Clustering") { concatSize = concatSize / 4; }
        if (concatSize > cmx_size)
        {
          // generally more performant to be in DDR
          head->set<bool>("avoid_cmx_concat", true);
          return false;
        }
        // input concat can be in CMX - no input to pipeline
        //NOTE: might be performant to spill input to DDR and pipeline
        if (head->hasAttr("explicitRelocate") && head->get<bool>("explicitRelocate"))
          return true;
        head->set<bool>("avoid_cmx_concat", true);
        return false;
      }
      else if (opType == "DMATask" || head->isImplicit())
        return activation_input_in_cmx(cmx_size, &(*head.leftmostParent()));
      else
        return false;
    }

    bool can_activation_pipelining_be_applied(operation_t dpu_op)
    {
      // for activation chain pipelining the op (inputs + output) must fit twice in cmx
      mv::Data::OpListIterator stream_1 = omodel_.getOp(dpu_op->getName());
      size_t input_size = 0UL;
      size_t output_size = 0UL;
      size_t shared_weight_size = 0UL;
      for (size_t idx=0UL; idx < stream_1->getInputTensor().size(); idx++)
      {
        if (stream_1->hasAttr("hasWeights") && stream_1->get<bool>("hasWeights") && idx==1UL)
          shared_weight_size = stream_1->getInputTensor()[idx]->getClusterSize();
        else
          input_size += stream_1->getInputTensor()[idx]->getClusterSize();
      }
      output_size = stream_1->getOutputTensor(0UL)->getClusterSize();

      auto globalParams = omodel_.getGlobalConfigParams();
      size_t clusterMemory = globalParams->get<int>("cmx");

      // allow a 15% cmx margin
      return (2 * input_size + 2 * output_size + shared_weight_size) < (clusterMemory * 0.85);
    }

    template<typename SubGraphContainer>
    bool is_network_candidate(SubGraphContainer& chain_subgraphs)
    {
      // networks with few chains are prefferred to have streams parallelized
      size_t activation_chain_count = 0UL;
      for (chain_subgraph_t chain_subgraph : chain_subgraphs)
        if (!(chain_subgraph.activation_reads_.empty()))
          ++activation_chain_count;
      return activation_chain_count > 1UL;
    }

    template<typename OutputIterator>
    void locate_longer_chains(OutputIterator output)
    {
      std::map<size_t, op_list_t> dpu_levels;

      mv::OpModel &model = omodel_;
      //////////////////////////////////////////////////////////////////////////
      std::list<operation_t> zero_in_degree_nodes[2UL];
      std::unordered_map<operation_t, size_t> in_degree_map;
      size_t curr_depth = 0;
      // STEP-0: compute the in-degree's of all nodes //
      //NOTE: in_degree means the number of inputs of an op, and the pseudo data flows
      //if an op is zero_in_degree goes to zero_in_degree_nodes, like constants
      for (auto op_itr = model.opBegin(); op_itr != model.opEnd(); ++op_itr)
      {
        size_t in_degree = 0;
        for (auto pitr=op_itr.leftmostParent(); pitr!=model.opEnd(); ++pitr)
          ++in_degree;

        operation_t op = &(*op_itr);
        in_degree_map[ op ] = in_degree;
        if (!in_degree)
          zero_in_degree_nodes[0].push_back(op);
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
          mv::Data::OpListIterator zop_itr = model.getOp((*zitr)->getName());
          for (auto citr=zop_itr.leftmostChild(); citr!=model.opEnd(); ++citr)
          {
            operation_t cop = &(*citr);
            auto ditr = in_degree_map.find(cop);
            if ( (ditr == in_degree_map.end()) || (ditr->second == 0UL) )
            {
              throw "Missing entry in the in-degree map (or)"
                  " invalid in-degree for op= " + cop->getName();
            }
            --(ditr->second);
            if (!(ditr->second))
            {
              zero_in_degree_nodes[!parity].push_back(cop);
              if (cop->getOpType() == "DPUTask")
              {
                dpu_levels[curr_depth].push_back(cop);
              }
              postProcessImplicitOperationsForDAG(dpu_levels, cop, curr_depth);
            }
          }
        }
        zero_in_degree_nodes[parity].clear();
        curr_depth++;
      }
      clearEmptyDepth(dpu_levels);

      // Activation and Vertical Fusion Chain Pipelining
      for (auto& level : dpu_levels)
      {
        const op_list_t &dpu_list = level.second;
        // create activation chains
        op_list_t activation_dpu_chain;
        std::unordered_map<std::string, bool> control_map;
        for (auto& opIt : dpu_list)
        {
          // from the head add the entire vertical subgraph
          if (opIt->hasAttr("verticalFusionSubgraphHead") && opIt->get<bool>("verticalFusionSubgraphHead"))
          {
            add_vertical_subgraph(opIt, std::back_inserter(activation_dpu_chain), control_map);
            // save the length of the subgraph
            omodel_.getOp(opIt->getName())->set<std::size_t>("subgraph_length", activation_dpu_chain.size());
          }
          // skip other operations as they should already be added above
          if (opIt->hasAttr("verticalFusion"))
            continue;
          // make sure the op will fit in CMX
          if (!can_activation_pipelining_be_applied(opIt))
          {
            activation_dpu_chain.clear();
            break;
          }
          // chains created from ops that are optimizable in activation streaming performance
          // a chain contains ops that belong to the same streaming - share the same weights
          if (opIt->hasAttr("performance_optimized") && opIt->get<bool>("performance_optimized") &&
              is_sharing_weights_operation(opIt) && share_the_same_weights(dpu_list))
            activation_dpu_chain.push_back(opIt);
        }

        // currently no overlapping chains, want to be able to prefetch more then half.
        // for smaller chains the scheduler parallelizes the ops
        if (activation_dpu_chain.size() > 8UL)
        {
          // create a subgraph //
          chain_subgraph_t chain_subgraph;
          chain_subgraph.dpu_chain_ = activation_dpu_chain;
          std::list<op_list_t> &activation_reads = chain_subgraph.activation_reads_;
          // now create reads //
          for (operation_t dpu_op : chain_subgraph.dpu_chain_)
          {
            op_list_t reads;
            if (dpu_op->hasAttr("verticalFusion"))
              get_vf_read_inputs(dpu_op, std::back_inserter(reads));
            else
              get_non_shared_read_inputs(dpu_op, std::back_inserter(reads));
            activation_reads.push_back(reads);
          }
          output = chain_subgraph;
        }
      }

      // Weight Pipelining
      auto level_itr = dpu_levels.begin();
      while (level_itr != dpu_levels.end())
      {
        const op_list_t & dpu_list = level_itr->second;

        //NOTE: if your dpu_list size is bigger than 1 but you belong to same streaming op, no problem!!
        if ((dpu_list.size() > 1UL) && !(comeFromTheSameParentStream(dpu_list)))
        {
          ++level_itr;
          continue;
        }

        /// try to create a chain subgraph ////
        auto next_level_itr = level_itr;
        op_list_t weight_dpu_chain;

        //NOTE: chain means a linear subgraph, when you find in the dpu_levels
        // a list of dpus with level > 1 you stop attaching in the chain
        do
        {
          for (auto& opIt : next_level_itr->second)
          {
            operation_t current_dpu_op = opIt;
            if (is_eltwise_with_soh_runtime_sparsity(current_dpu_op))
            {
              weight_dpu_chain.clear();
              break;
            }

            if (is_sharing_weights_operation(current_dpu_op))
              continue;
            
            // if current DPU is on output branch, no more DPUs can be added after it
            if (is_last_dpu_on_output_branch(current_dpu_op))
              break;

            //TODO(vamsikku): also avoid chains with breaks and pivots //
            //check if the current dpu_op has an incoming edge which is not yet
            //in the chain.

            weight_dpu_chain.push_back(current_dpu_op);
          }
          ++next_level_itr;
        }
        while ((next_level_itr != dpu_levels.end()) &&
                can_this_level_be_appended_to_chain(next_level_itr));

        if (weight_dpu_chain.size() > 1UL)
        {
          // create a subgraph //
          chain_subgraph_t chain_subgraph;
          chain_subgraph.dpu_chain_ = weight_dpu_chain;
          std::list<op_list_t> &weight_reads = chain_subgraph.weight_reads_;
          // now create reads //
          for (operation_t dpu_op : chain_subgraph.dpu_chain_)
          {
            op_list_t reads;
            get_weight_read_inputs(dpu_op, std::back_inserter(reads));
            weight_reads.push_back(reads);
          }
          output = chain_subgraph;
        }

        // make progress //
        if (level_itr == next_level_itr) { ++next_level_itr; }

        level_itr = next_level_itr;
      }
    }

  //NOTE: This function will check if all the dpus that are in the same
  //level are coming from different stream parent ops, or if they are
  //not streaming at all and the actual level of the ops are > 1
  bool comeFromTheSameParentStream(op_list_t operationLevelStream) const
  {
      bool comeFromTheSameParentStream = false;
      std::set<std::string> parentOpStreamNames;
      std::size_t numberOfOpsOnLevelWithNoStream = 0;
      parentOpStreamNames.clear();

      for (auto& i : operationLevelStream)
      {
          if (!i->hasAttr("parentOpName"))
          {
            numberOfOpsOnLevelWithNoStream++;
            continue;
          }
          else
            parentOpStreamNames.insert(i->get<std::string>("parentOpName"));
      }

      if ((parentOpStreamNames.size() == 1) &&
          (numberOfOpsOnLevelWithNoStream == 0))
          comeFromTheSameParentStream = true;

      if ((parentOpStreamNames.size() == 0) &&
          (numberOfOpsOnLevelWithNoStream == 1))
          comeFromTheSameParentStream = true;



      return comeFromTheSameParentStream;
  }

  bool vf_subgraph_can_be_pipelined(op_list_t dpu_chain) const
  {
    auto itr = dpu_chain.begin();
    auto head = omodel_.getOp((*itr)->getName());
    auto globalParams = omodel_.getGlobalConfigParams();
    size_t clusterMemory = globalParams->get<int>("cmx");
    size_t subgraph_size = 0;
    // if input already in cmx do not pipeline
    if (activation_input_in_cmx(clusterMemory, &(*head.leftmostParent())))
      return false;
    // the input activation from DDR
    subgraph_size += head->getInputTensor(mv::IO_TENSOR_INPUT)->getClusterSize();
    // iterate through one vertical fusion subgraph and calculate subgraph size
    do {
      subgraph_size += omodel_.getOp((*itr)->getName())->getOutputTensor(mv::IO_TENSOR_OUTPUT)->getClusterSize();
      ++itr;
    } while ((*itr)->hasAttr("verticalFusion") && !((*itr)->hasAttr("verticalFusionSubgraphTail") 
            && (*itr)->get<bool>("verticalFusionSubgraphTail")));
    // allow a 15% cmx fragmentation margin
    return subgraph_size < (clusterMemory * 0.85);
  }

  bool can_output_of_subgraph_be_cmx_concatinated(const op_list_t& dpu_chain) const
  {
    auto globalParams = omodel_.getGlobalConfigParams();
    size_t clusterMemory = globalParams->get<int>("cmx");
    size_t chain_output_size = 0;
    for (auto dpu_op : dpu_chain)
    {
      auto curr_op = omodel_.getOp(dpu_op->getName());
      // in vf only the tail ops can write to DDR
      if (curr_op->hasAttr("verticalFusion") && curr_op->hasAttr("verticalFusionSubgraphTail") 
        && curr_op->get<bool>("verticalFusionSubgraphTail"))
          chain_output_size += curr_op->getOutputTensor(mv::IO_TENSOR_OUTPUT)->getClusterSize();
      // in streaming over h all ops can write to DDR
      else
        chain_output_size += curr_op->getOutputTensor(mv::IO_TENSOR_OUTPUT)->getClusterSize();
    }
    if (chain_output_size > clusterMemory) { return false; }

    // check if concat has no cmx concat flag
    auto last_op = omodel_.getOp(dpu_chain.back()->getName());
    while (last_op->getOpType() != "ImplicitConcat")
      last_op = last_op.leftmostChild();

    // TODO: might be more performant to spill the concat to DDR
    last_op->set<bool>("avoid_cmx_concat", true);
      return false;
  }

  bool double_tail_vf_subgraph(op_list_t dpu_chain) const
  {
    size_t tail_count = 0UL;
    auto i = dpu_chain.begin();
    // iterate to the next head
    do {
      if ((*i)->hasAttr("verticalFusionSubgraphHead") && !(*i)->get<bool>("verticalFusionSubgraphHead"))
        ++tail_count;
      ++i;
    } while ((*i)->hasAttr("verticalFusion") && !((*i)->hasAttr("verticalFusionSubgraphHead") 
            && (*i)->get<bool>("verticalFusionSubgraphHead")));
    // if more than one tail it is double tail subgraph
    return tail_count > 1;
  }

  size_t calculate_vf_select_stages(op_list_t::const_iterator op_itr, bool double_tail_vf) const
  {
    size_t select_stages = 0;
    // simple analitical approach, could be optimized with a cost model
    mv::Data::OpListIterator temp_itr = omodel_.getOp((*op_itr)->getName());
    if (temp_itr->hasAttr("verticalFusionSubgraphTail") && temp_itr->get<bool>("verticalFusionSubgraphTail"))
      select_stages = 2;
    else if (temp_itr->hasAttr("taskOp"))
    {
      auto taskOp = temp_itr->get<std::string>("taskOp");
      if (taskOp == "Eltwise" || taskOp == "ImplicitConcat")
      {
        if (double_tail_vf)
          select_stages = 3;
        else
          select_stages = 2;
      }
    }
    return select_stages;
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

    void transform_op_model(FILE *fptr=stdout, size_t select_stages=0UL, mv::Target target_type=mv::Target::ma2490,
        bool activation_pipelining=false, bool vertical_fusion_pipelining=false) {
      std::list<control_edge_t> control_edges;
      std::list<chain_subgraph_t> subgraphs;
      transform_op_model(std::back_inserter(control_edges), subgraphs,
          select_stages, target_type, activation_pipelining, vertical_fusion_pipelining, fptr);
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
        if (weight_reads.empty())
          throw mv::RuntimeError("LpScheduler", "transform_op_model_old(): weight reads empty");
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
              (*curr_dpu_itr)->getName().c_str());
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
              om.pseudoOp(buf, inputs);
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
    void transform_op_model(ControlEdgeOutput,
        SubGraphContainer& chain_subgraphs, size_t select_stages=0UL, mv::Target target_type=mv::Target::ma2490, 
        bool activation_pipelining=false, bool vertical_fusion_pipelining=false, 
        FILE *fptr=stdout) {

      static_assert( std::is_same<chain_subgraph_t,
            typename SubGraphContainer::value_type>::value,
              "Invalid container for chain subgraphs");

      chain_subgraphs.clear();
      locate_longer_chains(std::back_inserter(chain_subgraphs));
      activation_pipelining = activation_pipelining && is_network_candidate(chain_subgraphs);

      for (chain_subgraph_t chain_subgraph : chain_subgraphs)
      {
        if(activation_pipelining && !(chain_subgraph.activation_reads_.empty()))
          pipeline_ops_in_chain(chain_subgraph, true, select_stages, target_type, vertical_fusion_pipelining, fptr);
        if(!(chain_subgraph.weight_reads_.empty()))
          pipeline_ops_in_chain(chain_subgraph, false, select_stages, target_type, false, fptr);
      }
    }

    void pipeline_ops_in_chain(chain_subgraph_t chain_subgraph,
        bool activation_pipelining, size_t select_stages=0UL, mv::Target target_type=mv::Target::ma2490,
        bool vertical_fusion_pipelining=false, FILE *fptr=stdout) {

        const op_list_t& dpu_chain = chain_subgraph.dpu_chain_;
        assert(!dpu_chain.empty() && "dpu_chain is empty");
        if (dpu_chain.size() <= 3UL) { return; }

        std::list<op_list_t>::iterator reads_begin;
        std::list<op_list_t>::iterator reads_end;
        bool vertical_fusion_subgraph = false;
        bool double_tail_vf = false;
        std::size_t vf_subgraph_length = 0;

        if (activation_pipelining)
        {
          auto dpu_flow_control = dpu_chain.begin();
          auto dpu_flow_control_prev = dpu_flow_control;
          ++dpu_flow_control;

          if ((*dpu_flow_control_prev)->hasAttr("verticalFusion"))
          {
            if (!vertical_fusion_pipelining || !vf_subgraph_can_be_pipelined(dpu_chain))
              return;
            vertical_fusion_subgraph = true;
            auto head_op = omodel_.getOp((*dpu_flow_control_prev)->getName());
            if (head_op->hasAttr("subgraph_length"))
              vf_subgraph_length = head_op->get<std::size_t>("subgraph_length");
            double_tail_vf = double_tail_vf_subgraph(dpu_chain);
            // issues with spills EISW-12020, extra vertical fusion pipelining not performant with 1 DMA controller
            if (double_tail_vf && target_type == mv::Target::ma2490)
              return;
          }

          // if output can be CMX-ed these flows can not be added
          bool out_cmxed = can_output_of_subgraph_be_cmx_concatinated(dpu_chain);
          while (!out_cmxed && dpu_flow_control != dpu_chain.end())
          {
            // set the flow: Ha0 -> Ha1 -> ... -> HaN -> Hb0 -> Hb1 -> ... -> HbN
            mv::Data::OpListIterator prev_op = omodel_.getOp((*dpu_flow_control_prev)->getName());
            mv::Data::OpListIterator curr_op = omodel_.getOp((*dpu_flow_control)->getName());
            if (!omodel_.pathExists(prev_op, curr_op) && !omodel_.pathExists(curr_op, prev_op))
            {
              std::size_t inputIdx = curr_op.parentsSize();
              omodel_.defineFlow(prev_op->getOutputTensor(0UL), curr_op, inputIdx + 1);
            }
            ++dpu_flow_control;
            ++dpu_flow_control_prev;
          }

          // curently no overlapping chains so chains with length < 4 better optimized by scheduler
          if (vertical_fusion_subgraph && vf_subgraph_length < 4)
              return;

          reads_begin = chain_subgraph.activation_reads_.begin();
          reads_end = chain_subgraph.activation_reads_.end();
        }
        else
        {
          reads_begin = chain_subgraph.weight_reads_.begin();
          reads_end = chain_subgraph.weight_reads_.end();
        }


        chain_subgraph.print(fptr);
        //compute_working_memory_for_eltwise_chain(chain_subgraph);
        auto curr_dpu_itr = dpu_chain.begin();
        auto curr_itr = reads_begin;

        auto pprev_dpu_itr = curr_dpu_itr;

        auto pprev_itr = curr_itr;

        if (curr_itr == reads_end) { return; }
        ++curr_itr;
        ++curr_dpu_itr;

        if (curr_itr == reads_end) { return; }
        ++curr_itr;
        ++curr_dpu_itr;

        size_t curr_dpu_index = 2UL;
        std::unordered_map<operation_t, size_t> stage_memory;
        while (curr_dpu_itr != dpu_chain.end())
        {
          if (curr_itr == reads_end)
          throw RuntimeError("transform_op_model", "curr_itr == reads_end");
          const op_list_t & curr_read_list = *curr_itr;
          if (pprev_itr == reads_end)
            throw RuntimeError("transform_op_model", "pprev_itr == reads_end");
          const op_list_t & pprev_read_list = *pprev_itr;
          if (!pprev_read_list.empty())
          {
            auto net_dpu_itr = pprev_dpu_itr;
            // calculate select stages based on pipelining type
            if (vertical_fusion_subgraph)
              select_stages = calculate_vf_select_stages(pprev_dpu_itr, double_tail_vf);
            else if (activation_pipelining)
              select_stages = 0;
            for (size_t sl=0; (sl < select_stages) &&
                  (net_dpu_itr != dpu_chain.begin()); ++sl) { --net_dpu_itr; }
            mv::Data::OpListIterator src_itr =
                omodel_.getOp((*net_dpu_itr)->getName());

            size_t demand = 0UL;
            for (operation_t curr_read_op : curr_read_list )
            {
              mv::Data::OpListIterator curr_read_op_itr =
                  omodel_.getOp(curr_read_op->getName());
              demand +=
                  (curr_read_op_itr->getOutputTensor(0UL))->getClusterSize();
            }

            if (stage_memory.find((*net_dpu_itr)) == stage_memory.end())
            {
              auto net_dpu_op_itr = omodel_.getOp((*net_dpu_itr)->getName());
              stage_memory[*net_dpu_itr] =
                (net_dpu_op_itr->getOutputTensor(0UL))->getClusterSize();
              auto next_dpu_itr = net_dpu_itr;
              ++next_dpu_itr;
              if (next_dpu_itr != dpu_chain.end())
              {
                auto next_dpu_op_itr =
                    omodel_.getOp((*next_dpu_itr)->getName());
                stage_memory[*net_dpu_itr] +=
                  (next_dpu_op_itr->getOutputTensor(0UL))->getClusterSize();
              }
            }

            //TODO(vamsikku): parameterize this based on CMX availablity //
            if ( !activation_pipelining && (stage_memory[*net_dpu_itr] + demand) > 700000)
              goto MOVE_TO_NEXT_SUBGRAPH;

            stage_memory[*net_dpu_itr] += demand;

            for (operation_t curr_read_op : curr_read_list )
            {
              mv::Data::OpListIterator sink_itr =
                  omodel_.getOp(curr_read_op->getName());
              sink_itr->set<bool>("pipeline_flow_control", true);
              if (activation_pipelining)
                sink_itr->set<bool>("pipelined_dma_task", true);
              mv::Data::TensorIterator src_tensor_itr
                  = src_itr->getOutputTensor(0UL);
              {
                if (!omodel_.pathExists(sink_itr, src_itr) && !omodel_.pathExists(src_itr, sink_itr))
                {
                  mv::Data::FlowListIterator flow_itr =
                      omodel_.defineFlow(src_tensor_itr, sink_itr, 0UL);
                  flow_itr->set<bool>("pseudo_data_flow", true);
                  src_itr->set<bool>("chain_pipelined_dpu", true);
                }
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

    void inplace_eltwise_pipeline_transforms() {
      std::list<chain_subgraph_t> chain_subgraphs;
      locate_longer_chains(std::back_inserter(chain_subgraphs));

    }

    void transform_inplace_eltwise_chain(const chain_subgraph_t& ){
    }

    size_t compute_working_memory_for_eltwise_chain(
        chain_subgraph_t& eltwise_chain,
        const std::string& inplace_attribute="inplace_eltwise_rep") const {

      if (eltwise_chain.dpu_chain_.size() < 2UL) {
        throw "chain lenght must be >= 2";
      }

      typedef typename chain_subgraph_t::read_size_map_t read_size_map_t;
      eltwise_chain.compute_total_read_sizes();
      read_size_map_t &read_size_map = eltwise_chain.total_read_size_map_;

      size_t max_read_memory_demand = 0UL;
      op_list_t dpu_chain = eltwise_chain.dpu_chain_;
      {
        size_t prev_non_zero_weight = 0UL;
        ///STEP-0: compute read memory demand //
        // init //
        auto dpu_itr = dpu_chain.begin(), dpu_itr_end = dpu_chain.end();
        while ((dpu_itr != dpu_itr_end) &&
              !(prev_non_zero_weight=read_size_map[*dpu_itr])) { ++dpu_itr; }
        size_t curr_read_memory_demand, curr_read_memory;
        for (;dpu_itr != dpu_itr_end; ++dpu_itr) {
          curr_read_memory = read_size_map[*dpu_itr];
          if (!curr_read_memory) { continue; }

          curr_read_memory_demand = (prev_non_zero_weight + curr_read_memory);

          if (curr_read_memory_demand > max_read_memory_demand) {
            max_read_memory_demand = curr_read_memory_demand;
          }
          prev_non_zero_weight = curr_read_memory;
        }
        printf("max_read_memory: %zu\n", max_read_memory_demand);
        fflush(stdout);
      }

      //STEP-1: compute dpu memory demand //
      //assumes that all dpus have non-zero memory //
      size_t max_dpu_demand = 0UL;
      {
        auto dpu_itr = dpu_chain.begin();
        auto prev_dpu_itr = dpu_itr;
        ++dpu_itr;

        size_t curr_dpu_demand, curr_dpu_memory;
        size_t prev_dpu_memory = op_memory_demand(*prev_dpu_itr);

        for (;dpu_itr != dpu_chain.end(); ++dpu_itr) {
          curr_dpu_memory = op_has_this_attribute(*dpu_itr, inplace_attribute)
              ? 0UL : op_memory_demand(*dpu_itr);
          curr_dpu_demand = (prev_dpu_memory + curr_dpu_memory);
          if (curr_dpu_demand > max_dpu_demand) {
            max_dpu_demand = curr_dpu_demand;
          }
          prev_dpu_memory = op_memory_demand(*dpu_itr);
        }
        printf("max_dpu_memory: %zu\n", max_dpu_demand);
      }

      size_t working_memory = (max_dpu_demand + max_read_memory_demand);

      printf("[CHAIN_WORKING_MEMORY]: %zu \n", working_memory);
      return working_memory;
    }


























  private:
    mv::OpModel& omodel_;
}; // class Pipeline_Chains //

} // namespace scheduler //
} // namespace mv//








#endif
