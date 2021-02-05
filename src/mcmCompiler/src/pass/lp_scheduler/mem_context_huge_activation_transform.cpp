#include "include/mcm/base/exception/runtime_error.hpp"
#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "lp_scheduler/lp_scheduler_pass.hpp"

static void MemContextForHugeActivations(
    const mv::pass::PassEntry& , mv::ComputationModel&, mv::TargetDescriptor&,
      mv::Element&, mv::Element&);

namespace mv {
  namespace pass {
    MV_REGISTER_PASS(MemContextForHugeActivations)
      .setFunc(MemContextForHugeActivations)
      .setDescription("Create Memory Context For Huge Activations.");
  } // namespace mv //
} // namespace pass //


typedef mv::scheduler::Operation_Dag<mv::OpModel> dag_t;
typedef typename dag_t::operation_t operation_t;

class Find_Mem_Contextable_Sequence {
  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef std::list<operation_t> op_list_t;
    typedef std::map<size_t, op_list_t> dpu_level_map_t;
    typedef typename dag_t::pseudo_edge_set_t edge_set_t;
    typedef typename dag_t::pseudo_edge_t edge_t;

    ////////////////////////////////////////////////////////////////////////////
    Find_Mem_Contextable_Sequence(mv::OpModel& model,
        size_t peak_threshold, size_t valley_threshold, size_t upper_bound,
        bool enable_cmx_concat_in_dag=false,
        bool enable_inplace_eltwise_in_dag=false)
      : model_(model), peak_threshold_(peak_threshold),
        valley_threshold_(valley_threshold), upper_bound_(upper_bound),
        edge_set_(), enable_cmx_concat_in_dag_(enable_cmx_concat_in_dag),
        enable_inplace_eltwise_in_dag_(enable_inplace_eltwise_in_dag)
    {}

    struct peak_valley_chain_subgraph_t {
      operation_t peak_dpu_;
      std::list<operation_t> valley_dpus_;

      bool is_valid() const { return (peak_dpu_ && !valley_dpus_.empty()); }
      void clear() { valley_dpus_.clear(); peak_dpu_ = NULL; }

      peak_valley_chain_subgraph_t() : peak_dpu_(NULL), valley_dpus_() {}

      void dump(FILE *fptr=stdout) const {
        fprintf(fptr, "==========================================\n");
        if (!is_valid()) { fprintf(fptr, "[INVALID]\n"); return; }
        fprintf(fptr, "Peak DPU: %s\n", peak_dpu_->getName().c_str());
        fprintf(fptr, "Valley DPUs:\n");
        for (operation_t dpu : valley_dpus_) {
          fprintf(fptr, "%s->", dpu->getName().c_str());
        }
        fprintf(fptr, "\n");
        fprintf(fptr, "==========================================\n");
      }

    }; // struct peak_valley_chain_subgraph_t //

    struct edge_attribute_transactions_t {
      std::list<mv::Data::FlowListIterator> flows_added_;

      // only pseudo data flows can be removed //
      std::list<mv::Data::FlowListIterator> flows_removed_;

      std::set<std::string> attribute_names_;
      std::list<operation_t> attribute_op_list_;
    }; // struct edge_attribute_transactions_t //

    template<typename OutputIterator>
    void locate_peak_valley_chains(OutputIterator output) {
      peak_valley_chain_subgraph_t result =
          locate_peak_valley_chain_extending_to_input();
      if (result.is_valid()) {
        *output = result;
      }
    }

    //Precondition: preak_valley_subgraph is valid //
    void transform(const peak_valley_chain_subgraph_t& peak_valley_subgraph,
          edge_attribute_transactions_t& transactions) {

      size_t max_valley_size = 0;
      for (operation_t dpu_op : peak_valley_subgraph.valley_dpus_) {
        max_valley_size = std::max(max_valley_size, resource_utility(dpu_op));
      }

      operation_t peak_dpu = peak_valley_subgraph.peak_dpu_;
      operation_t rep_dpu = peak_valley_subgraph.valley_dpus_.front();
      size_t mem_context_size = max_valley_size + resource_utility(peak_dpu);

     
      //STEP-0: first set offset for peak dpu //
      set_mem_context_attributes(peak_dpu, 0UL, max_valley_size, rep_dpu);

      size_t idx = 0UL;
      for (auto dpu_itr = peak_valley_subgraph.valley_dpus_.rbegin();
            dpu_itr != peak_valley_subgraph.valley_dpus_.rend();
              ++dpu_itr, ++idx) {
        operation_t valley_dpu = *dpu_itr;
        set_mem_context_attributes(valley_dpu,
            (valley_dpu == rep_dpu) ? mem_context_size : 0UL, 
            (idx%2UL) ? max_valley_size: 0UL, rep_dpu);
      }

      {
        create_edge_table_from_model(model_);
        // add resource control edges between rep and children of peak //
        mv::Data::OpListIterator peak_op_itr =
            model_.getOp(peak_dpu->getName());
        mv::Data::OpListIterator rep_op_itr = model_.getOp(rep_dpu->getName());

        for (auto citr=peak_op_itr.leftmostChild(); citr!=model_.opEnd();
              ++citr) {
          operation_t child_op = &(*citr);
          mv::Data::OpListIterator child_op_itr =
              model_.getOp(child_op->getName());

          if (!has_edge(peak_dpu, child_op) ) {
            mv::Data::FlowListIterator fitr =
                model_.defineFlow(rep_op_itr->getOutputTensor(0UL),
                      child_op_itr, 0UL);
            // add to transactions //
            transactions.flows_added_.push_back( fitr );
          }
        }
      }
    }

    void undo_transform(
        const peak_valley_chain_subgraph_t& peak_valley_subgraph,
          edge_attribute_transactions_t& transactions) {
      for (mv::Data::FlowListIterator fitr : transactions.flows_added_) {
        model_.undefineFlow(fitr);
      }

      for (operation_t dpu_op : peak_valley_subgraph.valley_dpus_) {
        unset_mem_context_attributes(dpu_op);
      }
      unset_mem_context_attributes(peak_valley_subgraph.peak_dpu_);
    }

    bool is_model_precondition_valid() const {
      dag_t input_dag;

      mv::OpModel &cm = model_;

      if (enable_cmx_concat_in_dag_) {
        input_dag.enable_cmx_concat_transforms(cm);
      } else {
        // input_dag.reset(cm);
      }

      if (enable_inplace_eltwise_in_dag_) {
        input_dag.enable_eltwise_transforms(cm);
      }

      std::list< std::pair<operation_t, size_t > > exceeding_ops;
      input_dag.find_all_ops_exceeding_resource_threshold(upper_bound_,
          std::back_inserter(exceeding_ops));
      return exceeding_ops.empty();
    }

  private:

    bool has_edge(operation_t src, operation_t sink) const {
      return (edge_set_.find( edge_t(src, sink) ) != edge_set_.end() );
    }

    void create_edge_table_from_model(mv::OpModel& om) {
      mv::DataModel dm(om);
      edge_set_.clear();
      for (auto eitr=dm.flowBegin(); eitr!=dm.flowEnd(); ++eitr) {
        edge_set_.insert( edge_t( &(*(eitr.source())), &(*(eitr.sink())) ) );
      }
    }

    void set_mem_context_attributes(operation_t curr_dpu,
        size_t context_utility, size_t offset, operation_t rep_dpu) {
      mv::Data::OpListIterator curr_dpu_itr = model_.getOp(curr_dpu->getName());
      curr_dpu_itr->set<size_t>("memory_context_utility", context_utility);
      curr_dpu_itr->set<size_t>("memory_context_offset", offset);
      curr_dpu_itr->set<std::string>("memory_context_rep",
            rep_dpu->getName());
      curr_dpu_itr->set<size_t>("actual_size_inside_memory_context",
              resource_utility(curr_dpu));
    }

    void unset_mem_context_attributes(operation_t curr_dpu) {
      std::vector<std::string> attributes = {"memory_context_utility",
        "memory_context_offset", "memory_context_rep",
        "actual_size_inside_memory_context" };
      mv::Data::OpListIterator curr_dpu_itr = model_.getOp(curr_dpu->getName());

      for (const std::string& attr : attributes) {
        if (curr_dpu_itr->hasAttr(attr)) {
          curr_dpu_itr->erase(attr);
        }
      }
    }

    void compute_dpu_level_map(dpu_level_map_t& dpu_levels) const {
      dpu_levels.clear();
      mv::OpModel &model = model_;

      //////////////////////////////////////////////////////////////////////////
      std::list<operation_t> zero_in_degree_nodes[2UL];
      std::unordered_map<operation_t, size_t> in_degree_map;
      size_t curr_depth = 0;
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
            }
          }
        }
        zero_in_degree_nodes[parity].clear();
        curr_depth++;
      }
    }

    peak_valley_chain_subgraph_t locate_peak_valley_chain_extending_to_input() {
      peak_valley_chain_subgraph_t subgraph;
      dpu_level_map_t dpu_levels;

      compute_dpu_level_map(dpu_levels);

      if (dpu_levels.empty()) { return subgraph; }

      // pattern : valley_rep->valley->...->peak //
      auto prev_itr = dpu_levels.begin(), curr_itr=prev_itr;

      if ((curr_itr->second).size() != 1UL) { return subgraph; }

      operation_t prev_dpu = curr_itr->second.front(), curr_dpu = prev_dpu;

      if (!is_valley(curr_dpu)) { return subgraph; }
      subgraph.valley_dpus_.push_back(curr_dpu);

      for (++curr_itr; curr_itr != dpu_levels.end(); ++curr_itr,++prev_itr) {
        // make sure curr_itr is a child of prev itr //

        // Following properites must be satistifed //
        // P1. only 1 DPU in level
        // P2. connected with prev and prev has only 1 DPU //
        // P4. valley or peak //

        /////////////////////////////[P1]///////////////////////////////////////
        if ((curr_itr->second).size() != 1UL) { 
          subgraph.clear();
          break;
        }
        /////////////////////////////[P2]///////////////////////////////////////
        {
          prev_dpu = (prev_itr->second).front();
          curr_dpu = (curr_itr->second).front();
          operation_t only_child_dpu = get_the_only_child_dpu(prev_dpu);
          bool connected = only_child_dpu && (only_child_dpu == curr_dpu);

          if (!connected) {
            subgraph.clear();
            break;
          }
        }
        /////////////////////////////[P3]///////////////////////////////////////
        if (!(is_valley(curr_dpu) || is_peak(curr_dpu))) {
          subgraph.clear();
          break;
        }

        if (is_valley(curr_dpu)) {
          subgraph.valley_dpus_.push_back(curr_dpu);
        }

        if (is_peak(curr_dpu)) {
          subgraph.peak_dpu_ = curr_dpu;
          break;
        }
      }

      return subgraph;
    }

    // If there are multiple DPU children this will return NULL //
    operation_t get_the_only_child_dpu(operation_t dpu_op) const {
      mv::Data::OpListIterator op_itr = model_.getOp(dpu_op->getName());
      operation_t only_child_dpu = NULL;
      for (auto child_itr = op_itr.leftmostChild(); child_itr != model_.opEnd();
            ++child_itr) {
        if (child_itr->getOpType() == "DPUTask") {
          if (only_child_dpu) { only_child_dpu = NULL; break;}
          else { only_child_dpu = &(*child_itr); }
        }
      }
      return only_child_dpu;
    }

   
    size_t resource_utility(operation_t dpu_op) const {
      size_t demand = (const_cast<mv::Op *>(dpu_op))->getOutputTensor(0UL)->
          getClusterSize();
      return demand;
    }
    bool is_valley(operation_t dpu_op) const {
      size_t demand = resource_utility(dpu_op);
      return (demand < peak_threshold_) && (demand >= valley_threshold_);
    }

    bool is_peak(operation_t dpu_op) const {
      size_t demand = resource_utility(dpu_op);
      return (demand >= peak_threshold_);
    }

    ////////////////////////////////////////////////////////////////////////////
    mv::OpModel& model_;
    size_t peak_threshold_;  // min //
    size_t valley_threshold_; // min //
    size_t upper_bound_;
    edge_set_t edge_set_;
    bool enable_cmx_concat_in_dag_;
    bool enable_inplace_eltwise_in_dag_;
    ////////////////////////////////////////////////////////////////////////////
};  // class Find_Mem_Contextable_Sequence //


void MemContextForHugeActivations(const mv::pass::PassEntry&,
    mv::ComputationModel& model, mv::TargetDescriptor& , mv::Element& passDesc,
    mv::Element&) {

  //////////////////////////////////////////////////////////////////////////////
  bool enabled = 
      passDesc.hasAttr("enable_pass") && passDesc.get<bool>("enable_pass");
  if (!enabled) { return; }

  size_t cmx_upper_bound = model.getGlobalConfigParam("totalCmx").get<unsigned>();
  double peak_percent = 0.5;
  double valley_percent = 0.25;
  size_t max_context_chain_size = 3UL;
  bool enable_cmx_concat_in_dag=false, enable_inplace_eltwise_in_dag=false;

  if (passDesc.hasAttr("max_context_chain_size")) {
    max_context_chain_size =
        size_t(passDesc.get<int>("max_context_chain_size"));
  }

  if (passDesc.hasAttr("peak_threshold_percent")) {
    peak_percent =
      double(passDesc.get<int>("peak_threshold_percent"))/double(100.0);
  }
  if (passDesc.hasAttr("valley_threshold_percent")) {
    valley_percent =
      double(passDesc.get<int>("valley_threshold_percent"))/double(100.0);
  }
  if (passDesc.hasAttr("enable_cmx_concat")) {
    enable_cmx_concat_in_dag = passDesc.get<bool>("enable_cmx_concat");
  }
  if (passDesc.hasAttr("enable_inplace_eltwise")) {
    enable_inplace_eltwise_in_dag=passDesc.get<bool>("enable_inplace_eltwise");
  }


  if (!( (peak_percent < 1.0)  && (valley_percent < 1.0) && 
         (valley_percent < peak_percent) ) ){
    throw mv::RuntimeError("MemContextPass", "Invalid Parameters");
  }
  //////////////////////////////////////////////////////////////////////////////


  mv::OpModel omodel(model);

  size_t peak_size = size_t( std::ceil(peak_percent*double(cmx_upper_bound)) );
  size_t valley_size =
      size_t( std::ceil(valley_percent*double(cmx_upper_bound)) );

  printf("[peak=%lu valley=%lu]\n", peak_size, valley_size);
  Find_Mem_Contextable_Sequence ctx_finder(omodel, peak_size, valley_size,
        cmx_upper_bound, enable_cmx_concat_in_dag,
        enable_inplace_eltwise_in_dag);

  // precondition check //
  if (!ctx_finder.is_model_precondition_valid()) {
    throw mv::RuntimeError("MemContextPass",
          "Model precondition failed due to exceeding ops.");
  }

  std::list<Find_Mem_Contextable_Sequence::peak_valley_chain_subgraph_t>
      peak_valley_subgraphs;
  ctx_finder.locate_peak_valley_chains(
        std::back_inserter(peak_valley_subgraphs));

  for (const auto & peak_valley_subgraph : peak_valley_subgraphs) {
    if (peak_valley_subgraph.valley_dpus_.size() > max_context_chain_size) {
      continue;
    }

    Find_Mem_Contextable_Sequence::edge_attribute_transactions_t transactions;
    ctx_finder.transform(peak_valley_subgraph, transactions);

    if (!ctx_finder.is_model_precondition_valid()) {
      // undo transform //
      ctx_finder.undo_transform(peak_valley_subgraph, transactions);
      if (!ctx_finder.is_model_precondition_valid()) {
        throw mv::RuntimeError("MemContextPass",
              "Unable to restore precondition validity");
      }
    }
    printf("[MemCtxTransform]: "
          "successfully applied to the following subgraph\n");
    peak_valley_subgraph.dump();
  }
}
