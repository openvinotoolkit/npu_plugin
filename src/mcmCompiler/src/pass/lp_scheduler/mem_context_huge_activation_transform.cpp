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

#if 0
class Find_Mem_Contextable_Sequence {
  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef std::list<operation_t> op_list_t;
    typedef std::map<size_t, op_list_t> dpu_level_map_t;
    typedef typename dag_t::pseudo_edge_t pseudo_edge_t;
    typedef typename dag_t::pseudo_edge_set_t pseudo_edge_set_T;
    typedef typename dag_t::pseudo_edge_set_t edge_set_T;
    typedef typename dag_t::edge_t edge_t;

    ////////////////////////////////////////////////////////////////////////////

    Find_Mem_Contextable_Sequence(mv::OpModel& model,
        size_t peak_threshold, size_t valley_threshold)
      : model_(model), peak_threshold_(peak_threshold),
        valley_threshold_(valley_threshold), pseudo_edge_set_()  { }

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

    struct edge_attribute_transaction_t {
      std::list<mv::Data::FlowListIterator> flows_added_;

      // only pseudo data flows can be removed //
      std::list<mv::Data::FlowListIterator> flows_removed_;

      std::set<std::string> attribute_names_;
      std::list<operation_t> attribute_op_list_;
    }; // struct edge_attribute_transaction_t //

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
      size_t idx = 0UL;
      for (auto dpu_itr = peak_valley_subgraph.valley_dpus_.begin();
            dpu_itr != peak_valley_subgraph.valley_dpus_.end();
              ++dpu_itr, ++idx) {
        set_mem_context_attributes(*dpu_itr, !idx ? mem_context_size : 0UL, 
            (idx%2UL) ? max_valley_size: 0UL, rep_dpu);
      }

      {
        add_pseudo_edges_from_model(model_);
        // add resource control edges between rep and children of peak //
        mv::Data::OpListIterator peak_op_itr =
            model_.getOp(peak_dpu->getName());
        mv::Data::OpListIterator rep_op_itr = model_.getOp(rep_dpu->getName());

        for (auto citr=peak_op_itr.leftmostChild(); citr!=model_.opEnd();
              ++citr) {
          operation_t child_op = &(*citr);
          mv::Data::OpListIterator child_op_itr =
              model_.getOp(child_op->getName());
          if (!is_pseudo_edge(peak_dpu, child_op) ) {
            mv::Data::FlowListIterator fitr =
                omodel.defineFlow(rep_op_itr->getOutputTensor(0UL),
                      child_op_itr, 0UL);
          }
        }
          
      }

    }

  private:

    bool is_pseudo_edge(operation_t src, operation_t sink) const {
      return (pseudo_edge_set_.find( pseudo_edge_t(src, sink) )
            != pseudo_edge_set_.end() );
    }

    void add_pseudo_edges_from_model(mv::OpModel& om) {
      mv::DataModel dm(om);
      pseudo_edge_set_.clear();
      edge_set_.clear();
      for (auto eitr=dm.flowBegin(); eitr!=dm.flowEnd(); ++eitr) {
        if (eitr->hasAttr("pseudo_data_flow")) {
          pseudo_edge_set_.insert(
              pseudo_edge_t( &(*(eitr.source())), &(*(eitr.sink())) )
          );
        }

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

      for (dpu_level_map_t::const_iterator dlitr=dpu_levels.begin();
            dlitr!=dpu_levels.end(); ++dlitr) {
        printf("level=%lu : { \n", dlitr->first);
        for (auto op : dlitr->second) {
          printf(" %s\n", op->getName().c_str());
        }
        printf(" }\n");
      }

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
        // P2. connected with prev.
        // P3. valley or peak //

        /////////////////////////////[P1]///////////////////////////////////////
        if ((curr_itr->second).size() != 1UL) { 
          subgraph.clear();
          break;
        }


        /////////////////////////////[P2]///////////////////////////////////////
        {
          prev_dpu = (prev_itr->second).front();
          curr_dpu = (curr_itr->second).front();
          bool connected = false;
          mv::Data::OpListIterator prev_op_itr = model_.getOp(prev_dpu->getName());
          for (auto child_itr = prev_op_itr.leftmostChild();
              child_itr != model_.opEnd(); ++child_itr) {
            if (child_itr->getName() == curr_dpu->getName()) {
              connected = true;
              break;
            }
          }
          if (!connected) {
            subgraph.clear();
            break;
          }
        }


        if (is_valley(curr_dpu)) {
          subgraph.valley_dpus_.push_back(curr_dpu);
        } else {
          if (is_peak(curr_dpu)) {
            subgraph.valley_dpus_.push_back(curr_dpu);
            subgraph.peak_dpu_ = curr_dpu;
          }
          else { subgraph.clear(); }
          break;
        }
      }

      return subgraph;
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
    pseudo_edge_set_t pseudo_edge_set_;
    edge_set_t edge_set_;
    ////////////////////////////////////////////////////////////////////////////
};  // class Find_Mem_Contextable_Sequence //
#endif



void MemContextForHugeActivations(const mv::pass::PassEntry&,
    mv::ComputationModel& model, mv::TargetDescriptor& , mv::Element& passDesc,
    mv::Element&) {

  //////////////////////////////////////////////////////////////////////////////
  bool enabled = 
      passDesc.hasAttr("enable_pass") && passDesc.get<bool>("enable_pass");
  if (!enabled) { return; }

  size_t cmx_upper_bound = model.getGlobalConfigParam("cmx").get<int>();
  double peak_percent = 0.5;
  double valley_percent = 0.25;

  if (passDesc.hasAttr("peak_threshold_percent")) {
    peak_percent =
      double(passDesc.get<int>("peak_threshold_percent"))/double(100.0);
  }
  if (passDesc.hasAttr("valley_threshold_percent")) {
    valley_percent =
      double(passDesc.get<int>("valley_threshold_percent"))/double(100.0);
  }

  if (!( (peak_percent < 1.0)  && (valley_percent < 1.0) && 
         (valley_percent < peak_percent) ) ){
    throw mv::RuntimeError("MemContextPass", "Invalid Parameters");
  }
  //////////////////////////////////////////////////////////////////////////////


  mv::OpModel omodel(model);
#if 0
  size_t peak_size = size_t( std::ceil(peak_percent*double(cmx_upper_bound)) );
  size_t valley_size =
      size_t( std::ceil(valley_percent*double(cmx_upper_bound)) );

  printf("[peak=%lu valley=%lu]\n", peak_size, valley_size);
  Find_Mem_Contextable_Sequence ctx_finder(omodel, peak_size, valley_size);

  std::list<Find_Mem_Contextable_Sequence::peak_valley_chain_subgraph_t> peaks;

  ctx_finder.locate_peak_valley_chains(std::back_inserter(peaks));

  if (!peaks.empty()) {
    for (auto peak : peaks) {
      peak.dump();
    }
  }
#endif


  {
    // Quick memory context implementation //
    // 
    // STEP-1: move resources around:
    // from : (mem_ctx_child)
    // "mobilenet_v1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/batchnorm/add_1
    // size=247808
    //
    // to : (mem_ctx_rep)
    // "mobilenet_v1/MobilenetV1/Conv2d_1_depthwise/BiasAdd/Add"
    //  size=495616 
    //
    // STEP-2: add control edges between the rep and children for mem_ctx_child
    //
    //
    try {

      mv::Data::OpListIterator mem_ctx_rep_op = omodel.getOp(
          "mobilenet_v1/MobilenetV1/Conv2d_0/BatchNorm/batchnorm/add_1");
      mv::Data::OpListIterator mem_ctx_left_valley_op = omodel.getOp(
          "mobilenet_v1/MobilenetV1/Conv2d_1_depthwise/BiasAdd/Add");
      mv::Data::OpListIterator mem_ctx_peak_op = omodel.getOp(
         "mobilenet_v1/MobilenetV1/Conv2d_1_pointwise/BatchNorm/batchnorm/add_1"
         );
      mv::Data::OpListIterator mem_ctx_right_valley_op = omodel.getOp(
          "mobilenet_v1/MobilenetV1/Conv2d_2_depthwise/BiasAdd/Add");

      // Precondition //
      size_t peak_size =
        mem_ctx_peak_op->getOutputTensor(0UL)->getClusterSize();
      size_t left_valley_size =
        mem_ctx_left_valley_op->getOutputTensor(0UL)->getClusterSize();
      size_t right_valley_size =
        mem_ctx_right_valley_op->getOutputTensor(0UL)->getClusterSize();
      size_t rep_orig_size =
        mem_ctx_rep_op->getOutputTensor(0UL)->getClusterSize();

      if ((left_valley_size >= right_valley_size) &&
          (left_valley_size == rep_orig_size)) {

        printf("[PeakValleyCondition] : passed\n");
        printf("[peak = %lu left_valley=%lu right_valley=%lu\n", peak_size,
            left_valley_size, right_valley_size);
        printf("[mem_context_utility=%lu]\n", (peak_size + left_valley_size));

        mem_ctx_rep_op->set<size_t>("memory_context_utility",
            peak_size + left_valley_size);
        mem_ctx_rep_op->set<size_t>("memory_context_offset", left_valley_size);
        mem_ctx_rep_op->set<std::string>("memory_context_rep",
              mem_ctx_rep_op->getName());
        mem_ctx_rep_op->set<size_t>("actual_size_inside_memory_context",
              rep_orig_size);

        mem_ctx_left_valley_op->set<size_t>("memory_context_utility", 0UL);
        mem_ctx_left_valley_op->set<size_t>("memory_context_offset", 0);
        mem_ctx_left_valley_op->set<std::string>("memory_context_rep",
              mem_ctx_rep_op->getName());
        mem_ctx_left_valley_op->set<size_t>("actual_size_inside_memory_context",
              left_valley_size);

        mem_ctx_peak_op->set<size_t>("memory_context_utility", 0UL);
        mem_ctx_peak_op->set<size_t>("memory_context_offset", left_valley_size);
        mem_ctx_peak_op->set<std::string>("memory_context_rep",
              mem_ctx_rep_op->getName());
        mem_ctx_peak_op->set<size_t>("actual_size_inside_memory_context",
              peak_size);

        // add resource control edges //
        // rep to peak and right_valley//
        mv::Data::FlowListIterator fitr1 = omodel.defineFlow(
            mem_ctx_rep_op->getOutputTensor(0UL), mem_ctx_right_valley_op, 0UL);
        fitr1->set<bool>("resource_control_flow", true);

        mv::Data::FlowListIterator fitr2 = omodel.defineFlow(
            mem_ctx_rep_op->getOutputTensor(0UL), mem_ctx_peak_op, 0UL);
        fitr2->set<bool>("resource_control_flow", true);

        printf("[ResourceControl] (%s,%s)\n", mem_ctx_rep_op->getName().c_str(),
              mem_ctx_right_valley_op->getName().c_str());

      } else {
        printf("[PeakValleyCondition]: failed\n");
      }
    } catch (mv::ArgumentError err) {
      // ignore argument error //
    }
  }

}
