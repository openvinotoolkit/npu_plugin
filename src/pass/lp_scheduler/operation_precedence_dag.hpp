#ifndef OPERATION_PRECEDENCE_DAG_HPP
#define OPERATION_PRECEDENCE_DAG_HPP

#include <cassert>
#include <unordered_map>
#include <unordered_set>

#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/iterator/control_context.hpp"
#include "include/mcm/logger/logger.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/target/kmb/runtime_model/runtime_model.hpp"
#include "include/mcm/utils/warning_manager.hpp"
#include "pass/lp_scheduler/cmx_concat_transform.hpp"
#include "pass/lp_scheduler/pipeline_transform.hpp"
#include "pass/lp_scheduler/pipeline_chains_transform.hpp"
#include "scheduler/feasible_scheduler.hpp"

namespace mv {

template<typename Model>
struct model_traits {
  typedef int const_operation_iterator_t;
  typedef int const_child_operation_iterator_t;
  typedef Model model_t;

  static const_operation_iterator_t begin_operations(model_t&);
  static const_child_operation_iterator_t
      begin_child_operations(const_operation_iterator_t&);
  static const_operation_iterator_t end_operations(model_t& model);
  static const_operation_iterator_t get_iterator(model_t&, const std::string&);
}; // struct model traits //


template<>
struct model_traits<mv::ControlModel> {
  typedef mv::ControlModel model_t;
  typedef mv::Control::OpListIterator const_operation_iterator_t;
  typedef mv::Control::OpChildIterator const_child_operation_iterator_t;

  //TODO(vamsikku): reference to model must be const here //
  static const_operation_iterator_t begin_operations(model_t& cm) {
    return cm.opBegin();
  }

  static const_child_operation_iterator_t begin_child_operations(
      const_operation_iterator_t& op) {
    return op.leftmostChild();
  }

  static const_operation_iterator_t end_operations(model_t& model) {
    return model.opEnd();
  }

  static const_operation_iterator_t get_iterator(model_t& model,
      const std::string& name) {
    auto op_itr = model.getOp(name);
    return model.switchContext(op_itr);
  }
}; // struct model_traits<mv::ControlModel> //

template<>
struct model_traits<mv::OpModel> {
  typedef mv::OpModel model_t;
  typedef mv::Data::OpListIterator const_operation_iterator_t;
  typedef mv::Data::OpChildIterator const_child_operation_iterator_t;


  static const_operation_iterator_t begin_operations(model_t& cm) {
    return cm.opBegin();
  }

  static const_child_operation_iterator_t begin_child_operations(
      const_operation_iterator_t& itr) {
    return itr.leftmostChild();
  }

  static const_operation_iterator_t end_operations(model_t& model) {
    return model.opEnd();
  }

  static const_operation_iterator_t get_iterator(model_t& model,
      const std::string& name) {
    return model.getOp(name);
  }
}; // struct model_traits<mv::OpModel> //

// Forward declaration //
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

  Scheduled_Op(const Scheduled_Op& o) : op_(o.op_),
  schedule_time_(o.schedule_time_), schedule_end_time_(o.schedule_end_time_),
  cmx_address_start_(o.cmx_address_start_),
  cmx_address_end_(o.cmx_address_end_), op_type_(o.op_type_) {}

  const Scheduled_Op& operator=(const Scheduled_Op& o) {
    op_ = o.op_;
    schedule_time_ = o.schedule_time_;
    schedule_end_time_ = o.schedule_end_time_;
    cmx_address_start_ = o.cmx_address_start_;
    cmx_address_end_ = o.cmx_address_end_;
    op_type_ = o.op_type_;
    return *this;
  }

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

  operator size_t() const { return schedule_time_; }

  typedef mv::Op const * operation_t;
  operator operation_t() const { return op_; }

  void set_start_address(size_t start_address) {
    cmx_address_start_ = start_address;
  }

  void set_end_address(size_t end_address) {
    cmx_address_end_ = end_address;
  }

  void invalidate_address() {
    cmx_address_start_= std::numeric_limits<size_t>::max();
    cmx_address_end_= std::numeric_limits<size_t>::min();
  }


  mv::Op const * op_;
  size_t schedule_time_;
  size_t schedule_end_time_;
  size_t cmx_address_start_;
  size_t cmx_address_end_;
  op_type_e op_type_;
}; // struct Scheduled_Op //

template<typename T> class DDR_Address_Generator;
} // namespace lp_scheduler //

namespace scheduler {

template<typename Model=mv::OpModel>
class Operation_Dag {
  template<typename T> friend class lp_scheduler::DDR_Address_Generator;

  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef Model model_t;
    typedef model_traits<model_t> mtraits;
    typedef typename mtraits::const_operation_iterator_t op_itr_t;
    typedef typename mtraits::const_child_operation_iterator_t child_op_itr_t;
    
    typedef mv::scheduler::CMX_Concatenation cmx_concat_algo_t;
    typedef typename cmx_concat_algo_t::control_edge_t
        cmx_concat_control_edge_t;
    typedef typename cmx_concat_algo_t::concat_subgraph_t cmx_concat_subgraph_t;

    typedef mv::scheduler::Pipelining_Transform pipeline_algo_t;
    typedef typename pipeline_algo_t::control_edge_t pipeline_control_edge_t;
    typedef typename pipeline_algo_t::pipeline_subgraph_t pipeline_subgraph_t;

    typedef mv::scheduler::Pipeline_Chains pipeline_chain_algo_t;
    typedef typename pipeline_chain_algo_t::chain_subgraph_t chain_subgraph_t;
    typedef typename pipeline_chain_algo_t::control_edge_t chain_control_edge_t;


    struct cmx_concat_subgraph_hash_t {
      std::size_t operator() (const cmx_concat_subgraph_t& a) const {
        return (size_t)(a.representative_dpu_);
      }
    }; // struct cmx_concat_subgraph_ordering_t //

    struct cmx_concat_subgraph_equality_t{
      bool operator() (const cmx_concat_subgraph_t& a,
            const cmx_concat_subgraph_t& b) const {
        return (a.representative_dpu_) == (b.representative_dpu_);
      }
    }; // struct cmx_concat_subgraph_ordering_t //


    typedef Operation_Dag dag_t;
    typedef mv::Op const * operation_t; // &(base_node_class::content_) //
    typedef operation_t const * const_op_ptr_t;

    typedef std::unordered_map<operation_t, cmx_concat_subgraph_t>
        cmx_concat_subgraphs_t;
    struct pseudo_edge_t {
      pseudo_edge_t(operation_t src, operation_t sink)
        : src_(src), sink_(sink) {}
      bool operator<(const pseudo_edge_t& o) const {
        return (src_ == o.src_) ? (sink_ < o.sink_) : (src_ < o.src_);
      }
      operation_t src_;
      operation_t sink_;
    }; // struct pseudo_edge_t //
    typedef std::set<pseudo_edge_t> pseudo_edge_set_t;
    // use the operation name as the hash key to reduce any non-determinism with
    // the virtual address //
    struct operation_hash_t {
      size_t operator()(const operation_t& op) const {
        return name_hash_(op->getName());
      }
      std::hash<std::string> name_hash_;
    }; // struct operation_hash_t //

    struct repackable_op_selector_t {
      repackable_op_selector_t() : dag_ptr_(NULL) {}
      repackable_op_selector_t(const dag_t& dag) : dag_ptr_(&dag) {}

      bool operator()(const operation_t& op) const {
        return dag_t::is_repackable_data_op(*dag_ptr_, op);
      }

      dag_t const *dag_ptr_;
    }; // struct repackable_op_selector_t //


    typedef std::list<const_op_ptr_t> op_ref_list_t;
    typedef op_ref_list_t::const_iterator const_ref_op_iterator_t;

    typedef std::unordered_set<operation_t, operation_hash_t> ops_set_t;
    typedef typename ops_set_t::const_iterator const_master_op_iterator_t;
    typedef typename ops_set_t::iterator master_op_iterator_t;

    typedef std::unordered_map<operation_t, op_ref_list_t> adjacency_map_t;
    typedef typename adjacency_map_t::const_iterator const_adj_map_iterator_t;
    typedef typename adjacency_map_t::iterator adj_map_iterator_t;

    typedef std::unordered_map<operation_t, unsigned> resource_utility_map_t;
    typedef typename resource_utility_map_t::const_iterator
        const_resource_map_iterator_t;
    typedef typename resource_utility_map_t::iterator resource_map_iterator_t;
    typedef std::unordered_map<operation_t, op_itr_t> op_to_iterator_lookup_t;

    typedef std::unordered_map<operation_t, size_t> in_degree_map_t;
    typedef typename in_degree_map_t::const_iterator const_in_degree_iterator_t;

    typedef std::unordered_map<std::string, operation_t> op_name_table_t;

    class const_operation_iterator_t {
      public:
        const_operation_iterator_t()
          : ref_itr_(), master_itr_(), is_ref_type_() {}

        const_operation_iterator_t(const const_ref_op_iterator_t& ref_itr)
          : ref_itr_(ref_itr), master_itr_(), is_ref_type_(true) {}

        const_operation_iterator_t(
            const const_master_op_iterator_t& master_itr) : ref_itr_(),
          master_itr_(master_itr), is_ref_type_(false) {}

        const operation_t& operator*() const {
          return is_ref_type_ ? *(*ref_itr_) : *master_itr_;
        }

        const_operation_iterator_t& operator++() {
          if (is_ref_type_) {
            ++ref_itr_;
          } else {
            ++master_itr_;
          }
          return *this;
        }

        bool operator==(const const_operation_iterator_t& o) const {
          if (is_ref_type_) {
            return ref_itr_ == o.ref_itr_;
          } else {
            return master_itr_ == o.master_itr_;
          }
        }

        bool operator!=(const const_operation_iterator_t& o) const {
          return !(*this == o);
        }

      private:

        const_ref_op_iterator_t ref_itr_;
        const_master_op_iterator_t master_itr_;
        bool is_ref_type_;
    }; // class const_operation_iterator_t //

    typedef size_t delay_t;
    typedef size_t resource_t;
    typedef mv::lp_scheduler::Producer_Consumer_Contiguous_Resource<resource_t,
            operation_t> resource_state_t;
    ////////////////////////////////////////////////////////////////////////////

    Operation_Dag(model_t& model) : adj_map_(), adj_map_rev_(),
      op_name_table_(), ops_(), resource_utility_map_(),
      op_to_iterator_lookup_(), in_degree_map_(), input_op_(),
      //NOTE: please add all implicit ops to this list -- except ImplicitConcat
      //All implicit ops are short-circuited during scheduling. ImplicitConcat
      //is left in place to reduce the edge blowup (quadratic) of dependencies.
      implicit_op_types_( {"Slice", "Crop", "Copy", "Align", "ImplicitReshape",
          "ImplicitPermute", "ImplicitOutput", "ImplicitUnion", "ImplicitInput",
          "ImplicitInputSlice", "ImplicitJoin"} ),
      cmx_concat_subgraphs_(), eltwise_rep_map_(), pseudo_edge_set_() {
        init_from_model(model);
    }

    Operation_Dag() : adj_map_(), adj_map_rev_(),
      op_name_table_(), ops_(), resource_utility_map_(),
      op_to_iterator_lookup_(), in_degree_map_(), input_op_(),
      //NOTE: please add all implicit ops to this list -- except ImplicitConcat
      //All implicit ops are short-circuited during scheduling. ImplicitConcat
      //is left in place to reduce the edge blowup (quadratic) of dependencies.
      implicit_op_types_( {"Slice", "Crop", "Copy", "Align", "ImplicitReshape",
          "ImplicitPermute", "ImplicitOutput", "ImplicitUnion", "ImplicitInput",
          "ImplicitInputSlice", "ImplicitJoin"} ),
        cmx_concat_subgraphs_(), eltwise_rep_map_(), pseudo_edge_set_() { }

    void reset(model_t& model) { init_from_model(model); }

    cmx_concat_subgraph_t const * does_this_dpu_have_cmx_concat_subgraph(
          operation_t op) const {
      auto itr = cmx_concat_subgraphs_.find(op);
      return (itr == cmx_concat_subgraphs_.end()) ? NULL : &(itr->second);
    }

    //TODO(vamsikku): please make mv::DataModel const correct we cannot even
    //scan through the edges in a read only fashion.
    void add_pseudo_edges_from_model(mv::OpModel& om) {
      mv::DataModel dm(om);
      pseudo_edge_set_.clear();
      for (auto eitr=dm.flowBegin(); eitr!=dm.flowEnd(); ++eitr) {
        if (eitr->hasAttr("pseudo_data_flow")) {
          pseudo_edge_set_.insert(
              pseudo_edge_t( &(*(eitr.source())), &(*(eitr.sink())) )
          );
        }
      }
    }

    bool is_pseudo_edge(operation_t src, operation_t sink) const {
      return (pseudo_edge_set_.find( pseudo_edge_t(src, sink) ) 
            != pseudo_edge_set_.end() );
    }

    template<typename ControlEdgeContainer>
    void reset_from_cmx_concat_control_edges(mv::OpModel& omodel,
          const ControlEdgeContainer cedge_container) {
      init_from_model(omodel);
      apply_control_edges(cedge_container.begin(), cedge_container.end());
      update_resource_utility_with_attribute(
          cmx_concat_algo_t::cmx_concat_attribute() );
    }

    template<typename ControlEdgeContainer>
    void reset_from_pipeline_control_edges(mv::OpModel& omodel,
          const ControlEdgeContainer cedge_container) {
      init_from_model(omodel);
      apply_control_edges(cedge_container.begin(), cedge_container.end());
      connect_all_non_unit_outdegree_dmas_to_input(omodel);
      update_resource_utility_with_attribute_all_ops(
          pipeline_algo_t::pipeline_resource_attribute() );
    }

    template<typename ControlEdgeContainer>
    void reset_from_chain_pipeline_control_edges(mv::OpModel& omodel,
          const ControlEdgeContainer cedge_container) {
      init_from_model(omodel);
      apply_control_edges(cedge_container.begin(), cedge_container.end());
    }


    void enable_cmx_concat_transforms(mv::OpModel& omodel,
          size_t cmx_size=917504UL) {
      std::list<cmx_concat_control_edge_t> cmx_control_edges;
      enable_cmx_concat_transforms(omodel, cmx_control_edges, cmx_size);
    }

    template<typename CmxConcatControlEdgeContainer>
    void enable_cmx_concat_transforms(mv::OpModel& omodel,
          CmxConcatControlEdgeContainer &cmx_concat_control_edges,
          size_t cmx_size=917504UL,
          std::string ignore_concat_list="") {

      static_assert(std::is_same<
          typename CmxConcatControlEdgeContainer::value_type,
            cmx_concat_control_edge_t>::value,
              "Invalid Control Edge container");

      // generate control edges for CMX concatenation //
      cmx_concat_algo_t cmx_concat_algo(omodel, ignore_concat_list);

      std::list<cmx_concat_subgraph_t> cmx_concat_subgraphs;
      cmx_concat_algo.transform_op_model(
          std::back_inserter(cmx_concat_control_edges), cmx_concat_subgraphs,
            cmx_size);

      cmx_concat_subgraphs_.clear();
      for (auto subg_itr=cmx_concat_subgraphs.begin();
            subg_itr!=cmx_concat_subgraphs.end(); ++subg_itr) {
        const cmx_concat_subgraph_t &subgraph = *subg_itr;
        cmx_concat_subgraphs_.insert(
            std::make_pair(subgraph.representative_dpu_, subgraph));
      }
      
      // reinit the DAG with fresh op model //
      reset_from_cmx_concat_control_edges(omodel, cmx_concat_control_edges);
    }

    bool is_inplace_op(operation_t op) const {
      return eltwise_rep_map_.find(op) != eltwise_rep_map_.end();
    }
    operation_t get_inplace_output_op(operation_t op) const {
      auto itr = eltwise_rep_map_.find(op);
      return itr->second;
    }

    template<typename T>
    bool does_this_op_generate_sparse_output(mv::OpModel& model, T op) {
      auto op_itr = model.getOp(op->getName());
      mv::Data::TensorIterator output_tensor_itr =
          op_itr->getOutputTensor(0UL);
      return output_tensor_itr->isSparse();
    }

    void enable_eltwise_transforms(mv::OpModel& model) {
      eltwise_rep_map_.clear();
      for (auto op_itr = model.opBegin(); op_itr != model.opEnd(); ++op_itr) {
        if (op_itr->hasAttr("inplace_eltwise_rep")) {
          std::string parent_name =
              op_itr->get<std::string>("inplace_eltwise_rep");
          mv::Data::OpListIterator parent_op_itr = model.getOp(parent_name);

          operation_t parent_op = &(*parent_op_itr);
          operation_t eltwise_op = &(*op_itr);

          eltwise_rep_map_.insert(std::make_pair(eltwise_op, parent_op));
        }
      }
    }

    operation_t does_this_dpu_have_eltwise_rep(operation_t op) const {
      auto itr = eltwise_rep_map_.find(op);
      return (itr == eltwise_rep_map_.end()) ? NULL : itr->second;
    }

  private:

    template<typename T>
    bool has_unit_out_degree(T op, mv::OpModel& model) {
      mv::Data::OpListIterator op_itr = model.getOp(op->getName());
      auto cop_itr = op_itr.leftmostChild();
      if (cop_itr == model.opEnd()) { return false; }
      ++cop_itr;
      return cop_itr == model.opEnd();
    }

  public:

    void enable_chain_pipeline_transforms(mv::OpModel& omodel) {
      pipeline_chain_algo_t algo(omodel);
      std::list<chain_subgraph_t> pipeline_subgraphs;
      std::list<chain_control_edge_t> chain_control_edges;

      algo.transform_op_model(std::back_inserter(chain_control_edges),
          pipeline_subgraphs);

      mv::GenerateDotFromModel(omodel, "OpModel",
            "pipeline_chain_transformed_model.dot");
      reset_from_chain_pipeline_control_edges(omodel, chain_control_edges);
    }

    template<typename PipeLineControlEdgeContainer>
    void enable_pipeline_transforms(mv::OpModel& omodel,
          PipeLineControlEdgeContainer &pipeline_control_edges,
            size_t cmx_size=917504UL) {

      static_assert(std::is_same<
          typename PipeLineControlEdgeContainer::value_type,
            pipeline_control_edge_t>::value, "Invalid Control Edge container");

      // generate control edges for CMX concatenation //
      pipeline_algo_t pipeline_algo(omodel);

      std::list<pipeline_subgraph_t> pipeline_subgraphs;
      pipeline_algo.transform_op_model(
          std::back_inserter(pipeline_control_edges), pipeline_subgraphs,
            cmx_size);
      mv::GenerateDotFromModel(omodel, "OpModel",
            "pipeline_transformed_model.dot");

      reset_from_pipeline_control_edges(omodel, pipeline_control_edges);
    }


    template<typename OpTypeIterator>
    void set_implicit_op_types(OpTypeIterator begin, OpTypeIterator end) {
      static_assert(std::is_same<typename OpTypeIterator::value_type,
          std::string>::value, "Invalid OpTypeIterator");
      implicit_op_types_.clear();
      for (; begin != end; ++begin) { implicit_op_types_.insert(*begin); }
    }

    const_operation_iterator_t begin_parent_nodes(const operation_t& op) const {
      typename adjacency_map_t::const_iterator itr = adj_map_rev_.find(op);

      return (itr == adj_map_rev_.end()) ?
        const_operation_iterator_t( ops_.end() ) :
        const_operation_iterator_t( (itr->second).begin() );
    }

    const_operation_iterator_t end_parent_nodes(const operation_t& op) const {
      typename adjacency_map_t::const_iterator itr = adj_map_rev_.find(op);

      return (itr == adj_map_rev_.end()) ?
        const_operation_iterator_t( ops_.end() ) :
        const_operation_iterator_t( (itr->second).end() );
    }

    const_operation_iterator_t begin_nodes() const {
      return const_operation_iterator_t( ops_.begin() );
    }
    const_operation_iterator_t end_nodes() const {
      return const_operation_iterator_t( ops_.end() );
    }

    bool is_valid_op(const operation_t& op) const {
      typename adjacency_map_t::const_iterator itr = adj_map_.find(op);
      return !(itr == adj_map_.end());
    }

    // operations on the outgoing edges //
    const_operation_iterator_t begin_nodes(const operation_t& op) const {
      typename adjacency_map_t::const_iterator itr = adj_map_.find(op);

      return (itr == adj_map_.end()) ?
        const_operation_iterator_t( ops_.end() ) :
        const_operation_iterator_t( (itr->second).begin() );
    }

    const_operation_iterator_t end_nodes(const operation_t& op) const {
      typename adjacency_map_t::const_iterator itr = adj_map_.find(op);

      return (itr == adj_map_.end()) ?
        const_operation_iterator_t( ops_.end() ) :
        const_operation_iterator_t( (itr->second).end() );
    }

    bool does_this_op_use_any_resources(const operation_t& op) const {
      return (resource_utility_map_.find(op) != resource_utility_map_.end());
    }

    resource_t resource_utility(model_t& model, const char* op_name) {
      typedef model_traits<model_t> mtraits;
      typedef typename mtraits::const_operation_iterator_t op_itr_t;

      op_itr_t itr = model.getOp(op_name);
      return itr == mtraits::end_operations(model) ?
          resource_t() : resource_utility(&(*itr));
    }


    resource_t resource_utility(const operation_t& op) const {
      auto itr = resource_utility_map_.find(op);
      assert(itr != resource_utility_map_.end());
      return itr->second;
    }

    void set_resource_utility(const operation_t& op, resource_t new_utility) {
      auto itr = resource_utility_map_.find(op);
      assert(itr != resource_utility_map_.end());
      itr->second = new_utility;
    }

    bool op_has_unit_out_degree(const operation_t& op) const {
      const_operation_iterator_t citr = begin_nodes(op),
                                 citr_end = end_nodes(op);
      if (citr == citr_end) { return false; }
      ++citr;
      return (citr == citr_end);
    }

    bool op_has_zero_in_degree(const operation_t& op) const {
      const_operation_iterator_t citr = begin_parent_nodes(op),
                                 citr_end = end_parent_nodes(op);
      return (citr == citr_end);
    }

    bool op_has_input_as_only_parent(const operation_t& op) const {
      const_operation_iterator_t citr = begin_parent_nodes(op),
                                 citr_end = end_parent_nodes(op);

      if (citr == citr_end) { return false; }
      operation_t pop = *citr;
      if (!is_input_op(pop)) { return false; }
      ++citr;
      bool ret_value = (citr == citr_end);
      return ret_value;
    }

    // Precondition: out degree of op >= 1 //
    operation_t get_first_child_op(const operation_t& op) const {
      const_operation_iterator_t citr = begin_nodes(op);
      return *citr;
    }


    // Checks if there is a DMATask which relocates the output of this op to DDR
    // Precondition: all implicit ops ("Slice" or "Crop") must be short
    // circuited.
    bool is_output_of_this_compute_op_relocated(const operation_t& op) const {
      if (!is_dpu_op(op) || !op_has_unit_out_degree(op)) { return false; }

      // does this have out degree 1 and connected to a DMATask which
      // moves data from CMX2DDR //

      operation_t cop = get_first_child_op(op);
      return is_dma_op_moving_data_from_cmx_to_ddr(cop);
    }

    // Precondition: is_output_of_this_compute_op_relocated(op) = true //
    operation_t get_output_relocating_dma_op(const operation_t& op) const {
      assert(is_output_of_this_compute_op_relocated(op));
      return get_first_child_op(op);
    }

    ////////////////////////////////////////////////////////////////////////////
    static const char* operation_name(const operation_t& op) {
      return (op->getName()).c_str();
    }

    static const_operation_iterator_t operations_begin(const dag_t& in) {
      return in.begin_nodes();
    }
    static const_operation_iterator_t operations_end(const dag_t& in) {
      return in.end_nodes();
    }

    static const_operation_iterator_t outgoing_operations_begin(const dag_t& in,
        const operation_t& op) {
      return in.begin_nodes(op);
    }
    static const_operation_iterator_t outgoing_operations_end(const dag_t& in,
        const operation_t& op) {
      return in.end_nodes(op);
    }


    static const_operation_iterator_t incoming_operations_begin(const dag_t& in,
        const operation_t& op) {
      return in.begin_parent_nodes(op);
    }
    static const_operation_iterator_t incoming_operations_end(const dag_t& in,
        const operation_t& op) {
      return in.end_parent_nodes(op);
    }

    static bool is_repackable_data_op(const dag_t& dag, const operation_t& op) {
      return dag.is_dma_op(op) &&
          (!(dag.is_dma_op_moving_data_from_cmx_to_ddr(op))) &&
          dag.op_has_zero_in_degree(op);
    }

    static bool is_pseudo_input_edge(const dag_t& dag, const operation_t& src,
        const operation_t& sink) { return dag.is_pseudo_edge(src, sink); }

    static bool is_inplace_op(const dag_t& dag, const operation_t& op) {
      return dag.is_inplace_op(op);
    }
    static operation_t get_inplace_output_op(
        const dag_t& dag, const operation_t& op) {
      return dag.get_inplace_output_op(op);
    }

    static bool is_data_operation(const dag_t& dag, const operation_t& op) {
      if (op->hasAttr("pipeline_flow_control")) { return false;}
      else if (op->hasAttr("pipeline_data_start")) { return true;}

      bool ret_value = (dag.is_dma_op(op) &&
          !(dag.is_dma_op_moving_data_from_cmx_to_ddr(op)) &&
            dag.op_has_unit_out_degree(op));
      return ret_value;
    }
    static bool is_compute_operation(const dag_t& dag, const operation_t& op) {
      // an implicit op is a compute op which takes 0 resources //
      return !(is_data_operation(dag, op));
    }

    static bool is_empty_demand(const resource_t& demand) {
      return (demand == resource_t(0UL));
    }



    static void initialize_resource_upper_bound(const resource_t& upper_bound,
        resource_state_t& state) {
      state.initialize_resource_upper_bound(upper_bound);
    }

    static bool is_resource_available(const resource_t& demand,
          const resource_state_t& state) {
      return state.is_resource_available(demand);
    }

    static bool schedule_operation(const operation_t& op,
        const resource_t& demand, resource_state_t& state,
        const_operation_iterator_t op_begin,
        const_operation_iterator_t op_end) {

        return (op->getOpType() == "Input") ?
          state.assign_resources(op, demand, op_end, op_end) :
          state.assign_resources(op, demand, op_begin, op_end);
    }

    static bool unschedule_operation(const operation_t& op,
        resource_state_t& state) {
      return state.unassign_resources(op);
    }

    static resource_t resource_utility(const dag_t& in, const operation_t& op) {
      return in.resource_utility(op);
    }

    static delay_t delay(const dag_t&, const operation_t&) {
      return delay_t(1UL);
    }

    //NOTE: if you want to specialize for different scheduled_op_t types
    //then add a function call extract<T>(in) and specialize it. //
    template<typename T>
    static size_t scheduled_op_time(const T& in) { return (size_t) in; }
    template<typename T>
    static operation_t scheduled_op(const T& in) { return (operation_t) in; }

    static size_t output_tensor_size(const operation_t& op) {
      if (!op->outputSlots()) { return 0UL; }

      mv::Op *op_ptr = const_cast<mv::Op *>(op);
      mv::Data::TensorIterator out_itr = op_ptr->getOutputTensor(0UL);
      mv::Tensor::MemoryLocation location =
          out_itr->get<mv::Tensor::MemoryLocation>("Location");

      return (location == mv::Tensor::MemoryLocation::NNCMX) ?
        out_itr->getClusterSize() : out_itr->computeTotalSize();
    }

    typedef mv::lp_scheduler::Scheduled_Op scheduled_op_t;
    struct scheduled_op_hash_t {
      size_t operator()(const scheduled_op_t& o) const {
        return hash_(o.op_);
      }
      std::hash<operation_t> hash_;
    }; // struct scheduled_op_hash_t //
    typedef repackable_op_selector_t data_op_selector_t;
    typedef size_t schedule_time_t;
    typedef size_t unit_t;

    static void set_new_schedule_time(scheduled_op_t& op,
          const schedule_time_t& t) {
      op.schedule_time_ = t;
    }

    static schedule_time_t scheduled_time(const scheduled_op_t& op) {
      return op.schedule_time_;
    }
    static operation_t scheduled_operation(const scheduled_op_t& sched_op) {
      return sched_op.op_;
    }

    static bool is_valid_scheduled_op(const scheduled_op_t& op) {
      return op.has_valid_address();
    }

    ////////////////////////////////////////////////////////////////////////////

    op_itr_t get_op_iterator(operation_t op) const {
      typename op_to_iterator_lookup_t::const_iterator itr =
          op_to_iterator_lookup_.find(op);

      assert(itr != op_to_iterator_lookup_.end());
      return itr->second;
    }
    size_t operation_in_degree(operation_t op) const {
      const_in_degree_iterator_t itr = in_degree_map_.find(op);
      return (itr == in_degree_map_.end()) ? 0UL : itr->second;
    }

    bool is_input_op(operation_t op) const {
      return op->getOpType() == "Input";
    }

    operation_t get_input_op() const { return input_op_; }

    bool is_dma_op(operation_t op) const {
      return op->getOpType() == "DMATask";
    }

    bool is_dpu_op(operation_t op) const {
      return op->getOpType() == "DPUTask";
    }

    bool is_upa_op(operation_t op) const {
      return op->getOpType() == "UPATask";
    }

    bool has_edge_between_ops(operation_t a, operation_t b) const {
      const_operation_iterator_t citr = begin_nodes(a), citr_end = end_nodes(a);
      for (; citr != citr_end; ++citr) {
        if (*citr == b) { return true; }
      }
      return false;
    }


    template<typename BackInsertIterator>
    size_t find_all_ops_exceeding_resource_threshold(resource_t threshold,
        BackInsertIterator output) {
      // TODO(vamsikku): the space can be improved by maintaining this table
      // for current level and previous level.
      typedef std::unordered_map<operation_t, resource_t> op_size_table_t;
      op_size_table_t op_size_table;
      std::list<operation_t> bfs_list;
      operation_t curr_op;

      // add all zero in-degree nodes //
      for (const_operation_iterator_t citr=begin_nodes(), citr_end=end_nodes();
            citr != citr_end; ++citr) {
        curr_op = *citr;
        size_t in_degree;
        if (!(in_degree=operation_in_degree(curr_op))) {
          printfInfo("LpScheduler:",
              "zero-degree-node=%s\n", curr_op->getName().c_str());
          bfs_list.push_back(curr_op);
        } else {
          printfInfo("LpScheduler:", "non-zero-degree-node=%s in-degree=%lu\n",
              curr_op->getName().c_str(), in_degree);
        }
      }


      while (!bfs_list.empty()) {
        curr_op = bfs_list.front();

        bfs_list.pop_front();
        op_size_table_t::iterator itr = op_size_table.find(curr_op);

        resource_t curr_op_utility = resource_utility(curr_op);

        if (itr == op_size_table.end()) {
          // initialize it with its output size //
          itr = op_size_table.insert(
                std::make_pair(curr_op, resource_t(0UL))).first;
        }
        itr->second += curr_op_utility;

        // for all the out-going edges
        const_operation_iterator_t citr=begin_nodes(curr_op);
        const_operation_iterator_t citr_end=end_nodes(curr_op);
        for (; citr != citr_end; ++citr) {
          operation_t child_op = *citr;
          
          if (is_pseudo_edge(curr_op, child_op)) {
            // pseudo edge does not contribute to the resource utility of //
            // the child op//
            continue;
          }

          itr = op_size_table.find(child_op);
          if (itr == op_size_table.end()) {
            // initialize it with its output size //
            itr = op_size_table.insert(
                std::make_pair(child_op, resource_t(0UL))).first;
            // newly discovered node //
            bfs_list.push_back(child_op);
          }
          itr->second += curr_op_utility;
        }

      } // while (!bfs_list.empty()) //

      size_t ret_value = 0UL;
      for (op_size_table_t::const_iterator itr=op_size_table.begin();
            itr != op_size_table.end(); ++itr) {
        if (itr->second >= threshold) {
          output = std::make_pair(itr->first, itr->second);
          ++output;
          ++ret_value;
        }
      }
      return ret_value;
    }

    operation_t get_op_by_name(const char *name) const {
      op_name_table_t::const_iterator itr = op_name_table_.find(name);
      return (itr != op_name_table_.end()) ? itr->second : operation_t(NULL);
    }

    bool is_spilled_op(operation_t op) const {
      return does_opname_ends_with(op, "spilledWrite") ||
        does_opname_have_substring(op, "_spilledRead");
    }
    bool ops_of_same_category(operation_t op_a, operation_t op_b) const {
      if (op_a->getOpType() != op_b->getOpType()) { return false; }

      if (op_a->getOpType() == "DMATask") {
        return (op_a->get<mv::DmaDirection>("direction")) ==
            (op_b->get<mv::DmaDirection>("direction"));
      }
      return true;
    }
    bool does_opname_have_substring(operation_t op, const char *substr) const {
      const std::string& op_name = op->getName();
      return !(op_name.find(substr) == std::string::npos);
    }

    bool does_opname_ends_with(operation_t op, const char *suffix) const {
      /*checks if the name ends with _spilledWrite*/
      const char *name = (op->getName()).c_str();
      size_t name_len = strlen(name), suffix_len = strlen(suffix);
      if (name_len < suffix_len) { return false; }
      if (!suffix_len) {  return true; }

      const char *rev_name_ptr = &name[name_len - 1UL];
      const char *rev_suffix_ptr = &suffix[suffix_len - 1UL];
      for (size_t i=0; i<suffix_len; i++, --rev_name_ptr, --rev_suffix_ptr) {
        if (*rev_name_ptr != *rev_suffix_ptr) { return false; }
      }
      return true;
    }

    bool is_output_op(operation_t op) const {
      return (op->getOpType() == "Output");
    }

    bool is_implicit_op(operation_t op) const {
      return (op->getOpType() == "ImplicitConcat") ||
          (op->getOpType() == "Slice") || (op->getOpType() == "Crop") ||
          (op->getOpType() == "Copy") || (op->getOpType() == "Align") ||
          (op->getOpType() == "ImplicitOutput") ||
          (op->getOpType() == "ImplicitUnion") ||
          (op->getOpType() == "ImplicitInput") ||
          (op->getOpType() == "ImplicitInputSlice");
    }


    bool short_circuit_unit_indegree_unit_outdegree_op(operation_t& op) {
      operation_t parent_op, child_op;
      {
        adjacency_map_t::const_iterator adj_rev_itr = adj_map_rev_.find(op);
        adjacency_map_t::const_iterator adj_itr = adj_map_.find(op);

        if ((adj_rev_itr == adj_map_rev_.end()) ||
              (adj_itr == adj_map_.end()) ) {
          return false;
        }

        const op_ref_list_t& parent_list = adj_rev_itr->second;
        const op_ref_list_t& child_list = adj_itr->second;

        if ((parent_list.size() != 1UL) || (child_list.size() != 1UL)) {
          return false;
        }

        parent_op = *(parent_list.front());
        child_op = *(child_list.front());
      }

      printfInfo("OperationPrecedenceDAG:",
          "[Short-Circuiting: (%s) -> (%s) -> (%s)]\n",
          parent_op->getName().c_str(), op->getName().c_str(),
          child_op->getName().c_str());

      // remove this op from the DAG //
      remove_op_from_dag(op);

      // add an edge between parent_op and child_op //
      return add_directed_edge(parent_op, child_op);
    }


    bool short_circuit_implicit_op(operation_t& op) {
      operation_t parent_op, child_op;
      adjacency_map_t::const_iterator adj_rev_itr = adj_map_rev_.find(op);
      adjacency_map_t::const_iterator adj_itr = adj_map_.find(op);

      if ((adj_rev_itr == adj_map_rev_.end()) ||
            (adj_itr == adj_map_.end()) ) {
        return false;
      }

      const op_ref_list_t& parent_list = adj_rev_itr->second;
      const op_ref_list_t& child_list = adj_itr->second;

      for (const_ref_op_iterator_t pitr=parent_list.begin();
          pitr!=parent_list.end(); ++pitr) {
        parent_op = *(*pitr);
        for (const_ref_op_iterator_t citr=child_list.begin();
            citr!=child_list.end(); ++citr) {
          child_op = *(*citr);

          printfInfo("OperationPrecedenceDAG",
              "[Short-Circuiting: (%s) -> (%s) -> (%s)]\n",
              parent_op->getName().c_str(), op->getName().c_str(),
              child_op->getName().c_str());
          if (!add_directed_edge(parent_op, child_op)) { return false; }
        }
      }
      // remove this op from the DAG //
      remove_op_from_dag(op);
      return true;
    }

    bool short_circuit_all_unit_indegree_outdegree_ops_of_this_type(
        const std::string& op_type) {
      std::list<operation_t> remove_list;

      for (const_operation_iterator_t itr=begin_nodes(); itr!=end_nodes();
            ++itr) {
        const operation_t& op = *itr;
        if (op->getOpType() == op_type) {
          remove_list.push_back(op);
        }
      }

      for (auto oitr=remove_list.begin(); oitr!=remove_list.end(); ++oitr) {
        bool short_circuited = short_circuit_implicit_op(*oitr);
        //TODO(vamsikku): investigate on why short cirucuiting failures are
        //commented out.
        //if (!short_circuited) {
        //  throw std::string("[ImplicitOp-Short-Circuting]: failed");
        //}
      }
      return true;
    }

    struct implicit_op_color_functor_t {
      bool operator()(const dag_t& dag, const operation_t& op) const {
        return dag.is_implicit_op(op);
      }
    }; // struct implicit_op_color_functor_t //


    // Takes a operation precedence DAG with some colored ops (implicit)
    // and removes them from the DAG by adding more edges to retain operation
    // precedence invariant.
    void perform_implicit_op_color_closure() {
      typedef mv::lp_scheduler::Color_Connected_Vertices<dag_t>
          color_closure_t;

      color_closure_t color_closure_algo(*this);
      for (const_operation_iterator_t itr=begin_nodes(); itr!=end_nodes();
          ++itr) {

        // compute the color-closure of DMATask or DPUTask //
        operation_t pop = *itr;
        if (is_implicit_op(pop)) { continue; }

        std::list<operation_t> color_closure;
        color_closure_algo.compute_connected_vertices(pop,
            std::back_inserter(color_closure), implicit_op_color_functor_t() );

        printfInfo("LpScheduler:", "[ColorClosure(%s) : {",
            (pop->getName()).c_str());

        if (!color_closure.empty()) {
          for (auto citr=color_closure.begin(); citr!=color_closure.end();
                ++citr) {
            const operation_t& cop = *citr;
            add_directed_edge(pop, cop);
            printfInfo("LpScheduler:", " %s ", (cop->getName()).c_str());
          }
        }
        printfInfo("LpScheduler:", "}\n");

      } // foreach implicit op in the input DAG //


      for (const_operation_iterator_t itr=begin_nodes(); itr!=end_nodes();) {
        operation_t pop = *itr;
        if (is_implicit_op(pop)) {
          const_operation_iterator_t itr_next = itr;
          ++itr_next;
          printfInfo("OperationPrecedenceDAG:", "[Removed %s]\n",
              ((*itr)->getName()).c_str());
          remove_op_from_dag(*itr);
          itr = itr_next;
        } else {
          ++itr;
        }
      }
    }

    bool is_aligned_dma_op(model_t& model, const char* op_name) const {
      typedef model_traits<model_t> mtraits;
      typedef typename mtraits::const_operation_iterator_t op_itr_t;

      op_itr_t itr = model.getOp(op_name);
      return itr == mtraits::end_operations(model) ? false :
          is_aligned_dma_op(model, &(*itr));
    }


  private:

    template<typename model_t>
    bool is_aligned_dma_op(model_t& model, operation_t op) const {
      typedef model_traits<model_t> mtraits;
      typedef typename mtraits::const_child_operation_iterator_t cop_itr_t;

      if (is_dma_op_moving_data_from_cmx_to_ddr(op)) { return false; }

      typename mtraits::const_operation_iterator_t pop_itr =
        mtraits::get_iterator(model, op->getName());

      // out degree should be one //
      size_t out_degree = 0UL;
      for (cop_itr_t cop_itr=mtraits::begin_child_operations(pop_itr);
            (cop_itr != mtraits::end_operations(model)) && (out_degree <= 1UL);
            ++cop_itr, ++out_degree) { }

      if (out_degree != 1UL) { return false; }

      op_itr_t cop_itr = mtraits::begin_child_operations(pop_itr);


      return (cop_itr->getOpType() == "Align");
    }

    // Precondition: is_aligned_dma_op() //
    size_t get_aligned_dma_op_resource_utility(mv::OpModel& model,
          operation_t op) const {
      typedef model_traits<model_t> mtraits;
      typedef typename mtraits::const_operation_iterator_t op_itr_t;


      assert(is_aligned_dma_op(model, op));

      op_itr_t pop_itr = model.getOp(op->getName());
      op_itr_t cop_itr = mtraits::begin_child_operations(pop_itr);
      return output_tensor_size(&(*cop_itr));
    }

    //TODO(vamsikku): the control model is used for barrier schedululing
    //ideally we need to take a functor which decides if the scheduler can
    //ignore the operation.
    bool is_operation_ignored(operation_t op,
          mv::ControlModel&) const {
      const std::string& op_type = op->getOpType();
      return (op_type == "ConstantInt") || (op_type == "ConstantDataElement") ||
        (op_type == "ImplicitConcat") ||
        (implicit_op_types_.find(op_type) != implicit_op_types_.end());
    }



    bool is_operation_ignored(operation_t op, mv::OpModel&) const {
      const std::string& op_type = op->getOpType();
      return (op_type == "ConstantInt") || (op_type == "ConstantDataElement");
    }

    // For all the DMA ops moving data from DDR2CMX followed by an Align
    // implicit op use the output of Align op as the resource utility of this
    // DMA op.
    template<typename model_t>
    void update_resource_utility_for_aligned_dma_ops(model_t& model) {
      for (typename resource_utility_map_t::iterator
            ritr = resource_utility_map_.begin();
            ritr != resource_utility_map_.end(); ++ritr) {
        if (is_aligned_dma_op(model, ritr->first)) {
          ritr->second =
              get_aligned_dma_op_resource_utility(model, ritr->first);
        }
      }
    }

    template<typename model_t>
    void build_adj_tables(model_t& model) {
      for (op_itr_t itr = mtraits::begin_operations(model);
            itr != mtraits::end_operations(model); ++itr) {
        operation_t op = &(*itr);
        if (is_operation_ignored(op, model)) { continue; }
        if (is_input_op(op)) { input_op_ = op; }


        master_op_iterator_t pop_itr = ops_.find(op), cop_itr;

        if (pop_itr == ops_.end()) {
          pop_itr = (ops_.insert(op)).first;
          // op should have an unique name //
          const char * const op_name = op->getName().c_str();
          assert(op_name_table_.find(op_name) == op_name_table_.end());
          op_name_table_.insert(std::make_pair(op_name, op));
        }
        op_to_iterator_lookup_.insert(std::make_pair(op, itr));

        adj_map_iterator_t adj_itr = adj_map_.find(op);
        assert(adj_itr == adj_map_.end());

        // create a new adjacency map entry //
        adj_itr = (adj_map_.insert(std::make_pair(op, op_ref_list_t()))).first;

        // adjacency list of the ops //
        op_ref_list_t &adj_list = adj_itr->second;
        for (child_op_itr_t citr = itr.leftmostChild();
              citr != model.opEnd(); ++citr) {
          operation_t child_op = &(*citr);
          if (is_operation_ignored(child_op, model)) { continue; }

          cop_itr = ops_.find(child_op);
          if (cop_itr == ops_.end()) {
            cop_itr = (ops_.insert(child_op)).first;
            const char * const child_op_name = child_op->getName().c_str();
            assert(op_name_table_.find(child_op_name) == op_name_table_.end());
            op_name_table_.insert(std::make_pair(child_op_name, child_op));
          }

          if (in_degree_map_.find(child_op) == in_degree_map_.end()) {
            in_degree_map_[child_op] = 0UL;
          }
          in_degree_map_[child_op]++;

          adj_list.push_back( &(*cop_itr) );
          adj_map_rev_[child_op].push_back( &(*pop_itr) );
        }
      }
    }

    void clear_state() {
      adj_map_.clear();
      adj_map_rev_.clear();
      op_name_table_.clear();
      ops_.clear();
      resource_utility_map_.clear();
      op_to_iterator_lookup_.clear();
      in_degree_map_.clear();
      input_op_ = NULL;
    }


    void create_resource_utility_table_for_cmx_scheduling(mv::OpModel& model) {
      for (op_itr_t itr = mtraits::begin_operations(model);
            itr != mtraits::end_operations(model); ++itr) {
        operation_t op = &(*itr);
        resource_t resource_utility;

        if ( !does_the_op_run_on_hardware(op) ||
            is_dma_op_moving_data_from_cmx_to_ddr(op) ) {
          resource_utility = 0UL;
        } else {
          resource_utility = output_tensor_size(op);
        }
        // resource utility //
        resource_utility_map_.insert(std::make_pair(op, resource_utility ));
      }
    }

    void create_resource_utility_table_for_barrier_scheduling(
        mv::ControlModel& model) {
      for (op_itr_t itr = mtraits::begin_operations(model);
            itr != mtraits::end_operations(model); ++itr) {
        operation_t op = &(*itr);
        resource_t resource_utility = 0UL;
        if (does_the_op_run_on_hardware(op)) {
          resource_utility =
            mv::RuntimeModel::countProducerConsumerTasks(model, itr);
        }
        // resource utility //
        resource_utility_map_.insert(std::make_pair(op, resource_utility ));
      }
    }

    void connect_all_non_unit_outdegree_dmas_to_input(mv::OpModel& model) {

      // connect all non-unit outdegree DMAS to input //
      for (op_itr_t itr = mtraits::begin_operations(model);
            itr != mtraits::end_operations(model); ++itr) {
        operation_t op = &(*itr);
        
        if (is_operation_ignored(op, model)) { continue; }
        if (!is_dma_op(op)) { continue; }
        if (is_dma_op_moving_data_from_cmx_to_ddr(op)) {continue;}
        if (op_has_unit_out_degree(op)) { continue; }

        add_directed_edge_from_input(op);
      }

    }

      // short circuit implicit ops //
    void shorting_implicit_ops() {
      for (auto short_circuit_itr=implicit_op_types_.begin();
          short_circuit_itr!=implicit_op_types_.end(); ++short_circuit_itr) {
        short_circuit_all_unit_indegree_outdegree_ops_of_this_type(
            *short_circuit_itr);
      }
    }

    void init_from_model(mv::ControlModel& model) {
      clear_state();
      build_adj_tables(model);
      create_resource_utility_table_for_barrier_scheduling(model);
    }

    template<typename ControlEdgeIterator>
    void apply_control_edges(ControlEdgeIterator cbegin, 
        ControlEdgeIterator cend) {
      while (cbegin != cend) {
        operation_t src_op = (&(*((*cbegin).source_itr_)));
        operation_t snk_op = (&(*((*cbegin).sink_itr_)));
        add_directed_edge(src_op, snk_op);
        ++cbegin;
      }
    }

    void update_resource_utility_with_attribute(
        const std::string& attribute="cmx_concatable") {
      for (typename resource_utility_map_t::iterator
            ritr = resource_utility_map_.begin();
            ritr != resource_utility_map_.end(); ++ritr) {
        operation_t op = ritr->first;
        if (is_dpu_op(op) && op->hasAttr(attribute)) {
          ritr->second = op->get<size_t>(attribute);
        }
      }
    }

    void update_resource_utility_with_attribute_all_ops(
        const std::string& attribute="cmx_concatable") {
      for (typename resource_utility_map_t::iterator
            ritr = resource_utility_map_.begin();
            ritr != resource_utility_map_.end(); ++ritr) {
        operation_t op = ritr->first;
        if (op->hasAttr(attribute)) {
          ritr->second = op->get<size_t>(attribute);
        }
      }
    }

    void init_from_model(mv::OpModel& model) {
      clear_state();
      build_adj_tables(model);
      create_resource_utility_table_for_cmx_scheduling(model);

      // Transform OpModel for scheduling //
      shorting_implicit_ops();

      connect_all_non_unit_outdegree_dmas_to_input(model);

      update_resource_utility_for_aligned_dma_ops(model);

      // add pseudo edges //
      add_pseudo_edges_from_model(model);
    }

  public:

    // Removes the op from the DAG and removes all incoming and outgoing edges
    void remove_op_from_dag(operation_t op) {
      // STEP-0: Find from op_set_ //
      master_op_iterator_t op_itr = ops_.find(op);
      assert(op_itr != ops_.end());

      // STEP-2: Remove from the indegree map //
      in_degree_map_.erase(op);

      // STEP-3: Remove this op from the adj_map_ of parent //
      {
        adjacency_map_t::iterator parent_itr = adj_map_rev_.find(op);

        if (parent_itr != adj_map_rev_.end()) {
          op_ref_list_t& parent_list = parent_itr->second;

          for(op_ref_list_t::const_iterator parent=parent_list.begin();
                parent!=parent_list.end(); ++parent) { //foreach parent //

            // find the adjacency list of this parent and remove op from it //
            adjacency_map_t::iterator parent_adj_itr =
                adj_map_.find(*(*parent));
            assert(parent_adj_itr != adj_map_.end());

            // remove op from this parent adjacency list //
            (parent_adj_itr->second).remove(&(*op_itr));
          }
        }

      }

      // STEP-4: Remove this op from the adj_map_rev_ of all its children //
      {
        adjacency_map_t::iterator child_itr = adj_map_.find(op);

        if (child_itr != adj_map_.end()) {
          op_ref_list_t& child_list = child_itr->second;

          for(op_ref_list_t::const_iterator child=child_list.begin();
                child!=child_list.end(); ++child) { //foreach child//

            // find the rev-adjacency list of this child and remove op from it
            adjacency_map_t::iterator child_adj_itr =
                adj_map_rev_.find(*(*child));
            assert(child_adj_itr != adj_map_rev_.end());

            // remove op from this child rev-adjacency list //
            (child_adj_itr->second).remove(&(*op_itr));

            // STEP-4.1: reduce the in-degree of the child //
            in_degree_map_t::iterator indegree_itr =
                in_degree_map_.find(*(*child));
            assert(indegree_itr != in_degree_map_.end());
            assert(indegree_itr->second >= 1);

            --(indegree_itr->second);
            if (!(indegree_itr->second)) {
              in_degree_map_.erase(indegree_itr);
            }

          }
        }
      }

      // STEP-1 //
      ops_.erase(op_itr);
      op_name_table_.erase(op->getName().c_str());
    }
  private:

    void clear_resource_model() { resource_utility_map_.clear(); }

  public:

    bool add_directed_edge(const std::string& src_op,
          const std::string& sink_op, mv::OpModel& model) {

      mv::Data::OpListIterator src_itr = model.getOp(src_op);
      mv::Data::OpListIterator sink_itr = model.getOp(sink_op);
      assert((src_itr != model.opEnd()) && (sink_itr != model.opEnd()));
      return add_directed_edge(&(*src_itr), &(*sink_itr));
    }

    bool add_directed_edge(operation_t source_op, operation_t sink_op) {

      master_op_iterator_t itr_source = ops_.find(source_op);
      master_op_iterator_t itr_sink = ops_.find(sink_op);

      if ((itr_source == ops_.end()) || (itr_sink == ops_.end())) {
        return false;
      }

      // add sink_op to adj_list of source_op //
      op_ref_list_t *child_list_ptr = NULL, *parent_list_ptr = NULL;
      {
        adjacency_map_t::iterator adj_itr = adj_map_.find(source_op);
        assert(adj_itr != adj_map_.end());

        op_ref_list_t& child_list = adj_itr->second;
        for (op_ref_list_t::const_iterator child=child_list.begin();
              child!=child_list.end(); ++child) {
          if (*child == &(*itr_sink)) { return false; }
        }
        child_list_ptr = &child_list;
      }

      // add source_op to rev_adj_list of sink_op //
      {
        adjacency_map_t::iterator adj_rev_itr = adj_map_rev_.find(sink_op);
        if (adj_rev_itr == adj_map_rev_.end()) {
          adj_rev_itr =
            adj_map_rev_.insert(std::make_pair(sink_op, op_ref_list_t())).first;
        }

        //assert(adj_rev_itr != adj_map_rev_.end());

        op_ref_list_t& parent_list = adj_rev_itr->second;
        for (op_ref_list_t::const_iterator parent=parent_list.begin();
              parent!=parent_list.end(); ++parent) {
          if (*parent == &(*itr_source)) { return false; }
        }
        parent_list_ptr = &parent_list;
      }

      child_list_ptr->push_back(&(*itr_sink));
      parent_list_ptr->push_back(&(*itr_source));

      // update the indegree of sink_op //
      in_degree_map_t::iterator in_degree_itr = in_degree_map_.find(sink_op);

      if (in_degree_itr == in_degree_map_.end()) {
        in_degree_map_.insert(std::make_pair(sink_op, 1UL));
      } else {
        in_degree_itr->second++;
      }
      return true;
    }

    void drop_all_pseudo_edges() {
      for (pseudo_edge_t edge : pseudo_edge_set_) {
        remove_directed_edge(edge.src_, edge.sink_);
      }
      pseudo_edge_set_.clear();
    }

    void drop_all_pseudo_edges_in_model(mv::OpModel& om) {
      mv::DataModel dm(om);
      std::list<mv::Data::FlowListIterator> edges_to_drop;
      for (mv::Data::FlowListIterator eitr=dm.flowBegin(); eitr!=dm.flowEnd();
            ++eitr) {
        if (eitr->hasAttr("pseudo_data_flow")) {
          edges_to_drop.push_back(eitr);
        }
      }

      for (mv::Data::FlowListIterator eitr : edges_to_drop) {
        om.undefineFlow(eitr);
      }
    }

    bool remove_directed_edge(operation_t source_op, operation_t sink_op) {
      master_op_iterator_t itr_source = ops_.find(source_op);
      master_op_iterator_t itr_sink = ops_.find(sink_op);

      if ((itr_source == ops_.end()) || (itr_sink == ops_.end())) {
        return false;
      }

      // remove sink_op from adj_list of source_op //
      op_ref_list_t *child_list_ptr = NULL, *parent_list_ptr = NULL;
      typename op_ref_list_t::iterator child_remove_iterator,
               parent_remove_iterator; 
      {
        adjacency_map_t::iterator adj_itr = adj_map_.find(source_op);
        if (adj_itr == adj_map_.end()) { return false; }

        op_ref_list_t& child_list = adj_itr->second;
        for (op_ref_list_t::iterator child=child_list.begin();
              child!=child_list.end(); ++child) {
          if (*child == &(*itr_sink)) {
            child_list_ptr = &child_list;
            child_remove_iterator = child;
            break;
          }
        }
      }

      // remove source_op from rev_adj_list of sink_op //
      {
        adjacency_map_t::iterator adj_rev_itr = adj_map_rev_.find(sink_op);

        if (adj_rev_itr == adj_map_rev_.end()) { return false; }

        op_ref_list_t& parent_list = adj_rev_itr->second;
        for (op_ref_list_t::iterator parent=parent_list.begin();
              parent!=parent_list.end(); ++parent) {
          if (*parent == &(*itr_source)) { 
            parent_remove_iterator = parent;
            parent_list_ptr = &parent_list;
          }
        }
      }

      bool ret_value = false;
      if (child_list_ptr && parent_list_ptr) {
        child_list_ptr->erase(child_remove_iterator);
        parent_list_ptr->erase(parent_remove_iterator);

        // update the indegree of sink_op //
        in_degree_map_t::iterator in_degree_itr = in_degree_map_.find(sink_op);

        assert(in_degree_itr != in_degree_map_.end());
        assert(in_degree_itr->second >= 1UL);

        in_degree_itr->second--;
        ret_value = true;
      }
      return ret_value;
    }


    bool add_directed_edge_from_input(operation_t sink_op) {
      operation_t source_op = get_input_op();
      return add_directed_edge(source_op, sink_op);
    }

    bool add_directed_edge_from_input_old(operation_t sink_op) {

      operation_t source_op = get_input_op();
      master_op_iterator_t itr_source = ops_.find(source_op);
      master_op_iterator_t itr_sink = ops_.find(sink_op);

      if ((itr_source == ops_.end()) || (itr_sink == ops_.end())) {
        return false;
      }

      // add sink_op to adj_list of source_op //
      op_ref_list_t *child_list_ptr = NULL;
      {
        adjacency_map_t::iterator adj_itr = adj_map_.find(source_op);
        assert(adj_itr != adj_map_.end());

        op_ref_list_t& child_list = adj_itr->second;
        for (op_ref_list_t::const_iterator child=child_list.begin();
              child!=child_list.end(); ++child) {
          if (*child == &(*itr_sink)) { return false; }
        }
        child_list_ptr = &child_list;
      }

      child_list_ptr->push_back(&(*itr_sink));

      // update the indegree of sink_op //
      in_degree_map_t::iterator in_degree_itr = in_degree_map_.find(sink_op);

      if (in_degree_itr == in_degree_map_.end()) {
        in_degree_map_.insert(std::make_pair(sink_op, 1UL));
      } else {
        in_degree_itr->second++;
      }
      return true;
    }

    bool is_op_reading_weights(const operation_t& op) const {
      mv::Op * op_ptr = const_cast<mv::Op *>(op);

      if (op->getOpType() != "DMATask") { return false; }

      mv::Data::TensorIterator input_tensor = op_ptr->getInputTensor(0UL);
      mv::Tensor::MemoryLocation location =
        input_tensor->get<mv::Tensor::MemoryLocation>("Location");

      return ( (location == std::string("BLOB")) ||
            (location == std::string("DEFAULT")) );
    }




    bool does_the_op_run_on_hardware(operation_t op) const {
      return (op->getOpType() == "DMATask") || (op->getOpType() == "DPUTask");
    }

    bool is_dma_op_moving_data_from_cmx_to_ddr(operation_t op) const {
      if ((op->getOpType()) != "DMATask") { return false; }

      mv::DmaDirectionEnum dma_dir = op->get<mv::DmaDirection>("direction");

      return (dma_dir == mv::DmaDirectionEnum::NNCMX2DDR) ||
          (dma_dir == mv::DmaDirectionEnum::UPACMX2DDR);
    }

    bool is_dma_op_moving_data_from_ddr_to_cmx(operation_t op) const {
      if ((op->getOpType()) != "DMATask") { return false; }

      mv::DmaDirectionEnum dma_dir = op->get<mv::DmaDirection>("direction");

      return (dma_dir == mv::DmaDirectionEnum::DDR2NNCMX) ||
          (dma_dir == mv::DmaDirectionEnum::DDR2UPACMX);
    }

  private:


    //TODO(vamsikku): consolidate ops_ and op_to_iterator_lookup_ tables. //
    adjacency_map_t adj_map_;
    adjacency_map_t adj_map_rev_;
    op_name_table_t op_name_table_;
    ops_set_t ops_;
    resource_utility_map_t resource_utility_map_;
    op_to_iterator_lookup_t op_to_iterator_lookup_;
    in_degree_map_t in_degree_map_;
    operation_t input_op_;
    std::unordered_set<std::string> implicit_op_types_;
    cmx_concat_subgraphs_t cmx_concat_subgraphs_;
    std::unordered_map<operation_t, operation_t> eltwise_rep_map_;
    pseudo_edge_set_t pseudo_edge_set_;
}; // class Operation_Dag //


typedef mv::lp_scheduler::Feasible_Schedule_Generator< Operation_Dag<> >
  mv_lp_scheduler_t;

typedef mv::lp_scheduler::Feasible_Schedule_Generator<
  Operation_Dag<mv::ControlModel> > mv_control_lp_scheduler_t;

} // namespace scheduler //
} // namespace mv //

namespace mv {
namespace lp_scheduler {

template<>
struct scheduler_traits< mv::scheduler::Operation_Dag<> >
  : public mv::scheduler::Operation_Dag<> {
  using mv::scheduler::Operation_Dag<>::Operation_Dag;
}; // scheduler_traits<mv::scheduler::Operation_Dag> //

}
}

namespace mv {
namespace lp_scheduler {

template<>
struct scheduler_traits< mv::scheduler::Operation_Dag<mv::ControlModel> >
  : public mv::scheduler::Operation_Dag<mv::ControlModel> {
  using mv::scheduler::Operation_Dag<mv::ControlModel>::Operation_Dag;
}; // scheduler_traits<mv::scheduler::Operation_Dag> //


typedef Feasible_Memory_Schedule_Generator< mv::scheduler::Operation_Dag<> >
  mv_memory_scheduler_with_spilling_t;

} // namespace lp_scheduler //
} // namespace mv //


#endif
