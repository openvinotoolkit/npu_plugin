#ifndef OPERATION_PRECEDENCE_DAG_HPP
#define OPERATION_PRECEDENCE_DAG_HPP

#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/iterator/control_context.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "scheduler/feasible_scheduler.hpp"
#include <unordered_map>

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
}; // struct model traits //


template<>
struct model_traits<mv::ControlModel> {
  typedef mv::ControlModel model_t;
  typedef mv::Control::OpListIterator const_operation_iterator_t;
  typedef mv::Control::OpChildIterator const_child_operation_iterator_t;

  //TODO(vamsikku): reference to model must be const here //
  static const_operation_iterator_t begin_operations(model_t& cm) {
    return cm.getFirst();
  }

  static const_child_operation_iterator_t begin_child_operations(
      const_operation_iterator_t& op) {
    return op.leftmostChild();
  }

  static const_operation_iterator_t end_operations(model_t& model) {
    return model.opEnd();
  }
}; // struct model_traits<mv::ControlModel> //

template<>
struct model_traits<mv::OpModel> {
  typedef mv::OpModel model_t;
  typedef mv::Data::OpListIterator const_operation_iterator_t;
  typedef mv::Data::OpChildIterator const_child_operation_iterator_t;


  static const_operation_iterator_t begin_operations(model_t& cm) {
    return cm.getInput();
  }

  static const_child_operation_iterator_t begin_child_operations(
      const_operation_iterator_t& itr) {
    return itr.leftmostChild();
  }

  static const_operation_iterator_t end_operations(model_t& model) {
    return model.opEnd();
  }
}; // struct model_traits<mv::OpModel> //


namespace scheduler {

template<typename Model=mv::OpModel>
class Operation_Dag {
  public:

    ////////////////////////////////////////////////////////////////////////////
    typedef Model model_t;
    typedef model_traits<model_t> mtraits;
    typedef typename mtraits::const_operation_iterator_t op_itr_t;
    typedef typename mtraits::const_child_operation_iterator_t child_op_itr_t;

    typedef Operation_Dag dag_t;
    typedef mv::Op const * operation_t; // &(base_node_class::content_) //
    typedef operation_t const * const_op_ptr_t;
    typedef std::hash<operation_t> operation_hash_t;


    typedef std::list<const_op_ptr_t> op_ref_list_t;
    typedef op_ref_list_t::const_iterator const_ref_op_iterator_t;

    typedef std::unordered_set<operation_t> ops_set_t;
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
      op_to_iterator_lookup_(), in_degree_map_(), input_op_() {
        init_from_model(model);
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

    resource_t resource_utility(const operation_t& op) const {
      auto itr = resource_utility_map_.find(op);
      assert(itr != resource_utility_map_.end());
      return itr->second; 
    }

    ////////////////////////////////////////////////////////////////////////////

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


    static bool is_data_operation(const dag_t& dag, const operation_t& op) {
      return dag.is_dma_op(op) &&
          !(dag.is_dma_op_moving_data_from_cmx_to_ddr(op));
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
          printf("zero-degree-node=%s\n", curr_op->getName().c_str());
          bfs_list.push_back(curr_op);
        } else {
          printf("non-zero-degree-node=%s in-degree=%lu\n",
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

  private:

    bool is_operation_ignored(operation_t op) const {
      const std::string& op_type = op->getOpType();
      return (op_type == "ConstantInt") || (op_type == "ConstantDataElement");
    }

    void init_from_model(model_t& model) {
      adj_map_.clear();
      ops_.clear();
      op_to_iterator_lookup_.clear();
      in_degree_map_.clear();
      input_op_ = NULL;


      for (op_itr_t itr = mtraits::begin_operations(model);
            itr != mtraits::end_operations(model); ++itr) {
        operation_t op = &(*itr);
        if (is_operation_ignored(op)) { continue; }
        if (is_input_op(op)) { input_op_ = op; }

        master_op_iterator_t pop_itr = ops_.find(op), cop_itr;

        if (pop_itr == ops_.end()) {
          pop_itr = (ops_.insert(op)).first;
          // op should have an unique name //
          const char * const op_name = op->getName().c_str();
          op_name_table_t::iterator nitr =
              op_name_table_.find(op->getName().c_str());
          assert(nitr == op_name_table_.end());
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
          if (is_operation_ignored(child_op)) { continue; }

          cop_itr = ops_.find(child_op);
          if (cop_itr == ops_.end()) {
            cop_itr = (ops_.insert(child_op)).first;
            const char * const child_op_name = child_op->getName().c_str();
            op_name_table_t::iterator nitr =
                op_name_table_.find(child_op->getName().c_str());
            assert(nitr == op_name_table_.end());
            op_name_table_.insert(std::make_pair(child_op_name, child_op));
          }

          if (in_degree_map_.find(child_op) == in_degree_map_.end()) {
            in_degree_map_[child_op] = 0UL;
          }
          in_degree_map_[child_op]++;

          adj_list.push_back( &(*cop_itr) );
          adj_map_rev_[child_op].push_back( &(*pop_itr) );
        }

        resource_t resource_utility; 

        if ( !does_the_op_run_on_hardware(op) ||
            is_dma_op_moving_data_from_cmx_to_ddr(op) ) {
          resource_utility = 0UL;
        } else {
          resource_utility = op->getOutputSize();
        }

        // resource utility //
        resource_utility_map_.insert(std::make_pair(op, resource_utility ));
      }
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


    //TODO(vamsikku): consolidate ops_ and op_to_iterator_lookup_ tables. //
    adjacency_map_t adj_map_;
    adjacency_map_t adj_map_rev_;
    op_name_table_t op_name_table_;
    ops_set_t ops_;
    resource_utility_map_t resource_utility_map_;
    op_to_iterator_lookup_t op_to_iterator_lookup_;
    in_degree_map_t in_degree_map_;
    operation_t input_op_;
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

} // namespace mv //
} // namespace lp_scheduler //



#endif
