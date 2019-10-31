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

    class const_operation_iterator_t {
      public:

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
    };


    typedef size_t delay_t;
    typedef size_t resource_t;
    typedef mv::lp_scheduler::Producer_Consumer_Contiguous_Resource<resource_t,
            operation_t> resource_state_t;
    ////////////////////////////////////////////////////////////////////////////

    Operation_Dag(model_t& model) : adj_map_(), ops_(), resource_utility_map_() 
      { init_from_model(model); }

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

  private:

    bool is_operation_ignored(operation_t op) const {
      const std::string& op_type = op->getOpType();
      return !( (op_type == "DMATask") || (op_type == "DPUTask")  ||
                (op_type == "Input") || (op_type == "Output") );
    }

    void init_from_model(model_t& model) {
      adj_map_.clear();
      ops_.clear();
      op_to_iterator_lookup_.clear();


      for (op_itr_t itr = mtraits::begin_operations(model);
            itr != mtraits::end_operations(model); ++itr) {
        operation_t op = &(*itr);
        if (is_operation_ignored(op)) { continue; }

        master_op_iterator_t op_itr = ops_.find(op);

        if (op_itr == ops_.end()) {
          ops_.insert(op);
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

          op_itr = ops_.find(child_op);
          if (op_itr == ops_.end()) {
            op_itr = (ops_.insert(child_op)).first;
          }

          adj_list.push_back( &(*op_itr) );
        }

        resource_t resource_utility; 

        if (((op->getOpType() == "DMATask") && 
             (op->get<mv::DmaDirection>("direction")
                == mv::DmaDirectionEnum::CMX2DDR)) ||
            ( (op->getOpType() == "Input") || (op->getOpType() == "Output")) ) {
          resource_utility = 0UL;
        } else {
          resource_utility = op->getOutputSize();
        }

        // resource utility //
        resource_utility_map_.insert(std::make_pair(op, 
              std::max( size_t(1UL), resource_utility) ));
      }
    }

    adjacency_map_t adj_map_;
    ops_set_t ops_;
    resource_utility_map_t resource_utility_map_;
    //TODO(vamsikku): consolidate ops_ and op_to_iterator_lookup_ tables. //
    op_to_iterator_lookup_t op_to_iterator_lookup_;
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

}
}

#endif
