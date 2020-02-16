#ifndef SCHEDULER_UNIT_TEST_UTILS_HPP
#define SCHEDULER_UNIT_TEST_UTILS_HPP

#include <iostream>

#include "scheduler/feasible_scheduler.hpp"

namespace mv_unit_tests {

struct interval_t {
  interval_t(int b, int e, const std::string& id="")
    : beg_(b), end_(e), id_(id) {}

  interval_t() : beg_(), end_(), id_() {}

  bool operator==(const interval_t& o) const {
    return (beg_ == o.beg_) && (end_ == o.end_) && (id_ == o.id_);
  }

  void print() const {
    std::cout << "[ " << beg_ << " " << end_ << " " << id_ << "]" << std::endl;
  }

  int beg_;
  int end_;
  std::string id_;
}; // struct interval_t //

} // namespace mv_unit_tests //

namespace mv {
namespace lp_scheduler {

template<>
struct interval_traits<mv_unit_tests::interval_t> {
  typedef int unit_t;
  typedef mv_unit_tests::interval_t interval_t;



  static unit_t interval_begin(const interval_t& interval) {
    return interval.beg_;
  }

  static unit_t interval_end(const interval_t& interval) {
    return interval.end_;
  }

  static void set_interval(interval_t& interval,
        const unit_t& beg, const unit_t& end) {
    interval.beg_ = beg; 
    interval.end_ = end;
  }

}; // struct interval_traits //

} // namespace lp_scheduler //
} // namespace mv //


namespace scheduler_unit_tests {


// Simple DAG for unit testing the core algorithm //
class Operation_Dag {
  public:
    typedef Operation_Dag dag_t;
    typedef std::string operation_t;
    typedef std::hash<std::string> operation_hash_t;
    typedef std::vector<operation_t> adjacency_list_t;
    typedef typename adjacency_list_t::const_iterator const_adj_list_iterator_t;
    typedef std::unordered_map<operation_t, adjacency_list_t> adjacency_map_t;
    typedef typename adjacency_map_t::const_iterator const_adj_map_iterator_t;

    // resource cost model //
    typedef size_t resource_t;
    typedef std::unordered_map<operation_t, resource_t> resource_cost_model_t;
    typedef std::unordered_set<operation_t> data_op_set_t;
    // delay cost model //
    typedef size_t delay_t;
    typedef std::unordered_map<operation_t, delay_t> delay_cost_model_t;

    class const_operation_iterator_t {
      public:
        const_operation_iterator_t() : adj_map_ptr_(NULL),
          itr_begin_(), itr_end_(), itr_adj_begin_(), itr_adj_end_(),
          iterate_edges_(false) {}

        const_operation_iterator_t(const const_operation_iterator_t& o)
          : adj_map_ptr_(), itr_begin_(), itr_end_(), itr_adj_begin_(),
            itr_adj_end_(), iterate_edges_(false) {
            adj_map_ptr_ = o.adj_map_ptr_;
            itr_begin_ = o.itr_begin_;
            itr_end_ = o.itr_end_;
            itr_adj_begin_ = o.itr_adj_begin_;
            itr_adj_end_ = o.itr_adj_end_;
            iterate_edges_ = o.iterate_edges_;
         }

        const_operation_iterator_t(const adjacency_map_t& adj_map,
            const_adj_map_iterator_t begin, const_adj_map_iterator_t end,
            bool iterate_edges=false) : adj_map_ptr_(&adj_map), itr_begin_(begin),
          itr_end_(end), itr_adj_begin_(), itr_adj_end_(),
          iterate_edges_(iterate_edges)
        {

          if (iterate_edges_ && (itr_begin_ != itr_end_)) {
            itr_adj_begin_ = (itr_begin_->second).cbegin();
            itr_adj_end_ = (itr_begin_->second).cend();
            if (itr_adj_begin_ == itr_adj_end_) {
              move_to_next_edge();
            }
          }
        }

        // only invalid iterators are equivalent //
        bool operator==(const const_operation_iterator_t& o) const {
          return !is_valid() && !o.is_valid();
        }

        bool operator!=(const const_operation_iterator_t& o) const {
          return !(*this == o);
        }

        const const_operation_iterator_t& operator++() {
          move_to_next_vertex_or_edge();
          return *this;
        }

        const const_operation_iterator_t& operator=(
            const const_operation_iterator_t& o) {
          if (this != &o) {
            adj_map_ptr_ = o.adj_map_ptr_;
            itr_begin_ = o.itr_begin_;
            itr_end_ = o.itr_end_;
            itr_adj_begin_ = o.itr_adj_begin_;
            itr_adj_end_ = o.itr_adj_end_;
            iterate_edges_ = o.iterate_edges_;
          }
          return *this;
        }

        const operation_t& operator*() const {
          const_adj_map_iterator_t op_itr = itr_begin_;
          if (iterate_edges_) {
            op_itr = adj_map_ptr_->find(*itr_adj_begin_);
            assert(op_itr != adj_map_ptr_->end());
          }
          return op_itr->first;
        }


      private:

        // Precondition: is_valid() //
        bool move_to_next_vertex_or_edge() {
          return iterate_edges_ ? move_to_next_edge() :
              (++itr_begin_) == itr_end_;
        }

        bool move_to_next_edge() {

          if (itr_adj_begin_ != itr_adj_end_) {
            // move to edge if possible //
            ++itr_adj_begin_;
          }

          if (itr_adj_begin_ == itr_adj_end_) {
            ++itr_begin_; 
            // find a node with atleast one out going edge //
            while ( (itr_begin_ != itr_end_) &&
                    ( (itr_adj_begin_ = (itr_begin_->second).begin()) ==
                      (itr_adj_end_ = (itr_begin_->second).end()) ) ) {
              ++itr_begin_;
            }
          }

          return (itr_adj_begin_ != itr_adj_end_);
        }

        bool is_valid(void) const { return itr_begin_ != itr_end_; }

        adjacency_map_t const * adj_map_ptr_;
        const_adj_map_iterator_t itr_begin_;
        const_adj_map_iterator_t itr_end_;
        const_adj_list_iterator_t itr_adj_begin_;
        const_adj_list_iterator_t itr_adj_end_;
        bool iterate_edges_;
    }; // class const_operation_iterator_t //


    Operation_Dag(const adjacency_map_t& in) : adj_map_(in) {init();} 
    Operation_Dag() : adj_map_() {} 


    const_operation_iterator_t begin(const operation_t& op) const {
      adjacency_map_t::const_iterator itr = adj_map_.find(op), itr_next;
      if (itr == adj_map_.end()) { return const_operation_iterator_t(); }

      itr_next = itr;
      ++itr_next;
      return const_operation_iterator_t(adj_map_, itr, itr_next, true);
    }

    const_operation_iterator_t begin_in(const operation_t& op) const {
      adjacency_map_t::const_iterator itr = inv_adj_map_.find(op), itr_next;
      if (itr == adj_map_.end()) { return const_operation_iterator_t(); }

      itr_next = itr;
      ++itr_next;
      return const_operation_iterator_t(adj_map_, itr, itr_next, true);
    }

    const_operation_iterator_t end(const operation_t& op) const {
      (void) op;
      return const_operation_iterator_t();
    }

    const_operation_iterator_t begin() const { 
      return const_operation_iterator_t(adj_map_, adj_map_.begin(),
          adj_map_.end(), false);
    }

    const_operation_iterator_t begin_edges() const {
      return const_operation_iterator_t(adj_map_, adj_map_.begin(),
          adj_map_.end(), true);
    }

    const_operation_iterator_t end() const {
      return const_operation_iterator_t();
    }

    // Precondition: new delay model most have an entry for every op. //
    void reset_delay_model(const delay_cost_model_t& delay_model) {
      delay_cost_model_.clear();
      delay_cost_model_ = delay_model;
    }

    // Precondition: new delay model most have an entry for every op. //
    void reset_resource_model(const resource_cost_model_t& resource_model) {
      resource_cost_model_.clear();
      resource_cost_model_ = resource_model;
    }

    const operation_t& get_operation(const std::string& name) const {
      adjacency_map_t::const_iterator itr = adj_map_.find(name);
      assert(itr != adj_map_.end());
      return itr->first;
    }


    delay_t get_operation_delay(const operation_t& op) const {
      resource_cost_model_t::const_iterator itr = delay_cost_model_.find(op);
      assert(itr != delay_cost_model_.end());
      return itr->second;
    }

    resource_t get_operation_resources(const operation_t& op) const {
      resource_cost_model_t::const_iterator itr = resource_cost_model_.find(op);
      assert(itr != resource_cost_model_.end());
      return itr->second;
    }

    size_t size() const { return adj_map_.size(); }

    bool is_data_op(const operation_t& op) const {
      return !(data_op_set_.find(op) == data_op_set_.end());
    }

    void reset_data_op_set(const data_op_set_t& in) { data_op_set_ = in; }

    ////////////////////////////////////////////////////////////////////////////
    // scheduler_traits //

    static const char * operation_name(const operation_t& op) {
      return op.c_str();
    }
    static const_operation_iterator_t operations_begin(const dag_t& g) {
      return g.begin();
    }

    static const_operation_iterator_t operations_end(const dag_t& g) {
      return g.end();
    }

    static const_operation_iterator_t outgoing_operations_begin(const dag_t& g,
        const operation_t& op) { return g.begin(op); }

    static const_operation_iterator_t outgoing_operations_end(const dag_t& g,
        const operation_t& op) { return g.end(op); }

    static const_operation_iterator_t incoming_operations_begin(const dag_t& g,
        const operation_t& op) { return g.begin_in(op); }

    static const_operation_iterator_t incoming_operations_end(const dag_t& g,
        const operation_t& op) { return g.end(op); }

    static delay_t delay(const dag_t& dag, const operation_t& op) {
      return dag.get_operation_delay(op);
    }

    // TODO(vamsikku) : change this to specific values of spilled delay //
    static delay_t spilled_read_delay(const dag_t& , const operation_t& ) {
      return delay_t(1);
    }

    static delay_t spilled_write_delay(const dag_t& , const operation_t& ){
      return delay_t(1);
    }

    static resource_t resource_utility(const dag_t& dag, const operation_t& op)
    {
      return dag.get_operation_resources(op);
    }

    static bool is_data_operation(const dag_t& dag, const operation_t& op) {
      return dag.is_data_op(op);
    }

    static bool is_compute_operation(const dag_t& dag, const operation_t& op) {
      return !dag.is_data_op(op);
    }

    ////////////////////////////////////////////////////////////////////////////

  private:

    void init() {
      reset_to_uniform_delay_model();
      reset_to_uniform_resource_model();
      init_inv_adj_map();
    }

    // Precondition: adj_map_ must be constructed //
    void init_inv_adj_map() {
      inv_adj_map_.clear();
      const_adj_map_iterator_t itr = adj_map_.begin(), itr_end = adj_map_.end();

      for (; itr != itr_end; ++itr) {
        const_adj_list_iterator_t inv_itr = (itr->second).begin(),
                                  inv_itr_end = (itr->second).end();
        for (; inv_itr != inv_itr_end; ++inv_itr) {
          inv_adj_map_[ *inv_itr ].push_back( itr->first);
        }
      }
    }

    void reset_to_uniform_delay_model(delay_t d=1) {
      delay_cost_model_.clear();

      const_operation_iterator_t itr=begin(), itr_end=end();

      while (itr != itr_end) {
        delay_cost_model_[*itr] = delay_t(d);
        ++itr;
      }
    }

    void reset_to_uniform_resource_model(resource_t r=1) {
      resource_cost_model_.clear();

      const_operation_iterator_t itr=begin(), itr_end=end();

      while (itr != itr_end) {
        resource_cost_model_[*itr] = delay_t(r);
        ++itr;
      }
    }

    adjacency_map_t adj_map_;
    adjacency_map_t inv_adj_map_; // all incoming edges of a node //
    delay_cost_model_t delay_cost_model_;
    resource_cost_model_t resource_cost_model_;
    data_op_set_t data_op_set_;
}; // class Operation_Dag //

// Using the Cumulative_Resource_State //
typedef mv::lp_scheduler::Cumulative_Resource_State<
  Operation_Dag::resource_t, Operation_Dag::operation_t> resource_state_t;

} // namespace scheduler_unit_tests //

namespace mv {
namespace lp_scheduler {

template<>
struct scheduler_traits<scheduler_unit_tests::Operation_Dag>
  : public scheduler_unit_tests::Operation_Dag {
    ////////////////////////////////////////////////////////////////////////////
    // input graph model and delay model are used from Operation_Dag itself   //
    using scheduler_unit_tests::Operation_Dag::Operation_Dag;

    ////////////////////////////////////////////////////////////////////////////
    // define resource update model //
    typedef scheduler_unit_tests::resource_state_t resource_state_t;

    static void initialize_resource_upper_bound(const resource_t& upper_bound,
        resource_state_t& state) {
      state.initialize_resource_upper_bound(upper_bound);
    }

    static bool is_empty_demand(const resource_t& demand) {
      return (demand == resource_t(0UL));
    }

    static bool is_resource_available(const resource_t& demand,
          const resource_state_t& state) {
      return state.is_resource_available(demand);
    }

    static bool schedule_operation(const operation_t& op,
        const resource_t& demand, resource_state_t& state,
        const_operation_iterator_t, const_operation_iterator_t) {
      return state.assign_resources(op, demand);
    }

    static bool unschedule_operation(const operation_t& op,
        resource_state_t& state) {
      return state.unassign_resources(op);
    }

    template<typename T>
    static size_t scheduled_op_time(const T& in) { return in.time_; }

    template<typename T>
    static operation_t scheduled_op(const T& in) { return in.op_; }

}; // specialization for scheduler_unit_tests::dag_t //

} // namespace lp_scheduler //
} // namespace mv //



#endif
