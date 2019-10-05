#ifndef FEASIBLE_SCHEDULER_HPP
#define FEASIBLE_SCHEDULER_HPP
#include <algorithm>
#include "disjoint_interval_set.hpp"
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mv {

namespace lp_scheduler {

template<typename schedule_concept_t>
struct scheduler_traits {
  //////////////////////////////////////////////////////////////////////////////
  // Input: G(V,E) //
  typedef int dag_t;
  typedef int operation_t;
  typedef int const_operation_iterator_t;

  // iterator v \in V //
  static const_operation_iterator_t operations_begin(const dag_t&);
  static const_operation_iterator_t operations_end(const dag_t&);

  // Given v \in V , iterator over { u | (v, u) \in E } 
  static const_operation_iterator_t outgoing_operations_begin(const dag_t&,
        const operation_t&);
  static const_operation_iterator_t outgoing_operations_end(const dag_t&,
        const operation_t&);
  //////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////
  // Delay function = d : V -> N^+ //
  typedef size_t delay_t;
  static delay_t delay(const dag_t&, const operation_t&);
  //////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////
  // Resource model //
  typedef size_t resource_t;

  // Resource utility = r : V->{1,2\ldots k} // 
  static resource_t resource_utility(const dag_t&, const operation_t&);
  //////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////
  // Resource update model:
  // 
  // Invariant: takes an operation and state of the resources and returns true
  // if this operation can be scheduled //

  typedef int resource_state_t; // this should encode all resource constraints

  static void initialize_resource_upper_bound(const resource_t& upper_bound,
      resource_state_t&);

  static bool is_resource_availble(const resource_t& demand,
        const resource_state_t&);
  // Precondition: is_resource_available(demand) = true. 
  // Invariant: makes an update in the resource usage of using the operations//
  static bool schedule_operation(const operation_t&, resource_t demand,
      resource_state_t&);
  // Precondition: updates the resource state by removing the operation form 
  // the schedule //
  static bool unschedule_operation(const operation_t&, resource_state_t&);
  //////////////////////////////////////////////////////////////////////////////
}; // struct scheduler_traits //


//NOTE: the operations should outlive this object //
template<typename Unit, typename Key>
class Cumulative_Resource_State {
  public:
  typedef Unit unit_t;
  typedef Key key_t;
  typedef const key_t * const_key_ptr_t;
  typedef std::unordered_map<const_key_ptr_t, unit_t> lookup_t;

  Cumulative_Resource_State(const unit_t& bound=unit_t(0))
    : resources_in_use_(unit_t(0)), resource_bound_(bound),
      active_resources_() {}

  void initialize_resource_upper_bound(const unit_t& upper_bound) {
    resource_bound_ = upper_bound;
    resources_in_use_ = unit_t(0);
    active_resources_.clear();
  }

  // This assumes a serial execution //
  bool is_resource_available(const unit_t& demand) const {
    return (resources_in_use_ + demand) <= resource_bound_;
  }

  // This assumes a serial execution //
  bool assign_resources(const key_t& op, const unit_t& demand) {
    const_key_ptr_t key = &op;
    if (!is_resource_available(demand)) { return false; }

    if (active_resources_.find(key) != active_resources_.end()) {
      return false;
    }

    resources_in_use_ += demand;
    active_resources_[key] = demand;
    return true;
  }

  bool unassign_resources(const key_t& op) {
    const_key_ptr_t key = &op;
    typename lookup_t::iterator itr = active_resources_.find(key);

    if (itr == active_resources_.end()) { return false; }

    resources_in_use_ -= itr->second;
    active_resources_.erase(itr);
    return true;
  }

  private:
  unit_t resources_in_use_;
  unit_t resource_bound_;
  lookup_t active_resources_;
}; // class Cumulative_Resource_State //

// This models the resource state as disjoint set of integral intervals//
template<typename Unit, typename Key>
class Contiguous_Resource_State {
  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef Unit unit_t;
    typedef Key key_t;
    typedef const key_t * const_key_ptr_t;
    typedef Disjoint_Interval_Set<unit_t, const_key_ptr_t> active_resources_t;
    typedef typename active_resources_t::free_interval_iterator_t
        free_interval_iterator_t;

    // closed interval [begin_, end_] = { x | begin_ <= x <= end_ }
    struct interval_info_t {
      unit_t begin_; 
      unit_t end_;
      interval_info_t(unit_t ibeg, unit_t iend) : begin_(ibeg), end_(iend) {}
    }; // struct interval_info_t //

    typedef std::unordered_map<const_key_ptr_t, interval_info_t> lookup_t;
    ////////////////////////////////////////////////////////////////////////////

    Contiguous_Resource_State() : active_resources_(),
      location_begin_(unit_t(1)), location_end_(unit_t(1)), lookup_() {}

    Contiguous_Resource_State(unit_t upper_bound) : active_resources_(),
      location_begin_(unit_t(1)), location_end_(unit_t(1)), lookup_() {
        initialize_resource_upper_bound(upper_bound);
    }


    void initialize_resource_upper_bound(const unit_t& upper_bound,
        unit_t location_begin=unit_t(1)) {
      assert(upper_bound > 0);
      active_resources_.clear();
      location_begin_ = location_begin;
      location_end_ = location_begin_ + upper_bound - 1; // its an index //
      assert(location_begin_ > std::numeric_limits<unit_t>::min());
      assert(location_end_ < std::numeric_limits<unit_t>::max());
    }

    bool is_resource_available(const unit_t& demand) const {
      unit_t ibeg, iend;
      return find_interval_satisfying_demand(demand, ibeg, iend);
    }

    bool assign_resources(const key_t& op, const unit_t& demand) {
      if (!demand) { return true;}

      const_key_ptr_t op_ptr = &op;
      typename lookup_t::iterator litr = lookup_.find(op_ptr);

      if (litr != lookup_.end()) { return false; }

      // no resources assigned for this operation //

      unit_t ibeg, iend;
      if (!find_smallest_interval_satisfying_demand(demand, ibeg, iend) ||
          !active_resources_.insert(ibeg, ibeg+demand-1, op_ptr)) {
        return false;
      }
      // found a feasible resource //
      lookup_.insert(std::make_pair(op_ptr,
              interval_info_t(ibeg, ibeg+demand-1)));
      return true;
    }

    bool unassign_resources(const key_t& op) {
      typename lookup_t::iterator litr = lookup_.find(&op);
      if (litr == lookup_.end()) { return false; }
      const interval_info_t &interval = litr->second; 

      if (!active_resources_.erase(interval.begin_, interval.end_)) {
        return false;
      }
      lookup_.erase(litr);
      return true;
    }

  private:

    // TODO(vamsikku): the cost of this O(k) where k is the number of active
    // tasking using the contiguous memory.
    bool find_interval_satisfying_demand(const unit_t& demand,
        unit_t& interval_start, unit_t& interval_end) const {
      free_interval_iterator_t itr, itr_end;
      itr = active_resources_.begin_free_intervals();
      itr_end = active_resources_.end_free_intervals();

      for (;itr != itr_end; ++itr) {
        if (does_interval_satisfy_demand(itr, demand,
                interval_start, interval_end)) { return true; }
      }
      return false;
    }

    bool does_interval_satisfy_demand(free_interval_iterator_t itr,
        const unit_t& demand,
        unit_t& interval_start, unit_t& interval_end) const {
      // note this always produces open intervals of the form:
      // (a,b) = { x | a < x < b } //
      unit_t a = std::max(location_begin_-1, itr.interval_begin());
      unit_t b = std::min(location_end_+1, itr.interval_end());
      assert(b >= a);
      unit_t available_capacity = (b - a) - 1;
      // note for open intervals (i,i+1) the above value is zero //

      if (available_capacity < demand) { return false; }

      interval_start = a+1; interval_end = b-1;
      assert(interval_start <= interval_end);
      return true;
    }

    // TODO(vamsikku): the cost of this O(k) where k is the number of active
    // tasking using the contiguous memory.
    bool find_smallest_interval_satisfying_demand(const unit_t& demand,
        unit_t& output_interval_start, unit_t& output_interval_end) const {
      typename active_resources_t::free_interval_iterator_t itr, itr_end;

      itr = active_resources_.begin_free_intervals();
      itr_end = active_resources_.end_free_intervals();
      unit_t min_capacity = std::numeric_limits<unit_t>::max();
      unit_t curr_start, curr_end;

      output_interval_start = std::numeric_limits<unit_t>::max();
      output_interval_end = std::numeric_limits<unit_t>::min();

      for (;itr != itr_end; ++itr) {
        if (does_interval_satisfy_demand(itr, demand, curr_start, curr_end) &&
            (min_capacity > ((curr_end - curr_start)+1)) ) {
          output_interval_start = curr_start;
          output_interval_end = curr_end;
          min_capacity = (curr_end - curr_start) + 1; 
        }
      }
      return output_interval_start <= output_interval_end;
    }

    unit_t upper_bound() const { return (location_end_ - location_begin_) + 1; }

    active_resources_t active_resources_;
    unit_t location_begin_;
    unit_t location_end_;
    lookup_t lookup_;
}; // class Contiguous_Resource_State //

template<typename T, typename SchedulerTraits=scheduler_traits<T>,
         typename Allocator=std::allocator<T> >
class Feasible_Schedule_Generator {
  public:
  //////////////////////////////////////////////////////////////////////////////
  //TODO(vamsikku): rebind and propagate the Allocator to all the containers.
  typedef SchedulerTraits traits;
  typedef typename traits::dag_t dag_t;
  typedef typename traits::operation_t operation_t;
  typedef const operation_t * const_op_ptr_t;
  typedef typename traits::resource_t resource_t;
  typedef typename traits::resource_state_t resource_state_t;
  typedef typename traits::delay_t delay_t;
  typedef typename traits::const_operation_iterator_t
      const_operation_iterator_t;
  typedef size_t schedule_time_t;

  struct heap_element_t {
    heap_element_t(const_op_ptr_t op=NULL, schedule_time_t t=0UL) :
      op_(op), time_(t) {}

    const_op_ptr_t op_;
    schedule_time_t time_;
  }; // struct heap_element_t //

  struct min_heap_ordering_t {
    bool operator()(const heap_element_t& a, const heap_element_t& b) {
      return a.time_ > b.time_;
    }
  }; // struct min_heap_ordering_t //

  typedef std::vector<heap_element_t> schedule_heap_t;
  typedef std::list<const_op_ptr_t> schedulable_ops_t;
  typedef typename schedulable_ops_t::const_iterator
      const_schedulable_ops_iterator_t;
  typedef typename schedulable_ops_t::iterator schedulable_ops_iterator_t;
  typedef std::unordered_map<const_op_ptr_t, size_t> operation_in_degree_t;
  typedef std::unordered_set<const_op_ptr_t> processed_ops_t;
  
  //////////////////////////////////////////////////////////////////////////////


  public:

  Feasible_Schedule_Generator(const dag_t& in, const resource_t& resource_bound)
    : heap_(), current_time_(0), candidates_(), resource_state_(),
    heap_ordering_(), schedulable_op_(), in_degree_(), processed_ops_(),
    input_ptr_(&in) { init(resource_bound); }

  Feasible_Schedule_Generator() : heap_(), current_time_(0), candidates_(),
    resource_state_(), heap_ordering_(), schedulable_op_(), in_degree_(),
    processed_ops_(), input_ptr_() {} 

  void operator++() { next_schedulable_operation(); }
  // Precondition: reached_end() is false //
  const operation_t& operator*() const { return *schedulable_op_; }

  bool operator==(const Feasible_Schedule_Generator& o) const {
    return reached_end() && o.reached_end();
  }

  bool operator!=(const Feasible_Schedule_Generator& o) const {
    return !(*this == o);
  }

  size_t current_time() const { return current_time_; }

  const_schedulable_ops_iterator_t begin_candidates() const {
    return candidates_.begin();
  }

  const_schedulable_ops_iterator_t end_candidates() const {
    return candidates_.end();
  }



  private:

  bool reached_end() const {
    return candidates_.empty()  && heap_.empty();
  }

  bool init(const resource_t& upper_bound) {
    traits::initialize_resource_upper_bound(upper_bound, resource_state_);
    in_degree_.clear();
    processed_ops_.clear();

    const_operation_iterator_t itr = traits::operations_begin(*input_ptr_);
    const_operation_iterator_t itr_end = traits::operations_end(*input_ptr_);

    // compute the in-degree of every node //
    while (itr != itr_end) {
      const_operation_iterator_t jtr = traits::outgoing_operations_begin(
          *input_ptr_, *itr);
      const_operation_iterator_t jtr_end = traits::outgoing_operations_end(
            *input_ptr_, *itr);

      while (jtr != jtr_end) { // foreach outgoing edge of *itr //
        const_op_ptr_t op_ptr = &(*jtr);
        typename operation_in_degree_t::iterator deg_itr =
            in_degree_.find(op_ptr);
        if (deg_itr == in_degree_.end()) {
          deg_itr = (in_degree_.insert(std::make_pair(op_ptr, 0))).first;
        }
        deg_itr->second++;
        ++jtr;
      }
      ++itr;
    }

    // collect the ones with zero-in degree into candidates //
    candidates_.clear();
    itr = traits::operations_begin(*input_ptr_);
    itr_end = traits::operations_end(*input_ptr_);

    while (itr != itr_end) {
      const_op_ptr_t op_ptr = &(*itr);
      if (in_degree_.find(op_ptr) == in_degree_.end()) {
        add_to_candidate_set(op_ptr);
      }
      ++itr;
    }

    if (candidates_.empty()) {
      fprintf(stderr, "Feasible_Schedule_Generator: no operations with ZERO"
            " in-degree means there must be a cycle in the input");
      return false;
    }
    return next_schedulable_operation();
  }

  // Precondition: reached_end() is false //
  bool next_schedulable_operation() {
    schedulable_op_ = NULL;
    do {
      schedulable_ops_iterator_t op_itr = find_schedulable_op();
      if (is_valid_op(op_itr)) {
        // found a schedulable operation //
        const operation_t &op = *(*op_itr);
        delay_t op_delay = traits::delay(*input_ptr_, op);
        resource_t op_resources = traits::resource_utility(*input_ptr_, op);

        schedule_time_t op_end_time = current_time_ + op_delay;
        push_to_heap( heap_element_t(&op, op_end_time) );
        candidates_.erase(op_itr);
        traits::schedule_operation(op, op_resources, resource_state_);
        schedulable_op_ = &op;
      } else if (!heap_.empty()) {
        // no-op found so move up the schedule time to the smallest completion
        // time among the active operations. //
        heap_element_t top_elem = pop_from_heap();
        const operation_t &op = *(top_elem.op_);

        assert(current_time_ <= top_elem.time_);
        current_time_ = top_elem.time_;
        // since operation is now complete update the schedule //
        traits::unschedule_operation(op, resource_state_);
        // since op has completed add all out-going ops to candidates //
        add_outgoing_operations_to_candidate_list(op);
      } else {
        // schedule is not feasible //
        candidates_.clear();
        break;
      }
    } while(!schedulable_op_ && !reached_end());

    return schedulable_op_ != NULL;
  }

  schedulable_ops_iterator_t find_schedulable_op() {
    schedulable_ops_iterator_t itr=candidates_.end();

    for (itr=candidates_.begin(); itr != candidates_.end(); ++itr){
      resource_t demand = traits::resource_utility(*input_ptr_, *(*itr));
      if (traits::is_resource_available(demand, resource_state_)) { break; }
    }

    return itr;
  }

  void add_outgoing_operations_to_candidate_list(const operation_t& op) {
    const_operation_iterator_t itr=input_ptr_->begin(op);
    const_operation_iterator_t itr_end=input_ptr_->end(op);
    for (;itr != itr_end; ++itr) {
      // decrement the in-degree of &(*itr) and only add to candidate set
      // if the indegree is zero. This means this op is ready to be scheduled.
      const_op_ptr_t dep_op_ptr = &(*itr);
      typename operation_in_degree_t::iterator deg_itr =
          in_degree_.find(dep_op_ptr);
      assert((deg_itr != in_degree_.end()) && (deg_itr->second > 0) );

      if (deg_itr->second == 1) {
        add_to_candidate_set( dep_op_ptr );
        in_degree_.erase(deg_itr);
      } else {
        --(deg_itr->second);
      }
    }
  }

  bool is_valid_op(schedulable_ops_iterator_t itr) const {
    return !(itr == candidates_.end());
  }

  const heap_element_t* top_element() const {
    return heap_.empty() ? NULL : heap_.back();
  }

  // Heap operations //
  void push_to_heap(const heap_element_t& elem) {
    heap_.push_back(elem);
    std::push_heap(heap_.begin(), heap_.end(), heap_ordering_);
  }

  // Precondition: !heap_.empty() //
  heap_element_t pop_from_heap() {
    std::pop_heap(heap_.begin(), heap_.end(), heap_ordering_);
    heap_element_t elem = heap_.back();
    heap_.pop_back();
    return elem;
  }

  void add_to_candidate_set(const_op_ptr_t op_ptr) {
    if (processed_ops_.find(op_ptr) != processed_ops_.end()) { return; }
    candidates_.push_back(op_ptr);
    processed_ops_.insert(op_ptr);
  }

 
  //////////////////////////////////////////////////////////////////////////////
  schedule_heap_t heap_;
  schedule_time_t current_time_;
  schedulable_ops_t candidates_;
  resource_state_t resource_state_;
  min_heap_ordering_t heap_ordering_;
  const_op_ptr_t schedulable_op_;
  operation_in_degree_t in_degree_;
  processed_ops_t processed_ops_;
  const dag_t *input_ptr_;
  //////////////////////////////////////////////////////////////////////////////

}; // class Feasible_Schedule_Generator //


} // namespace lp_scheduler //

} // namespace mv //

#endif
