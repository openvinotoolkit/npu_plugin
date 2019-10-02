#ifndef FEASIBLE_SCHEDULER_HPP
#define FEASIBLE_SCHEDULER_HPP
#include <algorithm>
#include <list>
#include <unordered_map>
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
  static delay_t delay(const operation_t&);
  //////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////
  // Resource model //
  typedef size_t resource_t;
  typedef int resource_state_t;
  // Resource utility = r : V->{1,2\ldots k} // 
  static resource_t resource_utility(const operation_t&);
  //////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////
  // Resource update model:
  // 
  // Invariant: takes an operation and state of the resources and returns true
  // if this operation can be scheduled //
  static bool is_operation_schedulable(const operation_t&,
        const resource_state_t&);
  // Precondition: is_operation_schedulable(op) = true. 
  // Invariant: makes an update in the resource usage of using the operations//
  static bool schedule_operation(const operation_t&, resource_state_t&);
  // Precondition: updates the resource state by removing the operation form 
  // the schedule //
  static bool unschedule_operation(const operation_t&, resource_state_t&);
  //////////////////////////////////////////////////////////////////////////////
}; // struct scheduler_traits //


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
    heap_element_t(const_op_ptr_t *op=NULL, schedule_time_t t=0UL) :
      op_(op), time_(t) {}

    schedule_time_t time_;
    const operation_t *op_;
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
  
  //////////////////////////////////////////////////////////////////////////////


  public:

  Feasible_Schedule_Generator(const dag_t& in=dag_t())
    : heap_(), current_time_(0), candidates_(), resource_state_(),
      heap_ordering_(), schedulable_op_(), in_degree_(), input_(in) { init(); }

  void operator++() { next_schedulable_operation(); }
  bool operator==(const Feasible_Schedule_Generator& o) const {
    return reached_end() && o.reached_end();
  }
  size_t current_time() const { return current_time_; }

  const_schedulable_ops_iterator_t begin_candidates() const {
    return candidates_.begin();
  }

  const_schedulable_ops_iterator_t end_candidates() const {
    return candidates_.end();
  }





  private:

  bool reached_end() const { return candidates_.empty() && heap_.empty(); }

  bool init() {
    in_degree_.clear();

    const_operation_iterator_t itr = traits::operations_begin(input_);
    const_operation_iterator_t itr_end = traits::operations_end(input_);

    // compute the in-degree of every node //
    while (itr != itr_end) {
      const_operation_iterator_t jtr = traits::outgoing_operations_begin(
          input_, *itr);
      const_operation_iterator_t jtr_end = traits::outgoing_operations_end(
            input_, *itr);

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
    itr = traits::operations_begin(input_);
    itr_end = traits::operations_end(input_);

    while (itr != itr_end) {
      const_op_ptr_t op_ptr = &(*itr);
      if (in_degree_.find(op_ptr) == in_degree_.end()) {
        candidates_.push_back(op_ptr);
      }
      ++itr;
    }

    if (candidates_.empty()) {
      fprintf(stderr, "Feasible_Schedule_Generator: no operations with ZERO"
            " in-degree means there must be a cycle in the input");
      return false;
    }
    return true;
  }

  // Precondition: reached_end() is false //
  bool next_schedulable_operation() {
    schedulable_op_ = NULL;
    do {
      schedulable_ops_iterator_t op_itr = find_schedulable_op();
      if (is_valid_op(op_itr)) {
        const operation_t &op = *op_itr;
        // found a schedulable operation //
        schedule_time_t op_end_time = current_time_ + traits::delay(op);
        push_to_heap(heap_element_t(&op, op_end_time));
        candidates_.erase(op_itr);
        traits::schedule_operation(op, resource_state_);
        schedulable_op_ = &op;
      } else {
        // no-op found so move up the schedule time to the smallest completion
        // time among the active operations. //
        assert(!heap_.empty());
        heap_element_t top_elem = pop_from_heap();
        const operation_t &op = *(top_elem.op_);

        assert(current_time_ <= top_elem.time_);
        current_time_ = top_elem.time_;
        // since operation is now complete update the schedule //
        traits::unschedule_operation(op, resource_state_);
      }
    } while(!schedulable_op_ && !reached_end());

    return schedulable_op_ != NULL;
  }

  schedulable_ops_iterator_t find_schedulable_op() {
    const_schedulable_ops_iterator_t itr=candidates_.end(), itr_end; 
    for (itr = candidates_.begin(), itr_end = candidates_.end();
          ( (itr != itr_end) &&
            !traits::is_operation_schedulable(*itr, resource_state_)); ++itr) {}
    return itr;
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
  }

 
  //////////////////////////////////////////////////////////////////////////////
  schedule_heap_t heap_;
  schedule_time_t current_time_;
  schedulable_ops_t candidates_;
  resource_state_t resource_state_;
  min_heap_ordering_t heap_ordering_;
  const_op_ptr_t schedulable_op_;
  operation_in_degree_t in_degree_;
  const dag_t &input_;
  //////////////////////////////////////////////////////////////////////////////

}; // class Feasible_Schedule_Generator //


} // namespace lp_scheduler //

} // namespace mv //

#endif
