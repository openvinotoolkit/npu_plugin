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
  // Invariant: &(*itr) should remain same constant irrespective of the iterator
  typedef int const_operation_iterator_t;

  // iterator v \in V //
  static const_operation_iterator_t operations_begin(const dag_t&);
  static const_operation_iterator_t operations_end(const dag_t&);

  // Data operations and compute operations //
  // data operations have no producers and just feed data to compute operations.
  static bool is_data_operation(const dag_t&, const operation_t&);
  static bool is_compute_operation(const dag_t&, const operation_t&);

  // Given v \in V , iterator over { u | (v, u) \in E } 
  static const_operation_iterator_t outgoing_operations_begin(const dag_t&,
        const operation_t&);
  static const_operation_iterator_t outgoing_operations_end(const dag_t&,
        const operation_t&);

  // Given v \in V , iterator over { u | (u, v) \in E } 
  static const_operation_iterator_t incoming_operations_begin(const dag_t&,
      const operation_t&);
  static const_operation_iterator_t incoming_operations_end(const dag_t&,
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

  static bool is_empty_demand(const resource_t& demand);
  static bool is_resource_available(const resource_t& demand,
        const resource_state_t&);

  template<typename DemandIterator> 
  static bool are_resources_available(
      DemandIterator ditr, DemandIterator ditr_end, const resource_state_t&);

  // Precondition: is_resource_available(demand) = true. 
  // Invariant: makes an update in the resource usage of using the operations//
  static bool schedule_operation(const operation_t&, resource_t demand,
      resource_state_t&);

  static bool schedule_operation(const operation_t& op, resource_t demand,
      resource_state_t& rstate,
      const_operation_iterator_t, const_operation_iterator_t) {
    // default schedule_operation ignores outgoing operations. //
    return schedule_operation(op, demand, rstate);
  }

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
  typedef std::unordered_map<key_t, unit_t> lookup_t;

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
    if (!is_resource_available(demand)) { return false; }

    if (active_resources_.find(op) != active_resources_.end()) {
      return false;
    }

    resources_in_use_ += demand;
    active_resources_[op] = demand;
    return true;
  }

  bool unassign_resources(const key_t& op) {
    typename lookup_t::iterator itr = active_resources_.find(op);

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
// 
// TODO(vkundeti): enforce hashable requirement for the Key //
template<typename Unit, typename Key>
class Contiguous_Resource_State {
  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef Unit unit_t;
    typedef Key key_t;
    typedef Disjoint_Interval_Set<unit_t, key_t> active_resources_t;
    typedef typename active_resources_t::free_interval_iterator_t
        free_interval_iterator_t;

    // closed interval [begin_, end_] = { x | begin_ <= x <= end_ }
    struct interval_info_t {
      void invalidate() { 
        begin_ = std::numeric_limits<unit_t>::max();
        end_ = std::numeric_limits<unit_t>::min();
      }

      size_t length() const {
        assert(begin_ <= end_);
        return (end_ - begin_ + 1);
      }

      interval_info_t() : begin_(), end_() { invalidate(); }
      interval_info_t(unit_t ibeg, unit_t iend) : begin_(ibeg), end_(iend) {}

      bool operator==(const interval_info_t& o) const {
        return (begin_ == o.begin_) && (end_ == o.end_);
      }

      unit_t begin_; 
      unit_t end_;
    }; // struct interval_info_t //

    struct no_op_back_insert_iterator_t {
      void operator++() {}
      void operator=(const interval_info_t&) {} 
    }; // struct no_op_back_insert_iterator_t //

    struct interval_length_ordering_t {
      bool operator()(const interval_info_t& a,
          const interval_info_t& b) const {
        return a.length() > b.length();
      }
    }; // struct interval_length_ordering_t //
    struct demand_ordering_t {
      bool operator()(const unit_t& a, const unit_t& b) const { return a >b; }
    }; // struct demand_ordering_t //

    typedef std::unordered_map<key_t, interval_info_t> lookup_t;
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


    // Given the current resource state checks if the demands can be
    // simultaneously (all at once) satisfied. It also returns the packing into
    // back insert iterator (of type interval_info_t ) //
    template<typename DemandIterator, typename OutputIterator>
    bool pack_demands_into_free_bins(DemandIterator ditr_in,
        DemandIterator ditr_in_end, OutputIterator output) const {

      typedef typename std::vector<interval_info_t> bins_t;
      typedef typename std::vector<unit_t> demands_t;

      //TODO(vkundeti): this currently uses a greedy algorithm to make
      // a decision about the packing. The algorithm is greedy on the size
      // of the free bins and packs as much demand as it can into the chosen
      // free bin and moves to the next bin. However its possible that this
      // can miss a valid packing. To solve it correctly we need to solve a 
      // 1D bin-packing problem.

      free_interval_iterator_t fitr = active_resources_.begin_free_intervals();
      free_interval_iterator_t fitr_end =
          active_resources_.end_free_intervals();
      if (fitr == fitr_end) { return false; }

      // sort the free bins based on their length //
      bins_t free_bins;
      for (; fitr != fitr_end; ++fitr) {
        // NOTE: the disjoint interval set is on open interval:
        // (-\infty, +\infty) and will return free intervals which
        // may not be in the range [location_begin_, location_end_] 
        unit_t a = std::max(location_begin_-1, fitr.interval_begin());
        unit_t b = std::min(location_end_+1, fitr.interval_end());

        if ((b-a) > 1) {
          // bin has capacity of at least one //
          free_bins.emplace_back( interval_info_t(a+1, b-1) );
        }
      }
      std::sort(free_bins.begin(), free_bins.end(),
            interval_length_ordering_t());

      demands_t demands;
      for (; ditr_in != ditr_in_end; ++ditr_in) {
        unit_t demand = *ditr_in;
        if (demand > 0) {
          demands.emplace_back(*ditr_in);
        }
      }
      std::sort(demands.begin(), demands.end(), demand_ordering_t());


      // we should have atleast one bin at this time //
      typename demands_t::const_iterator ditr = demands.begin();
      typename demands_t::const_iterator ditr_end = demands.end();
      typename bins_t::const_iterator bitr = free_bins.begin();
      typename bins_t::const_iterator bitr_end = free_bins.end();
      size_t remaining_space_in_curr_bin = (*bitr).length();
      bool is_fresh_bin = true;

      // In each iterator either we move to next demand or 
      // we move to next bin or we exit. Hence this loop terminates //
      while ( (ditr != ditr_end) && (bitr != bitr_end) ) {

        unit_t curr_demand = *ditr;
        if (curr_demand <=  remaining_space_in_curr_bin) {
          remaining_space_in_curr_bin -= curr_demand;

          {
            // compute the addresses of this demand //
            const interval_info_t& bin_info = *bitr;
            interval_info_t address_info;
            address_info.end_ =
                ((bin_info.end_) - remaining_space_in_curr_bin);
            address_info.begin_ = (address_info.end_ - curr_demand) + 1;
            output = address_info;
            ++output;
          }

          ++ditr; // move to next demand //
          if (is_fresh_bin) { is_fresh_bin = false; }
        } else if (!is_fresh_bin) {
          ++bitr; // move to next bin //
          is_fresh_bin = true;
          remaining_space_in_curr_bin =
              (bitr == bitr_end) ? 0UL : (*bitr).length();
        } else { // the demand does not fit in a fresh bin //
          break;
        }
      }

      return (ditr == ditr_end); // all bins packed //
    }

    template<typename DemandIterator>
    bool are_resources_available_simultaneously(
          DemandIterator ditr, DemandIterator ditr_end) const {
      return pack_demands_into_free_bins(ditr, ditr_end,
            no_op_back_insert_iterator_t());
    }

    /*
    // Precondition: are_resources_available_simultaneously //
    template<typename DemandIterator>
    bool assign_resources_simultaneously(
        const key_t& op, DemandIterator ditr, DemandIterator ditr_end) {

      typename lookup_t::iterator litr = lookup_.find(op);

      if (litr != lookup_.end()) { return false; }

      // no resources assigned for this operation //

      unit_t ibeg, iend;
      if (!find_smallest_interval_satisfying_demand(demand, ibeg, iend) ||
          !active_resources_.insert(ibeg, ibeg+demand-1, op)) {
        return false;
      }
      // found a feasible resource //
      lookup_.insert(std::make_pair(op, interval_info_t(ibeg, ibeg+demand-1)));
      return true;
    }
    */

    bool is_key_using_resources(const key_t& op) const {
      return lookup_.find(op) != lookup_.end();
    }

    bool assign_resources(const key_t& op, const unit_t& demand) {
      if (!demand) { return true;}

      typename lookup_t::iterator litr = lookup_.find(op);

      if (litr != lookup_.end()) { return false; }

      // no resources assigned for this operation //

      unit_t ibeg, iend;
      if (!find_smallest_interval_satisfying_demand(demand, ibeg, iend) ||
          !active_resources_.insert(ibeg, ibeg+demand-1, op)) {
        return false;
      }
      // found a feasible resource //
      lookup_.insert(std::make_pair(op, interval_info_t(ibeg, ibeg+demand-1)));
      return true;
    }

    bool unassign_resources(const key_t& op) {
      typename lookup_t::iterator litr = lookup_.find(op);
      if (litr == lookup_.end()) { return false; }
      const interval_info_t &interval = litr->second; 

      if (!active_resources_.erase(interval.begin_, interval.end_)) {
        return false;
      }
      lookup_.erase(litr);
      return true;
    }

    // Returns false if the operation is not using any resources //
    interval_info_t get_resource_usage_info(const key_t& op) const {
      typename lookup_t::const_iterator litr = lookup_.find(op);
      interval_info_t result;

      if (litr != lookup_.end()) {
        result = litr->second;
      }
      return result;
    }

  protected:

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

// This models a resource which can have consumers and its engaged until all the
// consumers are done consuming the resource. Both the producers and consumers
// are identified Key type objects.
// 
// TODO(vkundeti): enforce hashable requirement for the Key type.
template<typename Unit, typename Key>
class Producer_Consumer_Contiguous_Resource :
    public Contiguous_Resource_State<Unit, Key> {

  public:

    ////////////////////////////////////////////////////////////////////////////
    typedef Unit unit_t;
    typedef Key key_t;
    typedef Contiguous_Resource_State<unit_t, key_t> parent_t;
    typedef key_t consumer_t;
    typedef key_t producer_t;
    using parent_t::active_resources_t;
    using parent_t::active_resources_;
    typedef std::list<producer_t> producers_t;
    typedef std::unordered_map<consumer_t, producers_t> consumer_producer_map_t;
    typedef std::unordered_map<producer_t, size_t> producer_ref_count_t;
    typedef typename consumer_producer_map_t::const_iterator
        const_consumer_iterator_t;
    typedef typename consumer_producer_map_t::iterator consumer_iterator_t;
    typedef typename producer_ref_count_t::const_iterator
        const_ref_count_iterator_t;
    typedef typename producer_ref_count_t::iterator ref_count_iterator_t;
    ////////////////////////////////////////////////////////////////////////////

    Producer_Consumer_Contiguous_Resource() : parent_t(),
      consumer_producers_map_(),  producer_ref_count_() {}

    Producer_Consumer_Contiguous_Resource(unit_t upper_bound)
      : parent_t(upper_bound), consumer_producers_map_(),
        producer_ref_count_() {}

    // Assign resources to the op (producer) and also add a reference of this
    // producer to all consumers. Consumers for this producer are passed as an
    // iterator over the consumers.
    template<typename ConsumerIterator>
    bool assign_resources(const key_t& op, const unit_t& demand, 
        ConsumerIterator consumer_begin=ConsumerIterator(),
        ConsumerIterator consumer_end=ConsumerIterator()) {

      size_t consumer_count = 1UL;
      for (;consumer_begin != consumer_end;
            ++consumer_begin, ++consumer_count) {
        consumer_producers_map_[*consumer_begin].push_back(producer_t(op));
      }

      if (producer_ref_count_.find(op) != producer_ref_count_.end()) {
        return false;
      }

      producer_ref_count_[producer_t(op)] = consumer_count;
      // now update the interval tree data structure //
      parent_t::assign_resources(producer_t(op), demand);
      return true;
    }

    // Unassign resources for the op which may not actually unassign until all
    // its producers are done consuming it.
    bool unassign_resources(const key_t& op) {
      ref_count_iterator_t ref_itr, ref_itr_end=producer_ref_count_.end();

      ref_itr = producer_ref_count_.find(op);
      if ((ref_itr == producer_ref_count_.end()) || !(ref_itr->second)) {
        // ref count for entries in producer_ref_count_ has to atleast 1 //
        return false;
      }

      --(ref_itr->second);
      if (!(ref_itr->second)) {
        // no consumers for this op hence unassign its resources //
        producer_ref_count_.erase(ref_itr);
        parent_t::unassign_resources(op);
      } 

      // reduce the ref_count of all the producers //
      consumer_iterator_t citr = consumer_producers_map_.find(op);
      consumer_iterator_t citr_end = consumer_producers_map_.end();

      if (citr != citr_end) {
        const producers_t& producers = citr->second;
        typename producers_t::const_iterator pitr = producers.begin(),
                 pitr_next, pitr_end = producers.end();


        for(;pitr != pitr_end; ++pitr) {
          const producer_t& producer = *pitr;

          ref_itr = producer_ref_count_.find(producer); 
          if (ref_itr == ref_itr_end) { return false; }// Invariant violation //
          --(ref_itr->second);
          if (!(ref_itr->second)) {
            // no consumers for this op hence unassign its resources //
            producer_ref_count_.erase(ref_itr);
            parent_t::unassign_resources(producer);
          }
        }
        consumer_producers_map_.erase(citr); 
      }
      return true;
    }

  private:

    consumer_producer_map_t consumer_producers_map_;
    // ref_count is the number of consumers of this operation + 1 //
    producer_ref_count_t producer_ref_count_;
}; // class Producer_Consumer_Contiguous_Resource //



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
  typedef std::unordered_map<const_op_ptr_t, size_t> priority_map_t;
  typedef std::unordered_set<const_op_ptr_t> processed_ops_t;
  
  //////////////////////////////////////////////////////////////////////////////


  public:

  Feasible_Schedule_Generator(const dag_t& in, const resource_t& resource_bound)
    : heap_(), current_time_(0), candidates_(), resource_state_(),
    heap_ordering_(), schedulable_op_(), in_degree_(), processed_ops_(),
    input_ptr_(&in), priority_() { init(resource_bound); }

  Feasible_Schedule_Generator() : heap_(), current_time_(0), candidates_(),
    resource_state_(), heap_ordering_(), schedulable_op_(), in_degree_(),
    processed_ops_(), input_ptr_(), priority_() {} 

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

  const resource_state_t& resource_state() const { return resource_state_; }



  protected:

  bool reached_end() const {
    return candidates_.empty()  && heap_.empty();
  }

  bool init(const resource_t& upper_bound) {
    traits::initialize_resource_upper_bound(upper_bound, resource_state_);
    processed_ops_.clear();

    compute_op_indegree(in_degree_);

    // collect the ones with zero-in degree into candidates //
    candidates_.clear();
    const_operation_iterator_t itr = traits::operations_begin(*input_ptr_);
    const_operation_iterator_t itr_end = traits::operations_end(*input_ptr_);

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
    compute_operation_priorities();

    return next_schedulable_operation();
  }

  void compute_operation_priorities() {
    operation_in_degree_t in_degree;

    compute_op_indegree(in_degree);

    // assign topological sort level as priority to start with //

    std::list<const_op_ptr_t> zero_in_degree_nodes[2];
    priority_.clear();

    size_t curr_priority = 0;
    const_operation_iterator_t itr = traits::operations_begin(*input_ptr_);
    const_operation_iterator_t itr_end = traits::operations_end(*input_ptr_);

    while (itr != itr_end) {
      const_op_ptr_t op_ptr = &(*itr);
      if (in_degree.find(op_ptr) == in_degree_.end()) {
        zero_in_degree_nodes[curr_priority%2].push_back(op_ptr);
        priority_[op_ptr] = curr_priority;
      }
      ++itr;
    }

    while (!zero_in_degree_nodes[curr_priority%2].empty()) {
      // decrement the in-degree 
      for (auto zitr=zero_in_degree_nodes[curr_priority%2].begin();
            zitr != zero_in_degree_nodes[curr_priority%2].end(); ++zitr) {

        const_operation_iterator_t jtr = traits::outgoing_operations_begin(
            *input_ptr_, *(*zitr));
        const_operation_iterator_t jtr_end = traits::outgoing_operations_end(
            *input_ptr_, *(*zitr));

        while (jtr != jtr_end) {
          typename operation_in_degree_t::iterator deg_itr =
              in_degree.find(&(*jtr));
          assert((deg_itr != in_degree.end()) && (deg_itr->second > 0));
          (deg_itr->second)--;

          if (!(deg_itr->second)) {
            // in-degree of this node has become zero//
            priority_[deg_itr->first] = (curr_priority+1);
            zero_in_degree_nodes[(curr_priority+1)%2].push_back(deg_itr->first);
            in_degree.erase(deg_itr);
          }
          ++jtr;
        }
      }
      zero_in_degree_nodes[curr_priority%2].clear();
      ++curr_priority;
    }

    for (typename priority_map_t::iterator pitr=priority_.begin();
          pitr!=priority_.end(); ++pitr) {
      // set priority to max of all out going priorities //
      const_operation_iterator_t jtr = traits::outgoing_operations_begin(
          *input_ptr_, *(pitr->first));
      const_operation_iterator_t jtr_end = traits::outgoing_operations_end(
          *input_ptr_, *(pitr->first));

      if (!(pitr->second)) {
        size_t max=pitr->second;
        while (jtr != jtr_end) {
          max = std::max( priority_[ &(*jtr) ], max);
          ++jtr;
        }
        pitr->second = max;
      }
    }

  }

  void compute_op_indegree(operation_in_degree_t& in_degree) {
    in_degree.clear();
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
            in_degree.find(op_ptr);
        if (deg_itr == in_degree.end()) {
          deg_itr = (in_degree.insert(std::make_pair(op_ptr, 0))).first;
        }
        deg_itr->second++;
        ++jtr;
      }
      ++itr;
    }

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
        traits::schedule_operation(op, op_resources, resource_state_,
            traits::outgoing_operations_begin(*input_ptr_, op),
            traits::outgoing_operations_end(*input_ptr_, op));
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

    std::list<schedulable_ops_iterator_t> ready_list;

    for (itr=candidates_.begin(); itr != candidates_.end(); ++itr){
      resource_t demand = traits::resource_utility(*input_ptr_, *(*itr));
      if (traits::is_resource_available(demand, resource_state_)) {
        ready_list.push_back(itr);
      }
    }

    // find the one with lowest priority //
    if (!ready_list.empty()) {
      size_t min_priority = std::numeric_limits<size_t>::max();
      for (auto ritr=ready_list.begin(); ritr!=ready_list.end(); ++ritr) {
        size_t curr_priority = priority_[*(*ritr)];
        if (curr_priority < min_priority) {
          itr = *ritr;
          min_priority = curr_priority;
        }
      }
    }
    return itr;
  }

  void add_outgoing_operations_to_candidate_list(const operation_t& op) {
    const_operation_iterator_t itr=traits::outgoing_operations_begin(
          *input_ptr_, op);
    const_operation_iterator_t itr_end=traits::outgoing_operations_end(
        *input_ptr_, op);

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
  priority_map_t priority_;
  //////////////////////////////////////////////////////////////////////////////

}; // class Feasible_Schedule_Generator //


// Feasible_Memory_Schedule_Generator: generates a feasible schedule for an 
// operation DAG on a memory (contiguous resource) model. Also memory model
// falls into a producer-consumer resource model.
template<typename T, typename SchedulerTraits=scheduler_traits<T>,
         typename Allocator=std::allocator<T>>
class Feasible_Memory_Schedule_Generator {
  public:

    ////////////////////////////////////////////////////////////////////////////
    typedef SchedulerTraits traits;
    typedef typename traits::dag_t dag_t;
    typedef typename traits::operation_t operation_t;
    typedef typename traits::resource_t resource_t;
    typedef typename traits::delay_t delay_t;
    typedef typename traits::const_operation_iterator_t
        const_operation_iterator_t;
    typedef Producer_Consumer_Contiguous_Resource<resource_t, operation_t>
        resource_state_t;
    typedef size_t schedule_time_t;

    // The spill op is considered an implicit op //
    enum class op_type_e { ORIGINAL_OP=0, IMPLICIT_OP_READ=1,
        IMPLICIT_OP_WRITE=3 }; 
    // The output of the operation is active or spilled until its consumed.
    enum class operation_output_e { ACTIVE=0, SPILLED=1, CONSUMED=2} ;

    struct op_output_info_t {
      op_output_info_t(operation_output_e state=operation_output_e::CONSUMED,
          size_t outstanding_consumers=0UL)
        : state_(state), outstanding_consumers_(outstanding_consumers){}

      bool active() const { return state_ == operation_output_e::ACTIVE; }
      bool spilled() const { return state_ == operation_output_e::SPILLED; }
      bool consumed() const {
        return state_ == operation_output_e::CONSUMED;
      }

      operation_output_e state_;
      size_t outstanding_consumers_;
    }; // struct op_output_info_t //

    // This structure helps distinguish between original and implicit ops //
    struct scheduled_op_info_t {
      scheduled_op_info_t(operation_t op, op_type_e type)
        : op_(op), op_type_(type) {}
      scheduled_op_info_t() : op_(), op_type_() {}

      operation_t op_;
      op_type_e op_type_;
    }; // struct scheduled_op_info_t //

    //TODO(vamsikku): keep track of active results to help with an eviction 
    //policy.
    struct active_result_info_t {
      active_result_info_t(operation_t op, size_t demand_index,
          resource_t resource_usage) : op_(op), demand_index_(demand_index),
      resource_usage_(resource_usage) {}

      operation_t op_;
      size_t demand_index_; 
      resource_t resource_usage_;
    }; // struct active_result_info_t //

    struct heap_element_t {
      heap_element_t(operation_t op, schedule_time_t t=0UL)
        : op_(op), time_(t) {}

      bool operator<(const heap_element_t& o) const { return time_ > o.time_; }
      bool operator==(const heap_element_t& o) const {
        return (op_ == o.op_) && (time_ == o.time_);
      }

      operation_t op_;
      schedule_time_t time_;
    }; // struct heap_element_t //


    //TODO(vamsikku): consolidate all the lookup tables into one //
    typedef std::vector<heap_element_t> schedule_heap_t;
    typedef std::list<operation_t> op_list_t;
    typedef std::list<operation_t> ready_data_list_t;
    typedef std::unordered_set<operation_t> ready_active_list_t; 
    typedef std::unordered_set<operation_t> processed_ops_t; 
    typedef std::unordered_map<operation_t, size_t> op_in_degree_t;
    typedef std::unordered_map<operation_t, op_output_info_t> op_output_table_t;
    ////////////////////////////////////////////////////////////////////////////

    Feasible_Memory_Schedule_Generator(const dag_t& in, const resource_t& bound)
      : scheduled_ops_at_this_time_(), ready_list_(), op_in_degree_(),
      ready_active_list_(), ready_data_list_(), processed_ops_(), heap_(),
      current_time_(), memory_state_(), input_ptr_(&in) {}

    Feasible_Memory_Schedule_Generator()
      : scheduled_ops_at_this_time_(), ready_list_(), op_in_degree_(),
      ready_active_list_(), ready_data_list_(), processed_ops_(), heap_(),
      current_time_(), memory_state_(), input_ptr_(NULL) {}

    bool reached_end(void) const {
      return ready_list_.empty() && ready_active_list_.empty();
    }

    // Only terminated schedulers are equivalent //
    bool operator==(const Feasible_Memory_Schedule_Generator& o) const {
      return reached_end() && o.reached_end();
    }

    Feasible_Memory_Schedule_Generator& operator++() {
      move_to_next_schedule_op();
      return *this;
    }

  protected:

    //////////////////////// schedule heap /////////////////////////////////////
    heap_element_t const * top_element() const {
      return heap_.empty() ? NULL : &(heap_.front());
    }

    void push_to_heap(const heap_element_t& elem) {
      heap_.push_back(elem);
      std::push_heap(heap_.begin(), heap_.end());
    }

    // Precondition: !heap_.empty() //
    heap_element_t pop_from_heap() {
      std::pop_heap(heap_.begin(), heap_.end());
      heap_element_t elem = heap_.back();
      heap_.pop_back();
      return elem;
    }

    template<typename BackInsertIterator>
    void pop_all_elements_at_this_time(schedule_time_t time_step,
        BackInsertIterator output) {
      heap_element_t const *top_ptr = NULL;
      while ( (top_ptr = top_element()) && (top_ptr->time_ == time_step) ) {
        output = pop_from_heap();
        ++output;
      }
    }
    ////////////////////////////////////////////////////////////////////////////


    //Precondition: !heap_.empty() //
    //Note: this also updates the ready_list's //
    void unschedule_all_ops_ending_at_this_time_step(schedule_time_t time_step){
      std::vector<heap_element_t> unsched_ops;

      pop_all_elements_at_this_time(time_step, std::back_inserter(unsched_ops));

      for (auto itr = unsched_ops.begin(); itr != unsched_ops.end(); ++itr) {
        operation_t op = (*itr).op_;
        traits::unschedule_operation(op, memory_state_);
        add_outgoing_operations_to_ready_list(op);
      }

    }


    void move_to_next_schedule_op() {
      //TODO(vkundeti): take the op from the scheduled ops list if empty
      //try to refill //
    }
    

    // Also maintains the invariant that the map in_degree_ has no ops
    // with zero in-degree //
    void reduce_in_degree_of_adjacent_operations(const operation_t& op) {
      // reduce the in-degree of the adjacent operations //
      const_operation_iterator_t citr =
          traits::outgoing_operations_begin(*input_ptr_, op);
      const_operation_iterator_t citr_end =
          traits::outgoing_operations_end(*input_ptr_, op);
      typename op_in_degree_t::iterator deg_itr;

      for (; citr != citr_end; ++citr) {
        operation_t pop = *citr;
        deg_itr = op_in_degree_.find(pop);

        assert((deg_itr != op_in_degree_.end()) && (deg_itr->second > 0) );

        if (deg_itr->second == 1) {
          op_in_degree_.erase(deg_itr);
        } else {
          --(deg_itr->second);
        }

      }
    }

    void compute_op_in_degree() {
      op_in_degree_.clear();

      const dag_t& in = *input_ptr_;
      const_operation_iterator_t op_begin = traits::operations_begin(in);
      const_operation_iterator_t op_end = traits::operations_end(in);

      for (; op_begin != op_end; ++op_begin) {
        operation_t op = *op_begin;

        const_operation_iterator_t cop_begin =
            traits::outgoing_operations_begin(in, op);
        const_operation_iterator_t cop_end =
            traits::outgoing_operations_end(in, op);

        for (; cop_begin != cop_end; ++cop_begin) {
          operation_t cop = *cop_begin; // child op //
          typename op_in_degree_t::iterator ditr = op_in_degree_.find(cop);

          if (ditr == op_in_degree_.end()) {
            ditr = (op_in_degree_.insert(std::make_pair(cop, 0UL))).first;
          }
          ditr->second++;

        }
      }
    }

    bool is_zero_in_degree_op(const operation_t& op) const {
      return (op_in_degree_.find(op) == op_in_degree_.end());
    }

    // Precondition: operation must be in this DAG //
    size_t get_op_in_degree(const operation_t& op) const {
      typename op_in_degree_t::const_iterator itr = op_in_degree_.find(op);
      if (itr == op_in_degree_.end()) { return 0UL; }
      return (itr->second);
    }

    void compute_ready_data_list() {
      const_operation_iterator_t itr, itr_end;
        
      itr = traits::operations_begin(*input_ptr_);
      itr_end = traits::operations_end(*input_ptr_);

      for (;itr != itr_end; ++itr) {
        const operation_t & op = *itr;
        if ( traits::is_data_operation(*input_ptr_, op) &&
              is_zero_in_degree_op(op) ) {
          ready_data_list_.push_back(op);
          // this may create new ready compute-ops //
          reduce_in_degree_of_adjacent_operations(op); 
        }
      } // foreach op //
    }

    void compute_ready_compute_list() {
      const_operation_iterator_t itr, itr_end;
        
      itr = traits::operations_begin(*input_ptr_);
      itr_end = traits::operations_end(*input_ptr_);

      for (;itr != itr_end; ++itr) {
        operation_t op = *itr;
        if (traits::is_compute_operation(*input_ptr_, op) &&
            is_zero_in_degree_op(op)) {
          ready_list_.push_back(op);
        }
      } // foreach op //
    }

    bool init(const resource_t& upper_bound) {
      traits::initialize_resource_upper_bound(upper_bound, memory_state_);

      current_time_ = schedule_time_t(0);
      clear_lists();

      compute_op_in_degree();
      compute_ready_data_list();
      compute_ready_compute_list();

      return true;
    }

    bool is_operation_output_in_active_memory(operation_t op) const {
      return memory_state_.is_key_using_resources(op);
    }


    // A compute operation is schedulable if there are resources available for
    // all its inputs and output. Some of its inputs may be in active memory
    // which has demand 0 //
    //
    // Precondition: this must be a ready operation which means that all its
    // input compute operations are completed.
    bool is_ready_compute_operation_schedulable(operation_t op) const {
      // first check if output resources for this operation are available //
      resource_t demand = traits::resource_utility(*input_ptr_, op);

      if (!traits::is_resource_available(demand, memory_state_)) {
        return false;
      }

      // All its inputs should be active producers and there should be
      // resources available for this operation. //
      const_operation_iterator_t itr = traits::incoming_operations_begin(op);
      const_operation_iterator_t itr_end = traits::incoming_operations_end(op);

      std::list<resource_t> demand_list;
      if (!traits::is_empty_demand(demand)) {
        demand_list.push_back(demand); // push the demand of the current op //

      }

      // if any of the inputs is active then we don't need any demand for that
      // input.
      for (; itr != itr_end; ++itr) {
        const operation_t& pop = *itr;
        typename op_output_table_t::const_iterator out_itr =
            op_output_table_.find(pop);

        if ((out_itr == op_output_table_.end()) ||
              ((out_itr->second).spilled()) ) {
          demand = traits::resource_utility(*input_ptr_, pop);
          if (!traits::is_empty_demand(demand)) {
            demand_list.push_back(demand);
          }
        }
      }
      return memory_state_.are_resources_available(demand_list.begin(),
            demand_list.end());
    }

    // Returns the number of active ready ops which were scheduled //
    size_t schedule_all_possible_ops_in_active_ready_list() {
      size_t scount = 0UL;

      for (typename ready_active_list_t::iterator
            itr = ready_active_list_.begin(); itr != ready_active_list_.end();
              ++itr) {
        const operation_t& op = *itr;
        if (is_ready_compute_operation_schedulable(op)) {
          // schedule the op //
          // TODO(vamsikku): //
        }
      }
    }

    // Core Schedule Operation:
    // 1. Update the op_output_table_ to ACTIVE 
    // 2. Update the resource state 
    //
    // Precondition: is_ready_compute_operation_schedulable(op) is true //
    // TODO(vamsikku):
    bool schedule_op(const operation_t& op) { return false; }

    // Core Force Schedule Operation:
    // Try to make space for this operation by evicting all active inputs which
    // don't belong to this op.
    // TODO(vamsikku):
    bool force_schedule_op(const operation_t& op) { return false; }

    // Core Un-Schedule Operation:
    // 1. Notify all this producers
    // 2. Update op_output_table_ for all its producers
    //
    // TODO(vamsikku): this operation completed
    bool unschedule_completed_op( ) { return false; }


    // Precondition: scheduled_ops_at_this_time_.empty() //
    void find_all_schedulable_ops_at_next_time_step() {
      assert(scheduled_ops_at_this_time_.empty());

      // STEP-0: first establish the schedule time //
      if (!heap_.empty()) {
        heap_element_t const *top_ptr = top_element();
        assert(top_ptr);
        unschedule_all_ops_ending_at_this_time_step(top_ptr->time_);
      }

      // STEP-1: find all schedulable ops from active ready list. //

      // STEP-2: find all schedulable ops from. //
    }

    void reset_input(const dag_t& in) { input_ptr_ = &in; }
    void reset(const dag_t& in, const resource_t& upper_bound) {
      input_ptr_ = &in;
      clear_lists();
      traits::initialize_resource_upper_bound(upper_bound, memory_state_);
    }

    void clear_lists() {
      scheduled_ops_at_this_time_.clear();
      ready_list_.clear(); // ready to go compute ops with none active inputs.
      ready_data_list_.clear(); // all ready data inputs //
      ready_active_list_.clear(); // compute ops with at least one active input.
    }

    ////////////////////////////////////////////////////////////////////////////
    op_list_t scheduled_ops_at_this_time_;
    op_list_t ready_list_;
    // Invariant: op_in_degree_[op] > 0. If op is not in this map then its
    // in-degree is zero.//
    op_in_degree_t op_in_degree_;
    ready_active_list_t ready_active_list_;
    ready_data_list_t ready_data_list_;
    processed_ops_t processed_ops_;
    schedule_heap_t heap_;
    schedule_time_t current_time_;
    resource_state_t memory_state_; // state of the memory //
    op_output_table_t op_output_table_;

    const dag_t * input_ptr_;
    ////////////////////////////////////////////////////////////////////////////

}; // class Feasible_Memory_Schedule_Generator //


} // namespace lp_scheduler //

} // namespace mv //

#endif
