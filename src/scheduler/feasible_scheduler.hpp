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
  // should be a function which defines a hash function and is equivalent to
  // std::hash<operation_t> //
  typedef int operation_hash_t; 

  // Invariant: &(*itr) should remain same constant irrespective of the iterator
  typedef int const_operation_iterator_t;

  static const char* operation_name(const operation_t& op);

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
  static delay_t spilled_read_delay(const dag_t&, const operation_t&);
  static delay_t spilled_write_delay(const dag_t&, const operation_t&);
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
template< typename Unit, typename Key, typename KeyHash=std::hash<Key>,
         typename KeyEqual=std::equal_to<Key> >
class Contiguous_Resource_State {
  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef Unit unit_t;
    typedef Key key_t;
    typedef KeyHash key_hash_t;
    typedef KeyEqual key_equal_t;
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

    typedef std::unordered_map<key_t, interval_info_t, key_hash_t,
              key_equal_t> lookup_t;
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

    bool assign_resources(const key_t& key, unit_t ibeg, unit_t iend) {
      if ( !( (location_begin_ <= ibeg) && (ibeg <= iend) &&
              (iend <= location_end_) ) ) { return false; }

      typename lookup_t::iterator litr = lookup_.find(key);
      if (litr != lookup_.end()) { return false; }

      // insert into the interval tree //
      if ( !active_resources_.insert(ibeg, iend, key) ) { return false; }

      lookup_.insert( std::make_pair( key, interval_info_t(ibeg, iend) ) );
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

    void clear() {
      active_resources_.clear();
      lookup_.clear();
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
    typedef typename traits::operation_hash_t operation_hash_t;
    typedef typename traits::resource_t resource_t;
    typedef typename traits::delay_t delay_t;
    typedef typename traits::const_operation_iterator_t
        const_operation_iterator_t;
    typedef size_t schedule_time_t;

    // The spill op is considered an implicit op //
    enum class op_type_e { ORIGINAL_OP=0, IMPLICIT_OP_READ=1,
        IMPLICIT_OP_WRITE=2 };
    // The output of the operation is active or spilled until its consumed.
    enum class operation_output_e { ACTIVE=0, SPILLED=1, CONSUMED=2 };

    struct op_output_info_t {
      op_output_info_t(operation_output_e state=operation_output_e::CONSUMED,
          size_t outstanding_consumers=0UL)
        : state_(state), outstanding_consumers_(outstanding_consumers){}

      bool active() const { return state_ == operation_output_e::ACTIVE; }
      bool spilled() const { return state_ == operation_output_e::SPILLED; }
      bool consumed() const { return state_ == operation_output_e::CONSUMED; }

      void change_state_to_active() { state_ = operation_output_e::ACTIVE; }
      void change_state_to_consumed() { state_ = operation_output_e::CONSUMED; } 
      void change_state_to_spilled() { state_ = operation_output_e::SPILLED; } 

      void decrement_consumers() {
        assert(outstanding_consumers_ > 0UL);
        --outstanding_consumers_;
        if (!outstanding_consumers_) { state_ = operation_output_e::CONSUMED; }
      }

      operation_output_e state_;
      size_t outstanding_consumers_;
    }; // struct op_output_info_t //

    struct op_demand_info_t {
      op_demand_info_t(operation_t op, size_t demand_index)
        : op_(op), demand_index_(demand_index) {}

      bool operator==(const op_demand_info_t& o) const {
        return (op_ == o.op_) && (demand_index_ == o.demand_index_);
      }

      operation_t op_;
      size_t demand_index_;
    }; // struct op_demand_info_t //

    struct op_demand_info_hash_t {
      size_t operator()(const op_demand_info_t& op) const {
        operation_hash_t op_hash;
        std::hash<size_t> size_hash;

        return op_hash(op.op_) + size_hash(op.demand_index_);
      }
    }; // struct op_demand_info_hash_t //


    typedef resource_t unit_t;
    typedef Contiguous_Resource_State<resource_t,
            op_demand_info_t, op_demand_info_hash_t> resource_state_t;
    typedef typename resource_state_t::interval_info_t interval_info_t;

    struct active_result_info_t {

      active_result_info_t() : op_(), parent_op_(),
        demand_index_(), interval_info_() {}

      active_result_info_t(operation_t op, size_t demand_index,
            const interval_info_t& interval_info) : op_(op), parent_op_(op),
      demand_index_(demand_index), interval_info_(interval_info) {}

      active_result_info_t(operation_t op, operation_t pop, size_t demand_index,
            const interval_info_t& interval_info) : op_(op), parent_op_(pop),
      demand_index_(demand_index), interval_info_(interval_info) {}

      active_result_info_t& operator=(const active_result_info_t& o) {
        op_ = o.op_; parent_op_ = o.parent_op_;
        demand_index_ = o.demand_index_;
        interval_info_ = o.interval_info_;
      }

      operation_t op_;
      operation_t parent_op_;
      size_t demand_index_;
      interval_info_t interval_info_;
    }; // struct active_result_info_t //

    struct heap_element_t {
      heap_element_t() : op_(), time_(), op_type_() {}

      heap_element_t(operation_t op, schedule_time_t t=0UL,
          op_type_e op_type=op_type_e::ORIGINAL_OP)
        : op_(op), time_(t), op_type_(op_type) {}

      bool operator==(const heap_element_t& o) const {
        return (op_ == o.op_) && (time_ == o.time_);
      }

      bool is_original_op() const {
        return (op_type_ == op_type_e::ORIGINAL_OP);
      }
      bool is_implicit_write_op() const {
        return (op_type_ == op_type_e::IMPLICIT_OP_WRITE);
      }

      //TODO(vamsikku): use scheduled_op_info_t directly instead of
      // maintaining op_ and op_type_ seperately.
      operation_t op_;
      schedule_time_t time_;
      op_type_e op_type_; 
    }; // struct heap_element_t //


    // This structure helps distinguish between original and implicit ops //
    struct scheduled_op_info_t {
      scheduled_op_info_t(operation_t op, op_type_e type, schedule_time_t time)
        : op_(op), op_type_(type), time_(time) {}

      scheduled_op_info_t() : op_(), op_type_(), time_() {}

      bool operator==(const scheduled_op_info_t& o) const {
        return (o.op_ == op_) && (o.op_type_ == op_type_);
      }

      bool operator<(const scheduled_op_info_t& o) const {
        return (op_ != o.op_) ? (op_ < o.op_) : (op_type_ < o.op_type_);
      }

      const scheduled_op_info_t& operator=(const heap_element_t& helement) {
        op_ = helement.op_;
        op_type_ = helement.op_type_;
      }

      const char* op_type_name() const { 
        const char *ret = NULL;

        switch (op_type_) {

          case op_type_e::ORIGINAL_OP :
            ret = "ORIGINAL";
            break;
          case op_type_e::IMPLICIT_OP_READ:
            ret = "SPILLED_READ";
            break;
          default:
            ret = "SPILLED_WRITE";
            break;
        }
        return ret;
      }

      bool has_active_resource() const {
        return (resource_info_.begin_ <= resource_info_.end_);
      }

      resource_t begin_resource() const { return resource_info_.begin_; }
      resource_t end_resource() const { return resource_info_.end_; }

      operation_t op_; 
      op_type_e op_type_;
      schedule_time_t time_;
      interval_info_t resource_info_;
    }; // struct scheduled_op_info_t //

    struct completion_time_ordering_t {
      bool operator()(const heap_element_t& a, const heap_element_t& b) const {
        return a.time_ > b.time_;
      }
    }; // struct completion_time_ordering_t //

    struct start_time_ordering_t {
      bool operator()(const heap_element_t& a, const heap_element_t& b) const {
        return a.time_ > b.time_;
      }
    }; // struct start_time_ordering_t //



    //TODO(vamsikku): the purpose of a eviction policy is to choose the 
    //minimum active operation for eviction.
    struct default_eviction_policy_t {
      bool operator()(const operation_t& a, const operation_t& b) const {
        return false;
      }
    }; // struct default_eviction_policy_t //


    //TODO(vamsikku): consolidate all the lookup tables into one //
    typedef std::vector<active_result_info_t> eviction_heap_t;
    typedef std::vector<heap_element_t> heap_t;
    typedef std::list<operation_t> op_list_t;
    typedef std::list<scheduled_op_info_t> scheduled_op_info_list_t;

    // ready lists //
    typedef std::unordered_set<operation_t> ready_data_list_t;
    typedef std::unordered_set<operation_t> ready_active_list_t; 
    typedef std::unordered_set<operation_t> ready_list_t; 

    typedef std::unordered_set<operation_t> processed_ops_t; 
    typedef std::unordered_map<operation_t, size_t> op_in_degree_t;
    typedef std::unordered_map<operation_t, op_output_info_t> op_output_table_t;
    typedef std::vector<active_result_info_t> active_op_resources_t;
    // NOTE: the reason for the active_op_resources_t is to consider any future 
    // possibility of one operation generating multiple outputs.
    typedef std::unordered_map<operation_t, active_op_resources_t>
        active_resource_table_t;

    struct noop_op_back_insert_iterator_t {
      void operator++() {}
      void operator=(const operation_t&) {} 
    }; // struct noop_op_back_insert_iterator_t //

    struct decreasing_demand_ordering_t {

      decreasing_demand_ordering_t(const dag_t& dag) : dag_(dag) {}

      bool operator()(const resource_t& a, const resource_t& b) const {
        return a > b;
      }

      bool operator()(const operation_t& a, const operation_t& b) const {
        return traits::resource_utility(dag_, a) > 
            traits::resource_utility(dag_, b);
      }
      const dag_t& dag_;
    }; // struct decreasing_demand_ordering_t //

    ////////////////////////////////////////////////////////////////////////////

    Feasible_Memory_Schedule_Generator(const dag_t& in, const resource_t& bound)
      : current_scheduled_op_(), ready_list_(), op_in_degree_(),
      ready_active_list_(), ready_data_list_(), processed_ops_(),
      completion_time_heap_(), start_time_heap_(), current_time_(),
      memory_state_(), op_output_table_(), active_resource_table_(),
      input_ptr_(&in), total_compute_ops_(0UL), scheduled_compute_ops_(0UL){
        init(in, bound);
    }


    Feasible_Memory_Schedule_Generator()
      : ready_list_(), op_in_degree_(), ready_active_list_(),
      ready_data_list_(), processed_ops_(), completion_time_heap_(),
      start_time_heap_(), current_time_(), memory_state_(), op_output_table_(),
      active_resource_table_(), input_ptr_(NULL), total_compute_ops_(0UL),
      scheduled_compute_ops_(0UL) {}


    bool reached_end(void) const {
      return ready_list_.empty() && ready_active_list_.empty() &&
        start_time_heap_.empty() && completion_time_heap_.empty();
    }

    // Only terminated schedulers are equivalent //
    bool operator==(const Feasible_Memory_Schedule_Generator& o) const {
      return reached_end() && o.reached_end();
    }

    bool operator!=(const Feasible_Memory_Schedule_Generator& o) const {
      return !(*this == o);
    }

    const scheduled_op_info_t& operator*() const {
      return current_scheduled_op_;
    }

    Feasible_Memory_Schedule_Generator& operator++() {
      next_schedulable_op();
      return *this;
    }

    template<typename BackInsertIterator>
    size_t get_currently_scheduled_spill_read_operations(
        BackInsertIterator output) const {
      return get_scheduled_operations_at_time(current_time_, output,
            op_type_e::IMPLICIT_OP_READ);
    }

    template<typename BackInsertIterator>
    size_t get_currently_scheduled_spill_write_operations(
        BackInsertIterator output) const {
      return get_scheduled_operations_at_time(current_time_, output,
            op_type_e::IMPLICIT_OP_WRITE);
    }

    template<typename BackInsertIterator>
    size_t get_currently_scheduled_operations(BackInsertIterator output) const {
      return get_scheduled_operations_at_time(current_time_, output);
    }

    template<typename BackInsertIterator>
    size_t get_scheduled_operations_at_time(schedule_time_t time,
          BackInsertIterator output,
          op_type_e op_type=op_type_e::ORIGINAL_OP) const {
      size_t sched_op_count = 0;
      // all operations in the heap //
      for (typename heap_t::const_iterator hitr=start_time_heap_.begin();
          hitr != start_time_heap_.end(); ++hitr) {
        if ((hitr->time_ == time) && (hitr->op_type_ == op_type)) {
          output = hitr->op_;
          ++output;
          ++sched_op_count;
        }
      }
      return sched_op_count;
    }

  protected:

    const active_resource_table_t& get_current_active_resource_table() const {
      return active_resource_table_;
    }

    //////////////////////// schedule heap /////////////////////////////////////


    bool heap_empty() const { return completion_time_heap_.empty(); }

    heap_element_t const * top_element() const {
      return top_element_gen(completion_time_heap_);
    }

    void push_to_heap(const heap_element_t& elem) {
      push_to_heap_gen(elem, completion_time_heap_,
            completion_time_ordering_t());
    }

    heap_element_t pop_from_heap() {
      return pop_from_heap_gen(completion_time_heap_,
            completion_time_ordering_t());
    }

    template<typename BackInsertIterator>
    void pop_all_elements_at_this_time(schedule_time_t time_step,
        BackInsertIterator output) {
      heap_element_t const *top_ptr = NULL;
      while ( (top_ptr = top_element()) && (top_ptr->time_ == time_step) ) {
        output = pop_from_heap_gen(completion_time_heap_,
            completion_time_ordering_t());
        ++output;
      }
    }

    // ops on the start time heap //
    bool heap_empty_start_time() const { return start_time_heap_.empty(); }

    heap_element_t const *top_element_completion_time() const {
      return top_element();
    }

    //TODO(vamsikku): this is existing for backward compatability with unit-test
    //this should to changed to top_element_completion_time() in future.
    heap_element_t const * top_element_start_time() const {
      return top_element_gen(start_time_heap_);
    }

    void push_to_heap_start_time(const heap_element_t& elem) {
      push_to_heap_gen(elem, start_time_heap_, start_time_ordering_t());
    }

    heap_element_t pop_from_heap_start_time() {
      return pop_from_heap_gen(start_time_heap_, start_time_ordering_t());
    }

    template<typename BackInsertIterator>
    void pop_all_elements_at_this_start_time(schedule_time_t time_step,
        BackInsertIterator output) {
      heap_element_t const *top_ptr = NULL;
      while ( (top_ptr = top_element()) && (top_ptr->time_ == time_step) ) {
        output = pop_from_heap_gen(start_time_heap_, start_time_ordering_t());
        ++output;
      }
    }

    // gen heap operations //
    template<typename ordering_t>
    void push_to_heap_gen(const heap_element_t& elem, heap_t& heap,
        const ordering_t& order) {
      heap.push_back(elem);
      std::push_heap(heap.begin(), heap.end(), order);
    }


    heap_element_t const * top_element_gen(const heap_t& heap) const {
      return heap.empty() ? NULL : &(heap.front());
    }
    // Precondition: !heap_.empty() //
    template<typename ordering_t>
    heap_element_t pop_from_heap_gen(heap_t& heap, const ordering_t& order) {
      std::pop_heap(heap.begin(), heap.end(), order);
      heap_element_t elem = heap.back();
      heap.pop_back();
      return elem;
    }


    template<typename Ordering, typename BackInsertIterator>
    void pop_all_elements_at_this_time_gen(heap_t& heap, const Ordering& order, 
        schedule_time_t time_step, BackInsertIterator output) {
      heap_element_t const *top_ptr = NULL;
      while ( (top_ptr = top_element_gen(heap)) &&
            (top_ptr->time_ == time_step) ) {
        output = pop_from_heap_gen(heap, order);
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
        unschedule_operation(op, memory_state_);
        add_outgoing_operations_to_ready_list(op);
      }
    }


    void reduce_in_degree_of_adjacent_operations(const operation_t& op) {
      noop_op_back_insert_iterator_t noop;
      reduce_in_degree_of_adjacent_operations_gen(op, noop);
    }
    
    // Also maintains the invariant that the map in_degree_ has no ops
    // with zero in-degree. Also returns zero in degree ops via a back insert
    // iterator.
    template<typename BackInsertIterator>
    void reduce_in_degree_of_adjacent_operations_gen(const operation_t& op,
        BackInsertIterator output) {
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
          output = pop;
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
          ready_data_list_.insert(op);
          // this may create new ready compute-ops //
          reduce_in_degree_of_adjacent_operations(op); 
        }
      } // foreach op //
    }

    void compute_ready_compute_list() {
      const_operation_iterator_t itr, itr_end;
      total_compute_ops_ = 0UL;
        
      itr = traits::operations_begin(*input_ptr_);
      itr_end = traits::operations_end(*input_ptr_);

      for (;itr != itr_end; ++itr) {
        operation_t op = *itr;
        if (traits::is_compute_operation(*input_ptr_, op)) {
          ++total_compute_ops_;
          if (is_zero_in_degree_op(op)) {
            ready_list_.insert(op);
          }
        }
      } // foreach op //
    }

    bool init(const dag_t& input, const resource_t& upper_bound) {
      input_ptr_ = &input;
      memory_state_.clear();
      memory_state_.initialize_resource_upper_bound(upper_bound);

      current_time_ = schedule_time_t(1);
      clear_lists();

      compute_op_in_degree();
      compute_ready_data_list();
      compute_ready_compute_list();
      
      schedule_all_possible_ready_ops_and_update(ready_list_);
      next_schedulable_op();
      return true;
    }

    template<typename DemandBackInsertIterator, typename OpBackInsertIterator>
    size_t get_non_empty_op_demand_list(const operation_t& op,
        DemandBackInsertIterator output, OpBackInsertIterator output_op) const {
      resource_t demand = traits::resource_utility(*input_ptr_, op);
      size_t op_demand_count = 0UL;

      if (!traits::is_empty_demand(demand)) {
        output = demand;
        output_op = op;
        ++op_demand_count;
      }

      // All its inputs should be active producers and there should be
      // resources available for this operation. //
      const_operation_iterator_t itr =
          traits::incoming_operations_begin(*input_ptr_, op);
      const_operation_iterator_t itr_end =
          traits::incoming_operations_end(*input_ptr_, op);

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
            output = demand;
            output_op = pop;
            ++op_demand_count;
          }
        }
      }

      return op_demand_count;
    }


    template<typename DemandBackInsertIterator>
    size_t get_non_empty_op_demand_list(const operation_t& op,
        DemandBackInsertIterator output) const {
      noop_op_back_insert_iterator_t noop;
      return get_non_empty_op_demand_list(op, output, noop);
    }

    // A compute operation is schedulable if there are resources available for
    // all its inputs and output. Some of its inputs may be in active memory
    // which has demand 0 //
    //
    // Precondition: this must be a ready operation which means that all its
    // input compute operations are completed.
    bool is_ready_compute_operation_schedulable(operation_t op) const {
      if (!is_operation_using_non_empty_resources(op)) { return true; }
      // first check if output resources for this operation are available //
      resource_t demand = traits::resource_utility(*input_ptr_, op);


      if (!memory_state_.is_resource_available(demand)) {
        return false;
      }

      std::list<resource_t> demand_list;
      get_non_empty_op_demand_list(op, std::back_inserter(demand_list));

      return memory_state_.are_resources_available_simultaneously(
          demand_list.begin(), demand_list.end());
    }

    template<typename ReadyListIterator>
    size_t schedule_all_possible_ready_ops(ReadyListIterator ritr,
        ReadyListIterator ritr_end) {
      return schedule_all_possible_ready_ops_gen(ritr, ritr_end,
            noop_op_back_insert_iterator_t());
    }

    template<typename ReadyListContainer>
    size_t schedule_all_possible_ready_ops_and_update(
        ReadyListContainer& ready_ops) {
      std::list<operation_t> scheduled_ops;
      size_t ret;

      ret =
        schedule_all_possible_ready_ops_gen(ready_ops.begin(), ready_ops.end(),
          std::back_inserter(scheduled_ops));

      for (auto itr=scheduled_ops.begin(); itr!=scheduled_ops.end(); ++itr) {
        ready_ops.erase(*itr);
      }

      return ret;
    }

    // Returns the number of active ready compute ops which were scheduled //
    template<typename ReadyListIterator, typename BackInsertIterator>
    size_t schedule_all_possible_ready_ops_gen(ReadyListIterator ritr,
        ReadyListIterator ritr_end, BackInsertIterator output) {
      size_t scheduled_ops_count = 0UL;
      bool scheduled = false;

      for (; ritr != ritr_end; ++ritr) {
        const operation_t& op = *ritr;
        if (is_ready_compute_operation_schedulable(op)) {
          // schedule_compute_op() will also allocate resources for all missing
          // inputs for this compute op.
          scheduled = schedule_compute_op(op);
          assert(scheduled);
          scheduled_ops_count++;
          output = op;
        }
      }
      return scheduled_ops_count;
    }

    void schedule_input_op_for_compute_op(const operation_t& input_op) {
      // Does this op need any implicit read ops ? //
      //
      // 1. If input_op does not exist in op_output_table_ then this must be
      //    a data operation which may be need to be scheduled as original op.
      //
      // 2. Else an entry exists in the op_output_table then it must be in a
      //    spilled state.

      typename op_output_table_t::iterator op_out_itr =
          op_output_table_.find(input_op);

      op_type_e op_type;
      if (op_out_itr == op_output_table_.end()) {
        assert(traits::is_data_operation(*input_ptr_, input_op));
        //TODO(vamsikku): keep an outdegree table for each op //
        const_operation_iterator_t citr =
            traits::outgoing_operations_begin(*input_ptr_, input_op);
        const_operation_iterator_t citr_end =
            traits::outgoing_operations_end(*input_ptr_, input_op);
        size_t outstanding_consumers = 0UL;
        for (; citr != citr_end; ++citr, ++outstanding_consumers) {}

        op_output_table_.insert( std::make_pair(input_op,
              op_output_info_t(operation_output_e::ACTIVE,
                  outstanding_consumers)) );
        op_type = op_type_e::ORIGINAL_OP;
      } else {
        // just change the state //
        assert((op_out_itr->second).spilled());
        (op_out_itr->second).change_state_to_active();
        op_type = op_type_e::IMPLICIT_OP_READ;
      }

      // schedule the op //
      push_to_heap_start_time(heap_element_t(input_op, current_time_, op_type));
    }

    bool is_operation_using_non_empty_resources(const operation_t& op) const {
      return !( traits::is_empty_demand(traits::resource_utility(*input_ptr_,
                  op) ) );
    }

    bool is_output_of_this_operation_spilled(const operation_t& op) const {
      typename op_output_table_t::const_iterator out_itr =
          op_output_table_.find(op);

      return (out_itr != op_output_table_.end()) &&
          ((out_itr->second).spilled());
    }

    //assign resources and update active resources table //
    bool assign_resources_and_update_active_table(const interval_info_t& rinfo,
        const operation_t& output_op, size_t demand_index=0UL) {
      //NOTE: use demand_index if an operation can generate multiple outputs.
      //So the tuple (operation, demand_index) is the resource key.
      bool assigned = memory_state_.assign_resources(
          op_demand_info_t(output_op, demand_index), rinfo.begin_, rinfo.end_);

      if (!assigned) { return false; }

      // create an entry in the active resources table //
      typename active_resource_table_t::iterator aitr =
          active_resource_table_.find(output_op);
      if (aitr != active_resource_table_.end()) { return false; }

      aitr = active_resource_table_.insert(std::make_pair(output_op,
              active_op_resources_t())).first;
      (aitr->second).push_back(
          active_result_info_t(output_op, demand_index, rinfo) );
      return true;
    }

    //TODO: abstract the choice of picking a candidate with an eviction policy.
    bool choose_active_operation_for_eviction(operation_t& candidate) const {
      if (active_resource_table_.empty()) { return false; }

      auto aitr = active_resource_table_.begin();
      candidate = aitr->first;

      for (++aitr; aitr!=active_resource_table_.end(); ++aitr) {
        if (get_active_input_count(aitr->first) < 
              get_active_input_count(candidate)) {
          candidate = aitr->first; 
        }
      }
      return true;
    }

    bool force_schedule_active_op_eviction() {
      operation_t candidate;
      if (!choose_active_operation_for_eviction(candidate)) { return false; }
      if (!evict_active_op(candidate)) { return false; }

      //TODO(vamsikku): update ready lists since some of the ready ops in
      //ready_active_list_ may no longer be active.

      push_to_heap_start_time(heap_element_t(candidate, current_time_,
            op_type_e::IMPLICIT_OP_WRITE));
      return true;
    }

    // Core Evict Operation:
    // 1. Update the op_output_table_ and active_resource_table_ //
    // 2. Clear up the memory_state_ resources //
    bool evict_active_op(const operation_t& op) {
      typename op_output_table_t::iterator op_out_itr =
          op_output_table_.find(op);

      assert((op_out_itr != op_output_table_.end()) &&
            (op_out_itr->second).active());

      // op_output_table_ //
      op_output_info_t& op_output_info = op_out_itr->second;
      op_output_info.change_state_to_spilled();

      // active_resource_table_ //
      typename active_resource_table_t::iterator aitr =
          active_resource_table_.find(op);

      assert(aitr != active_resource_table_.end());
      // clear up the memory resources //
      const active_op_resources_t &active_op_resources = aitr->second;
      for (size_t i=0; i<active_op_resources.size(); i++) {
        // clear out its resources from memory_state_ //
        const active_result_info_t& active_result = active_op_resources[i];
        op_demand_info_t demand_key(active_result.op_,
              active_result.demand_index_);

        bool unassigned = memory_state_.unassign_resources(demand_key);
        assert(unassigned);
      }
      active_resource_table_.erase(aitr);
      return true;
    }

    // Core Schedule Operation:
    // 1. Update the op_output_table_ and active_resource_table_ 
    // 2. Assign resources simultaneously: updates memory_state_
    // 3. Schedule all non-active data ops which provide input to this op 
    //   3.1 add delay to the op equal to the max of the delays of these input
    //       ops
    // 4. Add the operation to the schedule heap : updates heap_
    //
    // Precondition: is_ready_compute_operation_schedulable(op) is true
    bool schedule_compute_op(const operation_t& op) {
      assert(traits::is_compute_operation(*input_ptr_, op));

      typename op_output_table_t::iterator out_itr = op_output_table_.find(op);
      assert(out_itr == op_output_table_.end());

      //////////////////////////////////////////////////////////////////////////
      //STEP-1: add to the output result table. //
      //TODO(vamsikku): add an additional out_degree datastructure so that this
      //can done in O(1) time.
      const_operation_iterator_t citr =
          traits::outgoing_operations_begin(*input_ptr_, op);
      const_operation_iterator_t citr_end =
          traits::outgoing_operations_end(*input_ptr_, op);
      size_t outstanding_consumers = 0UL;
      for (; citr != citr_end; ++citr, ++outstanding_consumers) {}

      op_output_table_.insert( std::make_pair(op,
            op_output_info_t(operation_output_e::ACTIVE,
                outstanding_consumers)) ); 
      //////////////////////////////////////////////////////////////////////////


      //////////////////////////////////////////////////////////////////////////
      //STEP-2: assign resources simultaneously //
      // get all the info about resource allocations //
      std::vector<resource_t> op_demands; 
      std::vector<operation_t> ops_corresponding_to_demands;
      decreasing_demand_ordering_t demand_ordering(*input_ptr_);

      get_non_empty_op_demand_list(op, std::back_inserter(op_demands),
          std::back_inserter(ops_corresponding_to_demands));

      assert(op_demands.size() == ops_corresponding_to_demands.size());

      //TODO(vamsikku): this is a small technicality since the function
      //pack_demands_into_free_bins gives a packing with decreasing demands
      //we can remove this by reconciling all demand
      std::sort(op_demands.begin(), op_demands.end(),
            decreasing_demand_ordering_t(*input_ptr_));
      std::sort(ops_corresponding_to_demands.begin(),
          ops_corresponding_to_demands.end(),
          decreasing_demand_ordering_t(*input_ptr_));

      std::vector<interval_info_t> resource_intervals;
      memory_state_.pack_demands_into_free_bins(op_demands.begin(),
          op_demands.end(), std::back_inserter(resource_intervals) );

      // resource_intervals are ordered not according to the demands //

      delay_t max_input_delay = delay_t(0);
      size_t demand_idx = 0UL;
      for (auto itr=resource_intervals.begin(); itr!=resource_intervals.end();
            ++itr, ++demand_idx) {
        assert(demand_idx < ops_corresponding_to_demands.size());

        const interval_info_t rinfo = *itr;
        const operation_t& input_op = ops_corresponding_to_demands[demand_idx];
        bool assigned = assign_resources_and_update_active_table(rinfo,
              input_op);

        if (!assigned) { return false; }

        if (input_op == op) { continue; }

        schedule_input_op_for_compute_op(input_op);

        // update the max delay to set the start time //
        max_input_delay = std::max(max_input_delay,
              traits::delay(*input_ptr_, input_op));
      } // foreach resource in the demand list //
      //////////////////////////////////////////////////////////////////////////


      //////////////////////////////////////////////////////////////////////////
      //STEP-3: schedule this compute op 
      schedule_time_t op_start_time = current_time_ + max_input_delay;
      // compute op is always original //
      push_to_heap_start_time(heap_element_t(op, op_start_time,
              op_type_e::ORIGINAL_OP));
      //////////////////////////////////////////////////////////////////////////

      //TODO(vamsikku): STEP-4: erase from the ready list //
      return true;
    }

    // Core Force Schedule Operation:
    // Try to make space for this operation by evicting all active inputs which
    // don't belong to this op.
    // TODO(vamsikku):
    bool force_schedule_op(const operation_t& op) { return false; }

    bool is_compute_op(const operation_t& op) const {
      return traits::is_compute_operation(*input_ptr_, op);
    }

    bool is_data_op(const operation_t& op) const {
      return traits::is_data_operation(*input_ptr_, op);
    }

    // Checks if this is a compute op with at least one active inputs //
    bool is_compute_op_with_some_active_inputs(const operation_t& op) const {
      if (!is_compute_op(op)) { return false; }

      const_operation_iterator_t itr =
        traits::incoming_operations_begin(*input_ptr_, op);
      const_operation_iterator_t itr_end =
          traits::incoming_operations_end(*input_ptr_, op);
      for (; itr!=itr_end; ++itr) {
        if (!(active_resource_table_.find(*itr) == active_resource_table_.end())) {
          return true;
        }
      }
      return false;
    }

    size_t get_active_input_count(const operation_t& op) const {
      const_operation_iterator_t itr =
        traits::incoming_operations_begin(*input_ptr_, op);
      const_operation_iterator_t itr_end =
          traits::incoming_operations_end(*input_ptr_, op);
      size_t acount = 0UL;
      for (; itr!=itr_end; ++itr) {
        if (!(active_resource_table_.find(*itr) == active_resource_table_.end())) {
          ++acount;
        }
      }
      return acount;
    }

    // Core Un-Schedule Operation:
    // 1. Update op_output_table_: for all its producers by decrementing
    //    the consumer count in the op_output_table_. If the state changes to
    //    consumed. Free its resources from memory_state_ note this operation
    //    must be active so lookup active_resources_table_ so totally clear it.
    //
    // 2. Clear all its input entries in the active_resources_table_ and if this
    //    op has no consumers just clear it from the active_resources_table_.
    //    This can also happen when the resource utility of the op is zero.
    void unschedule_op(const heap_element_t &helement) {

      const operation_t& op = helement.op_;
      const_operation_iterator_t pitr, pitr_end;

      if (helement.is_original_op()) {
        // for implicit read or write ops we need not decrement the consumers
        pitr = traits::incoming_operations_begin(*input_ptr_, op);
        pitr_end = traits::incoming_operations_end(*input_ptr_, op);
      }

      // decrement the consumer count for all the incoming edges into this
      // operation.

      for (; pitr != pitr_end; ++pitr) {
        const operation_t& pop = *pitr;

        typename op_output_table_t::iterator itr = op_output_table_.find(pop);
        assert(itr != op_output_table_.end());
        op_output_info_t &pop_output_info = itr->second;

        pop_output_info.decrement_consumers();

        if ( pop_output_info.consumed() &&
              is_operation_using_non_empty_resources(pop) ) {

          // clear all its active resources //
          typename active_resource_table_t::iterator aitr =
              active_resource_table_.find(pop);
          assert(aitr != active_resource_table_.end());
          
          const active_op_resources_t &active_op_resources = aitr->second;
          for (size_t i=0; i<active_op_resources.size(); i++) {
            // clear out its resources from memory_state_ //
            const active_result_info_t& active_result = active_op_resources[i];
            op_demand_info_t demand_key(active_result.op_,
                  active_result.demand_index_);

            bool unassigned = memory_state_.unassign_resources(demand_key);
            assert(unassigned);
          }

          // now delete this consumed operation from the active resources table
          active_resource_table_.erase(aitr);
        }
      }
      
      typename op_output_table_t::iterator itr = op_output_table_.find(op);
      assert(itr != op_output_table_.end());
      
      op_output_info_t &op_output_info = itr->second;

      if (op_output_info.consumed()) {
        op_output_info.change_state_to_consumed();
      }
    }

    bool is_operation_output_in_active_memory(const operation_t& o) const {
      return !(active_resource_table_.find(o) == active_resource_table_.end());
    }

    template<typename OpIterator>
    void distribute_ready_ops(OpIterator obegin, OpIterator oend) {
      for (; obegin != oend; ++obegin) {
        const operation_t& op = *obegin;

        if (is_data_op(op)) {
          assert(ready_data_list_.find(op) == ready_data_list_.end());
          ready_data_list_.insert(op);

          std::list<operation_t> ready_ops;
          reduce_in_degree_of_adjacent_operations_gen(op,
              std::back_inserter(ready_ops));
          // assert all the ready ops are compute ops //
          distribute_ready_ops(ready_ops.begin(), ready_ops.end());
        } else if (is_compute_op_with_some_active_inputs(op)) {
          assert(ready_active_list_.find(op) == ready_active_list_.end());
          ready_active_list_.insert(op);
        } else {
          ready_list_.insert(op);
        }
      }
    }

    void unschedule_all_completing_ops_at_next_earliest_time() {
      assert(top_element_completion_time());
      const heap_element_t *completion_top_ptr = top_element_completion_time();

      assert(completion_top_ptr);
      current_time_ = completion_top_ptr->time_;

      std::list<heap_element_t> unsched_ops;
      pop_all_elements_at_this_time(current_time_,
            std::back_inserter(unsched_ops));
      std::list<operation_t> ready_ops;

      for (auto uitr=unsched_ops.begin(); uitr != unsched_ops.end();
            ++uitr) {
        const operation_t& op = uitr->op_;

        unschedule_op(*uitr);

        if (is_compute_op(op) && (uitr->is_original_op())) {
          reduce_in_degree_of_adjacent_operations_gen(op,
              std::back_inserter(ready_ops));
        }
      }
      distribute_ready_ops(ready_ops.begin(), ready_ops.end());
    }


    // Precondition: at least some ready operations //
    void next_schedulable_op() {
      bool found_schedulable_op = false;


      do {
        // pick the min among the start time and completion time heaps //
        const heap_element_t *start_top_ptr = top_element_start_time();
        const heap_element_t *completion_top_ptr = top_element_completion_time();

        assert(start_top_ptr || completion_top_ptr);

        bool pop_from_start_heap = start_top_ptr && ( !completion_top_ptr || 
            (start_top_ptr->time_ < completion_top_ptr->time_) );

        heap_element_t helement;
        if (pop_from_start_heap) {
          assert(start_top_ptr);

          helement = pop_from_heap_start_time();

          current_time_ = helement.time_;

          // output this scheduled operation //
          current_scheduled_op_.op_ = helement.op_;
          current_scheduled_op_.op_type_ = helement.op_type_;
          current_scheduled_op_.time_ = current_time_;
          if (!helement.is_implicit_write_op() && 
                is_operation_using_non_empty_resources(helement.op_)) {
            current_scheduled_op_.resource_info_ =
                get_active_resource_info(helement.op_);
          } else {
            current_scheduled_op_.resource_info_.invalidate();
          }
          found_schedulable_op = true; /*break-out*/


          // now move this scheduled op to the completion heap //
          helement.time_ += traits::delay(*input_ptr_, helement.op_);
          push_to_heap(helement); // add to the completion heap //
        } else {

          do {
            // Move the time to next earliest time and unschedule all ops ending
            // at this time. This creates new ready lists.
            unschedule_all_completing_ops_at_next_earliest_time();

            // since we have unscheduled some ops try to see if we could 
            // schedule new ones //
            schedule_all_possible_ready_ops_and_update(ready_active_list_);
            schedule_all_possible_ready_ops_and_update(ready_list_);
            //TODO(vamsikku): we can also schedule data ops here //
          } while (!heap_empty() && heap_empty_start_time());


          if (heap_empty_start_time()) {
            // we are unable to schedule any ops so we need to force evict some
            // active ops //
            force_schedule_active_op_eviction();
          }
        }

      } while (!found_schedulable_op && !reached_end()); 

    }

    const interval_info_t& get_active_resource_info(
          const operation_t& op) const {
      typename active_resource_table_t::const_iterator aitr =
          active_resource_table_.find(op);
      assert(aitr != active_resource_table_.end());
      assert(!(aitr->second).empty());

      return (aitr->second).front().interval_info_;
    }

   
    void reset_input(const dag_t& in) { input_ptr_ = &in; }

    void reset(const dag_t& in, const resource_t& upper_bound) {
      input_ptr_ = &in;
      clear_lists();
      memory_state_.initialize_resource_upper_bound(upper_bound);
    }

    void clear_lists() {
      ready_list_.clear(); // ready to go compute ops with none active inputs.
      ready_data_list_.clear(); // all ready data inputs //
      ready_active_list_.clear(); // compute ops with at least one active input.
    }

    ////////////////////////////////////////////////////////////////////////////
    // 0 - write ops , 1 - read ops , 2 - compute ops
    scheduled_op_info_t current_scheduled_op_;
    ready_list_t ready_list_;
    // Invariant: op_in_degree_[op] > 0. If op is not in this map then its
    // in-degree is zero.//
    op_in_degree_t op_in_degree_;
    ready_active_list_t ready_active_list_;
    ready_data_list_t ready_data_list_;
    processed_ops_t processed_ops_;
    heap_t completion_time_heap_;
    heap_t start_time_heap_;
    schedule_time_t current_time_;
    resource_state_t memory_state_; // state of the memory //
    op_output_table_t op_output_table_; // output of only compute ops //
    active_resource_table_t active_resource_table_;
    const dag_t * input_ptr_;
    size_t total_compute_ops_;
    size_t scheduled_compute_ops_;
    ////////////////////////////////////////////////////////////////////////////

}; // class Feasible_Memory_Schedule_Generator //


// Given a DAG, G=(V,E) and a vertex v\in V returns an iterator over all
// vertices in V which are connected by some path (atleast one colored vertex)
// with only using the colored vertices.
template<typename DagType, typename DagTraits=scheduler_traits<DagType> >
class Color_Connected_Vertices {
  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef DagTraits traits;
    typedef typename traits::dag_t dag_t;
    typedef typename traits::operation_t vertex_t;
    typedef typename traits::operation_hash_t vertex_hash_t;
    typedef typename traits::const_operation_iterator_t const_vertex_iterator_t;
    ////////////////////////////////////////////////////////////////////////////

    Color_Connected_Vertices(const dag_t& input) : input_(input) {}

    template<typename VertexColorClassifier, typename BackInsertIterator>
    void compute_connected_vertices(const vertex_t& v,
        BackInsertIterator output,
        const VertexColorClassifier& vertex_selector=VertexColorClassifier()) {

      std::unordered_set<vertex_t, vertex_hash_t> explored;
      std::list<vertex_t> colored_bfs_list;
      const_vertex_iterator_t citr =
          traits::outgoing_operations_begin(input_, v);
      const_vertex_iterator_t citr_end =
          traits::outgoing_operations_end(input_, v);

      // init //
      for (; citr != citr_end; ++citr) {
        const vertex_t& u = *citr;
        if (vertex_selector(input_, u)) {
          colored_bfs_list.push_back(u);
        }
      }
      explored.insert(v); 

      while (!colored_bfs_list.empty()) {
        vertex_t w = colored_bfs_list.front();
        colored_bfs_list.pop_front();

        citr = traits::outgoing_operations_begin(input_, w);
        citr_end = traits::outgoing_operations_end(input_, w);

        for (; citr != citr_end; ++citr) {
          const vertex_t& u = *citr;

          if (explored.find(u) != explored.end()) { continue; }

          if (vertex_selector(input_, u)) {
            colored_bfs_list.push_back(u);
            explored.insert(u);
          } else {
            output = u;
          }
        }
        explored.insert(w); 
      } // while (!colored_bfs_list.empty() //
      
    }

  private:

    const dag_t& input_;
}; // class Color_Connected_Vertices //


} // namespace lp_scheduler //

} // namespace mv //

#endif
