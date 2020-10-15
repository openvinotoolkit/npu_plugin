#ifndef BARRIER_SCHEDULE_GENERATOR_H
#define BARRIER_SCHEDULE_GENERATOR_H

#include <algorithm>
#include <cassert>
#include <set>
#include <vector>

#include <scheduler/feasible_scheduler.hpp>

namespace mv {
namespace lp_scheduler {

class Barrier_Resource_State {
  public:

    ////////////////////////////////////////////////////////////////////////////
    typedef size_t slots_t;
    typedef size_t barrier_t;

    struct available_slot_key_t {

      available_slot_key_t(slots_t slots=slots_t(0UL),
            barrier_t barrier=barrier_t(0UL))
        : available_slots_(slots), total_slots_(slots), barrier_(barrier) {}

      bool operator<(const available_slot_key_t& o) const {
        return (o.available_slots_ != available_slots_) ?
          (available_slots_ < o.available_slots_) : (barrier_ < o.barrier_);
      }

      bool barrier_in_use() const { return total_slots_ > available_slots_; }

      slots_t available_slots_;
      const slots_t total_slots_;
      barrier_t barrier_;
    };  // struct available_slot_key_t //

    typedef std::set<available_slot_key_t> available_slots_t;
    typedef typename available_slots_t::const_iterator
        const_available_slots_iterator_t;
    typedef typename available_slots_t::iterator available_slots_iterator_t;
    typedef std::vector<available_slots_iterator_t> barrier_reference_t;
    ////////////////////////////////////////////////////////////////////////////


    Barrier_Resource_State() : barrier_reference_(), available_slots_() {}
    Barrier_Resource_State(size_t barrier_count, size_t slot_count)
      : barrier_reference_(), available_slots_() {
        init(barrier_count, slot_count);
    }
   
    void init(size_t bcount, slots_t slot_count) {
      assert(bcount && slot_count);
      available_slots_.clear();
      barrier_reference_.clear();

      // amortized O(1) insert cost //
      available_slots_iterator_t hint = available_slots_.end();
      for (size_t bid=1UL; bid <= bcount; bid++) {
        hint = available_slots_.insert(hint,
              available_slot_key_t(slot_count, barrier_t(bid)));
        barrier_reference_.push_back(hint);
      }
    }

    bool has_barrier_with_slots(slots_t slot_demand) const {
      available_slot_key_t key(slot_demand);
      const_available_slots_iterator_t itr = available_slots_.lower_bound(key);
      const_available_slots_iterator_t ret_itr = itr;

      // prefer a unused barrier to satisfy this slot demand //
      for(; (itr != available_slots_.end()) && (itr->barrier_in_use());
          ++itr) { }
      if (itr != available_slots_.end()) { ret_itr = itr; };

      return ret_itr != available_slots_.end();
    }

    
    // Precondition: has_barrier_with_slots(slot_demand) is true //
    barrier_t assign_slots(slots_t slot_demand) {
      available_slot_key_t key(slot_demand);
      available_slots_iterator_t itr = available_slots_.lower_bound(key);
      {
        available_slots_iterator_t ret_itr = itr;

        for(; (itr != available_slots_.end()) && (itr->barrier_in_use());
            ++itr) { }
        if (itr != available_slots_.end()) { ret_itr = itr; };
        itr = ret_itr;
      }

      if ((itr == available_slots_.end()) ||
            (itr->available_slots_ < slot_demand)) {
        return invalid_barrier();
      }

      barrier_t bid = itr->barrier_;
      return assign_slots(bid, slot_demand) ? bid : invalid_barrier();
    }

    bool assign_slots(barrier_t bid, slots_t slot_demand) {
      assert((bid <= barrier_reference_.size()) && (bid>=1UL));
      available_slots_iterator_t itr = barrier_reference_[bid-1UL];
      assert((itr->available_slots_) >= slot_demand);
      slots_t new_slot_demand = (itr->available_slots_) - slot_demand;
     
      itr = update(itr, new_slot_demand);
      return (itr != available_slots_.end());
    }

    bool unassign_slots(barrier_t bid, slots_t slot_demand) {
      assert((bid <= barrier_reference_.size()) && (bid>=1UL));
      available_slots_iterator_t itr = barrier_reference_[bid-1UL];
      slots_t new_slot_demand = (itr->available_slots_) + slot_demand;
     
      itr = update(itr, new_slot_demand);
      return (itr != available_slots_.end());
    }

    static barrier_t invalid_barrier() {
      return std::numeric_limits<barrier_t>::max();
    }

  private:

    //NOTE: will also update barrier_reference_ //
    void update(barrier_t bid, slots_t new_slots_value) {
      assert((bid <= barrier_reference_.size()) && (bid>=1UL));
      available_slots_iterator_t itr = barrier_reference_[bid-1UL];
      update(itr, new_slots_value);
    }

    available_slots_iterator_t update(available_slots_iterator_t itr,
          slots_t new_slots_value) {
      assert(itr != available_slots_.end());

      available_slot_key_t key = *itr;
      key.available_slots_ = new_slots_value;
      available_slots_.erase(itr);

      itr = (available_slots_.insert(key)).first;
      assert(itr != available_slots_.end());
      barrier_reference_[(itr->barrier_)-1UL] = itr;
      return itr;
    }

    barrier_reference_t barrier_reference_;
    available_slots_t available_slots_;
}; // class Barrier_Resource_State //


template<typename OpDag>
class Barrier_Schedule_Generator {
  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef OpDag dag_t;
    typedef scheduler_traits<dag_t> dag_traits;
    typedef typename dag_traits::operation_t operation_t;
    typedef typename dag_traits::const_operation_iterator_t
        const_operation_iterator_t;
    typedef typename dag_traits::operation_hash_t operation_hash_t;
    typedef typename dag_traits::resource_t resource_t;
    typedef typename dag_traits::schedule_time_t schedule_time_t;
    struct barrier_info_t {
      barrier_info_t(size_t bindex=0UL, size_t slot_count=0UL)
        : bindex_(bindex), slot_count_(slot_count) {}
      size_t bindex_;
      size_t slot_count_;
    }; // struct barrier_info_t //

    typedef std::unordered_map<operation_t, barrier_info_t, operation_hash_t>
        active_barrier_map_t;
    typedef typename active_barrier_map_t::const_iterator
        const_barrier_map_iterator_t;

    struct op_resource_state_t {

      op_resource_state_t(size_t n=0UL, size_t m=0UL)
        : barrier_map_(), state_(), barrier_count_(n), slots_per_barrier_(m) {}

      void init(const op_resource_state_t& other) {
        barrier_map_.clear();
        barrier_count_ = other.barrier_count_;
        slots_per_barrier_ = other.slots_per_barrier_;
        state_.init(barrier_count_, slots_per_barrier_);
      }

      bool is_resource_available(const resource_t& demand) const {
        return state_.has_barrier_with_slots(demand);
      }

      bool schedule_operation(const operation_t& op, resource_t& demand) {
        assert(is_resource_available(demand));
        if (barrier_map_.find(op) != barrier_map_.end()) { return false; }
        size_t bid = state_.assign_slots(demand);
        barrier_map_.insert(std::make_pair(op, barrier_info_t(bid, demand)));
        return true;
      }

      bool unschedule_operation(const operation_t& op) {
        const_barrier_map_iterator_t itr = barrier_map_.find(op);
        if (itr == barrier_map_.end()) { return false; }
        const barrier_info_t& binfo = itr->second;
        bool ret = state_.unassign_slots(binfo.bindex_, binfo.slot_count_);
        assert(ret);
        barrier_map_.erase(itr);
        return true;
      }

      const barrier_info_t& get_barrier_info(const operation_t& op) const {
        const_barrier_map_iterator_t itr = barrier_map_.find(op);
        assert(itr != barrier_map_.end());
        return itr->second;
      }

      active_barrier_map_t barrier_map_;
      Barrier_Resource_State state_;
      size_t barrier_count_;
      size_t slots_per_barrier_;
    }; // struct op_resource_state_t;
    typedef op_resource_state_t resource_state_t;

    ////////////// scheduler_traits : specialization ///////////////////////////
    struct barrier_scheduler_traits : public dag_traits {
      using dag_traits::dag_traits;

      typedef op_resource_state_t resource_state_t;

      static void initialize_resource_state(const resource_state_t& start_state,
          resource_state_t& state) {
        state.init(start_state);
      }

      static bool is_resource_available(const resource_t& demand,
            const resource_state_t& state) {
        return state.is_resource_available(demand);
      }

      static bool schedule_operation(const operation_t& op, resource_t demand,
          resource_state_t& state) {
        return state.schedule_operation(op, demand);
      }

      static bool schedule_operation(const operation_t& op, resource_t demand,
          resource_state_t& rstate,
          const_operation_iterator_t, const_operation_iterator_t) {
        return schedule_operation(op, demand, rstate);
      }


      static bool unschedule_operation(const operation_t& op,
          resource_state_t& rstate) {
        return rstate.unschedule_operation(op);
      }
    }; // struct barrier_scheduler_traits //

    struct schedule_info_t {
      schedule_time_t schedule_time_;
      operation_t op_;
      size_t barrier_index_;
      size_t slot_count_;
    }; // struct schedule_info_t //

    typedef Feasible_Schedule_Generator<dag_t, barrier_scheduler_traits>
        scheduler_t;
    ////////////////////////////////////////////////////////////////////////////

    Barrier_Schedule_Generator(const dag_t& input_dag, size_t n, size_t m=1UL)
      : barrier_count_(n), slots_per_barrier_(m), start_state_(n,m),
        scheduler_begin_(input_dag, start_state_), scheduler_end_(), sinfo_() {}

    Barrier_Schedule_Generator() : barrier_count_(0UL), slots_per_barrier_(0UL),
      start_state_(), scheduler_begin_(),
      scheduler_end_(), sinfo_() {}

    bool operator==(const Barrier_Schedule_Generator& o) const {
      return reached_end() && o.reached_end();
    }

    bool operator!=(const Barrier_Schedule_Generator& o) const {
      return !(*this == o);
    }

    void operator++() { ++scheduler_begin_; }

    const schedule_info_t&  operator*(void) const {
      sinfo_.op_ = *scheduler_begin_;
      sinfo_.schedule_time_ = scheduler_begin_.current_time();
      const resource_state_t& rstate = scheduler_begin_.resource_state();
      const barrier_info_t& binfo = rstate.get_barrier_info(sinfo_.op_);
      sinfo_.barrier_index_ = binfo.bindex_;
      sinfo_.slot_count_ = binfo.slot_count_;
      return sinfo_;
    }

  private:

    bool reached_end() const { return scheduler_begin_ == scheduler_end_; }

    dag_t const *input_dag_;
    size_t barrier_count_;
    size_t slots_per_barrier_;
    const resource_state_t start_state_;
    scheduler_t scheduler_begin_;
    scheduler_t scheduler_end_;
    mutable schedule_info_t sinfo_;
}; // class Barrier_Schedule_Generator //



} // namespace lp_scheduler //
} // namespace mv //





#endif
