#ifndef DAG_ADDRESS_GENERATOR_HPP 
#define DAG_ADDRESS_GENERATOR_HPP

#include <cstring>

#include "scheduler/feasible_scheduler.hpp"

namespace mv {
namespace lp_scheduler {

// Given a schedule (ordering of nodes) of a DAG(V,E) and subset of nodes
// $U \subset V$ with a utility function $f : U -> N^+$ this class generates
// address assignments with the following guarantees:
//
// 1. If u_1, u_2 \in U are scheduled at same time then:
//       address_range(u_1) \intersection address_range(u_2) = \Phi (empty)
// 
// 2. Let W = { v | (u, v) \in E } then:
//       address_range(u) is locked time t >= max{ schedule_time(v) | v \in W}
//
// TODO(vamsikku): currently a unit-delay model is assumed, need to use a heap
// to make it work delays > 1unit
template<typename OpDAG, typename Unit, typename OpSelector, typename OpUtility,
    typename DAGTraits=scheduler_traits<OpDAG> >
class DAG_Address_Generator {

  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef DAGTraits traits;
    typedef typename traits::dag_t dag_t;
    typedef typename traits::operation_t operation_t;
    typedef Unit unit_t;
    struct address_info_t {
      address_info_t() : op_(), address_begin_(), address_end_() {}

      address_info_t(operation_t op, unit_t begin, unit_t end)
        : op_(op), address_begin_(begin), address_end_(end) {}

      bool is_disjoint(const address_info_t& o) const {
        return !( std::max(address_begin_, o.address_begin_) <= 
          std::min(address_end_, o.address_end_) );
      }

      bool operator==(const address_info_t& o) const {
        return (op_ == o.op_) && (address_begin_ == o.address_begin_) &&
          (address_end_ == o.address_end_);
      }

      operation_t op_;
      unit_t address_begin_;
      unit_t address_end_;
    }; // struct address_info_t //
    typedef OpSelector op_selector_t; // should select ops with non-zero utility
    typedef OpUtility op_utility_t;
    typedef Producer_Consumer_Contiguous_Resource<unit_t, operation_t>
        resource_state_t;
    typedef typename resource_state_t::interval_info_t interval_info_t;

    struct scheduled_op_t {
      operation_t op_;
      size_t time_;
    }; // struct scheduled_op_t //
    ////////////////////////////////////////////////////////////////////////////

    DAG_Address_Generator(const dag_t& dag, unit_t upper_bound,
        const op_utility_t &utility=op_utility_t(),
        const op_selector_t& selector=op_selector_t())
      : input_dag_ptr_(&dag), rstate_(), high_water_mark_(0UL),
        op_utility_(utility), op_selector_(selector) { init(upper_bound); }

    void init(unit_t upper_bound) {
      rstate_.clear();
      rstate_.initialize_resource_upper_bound(upper_bound);
    }

    template<typename ScheduledOpIterator, typename OutputIterator>
    bool generate(ScheduledOpIterator sched_begin,
        ScheduledOpIterator sched_end, OutputIterator output) {
      assert(sched_begin != sched_end);

      size_t current_time = traits::scheduled_op_time(*sched_begin);
      std::list<scheduled_op_t> ops_scheduled_currently;

      for (; sched_begin != sched_end; ++sched_begin) {
        scheduled_op_t sop;
        sop.op_ = traits::scheduled_op(*sched_begin);
        sop.time_ = traits::scheduled_op_time(*sched_begin);


        if (sop.time_ != current_time) {
          // unassign all resources for completed ops in the previous time step
          for (auto pitr=ops_scheduled_currently.begin();
                pitr != ops_scheduled_currently.end(); ++pitr) {
            bool status = rstate_.unassign_resources(pitr->op_);
            assert(status);
          }
          current_time = sop.time_;
          ops_scheduled_currently.clear();
        }
        ops_scheduled_currently.push_back(sop);

        const operation_t& op = sop.op_;

        if (op_selector_(op)) {
          unit_t op_utility = op_utility_(op);
          if (!rstate_.is_resource_available(op_utility)) { return false; }
          bool assign = rstate_.assign_resources( op, op_utility,
              traits::outgoing_operations_begin(*input_dag_ptr_, op),
              traits::outgoing_operations_end(*input_dag_ptr_, op) );
          assert(assign);
          interval_info_t interval = rstate_.get_resource_usage_info(op);
          output = address_info_t(op, interval.begin_, interval.end_);
          high_water_mark_ = std::max(high_water_mark_, interval.end_);
        }
      }

      if (!ops_scheduled_currently.empty()) {
        // unassign all resources for completed ops in the previous time step/
        for (auto pitr=ops_scheduled_currently.begin();
              pitr != ops_scheduled_currently.end(); ++pitr) {
          rstate_.unassign_resources(pitr->op_);
        }
        ops_scheduled_currently.clear();
      } 

      return true;
    }

    unit_t get_high_watermark() const { return high_water_mark_; }

  private:
    const dag_t *input_dag_ptr_;
    resource_state_t rstate_;
    unit_t high_water_mark_;
    op_utility_t op_utility_;
    op_selector_t op_selector_;
}; // class DAG_Address_Generator //


} // namesapce lp_scheduler //
} // namespace mv //

#endif
