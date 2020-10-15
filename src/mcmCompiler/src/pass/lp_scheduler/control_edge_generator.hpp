#ifndef CONTROL_EDGE_GENERATOR_HPP
#define CONTROL_EDGE_GENERATOR_HPP

#include "scheduler/disjoint_interval_set.hpp"
#include "scheduler/feasible_scheduler.hpp"
#include "include/mcm/utils/warning_manager.hpp"

namespace mv {
namespace lp_scheduler {

// Given an iterator over sorted intervals the algorithm produces control
// edges which are defined in the following document:
// http://goto.intel.com/control-edge-generation-algo
template<typename T>
class Control_Edge_Generator {

  public:
    typedef mv::lp_scheduler::interval_traits<T> traits;
    typedef typename traits::unit_t unit_t;
    typedef typename traits::interval_t interval_t;
    typedef mv::lp_scheduler::Interval_Utils<interval_t> interval_utils_t;
    typedef mv::lp_scheduler::Disjoint_Interval_Set<unit_t, interval_t>
        interval_tree_t;
    typedef typename interval_tree_t::interval_iterator_t query_iterator_t;
    struct noop_functor_t {
      void operator()(const interval_t&, const interval_t&) {}
    }; // struct noop_functor_t //

    Control_Edge_Generator() : interval_tree_() {}


    template<typename IntervalIterator, typename OutputFunctor=noop_functor_t>
    size_t generate_control_edges(IntervalIterator begin, IntervalIterator end,
        OutputFunctor &output=OutputFunctor()) {
      size_t edge_count = 0UL;
      for (;begin != end; ++begin) {
        unit_t curr_beg = traits::interval_begin(*begin);
        unit_t curr_end = traits::interval_end(*begin);
        if (curr_beg > curr_end) { continue; }
        edge_count += process_next_interval(*begin, output);
      }
      return edge_count;
    }

  protected:

    bool overlaps(const query_iterator_t& qitr,
          unit_t ibegin, unit_t iend) const {
      return interval_utils_t::intersects(
          qitr.interval_begin(), qitr.interval_end(), ibegin, iend);
    }

    template<typename OutputFunctor=noop_functor_t>
    size_t process_next_interval(const interval_t& curr_interval,
        OutputFunctor &output=OutputFunctor()) {

      size_t edge_count = 0UL;
      unit_t curr_beg = traits::interval_begin(curr_interval);
      unit_t curr_end = traits::interval_end(curr_interval);
      assert(curr_beg <= curr_end);

      query_iterator_t qitr = interval_tree_.query(curr_beg, curr_end);
      query_iterator_t qitr_end = interval_tree_.end(), qitr_next = qitr_end;

      // Invariant: [curr_rem_beg, curr_rem_end] is the reminder of the
      // current interval which does not overlap intervals until qitr //
      unit_t curr_rem_beg = curr_beg, curr_rem_end = curr_end;
      while ((qitr != qitr_end) &&
            overlaps(qitr, curr_rem_beg, curr_rem_end)) { // foreach overlap //
        interval_t qinterval = *qitr;

        assert((curr_rem_beg <= curr_rem_end) && (curr_beg <= curr_rem_beg) &&
          (curr_rem_end <= curr_end));

        // output the control edge //
        output(qinterval, curr_interval);
        ++edge_count;

        // we now have an overlap between [qbeg,qend] and
        // [curr_rem_beg,curr_rem_end] //

        unit_t qbeg = qitr.interval_begin();
        unit_t qend = qitr.interval_end();

        // erase the current interval //
        qitr_next = qitr; ++qitr_next;
        interval_tree_.erase(qbeg, qend);
       
        // compute the intersecting interval //
        unit_t inter_beg = std::max(curr_rem_beg, qbeg);
        unit_t inter_end = std::min(curr_rem_end, qend);
        assert(inter_beg <= inter_end);

        unit_t result_beg[2UL], result_end[2UL];
        interval_t const * result_val[2UL];
        size_t rcount;

        // now compute the xor interval(s) //
        rcount = interval_utils_t::interval_xor(qbeg, qend,
            curr_rem_beg, curr_rem_end, result_beg, result_end);

        unit_t next_rem_beg = std::numeric_limits<unit_t>::max();
        unit_t next_rem_end = std::numeric_limits<unit_t>::min();
        assert(rcount <= 2UL);

        if (!rcount) {
          // no xor intervals //
          const bool status = interval_tree_.insert(inter_beg, inter_end, curr_interval);
          UNUSED(status);
          assert(status);
        } else {
          // NOTE: that first XOR interval comes before the second if there
          // are two XOR intervals. //

          // Determine the type of interval in the xor //
          // 
          //qbeg     qend
          // [--------]
          //        [----------------]
          //      curr_rem_beg       curr_rem_end
          //
          // the xor has atmost two parts:
          // [------)
          //          (--------------]
          for (size_t r=0; r<rcount; r++) {
            if (interval_utils_t::is_subset(result_beg[r], result_end[r],
                    curr_rem_beg, curr_rem_end)) {
              result_val[r] = &curr_interval;
            } else {
              assert(interval_utils_t::is_subset(result_beg[r], result_end[r],
                      qbeg, qend) );
              result_val[r] = &qinterval;
            }
          }

          size_t next_xor_interval_index = 0;

          if (result_end[next_xor_interval_index] < inter_beg) {
            // insert the interval part before [inter_beg, inter_end] //
            interval_tree_.insert(result_beg[next_xor_interval_index],
                result_end[next_xor_interval_index],
                *(result_val[next_xor_interval_index]) );
            ++next_xor_interval_index;
          }

          // insert the intersection part //
          interval_tree_.insert(inter_beg, inter_end, curr_interval);


          // process the interval above the intersection part //
          if (next_xor_interval_index < rcount) {
            if (result_val[next_xor_interval_index] != &curr_interval) {
              // need to insert this part back in the interval tree //
              interval_tree_.insert(result_beg[next_xor_interval_index],
                  result_end[next_xor_interval_index],
                  *(result_val[next_xor_interval_index]) );
            } else {
              // pass the remaining part of the curr_interval to the next
              // iteraton.
              next_rem_beg = result_beg[next_xor_interval_index];
              next_rem_end = result_end[next_xor_interval_index];
            }
          }
        }

        // update the remaining part of the current interval //
        curr_rem_beg = next_rem_beg;
        curr_rem_end = next_rem_end;
        qitr = qitr_next;
      } // foreach overlap //

      if (curr_rem_beg <= curr_rem_end) {
        // process the trailing part this also covers the trailing case.//
        const bool status = interval_tree_.insert(curr_rem_beg, curr_rem_end,
              curr_interval);
        UNUSED(status);
        assert(status);
      }

      //TODO(vamsikku): do the merging within the update itself //
      merge_abutting_intervals(curr_interval);
      return edge_count;
    }

    void merge_abutting_intervals(const interval_t& curr_interval) {
      unit_t ibeg = traits::interval_begin(curr_interval);
      unit_t iend = traits::interval_end(curr_interval);
      query_iterator_t qitr = interval_tree_.query(ibeg, iend);
      query_iterator_t qitr_end = interval_tree_.end(), qitr_next, qitr_start;

      if ((qitr == qitr_end) || !(*qitr == curr_interval)) { return; }

      qitr_start = qitr;
      unit_t prev_left_end = qitr.interval_begin();
      unit_t prev_right_end = qitr.interval_end();

      ++qitr;
      while ( (qitr != qitr_end) && ((*qitr == curr_interval) &&
                ((prev_right_end+1) == qitr.interval_begin()) ) ) {
        prev_right_end = qitr.interval_end();
        //TODO(vamsikku): implement an erase using iterators instead of
        //end point values so that this takes O(1) time.
        qitr_next = qitr; ++qitr_next;
        interval_tree_.erase(qitr.interval_begin(), qitr.interval_end());
        qitr = qitr_next;
      }


      if (prev_right_end > qitr_start.interval_end()) {
        interval_tree_.erase(qitr_start.interval_begin(),
              qitr_start.interval_end());
        interval_tree_.insert(prev_left_end, prev_right_end, curr_interval);
      }
    }

    interval_tree_t interval_tree_;
}; // class Control_Edge_Generator //


// Common control edge generation code for both DDR and CMX schedules. The
// call must pass appropriate ScheduleOpIterator which returns valid intervals
// depends on the resource which gets engaged.
//
// NOTE: OpSelector : should return false if the op need not have a memory
// control edge.
template<typename OpDag, typename OpSelector,
    typename SchedulerTraits=mv::lp_scheduler::scheduler_traits<OpDag> >
class Memory_Control_Edge_Generator {
  public:

  //////////////////////////////////////////////////////////////////////////////
    typedef OpDag dag_t;
    typedef SchedulerTraits traits;
    typedef OpSelector op_selector_t;
    typedef typename traits::operation_t operation_t;
    typedef typename traits::scheduled_op_t scheduled_op_t;
    typedef typename traits::resource_t resource_t;
    typedef typename traits::schedule_time_t schedule_time_t;
    typedef mv::lp_scheduler::Control_Edge_Generator<scheduled_op_t>
        memory_overlap_control_edge_generation_algo_t;
    typedef std::unordered_map<operation_t, scheduled_op_t>
        original_schedule_map_t;

    struct memory_control_edge_t {
      operation_t source_;
      operation_t sink_;
    }; // struct memory_control_edge_t //

    typedef std::list<memory_control_edge_t> overlap_edge_list_t;
    typedef std::back_insert_iterator<overlap_edge_list_t>
        overlap_edge_list_back_insert_iterator_t;

    struct noop_back_inserter_t {
      void operator()(const scheduled_op_t&, const scheduled_op_t&) {}
    }; // struct noop_back_inserter_t //

    template<typename BackInsertIterator=noop_back_inserter_t>
    struct memory_control_edge_extractor_t {
     
      memory_control_edge_extractor_t() : back_insert_iterator_(), medge_() {}
      memory_control_edge_extractor_t(BackInsertIterator itr)
        : back_insert_iterator_(itr), medge_() {}

      void operator()(const scheduled_op_t& source,
          const scheduled_op_t& sink) {
        medge_.source_ = traits::scheduled_operation(source);
        medge_.sink_ = traits::scheduled_operation(sink);
        *back_insert_iterator_ =  medge_;
      }

      BackInsertIterator back_insert_iterator_;
      memory_control_edge_t medge_;
    }; // struct memory_control_edge_extractor_t //

  //////////////////////////////////////////////////////////////////////////////


    template<typename ScheduledOpIterator, typename BackInsertIterator>
    void generate(const dag_t& input_dag,
        ScheduledOpIterator sbegin, ScheduledOpIterator send,
        BackInsertIterator output) {

      static_assert(std::is_same<typename ScheduledOpIterator::value_type,
          scheduled_op_t>::value, "Invalid ScheduledOpIterator");

      op_selector_t is_selectable_op;
      original_schedule_map_t original_schedule;

      // STEP-0: collect the overlap edge list based on the projection
      // algorithm.
      overlap_edge_list_t overlap_edges;
      memory_control_edge_extractor_t<overlap_edge_list_back_insert_iterator_t>
        extractor(std::back_inserter(overlap_edges));

      //STEP-1: generate the control edges based on the overlap in the
      //memory address space.
      memory_overlap_control_edge_generation_algo_t algo;
      algo.generate_control_edges(sbegin, send, extractor);

      // STEP-2: save the schedule //
      for (; sbegin != send; ++sbegin) {
        const scheduled_op_t& sop = *sbegin;
        assert(original_schedule.find(sop.op_) == original_schedule.end());
        original_schedule.insert(std::make_pair(sop.op_, sop));
      }


      // STEP-3: for each overlap control edge generated (u,v) do the following
      //  (a) call add_control_edge(u, v)
      //  (b) let t_u and t_v be the schedule times of u and v then 
      //      for all the nodes X = { w | (u, w) \in E and t_u <= t_w < t_v }
      //      call add_control_edge(x, v) , x \in X.
      memory_control_edge_t medge;
      for (auto eitr=overlap_edges.begin(); eitr!=overlap_edges.end(); ++eitr) {
        // control model node iterators //
        operation_t source_op = eitr->source_;
        operation_t sink_op = eitr->sink_;

        if (!is_selectable_op(input_dag, source_op) ||
            !is_selectable_op(input_dag, sink_op)) { continue; }

        medge.source_ = source_op; medge.sink_ = sink_op;
        *output = medge; // (a) //

        schedule_time_t source_time, sink_time;
        // source time //
        typename original_schedule_map_t::const_iterator itr;
        itr = original_schedule.find(source_op);
        assert(itr != original_schedule.end());
        source_time = (itr->second).schedule_time_;

        // sink time //
        itr = original_schedule.find(sink_op);
        assert(itr != original_schedule.end());
        sink_time = (itr->second).schedule_time_;

        medge.sink_ = sink_op;
        typename dag_t::const_operation_iterator_t cop_itr, cop_itr_end;

        cop_itr = traits::outgoing_operations_begin(input_dag, source_op);
        cop_itr_end = traits::outgoing_operations_end(input_dag, source_op);

        for (; cop_itr != cop_itr_end; ++cop_itr) {
          const operation_t& child_op = *(cop_itr);
          itr = original_schedule.find(child_op);
          if (itr == original_schedule.end()) { continue; }
          schedule_time_t child_time = (itr->second).schedule_time_;

          if ( !( (child_time > source_time) && 
                  (child_time < sink_time) ) ) { continue; }

          if (!is_selectable_op(input_dag, child_op)){ continue; }
          medge.source_ = child_op;
          *output = medge; // (b) //
        }
      }

    }

  private:

    bool does_this_op_use_resources(const dag_t& dag,
          const operation_t& op) const {
      return traits::is_empty_demand(traits::resource_utility(dag, op));
    }
}; // class Memory_Control_Edge_Generator //








} // namespace lp_scheduler//
} // namespace mv  //


#endif
