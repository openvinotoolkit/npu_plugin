#ifndef CONTROL_EDGE_GENERATOR_HPP
#define CONTROL_EDGE_GENERATOR_HPP

#include "scheduler/disjoint_interval_set.hpp"


namespace mv {
namespace pass {

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
      bool status = false;
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
          status = interval_tree_.insert(inter_beg, inter_end, curr_interval);
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
        status = interval_tree_.insert(curr_rem_beg, curr_rem_end,
              curr_interval);
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


} // namespace pass //
} // namespace mv  //


#endif
