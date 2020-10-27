#ifndef BARRIER_SIMULATOR_H
#define BARRIER_SIMULATOR_H

#include <list>
#include <map>
#include <unordered_map>
#include <utility>
#include "include/mcm/utils/warning_manager.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"

namespace mv {
namespace lp_scheduler { 

//
// Runs the runtime barrier simulator and adjusts the control model to unclog
// the barriers which are holding on resources due to consumer count not yet
// become zero. This pass can be removed once the runtime is resigned to not
// use consumer count.
//
// Precondition: an input DAG with maximum independent set size on its closure
// bounded by 'n'.
//
// NOTE: if we have 'n' real barriers then the barrier scheduler is only run
// as if 'n/2' barriers are available. Also the input DAG must have interleaved
// barrier and non-barrier ops //
//
// BarrierOpSelector: bool operator()(const dag_t&, const operation_t&)
// RealBarrierMapper: size_t operator()(const dag_t&, const operation_t&)
//
template<typename OpDAG, typename OpTypeSelector, typename RealBarrierMapper,
    typename DAGTraits=scheduler_traits<OpDAG> >
class Runtime_Barrier_Simulation_Checker {
  public:
  //////////////////////////////////////////////////////////////////////////////
    typedef DAGTraits traits;
    typedef OpTypeSelector op_type_selector_t;
    typedef RealBarrierMapper real_barrier_map_t;
    typedef typename traits::dag_t dag_t;
    typedef typename traits::operation_t operation_t;
    typedef typename traits::const_operation_iterator_t
        const_operation_iterator_t;
    typedef std::list<operation_t> op_list_t;
    typedef typename op_list_t::iterator op_list_iterator_t;
    typedef std::map<size_t, op_list_t> level_sets_t;
    typedef typename level_sets_t::iterator level_sets_iterator_t;
    typedef std::unordered_map<operation_t, size_t> in_degree_map_t;
    typedef std::unordered_map<operation_t, size_t> out_degree_map_t;
    typedef typename in_degree_map_t::iterator in_degree_iterator_t;
    typedef typename out_degree_map_t::iterator out_degree_iterator_t;

    enum edge_edit_e {REMOVE_EDGE=0, ADD_EDGE=1};
    struct edge_edits_t {
      operation_t source_;
      operation_t sink_;
    };

    struct active_barrier_info_t {
      size_t real_barrier_;
      size_t in_degree_;
      size_t out_degree_;
      active_barrier_info_t(size_t real, size_t in, size_t out)
        : real_barrier_(real), in_degree_(in), out_degree_(out) {}
    };
    typedef std::unordered_map<operation_t, active_barrier_info_t>
        active_barrier_table_t;
    typedef typename active_barrier_table_t::iterator
        active_barrier_table_iterator_t;
  //////////////////////////////////////////////////////////////////////////////

    Runtime_Barrier_Simulation_Checker(const dag_t& input,
        size_t barrier_bound)
      : input_ptr_(&input), level_sets_(), in_degree_map_(),
        real_barrier_list_(), barrier_bound_(barrier_bound),
        active_barrier_table_() {}

    Runtime_Barrier_Simulation_Checker()
      : input_ptr_(NULL), level_sets_(), in_degree_map_(), real_barrier_list_(),
      barrier_bound_(), active_barrier_table_() {}

    bool check() {
      init();
      build_level_sets();

      op_list_t barrier_list, compute_list, data_list;

      get_lists_from_level_sets(barrier_list, data_list, compute_list);

      bool filled_atleast_one = false;

      while (!data_list.empty() || !compute_list.empty()
            || !barrier_list.empty()) {
        filled_atleast_one = false;
        filled_atleast_one |= fill_barrier_tasks(barrier_list);
        filled_atleast_one |= process_tasks(data_list);
        filled_atleast_one |= process_tasks(compute_list);
        if (!filled_atleast_one) { return false; }
      }
      return true;
    }

  private:

    bool fill_barrier_tasks(op_list_t& barrier_task_list ) {
      active_barrier_table_iterator_t aitr;
      bool filled_atleast_once = false;
      op_list_iterator_t bcurr = barrier_task_list.begin(),
                         bend = barrier_task_list.end(), berase;

      while ( (bcurr != bend) && !real_barrier_list_.empty() ) {
        // atleast one barrier tasks and atleast one real barrier //
        operation_t bop = *bcurr;
        acquire_real_barrier(bop);
        filled_atleast_once = true;
        berase = bcurr; ++bcurr;
        barrier_task_list.erase(berase);
      }
      return filled_atleast_once;
    }

    bool is_task_ready(const operation_t& task) {
      const dag_t& dag = *input_ptr_;
      assert(!op_type_selector_t::is_barrier_op(dag, task));

      op_list_t wait_barriers, update_barriers;

      // wait barriers //
      for (const_operation_iterator_t
          pitr=traits::incoming_operations_begin(dag, task);
          pitr!=traits::incoming_operations_end(dag, task); ++pitr) {

        operation_t barrier_op = *pitr;
        assert(op_type_selector_t::is_barrier_op(dag, barrier_op));
        active_barrier_table_iterator_t aitr =
            active_barrier_table_.find(barrier_op);

        if ( (aitr == active_barrier_table_.end()) ||
              ((aitr->second).in_degree_ > 0) ) { return false; }
      }

      // update barriers //
      for (const_operation_iterator_t
          citr=traits::outgoing_operations_begin(dag, task);
          citr!=traits::outgoing_operations_end(dag, task); ++citr) {
        operation_t barrier_op = *citr;
        assert(op_type_selector_t::is_barrier_op(dag, barrier_op));
        if (active_barrier_table_.find(barrier_op) == 
            active_barrier_table_.end()) { return false; }
      }
      return true;
    }

    void process_task(const operation_t& task) {
      assert( is_task_ready(task) );

      const dag_t& dag = *input_ptr_;
      active_barrier_table_iterator_t aitr;

      // wait barriers //
      for (const_operation_iterator_t
          pitr=traits::incoming_operations_begin(dag, task);
          pitr!=traits::incoming_operations_end(dag, task); ++pitr) {
        operation_t barrier_op = *pitr;
        assert(op_type_selector_t::is_barrier_op(dag, barrier_op));

        aitr = active_barrier_table_.find(barrier_op);
        assert(aitr != active_barrier_table_.end());

        active_barrier_info_t &barrier_info = aitr->second;
        assert(barrier_info.in_degree_ == 0UL);
        assert(barrier_info.out_degree_ > 0UL);
        barrier_info.out_degree_--;

        if (barrier_info.out_degree_ == 0UL) {
          // return the barrier //
          return_real_barrier(barrier_op);
        }
      }

      // update barriers //
      for (const_operation_iterator_t
          citr=traits::outgoing_operations_begin(dag, task);
          citr!=traits::outgoing_operations_end(dag, task); ++citr) {
        operation_t barrier_op = *citr;
        assert(op_type_selector_t::is_barrier_op(dag, barrier_op));

        aitr = active_barrier_table_.find(barrier_op);
        assert(aitr != active_barrier_table_.end());

        active_barrier_info_t &barrier_info = aitr->second;
        assert(barrier_info.in_degree_ > 0UL);
        barrier_info.in_degree_--;
      }
    }


    bool process_tasks(op_list_t& task_list) {
      op_list_iterator_t tbegin = task_list.begin(),
                         tend = task_list.end(), terase;
      bool filled_atleast_once = false;

      while (tbegin != tend) {
        operation_t op = *tbegin;
        if (!is_task_ready(op) ) { break; }
        process_task(op);
        filled_atleast_once = true;
        terase = tbegin;
        ++tbegin;
        task_list.erase(terase);
      }
      return filled_atleast_once;
    }

    void get_lists_from_level_sets(op_list_t &barrier_list,
        op_list_t &data_list, op_list_t &compute_list) const {
      const dag_t &dag = *input_ptr_;

      size_t level=0;
      barrier_list.clear();
      data_list.clear();
      compute_list.clear();

      for (auto itr=level_sets_.begin(); itr!=level_sets_.end();
            ++itr, ++level) {
        const op_list_t& op_list = itr->second;
        for (auto olitr=op_list.begin(); olitr!=op_list.end(); olitr++) {
          operation_t op = *olitr;
          if (op_type_selector_t::is_barrier_op(dag, op)) {
            barrier_list.push_back(op);
          } else if (op_type_selector_t::is_data_op(dag, op)) {
            data_list.push_back(op);
          } else {
            compute_list.push_back(op);
          }
        }
      }
    }

    // acquires a real barrier for the input barrier task //
    void acquire_real_barrier(operation_t btask) {
      assert(!real_barrier_list_.empty());
      size_t real = real_barrier_list_.front();
      real_barrier_list_.pop_front();

      assert(active_barrier_table_.size() < (2*barrier_bound_) );

      in_degree_iterator_t in_itr = in_degree_map_.find(btask);
      out_degree_iterator_t out_itr = out_degree_map_.find(btask);

      assert((in_itr != in_degree_map_.end()) && 
            (out_itr != out_degree_map_.end()));

      assert(active_barrier_table_.find(btask) == active_barrier_table_.end());

      active_barrier_table_.insert(std::make_pair(btask,
            active_barrier_info_t(real, in_itr->second, out_itr->second)));
    }

    // returns a real barrier associated with the task //
    void return_real_barrier(operation_t btask) {
      active_barrier_table_iterator_t aitr = active_barrier_table_.find(btask);
      assert(aitr != active_barrier_table_.end());
      assert(((aitr->second).in_degree_==0UL) && 
            ((aitr->second).out_degree_==0UL));

      assert(real_barrier_list_.size() < (2*barrier_bound_));
      active_barrier_info_t &abinfo = aitr->second;
      real_barrier_list_.push_back(abinfo.real_barrier_);
      assert(real_barrier_list_.size() <= (2*barrier_bound_));
      active_barrier_table_.erase(aitr);
    }


    void init() {
      real_barrier_list_.clear();

      // 2n barriers //
      for (size_t i=0; i<2*barrier_bound_; i++) {
        real_barrier_list_.push_back(i);
      }
    }

    void build_level_sets() {
      level_sets_.clear();
      build_degree_table();

      in_degree_map_t in_degree_map = in_degree_map_;

      size_t level = 0UL;

      level_sets_iterator_t litr =
        (level_sets_.insert(std::make_pair(0UL, op_list_t()))).first;
      op_list_t *curr_level = &(litr->second), *next_level = NULL;
      in_degree_iterator_t in_itr;
      operation_t curr_op, next_op;
      const dag_t& dag = *input_ptr_;

      // initialize level 0 //
      for (auto itr=in_degree_map.begin(); itr!=in_degree_map.end(); ++itr) {
        if (itr->second == 0) {
          curr_level->push_back(itr->first);
        }
      }

      while (!(curr_level->empty())) {
        litr = (level_sets_.insert(std::make_pair(++level, op_list_t()))).first;
        next_level = &(litr->second);

        for (auto oitr=curr_level->begin(); oitr!=curr_level->end(); ++oitr) {
          curr_op = *oitr;
          for (const_operation_iterator_t
                citr=traits::outgoing_operations_begin(dag, curr_op); 
                citr!=traits::outgoing_operations_end(dag, curr_op); ++citr) {
            next_op = *citr;
            in_itr = in_degree_map.find( next_op );
            assert((in_itr != in_degree_map.end()) && (in_itr->second >= 1UL));
            in_itr->second--;
            if (!(in_itr->second) ) {
              next_level->push_back(in_itr->first);
              in_degree_map.erase(in_itr);
            }
          }
        }
        curr_level = next_level;
      }

      // remove the last level if its empty //
      litr = level_sets_.find(level);
      if (litr == level_sets_.end())
        throw RuntimeError("LpScheduler", "Runtime_Barrier_Simulation_Checker::build_level_sets(): level not found");
      if ((litr->second).empty()) {
        level_sets_.erase(litr);
      }
    }

    void build_degree_table() {
      in_degree_map_.clear();
      out_degree_map_.clear();

      const dag_t& dag = *input_ptr_;

      in_degree_iterator_t in_degree_itr, out_degree_itr;
      operation_t src_op, sink_op;

      for (const_operation_iterator_t oitr=traits::operations_begin(dag);
            oitr!=traits::operations_end(dag); ++oitr) {
        src_op = *oitr;

        in_degree_itr = in_degree_map_.find(src_op);
        if (in_degree_itr == in_degree_map_.end()) {
          in_degree_map_.insert(std::make_pair(src_op, 0UL));
        }

        size_t out_degree = 0;
        for (const_operation_iterator_t
              citr=traits::outgoing_operations_begin(dag, src_op);
              citr!=traits::outgoing_operations_end(dag, src_op); ++citr) {
          sink_op = *citr;

          in_degree_itr = in_degree_map_.find(sink_op);
          if (in_degree_itr == in_degree_map_.end()) {
            in_degree_itr =
              (in_degree_map_.insert(std::make_pair(sink_op,0UL))).first;
          }
          in_degree_itr->second++;
          out_degree++;
        }

        out_degree_itr = out_degree_map_.find(src_op);
        if (out_degree_itr == out_degree_map_.end()) {
          out_degree_map_.insert(std::make_pair(src_op, out_degree));
        }
      }
    }



    const dag_t * const input_ptr_;
    level_sets_t level_sets_;
    in_degree_map_t in_degree_map_;
    out_degree_map_t out_degree_map_;
    std::list<size_t> real_barrier_list_;
    size_t barrier_bound_;
    active_barrier_table_t active_barrier_table_;
}; // class Runtime_Barrier_Simulation_Checker //

} // namespace lp_scheduler //
} // namespace mv //
#endif
