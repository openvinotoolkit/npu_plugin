#include <list>
#include <map>
#include <unordered_map>
#include <utility>
#include "include/mcm/utils/warning_manager.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"
#include "pass/lp_scheduler/operation_precedence_dag.hpp"
#include "pass/lp_scheduler/barrier_schedule_generator.hpp"
#include "pass/lp_scheduler/barrier_simulator.hpp"

namespace mv {
namespace lp_scheduler {

/***
 * Runtime_Barrier_Simulation_Assigner inherits from Runtime_Barrier_Simulation_Checker
 * The main difference is 
 * 1) Runtime_Barrier_Simulation_Assigner seperate DPU Task and UPA Task into two lists
 * 2) Runtime_Barrier_Simulation_Assigner reorder task list according to their execution order on runtime side
 * 3) Runtime_Barrier_Simulation_Assigner assign physical ID to barriers
 ***/
template<typename OpDAG, typename OpTypeSelector, typename RealBarrierMapper,
    typename DAGTraits=scheduler_traits<OpDAG> >
class Runtime_Barrier_Simulation_Assigner : public Runtime_Barrier_Simulation_Checker<OpDAG, OpTypeSelector, RealBarrierMapper,
    DAGTraits> {
  public:
  //////////////////////////////////////////////////////////////////////////////
    typedef Runtime_Barrier_Simulation_Checker<OpDAG, OpTypeSelector, RealBarrierMapper,
    DAGTraits> Checker;
    typedef typename Checker::traits traits;
    typedef typename Checker::op_type_selector_t op_type_selector_t;
    typedef typename Checker::dag_t dag_t;
    typedef typename Checker::operation_t operation_t;
    typedef typename Checker::op_list_t op_list_t;
    typedef typename Checker::const_operation_iterator_t const_operation_iterator_t;
    typedef typename Checker::op_list_iterator_t op_list_iterator_t;
    typedef typename Checker::in_degree_map_t in_degree_map_t;
    typedef typename Checker::in_degree_iterator_t in_degree_iterator_t;
    typedef typename Checker::out_degree_iterator_t out_degree_iterator_t;
    typedef typename Checker::level_sets_iterator_t level_sets_iterator_t;

    using typename Checker::active_barrier_info_t;
    using typename Checker::active_barrier_table_t;
    using typename Checker::active_barrier_table_iterator_t;
    typedef model_traits<mv::OpModel> omtraits;
    typedef typename omtraits::const_operation_iterator_t op_iterator_t;
  //////////////////////////////////////////////////////////////////////////////

    Runtime_Barrier_Simulation_Assigner(const dag_t& input,
        size_t barrier_bound, mv::OpModel& om) : 
        Runtime_Barrier_Simulation_Checker<OpDAG, OpTypeSelector, RealBarrierMapper, DAGTraits>(input, barrier_bound), 
        om_(&om) {}

    bool assign() {
      init();
      build_level_sets();

      op_list_t barrier_list, compute_list, data_list, upa_list;

      get_lists_from_level_sets(barrier_list, data_list, compute_list, upa_list);

      // sort barriers
      barrier_list.sort([](const operation_t& a, const operation_t& b) -> bool { 
        mv::Attribute barrierA = a->getAttrs()["Barrier"];
        mv::Attribute barrierB = b->getAttrs()["Barrier"];
        return barrierA.get<mv::Barrier>().getIndex() < barrierB.get<mv::Barrier>().getIndex(); 
        });

      // sort dma/dpu/upa
      data_list.sort([](const operation_t& a, const operation_t& b) -> bool {
        std::map<std::string, mv::Attribute> attributesA = a->getAttrs();
        std::map<std::string, mv::Attribute> attributesB = b->getAttrs();
        unsigned DMALevelA = attributesA["DMALevel"].get<unsigned>();
        unsigned DMALevelB = attributesB["DMALevel"].get<unsigned>();
        unsigned DPUScheduleNumberA = attributesA["DPU-schedule-number"].get<unsigned>();
        unsigned DPUScheduleNumberB = attributesB["DPU-schedule-number"].get<unsigned>();
        unsigned schedulingNumberA = attributesA["schedulingNumber"].get<unsigned>();
        unsigned schedulingNumberB = attributesB["schedulingNumber"].get<unsigned>();
        //Sort based on DMA level first
        if(DMALevelA != DMALevelB) {
            return DMALevelA < DMALevelB;
        }
        //Then sort based on DPU scheduling number if the DMA level is equal
        if(DPUScheduleNumberA != DPUScheduleNumberB) {
            return DPUScheduleNumberA < DPUScheduleNumberB;
        }
        // If the DMA level and DPU scheduling number are equal then sort on the scheduling number assinged by the scheduler
        return schedulingNumberA < schedulingNumberB; 
        });

      compute_list.sort([](const operation_t& a, const operation_t& b) -> bool { 
        mv::Attribute schedulingNumberA = a->getAttrs()["schedulingNumber"];
        mv::Attribute schedulingNumberB = b->getAttrs()["schedulingNumber"];
        return schedulingNumberA.get<unsigned>() < schedulingNumberB.get<unsigned>(); 
        });

      upa_list.sort([](const operation_t& a, const operation_t& b) -> bool { 
        mv::Attribute schedulingNumberA = a->getAttrs()["schedulingNumber"];
        mv::Attribute schedulingNumberB = b->getAttrs()["schedulingNumber"];
        return schedulingNumberA.get<unsigned>() < schedulingNumberB.get<unsigned>(); 
        });

      logForOps(barrier_list, data_list, compute_list, upa_list);

      // ---simulate execuation---
      mv::Logger::log(mv::Logger::MessageType::Debug, "PassManager",
                          "Starting Runtime Simulation");
      bool filled_atleast_one = false;
      while (!data_list.empty() || !compute_list.empty()
            || !barrier_list.empty() || !upa_list.empty()) {
        filled_atleast_one = false;
        filled_atleast_one |= fill_barrier_tasks(barrier_list);
        filled_atleast_one |= process_tasks(data_list);
        filled_atleast_one |= process_tasks(compute_list);
        filled_atleast_one |= process_tasks(upa_list);
        if (!filled_atleast_one) { 
            logForBarrierTable(active_barrier_table_.begin(), active_barrier_table_.end());
            return false;
        }
      }
      return true;
    }
    
  protected:
    using Checker::input_ptr_;
    using Checker::level_sets_;
    using Checker::in_degree_map_;
    using Checker::out_degree_map_;
    using Checker::real_barrier_list_;
    using Checker::barrier_bound_;
    using Checker::active_barrier_table_;
    // member func
    using Checker::init;
    using Checker::build_level_sets;
    using Checker::process_task;
    using Checker::is_task_ready;
    using Checker::fill_barrier_tasks;
    using Checker::get_lists_from_level_sets;
    using Checker::return_real_barrier;
    using Checker::acquire_real_barrier;
    
  private:
    mv::OpModel * om_;

    bool is_task_ready(const operation_t& task) {
      const dag_t& dag = *input_ptr_;
      assert(!op_type_selector_t::is_barrier_op(dag, task));

      op_list_t wait_barriers, update_barriers;

      // wait barriers //
      for (const_operation_iterator_t
          pitr=traits::incoming_operations_begin(dag, task);
          pitr!=traits::incoming_operations_end(dag, task); ++pitr) {
        operation_t barrier_op = *pitr;
        if(op_type_selector_t::is_barrier_op(dag, barrier_op))
        {
            active_barrier_table_iterator_t aitr =
                active_barrier_table_.find(barrier_op);

            if ( (aitr == active_barrier_table_.end()) ||
              ((aitr->second).in_degree_ > 0) ) { return false; }
        }
      }

      // update barriers //
      for (const_operation_iterator_t
          citr=traits::outgoing_operations_begin(dag, task);
          citr!=traits::outgoing_operations_end(dag, task); ++citr) {
        operation_t barrier_op = *citr;
        // only check barrier op and skip op such as tailing upa 
        if(op_type_selector_t::is_barrier_op(dag, barrier_op)) {
            if (active_barrier_table_.find(barrier_op) == active_barrier_table_.end()) {
                return false;
            }
        }
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
        if(op_type_selector_t::is_barrier_op(dag, barrier_op))
        {
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
      }

      // update barriers //
      for (const_operation_iterator_t
          citr=traits::outgoing_operations_begin(dag, task);
          citr!=traits::outgoing_operations_end(dag, task); ++citr) {
        operation_t barrier_op = *citr;
        if (op_type_selector_t::is_barrier_op(dag, barrier_op)) {
            aitr = active_barrier_table_.find(barrier_op);
            assert(aitr != active_barrier_table_.end());

            active_barrier_info_t& barrier_info = aitr->second;
            assert(barrier_info.in_degree_ > 0UL);
            barrier_info.in_degree_--;
        }
      }
    }

    bool process_tasks(op_list_t& task_list) {
      bool filled_atleast_once = Checker::process_tasks(task_list);
      
      // print
      op_list_iterator_t tbegin = task_list.begin(), tend = task_list.end();
      if (Logger::getVerboseLevel()==VerboseLevel::Debug){
        while (tbegin != tend) {
          operation_t op = *tbegin;
          if (!is_task_ready(op) ) { break; }
          mv::Logger::log(mv::Logger::MessageType::Debug, "RuntimeSimulator",
                          "Process task: " + op->getName());
          ++tbegin;
        }
      }
      
      return filled_atleast_once;
    }

    void get_lists_from_level_sets(op_list_t &barrier_list,
        op_list_t &data_list, op_list_t &compute_list, op_list_t &upa_list) const {
      get_lists_from_level_sets(barrier_list, data_list, compute_list);
      
      const dag_t &dag = *input_ptr_;
      size_t level= 0;
      compute_list.clear();  // may include upa tasks, rebuild it
      upa_list.clear();
      for (auto itr=level_sets_.begin(); itr!=level_sets_.end(); ++itr, ++level) {
        const op_list_t& op_list = itr->second;
        for (auto olitr=op_list.begin(); olitr!=op_list.end(); olitr++) {
          operation_t op = *olitr;
          if (op_type_selector_t::is_compute_op(dag, op)){
            compute_list.push_back(op);
          }
          // WR for upa tailing, don't check tailing upas in runtime simulator
          if ((op->getOpType() == "UPATask") && (!op->hasAttr("trailing"))){
            upa_list.push_back(op);
          }
        }
      }
    }

    // acquires and assign a real barrier for the input barrier task //
    void acquire_real_barrier(operation_t btask) {
      assert(!real_barrier_list_.empty());
      size_t real = real_barrier_list_.front();
      op_iterator_t oitr = om_->getOp(btask->getName());      
      mv::Barrier &barrier = oitr->get<mv::Barrier>("Barrier");
      // assign physical id to barrier
      barrier.setRealBarrierIndex(real);
      mv::Logger::log(mv::Logger::MessageType::Debug, "RuntimeSimulator",
                          "Physical ID assignment: " + oitr->getName() + " : "+ std::to_string(barrier.getIndex())
                          + "->" + std::to_string(real));
      real_barrier_list_.pop_front();

      assert(active_barrier_table_.size() < (2*barrier_bound_));

      in_degree_iterator_t in_itr = in_degree_map_.find(btask);
      out_degree_iterator_t out_itr = out_degree_map_.find(btask);

      assert((in_itr != in_degree_map_.end()) && 
            (out_itr != out_degree_map_.end()));

      assert(active_barrier_table_.find(btask) == active_barrier_table_.end());

      active_barrier_table_.insert(std::make_pair(btask,
            active_barrier_info_t(real, in_itr->second, out_itr->second)));
    }
    
    void logForBarrierTable(active_barrier_table_iterator_t bcurr, active_barrier_table_iterator_t bend){
        if(Logger::getVerboseLevel()==VerboseLevel::Debug)
        {
            for(auto iter=bcurr; iter!=bend; iter++){
                mv::Logger::log(mv::Logger::MessageType::Debug, "RuntimeSimulator",
                  std::to_string((iter->second).real_barrier_) + ": " +
                  std::to_string((iter->second).in_degree_) + ", " +
                  std::to_string((iter->second).out_degree_) + "\n");
            }
        }
    }
    
    void logForOp(op_list_iterator_t tbegin, op_list_iterator_t tend, string op_type_name){
        while (tbegin != tend) {
            operation_t op = *tbegin;
            mv::Logger::log(mv::Logger::MessageType::Debug, "RuntimeSimulator", op_type_name + ": " + op->getName());
            ++tbegin;
        }
    }
    
    void logForOps(op_list_t barrier_list, op_list_t dma_list, op_list_t dpu_list, op_list_t upa_list){
        if(Logger::getVerboseLevel()==VerboseLevel::Debug)
        {
            logForOp(barrier_list.begin(), barrier_list.end(), "barrier_list");
            logForOp(dma_list.begin(), dma_list.end(), "dma_list");
            logForOp(dpu_list.begin(), dpu_list.end(), "dpu_list");
            logForOp(upa_list.begin(), upa_list.end(), "upa_list");
        }
    }
}; // class Runtime_Barrier_Simulation_Assigner //

struct Control_Model_Barrier_Assigner {
  //////////////////////////////////////////////////////////////////////////////
  typedef scheduler::Operation_Dag<mv::ControlModel> dag_t;
  typedef typename dag_t::operation_t operation_t;

  struct barrier_op_selector_t {
    static bool is_barrier_op(const dag_t&, const operation_t& op) {
      return (op->getOpType()) == "BarrierTask";
    }

    static bool is_data_op(const dag_t&, const operation_t& op) {
      return (op->getOpType()) == "DMATask";
    }

    static bool is_compute_op(const dag_t&, const operation_t& op) {
      return (op->getOpType()) == "DPUTask";
    }
  }; // struct barrier_op_selector_t //

  struct real_barrier_mapper_t {
    // Precondition: this must be a barrier op //
    size_t operator()(const dag_t&, const operation_t& op) const {
      const mv::Barrier& barrier = op->get<mv::Barrier>("Barrier");
      return barrier.getRealBarrierIndex();
    }
  }; // struct real_barrier_mapper_t //

  typedef mv::lp_scheduler::Runtime_Barrier_Simulation_Assigner<dag_t,
          barrier_op_selector_t, real_barrier_mapper_t> runtime_assigner_t;
  //////////////////////////////////////////////////////////////////////////////

  static bool assign_physical_id(mv::ControlModel& cmodel,
      size_t real_barrier_bound=8UL) {
    assert(real_barrier_bound%2UL == 0UL);

    dag_t dag(cmodel);
    mv::OpModel om(cmodel);
    runtime_assigner_t assigner(dag, real_barrier_bound/2UL, om);
    return assigner.assign();
  }
};

} // namespace lp_scheduler //
} // namespace mv //
