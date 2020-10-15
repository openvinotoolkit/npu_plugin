#include "pass/lp_scheduler/operation_precedence_dag.hpp"
#include "pass/lp_scheduler/barrier_schedule_generator.hpp"
#include "pass/lp_scheduler/barrier_simulator.hpp"

namespace mv_unit_testing { class Barrier_Control_Dag_Test; }

namespace mv {
namespace lp_scheduler {

// Barrier scheduler for mv::ControlModel //
// NOTE: the scheduler will set the real barrier index in
// Barrier::realBarrierIndex_ //
class Control_Model_Barrier_Scheduler {
  friend class mv_unit_testing::Barrier_Control_Dag_Test;
  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef scheduler::Operation_Dag<mv::ControlModel> control_op_dag_t;
    typedef typename control_op_dag_t::operation_t operation_t;
    typedef model_traits<mv::ControlModel> cmtraits;
    typedef model_traits<mv::OpModel> omtraits;
    typedef Barrier_Schedule_Generator<control_op_dag_t> scheduler_t;
    typedef typename scheduler_t::schedule_time_t schedule_time_t;
    typedef typename scheduler_t::schedule_info_t schedule_info_t;
    typedef typename scheduler_t::resource_t resource_t;
    typedef mv::Data::TensorIterator tensor_iterator_t;
    typedef typename omtraits::const_operation_iterator_t op_iterator_t;
    typedef typename omtraits::const_child_operation_iterator_t
        op_child_iterator_t;
    typedef typename cmtraits::const_operation_iterator_t control_op_iterator_t;
    typedef typename cmtraits::const_child_operation_iterator_t
        control_edge_iterator_t;

    // one of the source or sink is a barrier task //
    struct barrier_control_edge_t {
      barrier_control_edge_t(op_iterator_t src, op_iterator_t sink)
        : source_(src), sink_(sink) {}

      op_iterator_t source_;
      op_iterator_t sink_;
    }; // struct barrier_control_edge_t //

    struct noop_output_iterator_t {
      noop_output_iterator_t() {}
      template<typename T>
      void operator=(const T& o) {}
      void operator++(void) {}
    }; // struct noop_output_iterator_t //

    // One transition structure for each physical barrier //
    class barrier_transition_structure_t {
      public:
        ////////////////////////////////////////////////////////////////////////
        typedef std::set<operation_t> producers_t;
        typedef typename producers_t::const_iterator producer_iterator_t;
        ////////////////////////////////////////////////////////////////////////

        barrier_transition_structure_t(mv::OpModel *om_ptr=NULL,
            schedule_time_t time=std::numeric_limits<schedule_time_t>::max())
          : om_ptr_(NULL), time_(time), curr_barrier_task_(),
            prev_barrier_task_(), producers_() {}

        barrier_transition_structure_t(
            const barrier_transition_structure_t& o) { (*this) = o; }

        barrier_transition_structure_t& operator=(
            const barrier_transition_structure_t& o) {
          om_ptr_ = o.om_ptr_;
          time_ = o.time_;
          curr_barrier_task_ = o.curr_barrier_task_;
          prev_barrier_task_ = o.prev_barrier_task_;
          producers_ = o.producers_;
          return *this;
        }

        bool is_valid_but_redundant_barrier() const {
          assert(om_ptr_);
          if (curr_barrier_task_ == om_ptr_->opEnd()) { return false; }

          const mv::Barrier& barrier =
              curr_barrier_task_->get<mv::Barrier>("Barrier");
          return !barrier.hasConsumers();
        }

        op_iterator_t curr_barrier_task() const { return curr_barrier_task_; }

        void init(mv::OpModel& om) {
          om_ptr_ = &om;
          time_ = std::numeric_limits<schedule_time_t>::max();
          prev_barrier_task_ = om.opEnd();
          curr_barrier_task_ = om.opEnd();
          producers_.clear();
        }


        // returns true if this call has resulted in creating a new barrier
        // task (also means there is a temporal change) //
        template<typename BackInsertOutputIterator=noop_output_iterator_t>
        bool process_next_scheduled_op(const schedule_info_t& sinfo,
            BackInsertOutputIterator output=BackInsertOutputIterator()) {
          (void) output;
          schedule_time_t curr_time = sinfo.schedule_time_;
          bool created_new_barrier_task = false;

          if (time_ != curr_time) {
            // CASE-1: temporal transition happened //
            created_new_barrier_task = true;
            maintain_invariant_temporal_change(sinfo);
            time_ = curr_time;
          } else {
            // CASE-2: trival case //
            add_scheduled_op_to_producer_list(sinfo);
          }
          return created_new_barrier_task;
        }

        void close_barrier_producer_list() {
          if (curr_barrier_task_ == om_ptr_->opEnd()) { return; }
          process_current_barrier_producer_list_close_event(curr_barrier_task_,
              prev_barrier_task_);
        }

      private:
   
        // Maintain the invariant given that that we are closing the producer
        // list of the current barrier.
        inline void process_current_barrier_producer_list_close_event(
            op_iterator_t bop_curr, op_iterator_t bop_prev) {

          mv::OpModel &om = *om_ptr_;
          mv::ControlModel cm(om);

          op_iterator_t bop_end = om.opEnd();
          assert(bop_curr != bop_end);


          // Get the barrier object for the three barrier tasks //
          mv::Barrier *b_prev = NULL;

          if (bop_prev != bop_end) {
            assert(is_barrier_task(bop_prev));
            b_prev = &(bop_prev->get<mv::Barrier>("Barrier"));
          }
          mv::Barrier *b_curr = &(bop_curr->get<mv::Barrier>("Barrier"));

          for (producer_iterator_t itr=producers_.begin();
              itr!=producers_.end(); ++itr) {
            operation_t source = *itr;

            // STEP-1.2 (a): producers //
            b_curr->addProducer(source->getName());

            // STEP-1.2 (b): consumers //
            op_iterator_t oitr = om_ptr_->getOp(source->getName());
            typename cmtraits::const_operation_iterator_t pitr =
                cm.switchContext(oitr);
            control_edge_iterator_t eitr =
                cmtraits::begin_child_operations(pitr);
            for (; eitr != cm.opEnd(); ++eitr) {
              operation_t sink = &(*eitr);
              b_curr->addConsumer(sink->getName());
            } // foreach edge (p, u) \in E //

            // STEP-1.3 //
            if (b_prev) {
              b_prev->addConsumer(source->getName());
            }
          } // foreach producer //

        }


        void maintain_invariant_temporal_change(const schedule_info_t& sinfo) {
          //              B_prev
          // curr_state : Prod_list={p_0, p_1, ... p_n}-->B_curr
          // event: Prod_list={q_0}->B_curr_new
          // 
          // scheduler says it want to associate both B_old and B_new to the
          // same physical barrier.
          //
          // Restore Invariant:
          // STEP-1.1: create a new barrier task (B_new).
          // STEP-1.2: update B_curr
          //        a. producers: B_curr is now closed so update its producers
          //        b. consumers: for each (p_i, u) \in P_old x V 
          //                      add u to the consumer list of B_old
          // STEP-1.3: update B_prev
          //           consumers: add p_i \in P_old to the consumer list of
          //                      B_prev. This is because B_prev and B_curr
          //                      are associated with same physical barrier.
          // STEP-2: B_prev = B_curr , B_curr = B_curr_new , Prod_list ={q0}
          mv::ControlModel cm(*om_ptr_);
          op_iterator_t bop_prev = prev_barrier_task_, 
                          bop_curr = curr_barrier_task_,
                          bop_end = om_ptr_->opEnd(), bop_curr_new = bop_end;

          bop_curr_new = create_new_barrier_task(sinfo);
          assert(bop_curr_new != bop_end);
          assert(is_barrier_task(bop_curr_new));

          // STEP-1 //
          if (bop_curr != bop_end) {
            process_current_barrier_producer_list_close_event(bop_curr,
                  bop_prev);
          }

          // STEP-2 //
          prev_barrier_task_ = curr_barrier_task_;
          curr_barrier_task_ = bop_curr_new;
          producers_.clear();
          add_scheduled_op_to_producer_list(sinfo);
        }

        void add_scheduled_op_to_producer_list(const schedule_info_t& sinfo) {
          operation_t scheduled_op = sinfo.op_;
          producers_.insert(scheduled_op);
        }

        op_iterator_t create_new_barrier_task(
            const schedule_info_t&) const {

          static size_t barrier_task_id=0UL;
          char barrier_name[64UL]; 
          // digits in barrier_task_id <= 64xlog(2)_10 < 20 so the barrier_name
          // cannot exceed 64 chars //
          sprintf(barrier_name, "Barrier_%lu", barrier_task_id++);

          std::set<std::string> empty_set;
          struct mv::Barrier new_barrier(empty_set, empty_set);

          om_ptr_->barrierTask(new_barrier, barrier_name);
          op_iterator_t barrier_new = om_ptr_->getOp(barrier_name);
          assert((barrier_new != om_ptr_->opEnd()) &&
                  barrier_new->hasAttr("Barrier"));
          mv::Barrier &barrier = barrier_new->get<mv::Barrier>("Barrier");
          barrier.setID(barrier_task_id - 1UL);
          barrier.setIndex(barrier.getID());
          return barrier_new;
        }

        template<typename T>
        bool is_barrier_task(T op_itr) const {
          return (op_itr->getOpType()) == "BarrierTask";
        }

        mv::OpModel *om_ptr_;
        schedule_time_t time_;
        op_iterator_t curr_barrier_task_;
        op_iterator_t prev_barrier_task_;
        producers_t producers_; 
    };  // class barrier_transition_structure_t //
    typedef std::unordered_map<size_t, barrier_transition_structure_t>
        barrier_association_table_t;
    typedef typename barrier_association_table_t::iterator
        barrier_association_iterator_t;
    ////////////////////////////////////////////////////////////////////////////

    Control_Model_Barrier_Scheduler(mv::ControlModel& cmodel,
        size_t barrier_count, size_t slot_count) : control_model_(cmodel),
      bcount_(barrier_count), scount_(slot_count) {}

    template<typename BackInsertControlEdgeIterator=noop_output_iterator_t>
    size_t schedule(
        BackInsertControlEdgeIterator output=noop_output_iterator_t()) {

      size_t btask_count = 0UL;
      control_op_dag_t input_dag(control_model_);
      mv::OpModel om(control_model_);

      // last associated barrier task associated with index //
      barrier_association_table_t barrier_association;

      //STEP-0: initialize the association table//
      for (size_t bid=1; bid<=bcount_; bid++) {
        auto bitr = barrier_association.insert(std::make_pair(bid,
              barrier_transition_structure_t()));
        barrier_transition_structure_t& bstructure = (bitr.first)->second;
        bstructure.init(om);
      }

      {
        //STEP-1: run the scheduler //
        scheduler_t scheduler_begin(input_dag, bcount_, scount_), scheduler_end;
        size_t scheduling_number = 0UL;
        for ( ;scheduler_begin != scheduler_end; ++scheduler_begin) {
          const schedule_info_t& sinfo = *scheduler_begin;
          auto bitr = barrier_association.find(sinfo.barrier_index_);
          assert(bitr != barrier_association.end());
          barrier_transition_structure_t& bstructure = bitr->second;
          operation_t sop = sinfo.op_;

          if (is_output_or_input_op(sinfo)) {
            continue;
          }

          om.getOp(sop->getName())->set<unsigned>("schedulingNumber",
                scheduling_number++);

          // STEP-2: update barrier structure invariant //
          bool new_barrier_task_created =
              bstructure.process_next_scheduled_op(sinfo, output);

          if (new_barrier_task_created) { ++btask_count; }
        }
      }

      // STEP-2.5: process trailing barrier control structures //
      {

        for (auto bitr=barrier_association.begin();
              bitr!=barrier_association.end(); ++bitr) {
          barrier_transition_structure_t &bstruct = bitr->second;
          bstruct.close_barrier_producer_list();
        }
      }

      {
        // STEP-3: Change the control model by removing the old control edges
        // and add only control edges between tasks and barriers//
        clear_all_edges_in_control_model();

        // for each barrier task define control flows //
        for (op_iterator_t oitr=omtraits::begin_operations(om);
              oitr!=omtraits::end_operations(om); ++oitr) {
          if ( !(oitr->getOpType() == "BarrierTask") )  { continue; }

          // add flow edges between producers and this barrier //
          const mv::Barrier& barrier = oitr->get<mv::Barrier>("Barrier");
          const std::set<std::string> &producers = barrier.getProducers();
          const std::set<std::string> &consumers = barrier.getConsumers();

          op_iterator_t barrier_sink = oitr, barrier_source = oitr;

          for (auto pitr=producers.begin(); pitr!=producers.end(); ++pitr) {
            op_iterator_t source = om.getOp(*pitr);
            assert(source != om.opEnd());
            control_model_.defineFlow(source, barrier_sink);

            // [STEP-4] BarrierDeps: add update barrier to source//
            if (!source->hasAttr("BarrierDeps")) {
              source->set("BarrierDeps", mv::BarrierDependencies());
            }
            auto& barrierRef =
                source->get<mv::BarrierDependencies>("BarrierDeps");
            barrierRef.addUpdateBarrier(barrier.getIndex());
          }

          for (auto citr=consumers.begin(); citr!=consumers.end(); ++citr) {
            op_iterator_t sink = om.getOp(*citr);
            assert(sink != om.opEnd());
            if (sink->getOpType() == "Output") { continue; }

            control_model_.defineFlow(barrier_source, sink);

            // [STEP-4] BarrierDeps: add wait barrier to sink//
            if (!sink->hasAttr("BarrierDeps")) {
              sink->set("BarrierDeps", mv::BarrierDependencies());
            }
            auto& barrierRef =
                sink->get<mv::BarrierDependencies>("BarrierDeps");
            barrierRef.addWaitBarrier(barrier.getIndex());
          }
        }
      }

      control_model_.transitiveReduction();
      recomputeProducerConsumerCounts();

      size_t removed_barriers =
          removeBarriersWithNoConsumers(barrier_association, om);
      assert(btask_count >= removed_barriers);

      renumberBarrierTasks(om);
      recomputeProducerConsumerCounts();

      btask_count -= removed_barriers;

      return btask_count;
    }

    void remove_redundant_wait_barriers() {
      mv::OpModel om(control_model_);
      locate_and_remove_redundant_wait_barriers();
      renumberBarrierTasks(om);
      recomputeProducerConsumerCounts();
    }

    void remove_barriers_in_upa_chain_connected_to_output() {
      mv::ControlModel &cm = control_model_;
      mv::OpModel om(cm);
      std::list<operation_t> upa_chain;

      get_tailing_upa_chain(std::back_inserter(upa_chain));

      if (!upa_chain.empty() && is_barrier_op(upa_chain.back())) {
        //Updating barriers of the parents of the barrier task of first UPA Layer //
        mv::Data::OpListIterator upaBitr = om.getOp(upa_chain.back()->getName());
        const mv::Barrier& currB =
          upaBitr->get<mv::Barrier>("Barrier");
        unsigned currB_index = currB.getIndex();
        mv::Control::OpListIterator upaCBitr = cm.switchContext(upaBitr);

        for (mv::Control::OpParentIterator pitr=upaCBitr.leftmostParent();
            pitr!=cm.opEnd(); ++pitr){
          mv::BarrierDependencies& barrierRefParent =
            pitr->get<mv::BarrierDependencies>("BarrierDeps");
          std::vector<unsigned> updateB = barrierRefParent.getUpdate();
          barrierRefParent.clearUpdateBarriers();

          for(std::size_t t = 0; t < updateB.size(); t++){
            if(updateB[t] == currB_index)
              continue;
            barrierRefParent.addUpdateBarrier(updateB[t]);
          }
          // Adding control edge between parents of barrier and first UPA Layer //
          for(mv::Control::OpChildIterator childItr=upaCBitr.leftmostChild();
                childItr!=cm.opEnd(); ++childItr) {
            cm.defineFlow(om.getOp(pitr->getName()), om.getOp(childItr->getName()));
          }
        }
        om.removeOp(om.getOp(upa_chain.back()->getName()));
        upa_chain.pop_back();
      }
      if (upa_chain.empty()) { return; }
     
      operation_t prev_upa_op = NULL, curr_op;

      for (auto ritr=upa_chain.rbegin(); ritr!=upa_chain.rend(); ++ritr) {
        curr_op = *ritr;
        assert( is_upa_op(curr_op) || is_barrier_op(curr_op) );

        if (is_upa_op(curr_op)) {
          //removing barrier references for UPA Tailing Layers //
          mv::BarrierDependencies& barrierRef =
             om.getOp(curr_op->getName())->get<mv::BarrierDependencies>("BarrierDeps");
          barrierRef.clear();
          om.getOp(curr_op->getName())->set<bool>("trailing", true);
          // add control edge prev_upa->curr_upa //
          if (prev_upa_op) {
            auto prev_upa_itr = om.getOp(prev_upa_op->getName());
            auto curr_upa_itr = om.getOp(curr_op->getName());
            cm.defineFlow(prev_upa_itr, curr_upa_itr);
	  }
          prev_upa_op = curr_op;
          } else {
          om.removeOp(om.getOp(curr_op->getName()));
        }
      }
      renumberBarrierTasks(om);
      recomputeProducerConsumerCounts();
    }

  private:
    
    template<typename T>
    bool is_barrier_op(T op_itr) const { 
      return op_itr->getOpType() == "BarrierTask";
    }


    bool has_zero_out_degree(control_op_iterator_t itr) const {
      return itr.leftmostChild() ==  control_model_.opEnd();
    }

    bool has_in_degree_equal_to_k(control_op_iterator_t itr, size_t k) const {
      mv::Control::OpParentIterator pitr=itr.leftmostParent(), 
          pitr_end=control_model_.opEnd();

      size_t degree=0UL; 
      for (;(pitr!=pitr_end) && (degree < k); ++pitr, ++degree) {}

      return degree == k;
    }

    bool has_unit_in_degree(control_op_iterator_t itr) const {
      return has_in_degree_equal_to_k(itr, 1UL);
    }

    bool has_unit_out_degree(control_op_iterator_t itr) const {
      mv::Control::OpChildIterator citr = itr.leftmostChild(), 
          citr_end=control_model_.opEnd();
      if (citr == citr_end) { return false; }
      ++citr;
      return (citr == citr_end);
    }

    bool has_non_zero_in_degree(control_op_iterator_t itr) const {
      return itr.leftmostParent() !=  control_model_.opEnd();
    }

    template<typename T>
    bool is_upa_op(T itr) const {
      return itr->getOpType() == "UPATask";
    }

    template<typename T>
    bool is_real_task_in_blob(T itr) const {
      return (itr->getOpType() == "UPATask") || (itr->getOpType() == "DMATask")
        || (itr->getOpType() == "DPUTask");
    }

    bool is_input_or_output_op(control_op_iterator_t itr) const {
      return (itr->getOpType() == "Input") || (itr->getOpType() == "Output");
    }

    // returns in the order of tail -> head //
    template<typename OutputIterator>
    void get_tailing_upa_chain(OutputIterator output) const {
      typedef std::unordered_map<operation_t, operation_t>
          unit_in_degree_parent_t;

      mv::ControlModel &cm = control_model_;
      unit_in_degree_parent_t unit_in_degree_parent;
      std::unordered_set<operation_t> non_zero_out_degree_set;

      for (mv::Control::FlowListIterator eitr=cm.flowBegin();
            eitr!=cm.flowEnd(); ++eitr ) {
        mv::Control::OpListIterator src_itr = eitr.source();
        mv::Control::OpListIterator sink_itr = eitr.sink();

        operation_t source = &(*src_itr);
        operation_t sink = &(*sink_itr);

        auto parent_itr = unit_in_degree_parent.find(sink);
        if (parent_itr == unit_in_degree_parent.end()) {
          unit_in_degree_parent.insert(std::make_pair(sink, source)); 
        } else {
          parent_itr->second = NULL;
        }

        non_zero_out_degree_set.insert(source);
      }

      operation_t tail = NULL;
      // locate the task with zero outdegree and in-degree = 1 //
      for (control_op_iterator_t op_itr=cm.opBegin(); op_itr!=cm.opEnd();
            ++op_itr) {
        operation_t op = &(*op_itr);

        if ( !(is_upa_op(op) || is_barrier_op(op)) ) { continue; }
        if (non_zero_out_degree_set.find(op) != non_zero_out_degree_set.end()){
          continue;
        }

        auto parent_itr = unit_in_degree_parent.find(op);
        if ((parent_itr != unit_in_degree_parent.end()) &&
              (parent_itr->second != NULL)) {
          tail = op; 
          break;
        }
      }

      if (tail == NULL) { return; } 

      output = tail; 

      operation_t curr_op = tail;
      auto parent_itr = unit_in_degree_parent.find(curr_op);
      while ( (parent_itr != unit_in_degree_parent.end()) &&
          (parent_itr->second != NULL) ) {
        curr_op = parent_itr->second;
        if ( !(is_upa_op(curr_op) || is_barrier_op(curr_op)) ) { break; }
        output = curr_op;
        parent_itr = unit_in_degree_parent.find(curr_op);
      }
    }


    size_t locate_and_remove_redundant_wait_barriers() {
      mv::OpModel om(control_model_);

      FILE *fptr = nullptr;
      if(mv::isDebugFilesEnabled()) {
        fptr = fopen("redundant_barriers.txt", "w");
        if(nullptr == fptr) {
          throw std::string("Cannot open file for writing");
        }
      }
      size_t total = 0;
      for (op_iterator_t oitr=omtraits::begin_operations(om);
            oitr!=omtraits::end_operations(om); ++oitr) {
        if (!is_real_task_in_blob(oitr)) { continue; }
        size_t rcount = remove_redundant_wait_barriers(oitr);

        if (fptr && rcount) {
          fprintf(fptr, "op=%s rcount=%lu\n", oitr->getName().c_str(),
              rcount);
        }
        total += rcount;
      }
      if(fptr) {
        fprintf(fptr, "total=%lu\n", total);
        fclose(fptr);
      }

      return total;
    }

    size_t remove_redundant_wait_barriers(mv::Data::OpListIterator op_itr) {
      mv::ControlModel &cm = control_model_;
      mv::Control::OpListIterator cop_itr = cm.switchContext(op_itr);
      mv::OpModel om(control_model_);


      std::vector<std::string> barriers_with_unit_out_degree;

      for (mv::Control::OpParentIterator pitr=cop_itr.leftmostParent();
            pitr!=cm.opEnd(); ++pitr) {
        const mv::Barrier& barrier = pitr->get<mv::Barrier>("Barrier");
        size_t consumer_count = (barrier.getConsumers()).size();
        if (consumer_count == 1UL) {
          barriers_with_unit_out_degree.push_back(pitr->getName());
        }
      }

      if (barriers_with_unit_out_degree.size() > 1UL) {
        //STEP-0 //
        mv::Data::OpListIterator canonical_barrier_itr =
            om.getOp(barriers_with_unit_out_degree[0]);
        mv::Barrier& canonical_barrier =
            canonical_barrier_itr->get<mv::Barrier>("Barrier");
        unsigned canonical_barrier_index = canonical_barrier.getIndex();

        for (size_t i=1; i<barriers_with_unit_out_degree.size(); ++i) {
          // STEP-1.1: add parents of this barrier to the producer list of
          // the canonical barrier.
          // STEP-1.2: remove this barrier from update list of each parent.
          // STEP-1.3: remove this barrier.
          // STEP-1.4: add the canonical barrier to the update list of each 
          // parent 
          // STEP-1.5: add control edges between the parents and canonical 
          // barrier

          mv::Data::OpListIterator curr_op_bitr =
              om.getOp(barriers_with_unit_out_degree[i]);
          const mv::Barrier& curr_barrier =
              curr_op_bitr->get<mv::Barrier>("Barrier");
          unsigned curr_barrier_index = curr_barrier.getIndex();
          mv::Control::OpListIterator cbitr = cm.switchContext(curr_op_bitr);
          
          for (mv::Control::OpParentIterator pitr=cbitr.leftmostParent();
                pitr!=cm.opEnd(); ++pitr) {
            mv::BarrierDependencies& barrier_deps =
                pitr->get<mv::BarrierDependencies>("BarrierDeps");
            std::vector<unsigned> update_barriers = barrier_deps.getUpdate();
            barrier_deps.clearUpdateBarriers();

            // STEP-1.1 //
            canonical_barrier.addProducer(pitr->getName());

            for (size_t idx=0; idx<update_barriers.size(); ++idx) {
              if (update_barriers[idx] == curr_barrier_index) {
                // STEP-1.2 , 1.4 //
                update_barriers[idx] = canonical_barrier_index;
              }
              barrier_deps.addUpdateBarrier(update_barriers[idx]);
            }

            mv::Data::OpListIterator source = om.getOp(pitr->getName());
            mv::Data::OpListIterator sink = canonical_barrier_itr;
            // STEP-1.5 //
            cm.defineFlow(source, sink);
          }
          // STEP-1.3 //
          om.removeOp(curr_op_bitr);
        }
      }

      return barriers_with_unit_out_degree.empty()
          ? 0UL : (barriers_with_unit_out_degree.size()-1UL);
    }


    // mostly barriers connected to output //
    size_t removeBarriersWithNoConsumers(
        const barrier_association_table_t& barrier_association, 
        mv::OpModel& om) {
      size_t removed_count = 0UL;
      // remove any redundant barriers //
      for (auto bitr=barrier_association.begin();
            bitr!=barrier_association.end(); ++bitr) {
        const barrier_transition_structure_t &bstruct = bitr->second;
        if (bstruct.is_valid_but_redundant_barrier()) {
          om.removeOp(bstruct.curr_barrier_task());
          ++removed_count;
        }
      }
      return removed_count;
    }

    void renumberBarrierTasks(mv::OpModel& om) {
      size_t bid = 0UL;
      for (op_iterator_t oitr=omtraits::begin_operations(om);
            oitr!=omtraits::end_operations(om); ++oitr) {
        if ( !(oitr->getOpType() == "BarrierTask") )  { continue; }
        mv::Barrier &barrier = oitr->get<mv::Barrier>("Barrier");

        barrier.setID(bid);
        barrier.setIndex(bid++);
      }
    }

    void clearProducerConsumerReferences() {
      mv::ControlModel &cm = control_model_;
      mv::Control::FlowListIterator fitr, fitr_next;
      for (fitr=cm.flowBegin(); fitr!=cm.flowEnd(); ++fitr) {
        mv::Control::OpListIterator src_itr = fitr.source();
        mv::Control::OpListIterator sink_itr = fitr.sink(); 
        mv::Control::OpListIterator bar_itr, op_itr;

        assert( (src_itr != cm.opEnd()) && (sink_itr != cm.opEnd()) );
        if (!is_barrier_op(src_itr) && !is_barrier_op(sink_itr)) { continue; }

        if (is_barrier_op(src_itr)) {
          assert(!is_barrier_op(sink_itr));
          bar_itr = src_itr;
          op_itr = sink_itr;
        } else {
          assert(is_barrier_op(sink_itr));
          bar_itr = sink_itr;
          op_itr = src_itr;
        }

       
        mv::Barrier& barrier = bar_itr->get<mv::Barrier>("Barrier");
        barrier.clearProducersConsumers();

        mv::BarrierDependencies& barrierRef =
          op_itr->get<mv::BarrierDependencies>("BarrierDeps");
        barrierRef.clear();
      }
    }

    void recomputeProducerConsumerCounts() {
      // STEP-1: clear all the references //
      clearProducerConsumerReferences();

      // STEP-2:
      // foreach control edge (u, v) 
      //
      // CASE-1: (bar->op)
      //    op.addWaitBarrier(bar)
      //    bar.addConsumer(op) 
      //   
      // CASE-2: (op->bar)
      mv::ControlModel &cm = control_model_;
      mv::Control::FlowListIterator fitr, fitr_next;
      for (fitr=cm.flowBegin(); fitr!=cm.flowEnd(); ++fitr) {
        mv::Control::OpListIterator src_itr = fitr.source();
        mv::Control::OpListIterator sink_itr = fitr.sink(); 
        assert( (src_itr != cm.opEnd()) && (sink_itr != cm.opEnd()) );

        if (!is_barrier_op(src_itr) && !is_barrier_op(sink_itr)) { continue; }

        if (is_barrier_op(src_itr)) {
          assert(!is_barrier_op(sink_itr));

          mv::Barrier& barrier = src_itr->get<mv::Barrier>("Barrier");
          mv::BarrierDependencies& barrierRef =
            sink_itr->get<mv::BarrierDependencies>("BarrierDeps");


          barrierRef.addWaitBarrier(barrier.getIndex());
          barrier.addConsumer(sink_itr->getName());
        } else {
          assert(!is_barrier_op(src_itr));

          mv::Barrier& barrier = sink_itr->get<mv::Barrier>("Barrier");
          mv::BarrierDependencies& barrierRef =
            src_itr->get<mv::BarrierDependencies>("BarrierDeps");


          barrierRef.addUpdateBarrier(barrier.getIndex());
          barrier.addProducer(src_itr->getName());
        }
      } // foreach control edge //
    }

    bool is_output_or_input_op(const schedule_info_t& sinfo) const {
      operation_t op = sinfo.op_;
      return (op->getOpType() == "Output") || (op->getOpType() == "Input");
    }

    void remove_redundant_output_barrier(mv::OpModel& om, operation_t output) {
      if (!output) { return; }

      // remove the parent barrier connected to the output //
      mv::Data::OpListIterator output_itr = om.getOp(output->getName());
      assert(output_itr != om.opEnd());

      mv::ControlModel cm(om);
      size_t in_degree = 0UL;
      mv::Control::OpListIterator cop_itr = cm.switchContext(output_itr);
      for (mv::Control::OpParentIterator pitr=cop_itr.leftmostParent();
          pitr!=cm.opEnd(); ++pitr) {
        in_degree++;
        if (in_degree > 1) { return; }
      }


      std::string remove_barrier_name;
      std::string remove_barrier_parent_name;
      {
        mv::Control::OpParentIterator bitr_p=cop_itr.leftmostParent();
        mv::Data::OpListIterator bitr_o = om.getOp(bitr_p->getName());
        mv::Control::OpListIterator bitr = cm.switchContext(bitr_o);
        assert(bitr != cm.opEnd());

        in_degree = 0UL;
        for (mv::Control::OpParentIterator pitr=bitr.leftmostParent();
            pitr!=cm.opEnd(); ++pitr) {
          in_degree++;
          assert(in_degree == 1UL);
        }
        remove_barrier_name = bitr_o->getName();
        remove_barrier_parent_name = (bitr.leftmostParent())->getName();
      }


      // add control edge between parent of the barrier and output //
      mv::Data::OpListIterator source = om.getOp(remove_barrier_parent_name);
      cm.defineFlow(source, output_itr);

      mv::Data::OpListIterator rm_itr = om.getOp(remove_barrier_name);
      om.removeOp(rm_itr);
    }

    void clear_all_edges_in_control_model() {
      mv::ControlModel &cm = control_model_;
      mv::Control::FlowListIterator fitr, fitr_next;
      for (fitr=cm.flowBegin(); fitr!=cm.flowEnd();) {
        fitr_next = fitr; ++fitr_next;
        cm.undefineFlow(fitr);
        fitr = fitr_next;
      }
    }

    ///////////////////////////////////////////////////////////////////////////
    mv::ControlModel &control_model_;
    size_t bcount_;
    size_t scount_;
}; // class Control_Model_Barrier_Scheduler //

struct Control_Model_Barrier_Checker {
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

  typedef mv::lp_scheduler::Runtime_Barrier_Simulation_Checker<dag_t,
          barrier_op_selector_t, real_barrier_mapper_t> runtime_checker_t;
  //////////////////////////////////////////////////////////////////////////////

  static bool check_schedule(mv::ControlModel& cmodel,
      size_t real_barrier_bound=8UL) {
    assert(real_barrier_bound%2UL == 0UL);

    dag_t dag(cmodel);
    runtime_checker_t checker(dag, real_barrier_bound/2UL);
    return checker.check();
  }

};  // class Control_Model_Barrier_Checker //


class Save_Restore_Control_Model {
  public:
    ////////////////////////////////////////////////////////////////////////////
    typedef mv::Data::OpListIterator op_iterator_t;
    struct control_edge_t {
      control_edge_t(op_iterator_t src, op_iterator_t sink)
        : source_(src), sink_(sink) {}
      op_iterator_t source_;
      op_iterator_t sink_;
    }; // struct control_edge_t //
    typedef std::list<control_edge_t> control_edge_list_t;
    ////////////////////////////////////////////////////////////////////////////

    Save_Restore_Control_Model(mv::ControlModel& cmodel)
      : cmodel_(cmodel), saved_control_edges_() {}


    void save() {
      saved_control_edges_.clear();

      mv::ControlModel &cm = cmodel_;
      mv::OpModel om(cm);

      mv::Control::FlowListIterator fitr, fitr_next;
      for (fitr=cm.flowBegin(); fitr!=cm.flowEnd(); ++fitr) {
        mv::Control::OpListIterator src_itr = fitr.source();
        mv::Control::OpListIterator sink_itr = fitr.sink(); 
        assert(!is_barrier_op(src_itr) && !is_barrier_op(sink_itr));

        op_iterator_t src_oitr = om.getOp(src_itr->getName());
        op_iterator_t sink_oitr = om.getOp(sink_itr->getName());

        saved_control_edges_.push_back(control_edge_t(src_oitr, sink_oitr));
      }
    }

    void restore() {
      std::list<op_iterator_t> barrier_ops;

      // STEP-1: clear all current control edges //
      clear_all_control_edges();
      mv::OpModel om(cmodel_);

      // STEP-2: remove barrier ops //
      for (op_iterator_t oitr=om.getInput(); oitr!=om.opEnd(); ++oitr) {
        if (is_barrier_op(oitr)) { barrier_ops.push_back(oitr); }
      }
      for (auto bitr=barrier_ops.begin(); bitr!=barrier_ops.end(); ++bitr) {
        om.removeOp(*bitr);
      }

      // STEP-3: add saved control edges //
      for (auto eitr=saved_control_edges_.begin();
            eitr!=saved_control_edges_.end(); ++eitr) {
        cmodel_.defineFlow(eitr->source_, eitr->sink_);
      }
    }

    template<typename T>
    bool is_barrier_op(T itr) const {
      return (itr->getOpType() == "BarrierTask");
    }


  private:

    void clear_all_control_edges() {
      mv::ControlModel cm(cmodel_);
      mv::Control::FlowListIterator fitr, fitr_next;
      for (fitr=cm.flowBegin(); fitr!=cm.flowEnd();) {
        fitr_next = fitr; ++fitr_next;
        cm.undefineFlow(fitr);
        fitr = fitr_next;
      }
    }

    mv::ControlModel &cmodel_;
    control_edge_list_t saved_control_edges_;
}; // class Save_Restore_Control_Model //



} // namespace lp_scheduler //
} // namespace mv //
