#ifndef PIPELINE_SCHEDULE_TRANSFORMS_H
#define PIPELINE_SCHEDULE_TRANSFORMS_H

#include <unordered_set>
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/model/iterator/tensor.hpp"
#include "include/mcm/op_model.hpp"
#include "pass/lp_scheduler/cmx_concat_transform.hpp"


namespace mv {
namespace scheduler {


class Pipelining_Transform {
  public:

    ////////////////////////////////////////////////////////////////////////////
    typedef const mv::Op* operation_t;
    typedef mv::Op* operation_non_const_t;
    typedef std::list<operation_t> op_list_t;
    typedef CMX_Concatenation concat_subgraph_finder_t;
    typedef typename concat_subgraph_finder_t::concat_subgraph_t
        concat_subgraph_t;
    typedef std::map<operation_t, size_t> depth_map_t;

    class exception_t : std::string {
      public:
        exception_t(const std::string& msg) : std::string(msg) {}
        exception_t(const char *msg) : std::string(msg) {}
        const std::string& getMessage() const { return  *this; }
    }; // class exception_t //

    struct control_edge_t {
      control_edge_t(mv::Data::OpListIterator src,
            mv::Data::OpListIterator sink)
          : source_itr_(src) , sink_itr_(sink) { } 

      control_edge_t(const control_edge_t& o) : source_itr_(o.source_itr_), 
        sink_itr_(o.sink_itr_) {}

      control_edge_t& operator=(const control_edge_t& o) {
        source_itr_ = o.source_itr_;
        sink_itr_ = o.sink_itr_;
        return *this;
      }

      mv::Data::OpListIterator source_itr_;
      mv::Data::OpListIterator sink_itr_;
    };

    struct stream_operation_t {
      operation_t dpu_;
      op_list_t weight_reads_;
      operation_t write_;

      bool operator<(const stream_operation_t& o) const {
        return dpu_->getName() < o.dpu_->getName();
      }

      size_t compute_non_read_input_size(mv::OpModel& om) const {
        std::unordered_set<std::string> weight_map;

        for (auto itr=weight_reads_.begin(); itr!=weight_reads_.end(); ++itr) {
          weight_map.insert((*itr)->getName());
        }

        size_t input_size = 0UL;
        mv::Data::OpListIterator dpu_itr = om.getOp(dpu_->getName());
        for (auto pitr=dpu_itr.leftmostParent(); pitr!=om.opEnd(); ++pitr) {
          if (weight_map.find(pitr->getName()) != weight_map.end()) {
            continue;
          }
          mv::Data::TensorIterator titr = pitr->getOutputTensor(0UL);

          if (titr->get<mv::Tensor::MemoryLocation>("Location") == 
                mv::Tensor::MemoryLocation::NNCMX) {
            input_size += titr->getClusterSize();
          }
        }
        return input_size;
      }


      operation_t compute_non_read_input_operation(mv::OpModel& om) const {
        std::unordered_set<std::string> weight_map;
        operation_t non_read_input = NULL;

        for (auto itr=weight_reads_.begin(); itr!=weight_reads_.end(); ++itr) {
          weight_map.insert((*itr)->getName());
        }

        size_t input_size = 0UL;
        mv::Data::OpListIterator dpu_itr = om.getOp(dpu_->getName());
        for (auto pitr=dpu_itr.leftmostParent(); pitr!=om.opEnd(); ++pitr) {
          if (weight_map.find(pitr->getName()) != weight_map.end()) {
            continue;
          }
          non_read_input = &(*pitr);
          break;
        }
        return non_read_input;
      }

      size_t compute_read_input_size(mv::OpModel& om) const {
        size_t input_size = 0UL;
        for (auto weight_read_itr=weight_reads_.begin();
              weight_read_itr!=weight_reads_.end(); ++weight_read_itr) {
          mv::Data::OpListIterator rop_itr =
              om.getOp((*weight_read_itr)->getName());
          mv::Data::TensorIterator titr = rop_itr->getOutputTensor(0UL);
          input_size += titr->getClusterSize();
        }
        return input_size;
      }

      size_t compute_output_size(mv::OpModel& om) const {
        mv::Data::OpListIterator dpu_op_itr = om.getOp(dpu_->getName());
        mv::Data::TensorIterator titr = dpu_op_itr->getOutputTensor(0UL);
        return titr->getClusterSize();
      }

    }; // struct stream_operation_t //
    typedef std::map<std::string, stream_operation_t> streamed_operation_map_t;

    struct pipeline_subgraph_t {
      op_list_t dpus_;
      op_list_t weight_reads_;
      op_list_t writes_;
      operation_t concat_root_;
      streamed_operation_map_t stream_map_;
      size_t id_;
      size_t max_weight_size_;
      size_t max_output_size_;
      size_t max_input_size_;

      pipeline_subgraph_t() : dpus_(), weight_reads_(), writes_(), id_(),
        concat_root_() {}


      bool is_valid() const {
        return !dpus_.empty() && !weight_reads_.empty() &&
          !weight_reads_.empty() && (concat_root_ != NULL);
      }

      const std::string& name() const {
        return concat_root_->getName(); 
      }


      bool normalize(mv::OpModel& om) {
        stream_map_.clear();

        for (auto ditr=dpus_.begin(); ditr!=dpus_.end(); ++ditr) {
          stream_map_[ (*ditr)->getName() ].dpu_ = *ditr;
        }

        // weight reads //
        for (auto witr=weight_reads_.begin(); witr!=weight_reads_.end();
              ++witr) {
          mv::Data::OpListIterator weight_op_itr = om.getOp((*witr)->getName());


          size_t out_degree = 0UL;
          for (auto citr=weight_op_itr.leftmostChild(); citr!=om.opEnd();
                ++citr) {
            ++out_degree;
            if (out_degree > 1UL) {
              fprintf(stdout, "Reads ccannot have outdegree > 1 %s\n",
                    (citr->getName()).c_str());
              fflush(stdout);
              return false;
            }
          }

          auto citr = weight_op_itr.leftmostChild();
          if (stream_map_.find(citr->getName()) == stream_map_.end()) {
              fprintf(stdout, "Missing DPU for read %s\n",
                    (citr->getName()).c_str());
              fflush(stdout);
              return false;
          }
          stream_map_[citr->getName()].weight_reads_.push_back(*witr);
        }

        // writes //
        for (auto witr=writes_.begin(); witr!=writes_.end(); ++witr) {
          mv::Data::OpListIterator write_op_itr = om.getOp((*witr)->getName());

          size_t out_degree = 0UL;
          for (auto pitr=write_op_itr.leftmostParent(); pitr!=om.opEnd();
                ++pitr) {
            ++out_degree;
            if (out_degree > 1UL) {
              fprintf(stdout, "Writes cannot have indegree > 1 %s\n",
                    (pitr->getName()).c_str());
              fflush(stdout);
              return false;
            }
          }

          auto pitr = write_op_itr.leftmostParent();
          if (stream_map_.find(pitr->getName()) == stream_map_.end()) {
            fprintf(stdout, "Missing DPU for write : %s\n",
                  (pitr->getName()).c_str());
            fflush(stdout);
            return false;
          }
          stream_map_[ pitr->getName() ].write_ = *witr;
        }

        max_weight_size_ = std::numeric_limits<size_t>::min();
        max_output_size_ = std::numeric_limits<size_t>::min();
        max_input_size_ = std::numeric_limits<size_t>::min();

        for (auto sitr=stream_map_.begin(); sitr!=stream_map_.end(); ++sitr) {
          max_weight_size_ = std::max(max_weight_size_, 
              (sitr->second).compute_read_input_size(om) );
          max_input_size_ = std::max( max_input_size_,
              (sitr->second).compute_non_read_input_size(om) );
          max_output_size_ = std::max(max_output_size_,
              (sitr->second).compute_output_size(om) );
        }
        return true;
      }

      bool is_pipelineable(size_t memory_upper_bound) const {
        return (max_input_size_ + 2*(max_weight_size_) + max_output_size_) 
            < memory_upper_bound;
      }

      void print() const {
        printf("=====================================\n");
        printf("pipeline_subgraph_id : %lu root: %s\n", id_,
              concat_root_->getName().c_str());

        for (auto itr=dpus_.begin(); itr!=dpus_.end(); ++itr) {
          printf("dpu: %s\n", (*itr)->getName().c_str());
        }

        for (auto itr=weight_reads_.begin(); itr!=weight_reads_.end(); ++itr) {
          printf("read: %s\n", (*itr)->getName().c_str());
        }

        for (auto itr=writes_.begin(); itr!=writes_.end(); ++itr) {
          printf("write: %s\n", (*itr)->getName().c_str());
        }
        printf("=====================================\n");
      }

      void print_stream_map(mv::OpModel& om) const {
        for (auto sitr=stream_map_.begin(); sitr!=stream_map_.end(); ++sitr) {
          printf("=================\n");
          printf("dpu = %s\n", (sitr->first).c_str());
          printf("non_read_input_size=%lu\n",
                (sitr->second).compute_non_read_input_size(om));
          printf("read_input_size=%lu\n",
                (sitr->second).compute_read_input_size(om));
          printf("output_size=%lu\n", (sitr->second).compute_output_size(om) );
          printf("reads = ");
          for (auto ritr=(sitr->second).weight_reads_.begin();
              ritr!=(sitr->second).weight_reads_.end(); ++ritr) {
            printf(" %s ", (*ritr)->getName().c_str());
          }
          printf("\n");
          printf("write = %s\n", (sitr->second).write_->getName().c_str());
          printf("=================\n");
        }

      }


      // Precondition: the op must have pipeline attribute //
      void add(operation_t op) {
        if (op->getOpType() == "DPUTask") {
          dpus_.push_back(op);
        } else if (op->getOpType() == "DMATask" ) {
          mv::DmaDirectionEnum dma_dir =
              op->get< mv::DmaDirection >("direction");

          if (dma_dir == mv::DmaDirectionEnum::DDR2NNCMX) {
            mv::Data::TensorIterator itr =
              (const_cast<operation_non_const_t>(op))->getOutputTensor(0UL);
            if (itr->isPopulated()) { weight_reads_.push_back(op); }
          } else if (dma_dir == mv::DmaDirectionEnum::NNCMX2DDR) {
            writes_.push_back(op);
          } else {
            throw exception_t("Unrecognized dma direction in op: " +
                  op->getName() + "\n" );
          }
        } else if(op->getOpType() == "ImplicitConcat") {
          concat_root_ = op;
        } else {
          throw exception_t("Unrecognized optype " + op->getOpType() +
                "in pipeline subgraph\n");
        }
      }

    }; // struct pipeline_subgraph_t //
    ////////////////////////////////////////////////////////////////////////////

    Pipelining_Transform(mv::OpModel& omodel,
        const std::string pipeline_layer_attribute="schedule_for_dpu_dma_overlap")
      : omodel_(omodel), pipeline_attribute_(pipeline_layer_attribute),
        depth_map_() {}

    // "pipeline_read_rep" : "dma_read_op_name" //
    static const std::string pipeline_read_representative_attribute() {
      return "pipeline_read_rep";
    }

    // "pipeline_read_offset" : size_t // final address = rep_address + offset//
    static const std::string pipeline_read_offset_attribute() {
      return "pipeline_read_offset";
    }

    // "pipeline_dpu_rep" : "dpu_rep_op" //
    static const std::string pipeline_dpu_representative_attribute() {
      return "pipeline_dpu_rep";
    }

    // "pipeline_resource_utility" : //
    static const std::string pipeline_resource_attribute() {
      return "pipelined_resource_utility";
    }

    template<typename OutputIterator>
    size_t locate_pipeline_subgraphs_with_attribute(OutputIterator output) {
      std::unordered_map<size_t, pipeline_subgraph_t> subgraphs;

      for (mv::Data::OpListIterator oitr=omodel_.opBegin();
            oitr!=omodel_.opEnd(); ++oitr) {
        if (oitr->hasAttr(pipeline_attribute_)) {
          size_t id = (size_t) oitr->get<unsigned>(pipeline_attribute_);
          subgraphs[id].add( &(*oitr) );
        }
      }

      for (auto gitr=subgraphs.begin(); gitr!=subgraphs.end(); ++gitr) {
        (gitr->second).id_ = gitr->first;
        *output = gitr->second;
      }
      return subgraphs.size();
    }

    template<typename T>
    bool is_the_output_of_this_op_in_cmx(T op) const {
      if (op->outputSlots() != 1UL) { return false; }
      auto op_itr = omodel_.getOp(op->getName());
      mv::Data::TensorIterator tensor_itr = op_itr->getOutputTensor(0UL);
      return tensor_itr->get<mv::Tensor::MemoryLocation>("Location") ==
        mv::Tensor::MemoryLocation::NNCMX;
    }

    template<typename T>
    bool is_the_output_of_this_op_populated_tensor(T op) const {
      if (op->outputSlots() != 1UL) { return false; }
      auto op_itr = omodel_.getOp(op->getName());
      mv::Data::TensorIterator tensor_itr = op_itr->getOutputTensor(0UL);
      return tensor_itr->isPopulated();
    }


    pipeline_subgraph_t create_pipeline_subgraph_from_concat_subgraph(
        const concat_subgraph_t& subgraph) const {
      pipeline_subgraph_t pipeline_subgraph;
      static size_t id = 0UL;

      pipeline_subgraph.dpus_ = subgraph.dpu_in_;
      pipeline_subgraph.concat_root_ = subgraph.concat_root_;
      pipeline_subgraph.writes_ = subgraph.writes_;
      pipeline_subgraph.id_ = ++id;


      // fill the weight_reads_ //
      const op_list_t &dpus = pipeline_subgraph.dpus_;
      for (operation_t dpu : dpus) {
        mv::Data::OpListIterator dpu_itr = omodel_.getOp(dpu->getName());
        for (auto pitr=dpu_itr.leftmostParent(); pitr!=omodel_.opEnd();
              ++pitr) {
          //TODO(vamsikku): handle the case where the same DMA read is the 
          //weight input for several DPU tasks.
          if ( (pitr->getOpType() == "DMATask") &&
                (is_the_output_of_this_op_in_cmx(pitr)) && 
                (is_the_output_of_this_op_populated_tensor(pitr)) ) {
            pipeline_subgraph.weight_reads_.push_back( &(*pitr) );
          }
        }
      }

      printf("pipeline_id = %lu is_valid=%s\n", id,
            (pipeline_subgraph.is_valid()) ? "YES" : "NO");

      return pipeline_subgraph;
    }

    template<typename OutputIterator>
    void locate_pipeline_subgraphs(OutputIterator output) {
      // STEP-0: locate all concat subgraphs //
      concat_subgraph_finder_t subgraph_finder(omodel_);
      std::list<concat_subgraph_t> concat_subgraphs;

      subgraph_finder.locate_all_concat_subgraphs(
          std::back_inserter(concat_subgraphs));

      for (auto subgraph : concat_subgraphs) {
        pipeline_subgraph_t pipelined_subgraph =
            create_pipeline_subgraph_from_concat_subgraph(subgraph);
        if (pipelined_subgraph.is_valid()) {
          output = pipelined_subgraph;
        }
      }
    }



    template<typename ControlEdgeOutput, typename SubGraphContainer>
    void transform_op_model(ControlEdgeOutput output,
        SubGraphContainer& pipeline_subgraphs, size_t cmx_size=917504UL) {

      static_assert( std::is_same<pipeline_subgraph_t,
            typename SubGraphContainer::value_type>::value,
              "Invalid container for pipeline subgraphs");

      compute_depth_map();
      mv::OpModel &om = omodel_;
      pipeline_subgraphs.clear();
      locate_pipeline_subgraphs(std::back_inserter(pipeline_subgraphs));
      for (auto subitr=pipeline_subgraphs.begin();
            subitr!=pipeline_subgraphs.end(); ++subitr) {
        // Foreach pipelineable subgraph //

        //TODO(vamsikku): to avoid any kind of spilling avoid pipelining if the
        //dpus are not all at the same depth.
        if (!are_all_dpus_at_same_depth(*subitr)) { continue; }

        // STEP-0: normalize the subgraph //
        bool normalized = subitr->normalize(omodel_);
        if (!normalized) { continue; }

        subitr->print();
        if (!(subitr->is_pipelineable(cmx_size))) { continue; }


        // move the max_output to first dpu in the stream 
        // move 2*(max_weight_size) to the first read //
        const pipeline_subgraph_t &curr_subgraph = *subitr;
        const streamed_operation_map_t &stream_map = curr_subgraph.stream_map_;

        size_t max_dpu_output = subitr->max_output_size_;
        size_t max_weight_size = subitr->max_weight_size_;

        auto stream_itr = stream_map.begin();
        const stream_operation_t& stream_op = stream_itr->second;
        const std::string dpu_rep_name = (stream_op.dpu_)->getName();
        auto dpu_rep_itr = omodel_.getOp(dpu_rep_name);

        dpu_rep_itr->set<std::string>(pipeline_dpu_representative_attribute(),
              dpu_rep_name);
        dpu_rep_itr->set<size_t>(pipeline_resource_attribute(), max_dpu_output);

        auto weight_reads_itr = (stream_op.weight_reads_).begin();
        operation_t weight_read_rep = *weight_reads_itr;
        std::string weight_read_rep_name = weight_read_rep->getName();
        mv::Data::OpListIterator weight_read_rep_itr =
            omodel_.getOp(weight_read_rep_name);
        size_t rep_read_offset =
            (weight_read_rep_itr->getOutputTensor(0UL))->getClusterSize();


        weight_read_rep_itr->set<std::string>(
            pipeline_read_representative_attribute(), weight_read_rep_name);
        weight_read_rep_itr->set<size_t>(pipeline_resource_attribute(),
            (2UL*max_weight_size) );

        // Compute offsets of any reads associated with this representative DPU
        // task.
        ++weight_reads_itr;
        while (weight_reads_itr != (stream_op.weight_reads_).end()) {
          auto weight_read_itr = omodel_.getOp((*weight_reads_itr)->getName());
          weight_read_itr->set<std::string>(
              pipeline_read_representative_attribute(), weight_read_rep_name);
          weight_read_itr->set<size_t>(pipeline_resource_attribute(), 0UL);
          weight_read_itr->set<size_t>(pipeline_read_offset_attribute(),
               rep_read_offset);
          rep_read_offset +=
            (weight_read_itr->getOutputTensor(0UL))->getClusterSize();

          // Add control edges between read-rep and rest of the reads.//
          assert(weight_read_rep_itr != omodel_.opEnd());
          assert(weight_read_itr != omodel_.opEnd());

          output = control_edge_t(weight_read_rep_itr, weight_read_itr);
          output = control_edge_t(weight_read_itr, dpu_rep_itr);

          ++weight_reads_itr;
        }

        // Add a control edge between non-weight read input of DPU and the 
        // weight_read_rep_itr //
        operation_t non_read_dpu_rep_input =
            stream_op.compute_non_read_input_operation(omodel_);

        if (non_read_dpu_rep_input) {
          auto non_read_dpu_rep_input_itr =
              omodel_.getOp(non_read_dpu_rep_input->getName());
          printf("non_read_dpu_rep: %s -> %s\n",
                non_read_dpu_rep_input->getName().c_str(),
                weight_read_rep_itr->getName().c_str());
          output =
              control_edge_t(non_read_dpu_rep_input_itr, weight_read_rep_itr);
        }

        // add the read_offset for all even streams //
        auto prev_prev_stream_itr = stream_map.end(); 
        auto prev_stream_itr = stream_itr;
        size_t curr_stream_idx = 1UL;

        for (++stream_itr; stream_itr != stream_map.end();
              ++stream_itr, ++curr_stream_idx) {

          const stream_operation_t& stream_op = stream_itr->second;
          const stream_operation_t& prev_stream_op = prev_stream_itr->second;
          auto dpu_itr = omodel_.getOp(stream_op.dpu_->getName());
         
          // DPU RESOURCE SETTING //
          dpu_itr->set<std::string>(pipeline_dpu_representative_attribute(),
              dpu_rep_name);
          dpu_itr->set<size_t>(pipeline_resource_attribute(), 0UL);


          // OFFSETS: //
          const op_list_t& curr_reads = stream_op.weight_reads_;
          auto weight_reads_itr = curr_reads.begin();
          size_t read_offset = (curr_stream_idx)%2UL ? rep_read_offset : 0UL;
          while (weight_reads_itr != curr_reads.end()) {
            auto weight_read_itr =
              omodel_.getOp((*weight_reads_itr)->getName());
            weight_read_itr->set<std::string>(
                pipeline_read_representative_attribute(), weight_read_rep_name);
            weight_read_itr->set<size_t>(pipeline_resource_attribute(), 0UL);
            weight_read_itr->set<size_t>(pipeline_read_offset_attribute(),
                  read_offset);
            read_offset +=
              (weight_read_itr->getOutputTensor(0UL))->getClusterSize();
            ++weight_reads_itr;
          }

          // READ_CONTROL_EDGES: add control edges (odd->odd) and (even->even)
          for (auto curr_reads_itr=curr_reads.begin();
                curr_reads_itr!=curr_reads.end(); ++curr_reads_itr) {
            mv::Data::OpListIterator curr_weight_read_itr =
                om.getOp((*curr_reads_itr)->getName());

            if (prev_prev_stream_itr != stream_map.end()) {
              const stream_operation_t& pprev_stream_op =
                  prev_prev_stream_itr->second;
              // ODD EVEN CONTROL EDGE //
              mv::Data::OpListIterator write_itr =
                  om.getOp((pprev_stream_op.write_)->getName());

              assert(write_itr != omodel_.opEnd());
              assert(curr_weight_read_itr != omodel_.opEnd());

              output = control_edge_t( write_itr, curr_weight_read_itr);
            } else {
              //This is a special case of first set of odd reads //

              assert(weight_read_rep_itr != omodel_.opEnd());
              assert(curr_weight_read_itr != omodel_.opEnd());

              output = control_edge_t(weight_read_rep_itr,
                    curr_weight_read_itr);
            }

            // since these are all reads are using ZERO demand add control
            // control edges between the READ and DPU. //
            output = control_edge_t(curr_weight_read_itr, dpu_itr);
          }

          // DPU_CONTROL_EDGES: add control edges between write of
          // prev_stream_itr and stream_itr //
          mv::Data::OpListIterator prev_write_itr =
              om.getOp(prev_stream_op.write_->getName());

          assert(prev_write_itr != omodel_.opEnd());
          assert(dpu_itr != omodel_.opEnd());

          output = control_edge_t(prev_write_itr, dpu_itr);

          prev_prev_stream_itr = prev_stream_itr;
          prev_stream_itr = stream_itr;
        } // foreach stream operation in this subgraph//

        // BUFFER CONTROL EDGES: add control edges between the rep_dpu and 
        // rep_weight and concat root. Buffer should remain until the concat is
        // scheduled.
        assert(weight_read_rep_itr != omodel_.opEnd());
        assert(dpu_rep_itr != omodel_.opEnd());

        if (prev_stream_itr != stream_itr) {
          const stream_operation_t& prev_stream_op = prev_stream_itr->second;
          mv::Data::OpListIterator last_dpu_in_stream =
              om.getOp((prev_stream_op.dpu_)->getName());
          mv::Data::OpListIterator last_write_in_stream =
              om.getOp((prev_stream_op.write_)->getName());

          output = control_edge_t(weight_read_rep_itr, last_dpu_in_stream);
          output = control_edge_t(dpu_rep_itr, last_write_in_stream);
        }
      } // foreach subgraph //

    }

    size_t set_pipelined_operation_addresses() {
      std::unordered_map<std::string, size_t> pipeline_dpu_reps;
      std::unordered_map<std::string, size_t> pipeline_read_reps;

      for (mv::Data::OpListIterator oitr=omodel_.opBegin();
            oitr!=omodel_.opEnd(); ++oitr) {
        if (oitr->hasAttr(pipeline_read_representative_attribute())) {
          std::string rep_name =
            oitr->get<std::string>(pipeline_read_representative_attribute());
          if (rep_name == oitr->getName()) {
            size_t final_address = get_address(oitr);
            pipeline_read_reps[ rep_name ] = final_address;
            oitr->set<size_t>("final_pipeline_address", final_address);
          }
        } else if (oitr->hasAttr(pipeline_dpu_representative_attribute())) {
          std::string rep_name =
            oitr->get<std::string>(pipeline_dpu_representative_attribute());
          if (rep_name == oitr->getName()) {
            size_t final_address = get_address(oitr);
            pipeline_dpu_reps[ rep_name ] = final_address;
            oitr->set<size_t>("final_pipeline_address", final_address);
          }
        }
      } // foreach //


      for (mv::Data::OpListIterator oitr=omodel_.opBegin();
            oitr!=omodel_.opEnd(); ++oitr) {
        if (oitr->hasAttr(pipeline_read_representative_attribute())) {
          std::string rep_name =
            oitr->get<std::string>(pipeline_read_representative_attribute());
          if (rep_name != oitr->getName()) {
            size_t final_address = pipeline_read_reps[rep_name];
            size_t offset = oitr->get<size_t>(pipeline_read_offset_attribute());
            final_address += offset;
            set_address(oitr, final_address);
            oitr->set<size_t>("final_pipeline_address", final_address);
          }
        } else if (oitr->hasAttr(pipeline_dpu_representative_attribute())) {
          std::string rep_name =
            oitr->get<std::string>(pipeline_dpu_representative_attribute());
          if (rep_name != oitr->getName()) {
            size_t final_address = pipeline_dpu_reps[rep_name];
            set_address(oitr, final_address);
            oitr->set<size_t>("final_pipeline_address", final_address);
          }
        }
      } // foreach //
    }

  private:

    bool are_all_dpus_at_same_depth(const pipeline_subgraph_t& subgraph) {
      size_t depth = depth_map_[subgraph.dpus_.front()];
      for (operation_t dpu_op : subgraph.dpus_) {
        if (depth_map_[dpu_op] != depth) { return false; }
      }
      return true;
    }

    void compute_depth_map() {
      depth_map_.clear();
      std::unordered_map<operation_t, size_t> in_degree_map;

      for (mv::Data::OpListIterator oitr=omodel_.opBegin();
            oitr!=omodel_.opEnd(); ++oitr) {
        size_t in_degree = 0UL;
        for (auto pitr=oitr.leftmostParent(); pitr!=omodel_.opEnd(); ++pitr) {
          ++in_degree;
        }
        in_degree_map[ &(*oitr) ] = in_degree;
      }

      std::vector<operation_t> ops_in_level[2UL];
      size_t curr_depth = 0UL;
      for (auto ditr=in_degree_map.begin(); ditr!=in_degree_map.end(); ++ditr) {
        if (!(ditr->second)) {
          ops_in_level[curr_depth%2UL].push_back(ditr->first);
        }
      }

      while (!(ops_in_level[curr_depth%2UL].empty())) {
        std::vector<operation_t> &curr_level_ops = ops_in_level[curr_depth%2UL];
        std::vector<operation_t> &next_level_ops =
            ops_in_level[(curr_depth+1UL)%2UL];

        next_level_ops.clear();
        for (operation_t op : curr_level_ops) {
          depth_map_[op] = curr_depth;
          auto op_itr = omodel_.getOp(op->getName());
          for (auto cop_itr = op_itr.leftmostChild();
                cop_itr != omodel_.opEnd(); ++cop_itr) {
            in_degree_map[ &(*cop_itr) ]--;
            if (!in_degree_map[ &(*cop_itr) ]) {
              next_level_ops.push_back( &(*cop_itr) );
            }
          }
        }
        curr_level_ops.clear();
        ++curr_depth;
      }

    }

    template<typename OperationIterator>
    size_t get_address(OperationIterator op_itr) {
      mv::Data::TensorIterator tensor_itr = op_itr->getOutputTensor(0UL);
      return tensor_itr->get<size_t>("address");
    }

    template<typename OperationIterator>
    void set_address(OperationIterator op_itr, size_t final_address) {
      mv::Data::TensorIterator tensor_itr = op_itr->getOutputTensor(0UL);
      tensor_itr->setAddress(final_address);

      mv::DataModel dm(omodel_);
      auto tensor_allocators = tensor_itr->get<std::set<std::string>>("allocators");
      if (tensor_allocators.empty())
        throw mv::RuntimeError("LpScheduler", "Pipelining_Transform: Tensor Allocators empty");
      auto tensor_alloc_name=tensor_allocators.begin();
      auto tensor_alloc= dm.getAllocator(*tensor_alloc_name);
      mv::Data::BufferIterator tensor_buffer_itr =
          tensor_alloc.getBuffer(0, tensor_itr);
      mv::Data::BufferIterator master_tensor_buffer_itr =
          tensor_alloc.getTopMasterBuffer(tensor_buffer_itr);
      master_tensor_buffer_itr->setOffset((tensor_itr->get<size_t>("address")));
    }

    bool is_pipelineable_in_current_opmodel(
          const pipeline_subgraph_t& subgraph) const {
    }

    mv::OpModel& omodel_;
    const std::string pipeline_attribute_;
    depth_map_t depth_map_;
}; // class Pipelining_Transform //



} // namespace scheduler
} // namespace mv
#endif
