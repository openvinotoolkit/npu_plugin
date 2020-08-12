#ifndef PIPELINE_SCHEDULE_TRANSFORMS_H
#define PIPELINE_SCHEDULE_TRANSFORMS_H

#include <unordered_set>

#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/model/iterator/tensor.hpp"


namespace mv {
namespace scheduler {


class Pipelining_Transform {
  public:

    ////////////////////////////////////////////////////////////////////////////
    typedef const mv::Op* operation_t;
    typedef mv::Op* operation_non_const_t;
    typedef std::list<operation_t> op_list_t;

    class exception_t : std::string {
      public:
        exception_t(const std::string& msg) : std::string(msg) {}
        exception_t(const char *msg) : std::string(msg) {}
        const std::string& getMessage() const { return  *this; }
    }; // class exception_t //

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
          input_size += titr->getClusterSize();
        }
        return input_size;
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

      const std::string& name() const {
        return concat_root_->getName(); 
      }
      void normalize(mv::OpModel& om) {
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
              throw exception_t("Reads cannot have outdegree > 1\n");
            }
          }

          auto citr = weight_op_itr.leftmostChild();
          if (stream_map_.find(citr->getName()) == stream_map_.end()) {
            throw exception_t("Missing DPU for read: " + citr->getName());
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
              throw exception_t("Writes cannot have indegree > 1\n");
            }
          }

          auto pitr = write_op_itr.leftmostParent();
          if (stream_map_.find(pitr->getName()) == stream_map_.end()) {
            throw exception_t("Missing DPU for write: " + pitr->getName());
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
      : omodel_(omodel), pipeline_attribute_(pipeline_layer_attribute) {}


    template<typename OutputIterator>
    size_t locate_pipeline_subgraphs(OutputIterator output) {
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

    template<typename ControlEdgeOutput, typename SubGraphContainer>
    void transform_op_model(ControlEdgeOutput output,
        SubGraphContainer& pipeline_subgraphs, size_t cmx_size=917504UL) {

      static_assert( std::is_same<pipeline_subgraph_t,
            typename SubGraphContainer::value_type>::value,
              "Invalid container for pipeline subgraphs");

      pipeline_subgraphs.clear();
      locate_pipeline_subgraphs(std::back_inserter(pipeline_subgraphs));
      for (auto subitr=pipeline_subgraphs.begin();
            subitr!=pipeline_subgraphs.end(); ++subitr) {
        subitr->normalize();
        if (!(subitr->is_pipelineable())) { continue; }

        // move the max_output to first dpu in the stream 
        // move 2*(max_weight_size) to the first read //

        
        
      }

    }

  private:

    bool is_pipelineable_in_current_opmodel(
          const pipeline_subgraph_t& subgraph) const {
    }

    mv::OpModel& omodel_;
    const std::string pipeline_attribute_;
}; // class Pipelining_Transform //



} // namespace scheduler
} // namespace mv
#endif
