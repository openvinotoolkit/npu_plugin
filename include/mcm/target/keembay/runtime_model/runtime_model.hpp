#ifndef MV_RUNTIME_MODEL_
#define MV_RUNTIME_MODEL_

#include "meta/schema/graphfile/graphfile_generated.h"
#include "meta/schema/graphfile/memoryManagement_generated.h"
#include "include/mcm/computation/resource/memory_allocator.hpp"
#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/target_descriptor.hpp"

namespace mv
{
    class RuntimeModel
    {
        private:
            MVCNN::GraphFileT graphFile_;
            static const std::unordered_map<std::string, MVCNN::DType> dTypeMapping_;
            static const std::unordered_map<std::string, MVCNN::MemoryLocation> memoryLocationMapping_;

        public:
            RuntimeModel();
            ~RuntimeModel();

            static MVCNN::MemoryLocation convertAllocatorToMemoryLocale(const std::string& allocatorName);
            static MVCNN::DType convertDtype(const DType& dtype);
            static void buildTensorReferenceT(ComputationModel &cm, Data::TensorIterator t, std::unique_ptr<MVCNN::TensorReferenceT> toBuild);
            static void buildGraphNodeT(ComputationModel &cm, Data::OpListIterator op, std::unique_ptr<MVCNN::GraphNodeT> toBuild);
            static void buildSourceStructureT(ComputationModel &cm, std::unique_ptr<MVCNN::SourceStructureT> toBuild);
            static void buildSummaryHeaderT(ComputationModel& cm, json::Object& compilationDescriptor, std::unique_ptr<MVCNN::SummaryHeaderT> toBuild);
            static void buildVersionT(json::Object &compilationDescriptor, std::unique_ptr<MVCNN::VersionT> toBuild);
            static void buildResourcesT(json::Object &compilationDescriptor, std::unique_ptr<MVCNN::ResourcesT> toBuild);
            static void buildBinaryDataT(Data::TensorIterator t, std::unique_ptr<MVCNN::BinaryDataT> toBuild);
            static void buildTaskListT(ComputationModel& cm, std::unique_ptr<MVCNN::TaskListT> toBuild);
            static void buildTaskT(ComputationModel& cm, Data::OpListIterator opIt, std::unique_ptr<MVCNN::TaskT> toBuild);
            static void buildSpecificTaskUnion(ComputationModel& cm, Data::OpListIterator opIt, MVCNN::SpecificTaskUnion& specificTask);

            // TASKS
            static void buildMvTensorTaskT(ComputationModel& cm, Data::OpListIterator opIt, MVCNN::MvTensorTaskT* toBuild);
            static void buildUPADMATaskT(ComputationModel& cm, Data::OpListIterator opIt, MVCNN::UPADMATaskT* toBuild);
            static void buildNNDMATaskT(ComputationModel& cm, Data::OpListIterator opIt, MVCNN::NNDMATaskT* toBuild);
            static void buildNCE1TaskT(ComputationModel& cm, Data::OpListIterator opIt, MVCNN::NCE1TaskT* toBuild);
            static void buildNCE2TaskT(ComputationModel& cm, Data::OpListIterator opIt, MVCNN::NCE2TaskT* toBuild);
            static void buildNNTensorTaskT(ComputationModel& cm, Data::OpListIterator opIt, MVCNN::NNTensorTaskT* toBuild);
            static void buildControllerTaskT(ComputationModel& cm, Data::OpListIterator opIt, MVCNN::ControllerTaskT* toBuild);

            void serialize(const std::string& path);
            char * serialize(int& bufferSize);
            void deserialize(const std::string& path);
            void deserialize(char * buffer, int length);
            void buildGraphFileT(ComputationModel& cm, json::Object& compilationDescriptor);
    };
}

#endif
