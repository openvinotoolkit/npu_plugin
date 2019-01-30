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

            void serialize(const std::string& path);
            char * serialize(int& bufferSize);
            void deserialize(const std::string& path);
            void deserialize(char * buffer, int length);
            void buildGraphFileT(ComputationModel& cm, json::Object& compilationDescriptor);
    };
}

#endif
