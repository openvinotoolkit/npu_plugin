#ifndef MV_RUNTIME_MODEL_
#define MV_RUNTIME_MODEL_

#include "meta/schema/graphfile/graphfile_generated.h"
#include "meta/schema/graphfile/memoryManagement_generated.h"
#include "include/mcm/computation/resource/memory_allocator.hpp"
#include "include/mcm/computation/model/base_op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

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
            static MVCNN::TensorReferenceT convertTensorRepresentation(MemoryAllocator &allocator, Data::TensorIterator t);
            static MVCNN::DType convertDtype(const DType& dtype);
            static MVCNN::GraphNodeT convertOperationToGraphNodeT(BaseOpModel &om, Data::OpListIterator op);

            void serialize(const std::string& path);
            char * serialize(int& bufferSize);
            void deserialize(const std::string& path);
            void deserialize(char * buffer, int length);
    };
}

#endif
