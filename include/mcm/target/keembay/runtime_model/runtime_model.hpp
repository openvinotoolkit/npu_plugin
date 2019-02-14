#ifndef MV_RUNTIME_MODEL_
#define MV_RUNTIME_MODEL_

#include "meta/schema/graphfile/graphfile_generated.h"
#include "meta/schema/graphfile/memoryManagement_generated.h"
#include "include/mcm/computation/resource/memory_allocator.hpp"
#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/target_descriptor.hpp"
#include "include/mcm/target/keembay/workloads.hpp"
#include "include/mcm/target/keembay/ppe_layer_type.hpp"
#include "include/mcm/target/keembay/ppe_fixed_function.hpp"

namespace mv
{
    struct EnumClassHash
    {
        template <typename T>
        std::size_t operator()(T t) const
        {
            return static_cast<std::size_t>(t);
        }
    };

    class RuntimeModel
    {
        private:
            MVCNN::GraphFileT graphFile_;
            static const std::unordered_map<std::string, MVCNN::DType> dTypeMapping_;
            static const std::unordered_map<std::string, MVCNN::MemoryLocation> memoryLocationMapping_;
            static const std::unordered_map<std::string, MVCNN::DPULayerType> dpuLayerMapping_;
            static const std::unordered_map<PpeLayerTypeEnum, MVCNN::PPELayerType, EnumClassHash> ppeLayerTypeMapping_;

        public:
            RuntimeModel();
            ~RuntimeModel();

            // CONVERT METHODS (String to enums, enums to strings, enums mapping etc)
            static MVCNN::MemoryLocation convertAllocatorToMemoryLocale(const std::string& allocatorName);
            static MVCNN::DType convertDtype(const DType& dtype);
            static MVCNN::DPULayerType convertTaskOp(const std::string& opName);
            static MVCNN::MPE_Mode convertMPEMode(MPE_Mode mpe);
            static MVCNN::PPELayerType convertPPELayerType(PpeLayerTypeEnum ppe);

            static void buildTensorReferenceT(ComputationModel &cm, json::Object, Data::TensorIterator t, std::unique_ptr<MVCNN::TensorReferenceT> toBuild);
            static void buildGraphNodeT(ComputationModel &cm, json::Object, Data::OpListIterator op, std::unique_ptr<MVCNN::GraphNodeT> toBuild);
            static void buildSourceStructureT(ComputationModel &cm, json::Object& compilationDescriptor, std::unique_ptr<MVCNN::SourceStructureT> toBuild);
            static void buildSummaryHeaderT(ComputationModel& cm, json::Object& compilationDescriptor, std::unique_ptr<MVCNN::SummaryHeaderT> toBuild);
            static void buildVersionT(ComputationModel&, json::Object &compilationDescriptor, std::unique_ptr<MVCNN::VersionT> toBuild);
            static void buildResourcesT(ComputationModel&, json::Object &compilationDescriptor, std::unique_ptr<MVCNN::ResourcesT> toBuild);
            static void buildBinaryDataT(ComputationModel&, json::Object, Data::TensorIterator t, std::unique_ptr<MVCNN::BinaryDataT> toBuild);
            static void buildTaskListT(ComputationModel& cm, json::Object& compilationDescriptor, std::unique_ptr<MVCNN::TaskListT> toBuild);
            static void buildTaskT(ComputationModel& cm, json::Object& compilationDescriptor, Data::OpListIterator opIt, std::unique_ptr<MVCNN::TaskT> toBuild);
            static void buildSpecificTaskUnion(ComputationModel& cm, json::Object& compilationDescriptor, Data::OpListIterator opIt, MVCNN::SpecificTaskUnion& specificTask);

            // TASKS
            static void buildMvTensorTaskT(ComputationModel& cm, json::Object& compilationDescriptor, Data::OpListIterator opIt, MVCNN::MvTensorTaskT* toBuild);
            static void buildUPADMATaskT(ComputationModel& cm, json::Object& compilationDescriptor, Data::OpListIterator opIt, MVCNN::UPADMATaskT* toBuild);
            static void buildNNDMATaskT(ComputationModel& cm, json::Object& compilationDescriptor, Data::OpListIterator opIt, MVCNN::NNDMATaskT* toBuild);
            static void buildNCE1TaskT(ComputationModel& cm, json::Object& compilationDescriptor, Data::OpListIterator opIt, MVCNN::NCE1TaskT* toBuild);
            static void buildNCE2TaskT(ComputationModel& cm, json::Object& compilationDescriptor, Data::OpListIterator opIt, MVCNN::NCE2TaskT* toBuild);
            static void buildNNTensorTaskT(ComputationModel& cm, json::Object& compilationDescriptor, Data::OpListIterator opIt, MVCNN::NNTensorTaskT* toBuild);
            static void buildControllerTaskT(ComputationModel& cm, json::Object& compilationDescriptor, Data::OpListIterator opIt, MVCNN::ControllerTaskT* toBuild);

            // NCE2 TASK
            static void buildNCEInvariantFieldsT(ComputationModel& cm, json::Object& compilationDescriptor, Data::OpListIterator opIt, std::unique_ptr<MVCNN::NCEInvariantFieldsT> toBuild);
            static void buildNCEVariantFieldsTVector(ComputationModel& cm, json::Object& compilationDescriptor, Data::OpListIterator opIt, std::vector<std::unique_ptr<MVCNN::NCEVariantFieldsT>>& toBuild);
            static void buildNCEVariantFieldsT(ComputationModel& cm, json::Object& compilationDescriptor, Data::OpListIterator opIt, Workload workload, std::unique_ptr<MVCNN::NCEVariantFieldsT> toBuild);
            static void buildPPETaskT(ComputationModel& cm, json::Object& compilationDescriptor, Data::OpListIterator opIt, std::unique_ptr<MVCNN::PPETaskT> toBuild);
            static void buildPPEFixedFunctionT(ComputationModel&, json::Object&, const PPEFixedFunction &ppe, std::unique_ptr<MVCNN::PPEFixedFunctionT> toBuild);

            void serialize(const std::string& path);
            char * serialize(int& bufferSize);
            void deserialize(const std::string& path);
            void deserialize(char * buffer, int length);
            void buildGraphFileT(ComputationModel& cm, json::Object& compilationDescriptor);
    };
}

#endif
