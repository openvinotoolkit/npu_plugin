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

            static std::unique_ptr<MVCNN::TensorReferenceT> buildTensorReferenceT(ComputationModel &cm, mv::Element&, Data::TensorIterator t);
            static std::unique_ptr<MVCNN::GraphNodeT> buildGraphNodeT(ComputationModel &cm, mv::Element&, Data::OpListIterator op);
            static std::unique_ptr<MVCNN::SourceStructureT> buildSourceStructureT(ComputationModel &cm, mv::Element& compilationDescriptor);
            static std::unique_ptr<MVCNN::SummaryHeaderT> buildSummaryHeaderT(ComputationModel& cm, mv::Element& compilationDescriptor);
            static std::unique_ptr<MVCNN::VersionT> buildVersionT(ComputationModel&, mv::Element& compilationDescriptor);
            static std::unique_ptr<MVCNN::ResourcesT> buildResourcesT(ComputationModel&, mv::Element& compilationDescriptor);
            static std::unique_ptr<MVCNN::BinaryDataT> buildBinaryDataT(ComputationModel&, mv::Element&, Data::TensorIterator t);
            static std::unique_ptr<MVCNN::TaskListT> buildTaskListT(ComputationModel& cm, mv::Element& compilationDescriptor);
            static std::unique_ptr<MVCNN::TaskT> buildTaskT(ComputationModel& cm, mv::Element& compilationDescriptor, Data::OpListIterator opIt);
            static void buildSpecificTaskUnion(ComputationModel& cm, mv::Element& compilationDescriptor, Data::OpListIterator opIt, MVCNN::SpecificTaskUnion& specificTask);

            // TASKS
            static void buildMvTensorTaskT(ComputationModel& cm, mv::Element& compilationDescriptor, Data::OpListIterator opIt, MVCNN::MvTensorTaskT* toBuild);
            static void buildUPADMATaskT(ComputationModel& cm, mv::Element& compilationDescriptor, Data::OpListIterator opIt, MVCNN::UPADMATaskT* toBuild);
            static void buildNNDMATaskT(ComputationModel& cm, mv::Element& compilationDescriptor, Data::OpListIterator opIt, MVCNN::NNDMATaskT* toBuild);
            static void buildNCE1TaskT(ComputationModel& cm, mv::Element& compilationDescriptor, Data::OpListIterator opIt, MVCNN::NCE1TaskT* toBuild);
            static void buildNCE2TaskT(ComputationModel& cm, mv::Element& compilationDescriptor, Data::OpListIterator opIt, MVCNN::NCE2TaskT* toBuild);
            static void buildNNTensorTaskT(ComputationModel& cm, mv::Element& compilationDescriptor, Data::OpListIterator opIt, MVCNN::NNTensorTaskT* toBuild);
            static void buildControllerTaskT(ComputationModel& cm, mv::Element& compilationDescriptor, Data::OpListIterator opIt, MVCNN::ControllerTaskT* toBuild);

            // NCE2 TASK
            static std::unique_ptr<MVCNN::NCEInvariantFieldsT> buildNCEInvariantFieldsT(ComputationModel& cm, mv::Element& compilationDescriptor, Data::OpListIterator opIt);
            static std::vector<std::unique_ptr<MVCNN::NCEVariantFieldsT> > buildNCEVariantFieldsTVector(ComputationModel& cm, mv::Element& compilationDescriptor, Data::OpListIterator opIt);
            static std::unique_ptr<MVCNN::NCEVariantFieldsT> buildNCEVariantFieldsT(ComputationModel& cm, mv::Element& compilationDescriptor, Data::OpListIterator opIt, Workload workload);
            static std::unique_ptr<MVCNN::PPETaskT> buildPPETaskT(ComputationModel& cm, mv::Element& compilationDescriptor, Data::OpListIterator opIt);
            static std::unique_ptr<MVCNN::PPEFixedFunctionT> buildPPEFixedFunctionT(ComputationModel&, mv::Element&, const PPEFixedFunction &ppe);

            void serialize(const std::string& path);
            char * serialize(int& bufferSize);
            void deserialize(const std::string& path);
            void deserialize(char * buffer, int length);
            void buildGraphFileT(ComputationModel& cm, mv::Element& compilationDescriptor);
    };
}

#endif
