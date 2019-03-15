#ifndef MV_RUNTIME_MODEL_
#define MV_RUNTIME_MODEL_

#include "meta/schema/graphfile/graphfile_generated.h"
#include "meta/schema/graphfile/memoryManagement_generated.h"
#include "include/mcm/computation/resource/memory_allocator.hpp"
#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
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
            RuntimeModel() {}
            MVCNN::GraphFileT graphFile_;
            static const std::unordered_map<std::string, MVCNN::DType> dTypeMapping_;
            static const std::unordered_map<std::string, MVCNN::MemoryLocation> memoryLocationMapping_;
            static const std::unordered_map<std::string, MVCNN::DPULayerType> dpuLayerMapping_;
            static const std::unordered_map<PpeLayerTypeEnum, MVCNN::PPELayerType, EnumClassHash> ppeLayerTypeMapping_;

        public:
            static RuntimeModel& getInstance()
            {
                static RuntimeModel instance;
                return instance;
            }
            RuntimeModel(RuntimeModel const&) = delete;
            void operator=(RuntimeModel const&) = delete;

            // CONVERT METHODS (String to enums, enums to strings, enums mapping etc)
            static MVCNN::MemoryLocation convertAllocatorToMemoryLocale(const std::string& allocatorName);
            static MVCNN::DType convertDtype(const DType& dtype);
            static MVCNN::DPULayerType convertTaskOp(const std::string& opName);
            static MVCNN::MPE_Mode convertMPEMode(MPE_Mode mpe);
            static MVCNN::PPELayerType convertPPELayerType(PpeLayerTypeEnum ppe);

            static std::unique_ptr<MVCNN::TensorReferenceT> buildTensorReferenceT(ComputationModel &cm, Element&, Data::TensorIterator t);
            static std::unique_ptr<MVCNN::GraphNodeT> buildGraphNodeT(ComputationModel &cm, Element&, Data::OpListIterator op);
            static std::unique_ptr<MVCNN::SourceStructureT> buildSourceStructureT(ComputationModel &cm, Element& compilationDescriptor);
            static std::unique_ptr<MVCNN::SummaryHeaderT> buildSummaryHeaderT(ComputationModel& cm, Element& compilationDescriptor, std::unique_ptr<MVCNN::SummaryHeaderT> originalHeader);
            static std::unique_ptr<MVCNN::SummaryHeaderT> buildSummaryHeaderMetaInformations(ComputationModel& cm, mv::Element& compilationDescriptor);
            static std::unique_ptr<MVCNN::VersionT> buildVersionT(ComputationModel&, Element& compilationDescriptor);
            static std::unique_ptr<MVCNN::ResourcesT> buildResourcesT(ComputationModel&, Element& compilationDescriptor);
            static std::unique_ptr<MVCNN::BinaryDataT> buildBinaryDataT(ComputationModel&, Element&, mv::Tensor& t);
            static std::vector<std::unique_ptr<MVCNN::TaskListT>> buildTaskListT(ComputationModel& cm, Element& compilationDescriptor);
            static std::vector<std::unique_ptr<MVCNN::BarrierT>> buildBarrierTable(ComputationModel& cm, Element& compilationDescriptor);
            static std::unique_ptr<MVCNN::BarrierReferenceT> buildBarrierReferenceT(ComputationModel& cm, Element& compilationDescription, BarrierDependencies dep);
            static std::unique_ptr<MVCNN::BarrierT> buildBarrierT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt);
            static std::unique_ptr<MVCNN::TaskT> buildTaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt);
>>>>>>> Changed all RuntimeModel static methods to use Control::OpListIterator.

            // TASKS
            static void buildSpecificTaskUnion(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt, MVCNN::SpecificTaskUnion& specificTask);
            static void buildMvTensorTaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt, MVCNN::MvTensorTaskT* toBuild);
            static void buildUPADMATaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt, MVCNN::UPADMATaskT* toBuild);
            static void buildNNDMATaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt, MVCNN::NNDMATaskT* toBuild);
            static void buildNCE1TaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt, MVCNN::NCE1TaskT* toBuild);
            static void buildNCE2TaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt, MVCNN::NCE2TaskT* toBuild);
            static void buildNNTensorTaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt, MVCNN::NNTensorTaskT* toBuild);
            static void buildControllerTaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt, MVCNN::ControllerTaskT* toBuild);
            static void buildBarrierTaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt, MVCNN::ControllerTaskT* toBuild);

            // NCE2 TASK
            static std::unique_ptr<MVCNN::NCEInvariantFieldsT> buildNCEInvariantFieldsT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt);
            static std::vector<std::unique_ptr<MVCNN::NCEVariantFieldsT> > buildNCEVariantFieldsTVector(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt);
            static std::unique_ptr<MVCNN::NCEVariantFieldsT> buildNCEVariantFieldsT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt, Workload workload);
            static std::unique_ptr<MVCNN::PPETaskT> buildPPETaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt);
            static std::unique_ptr<MVCNN::PPEFixedFunctionT> buildPPEFixedFunctionT(ComputationModel&, Element&, const PPEFixedFunction &ppe);

            void serialize(const std::string& path);
            char * serialize(int& bufferSize);
            void deserialize(const std::string& path);
            void deserialize(char * buffer, int length);
            void buildGraphFile(ComputationModel& cm, Element& compilationDescriptor);
            void buildHeader(ComputationModel& cm, Element& compilationDescriptor);
    };
}

#endif
