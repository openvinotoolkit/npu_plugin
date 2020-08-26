#ifndef MV_RUNTIME_MODEL_
#define MV_RUNTIME_MODEL_

#include "schema/graphfile/graphfile_generated.h"
#include "schema/graphfile/memoryManagement_generated.h"
#include "include/mcm/computation/resource/memory_allocator.hpp"
#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/target/target_descriptor.hpp"
#include "include/mcm/target/kmb/workloads.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"
#include "include/mcm/target/kmb/ppe_layer_type.hpp"
#include "include/mcm/target/kmb/ppe_fixed_function.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/compression/hde.hpp"
#include "include/mcm/pass/pass_utils.hpp"


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
            RuntimeModel(const mv::TargetDescriptor& td)
            {
                auto hdeDef = td.hdeDef();
                hde_.reset(new Hde(hdeDef.bitPerSymbol, hdeDef.maxNumberEncodedSymbols, 0, hdeDef.blockSize, false, hdeDef.bypassMode));
            }
            
            std::unique_ptr<Hde> hde_ = nullptr;
            MVCNN::GraphFileT graphFile_;
            std::shared_ptr<std::vector<char>> binaryData_;
            static const std::unordered_map<std::string, MVCNN::DType> dTypeMapping_;
            static const std::unordered_map<std::string, MVCNN::MemoryLocation> memoryLocationMapping_;
            static const std::unordered_map<std::string, MVCNN::DPULayerType> dpuLayerMapping_;
            static const std::unordered_map<PPELayerTypeEnum, MVCNN::PPELayerType, EnumClassHash> ppeLayerTypeMapping_;

            static std::vector<unsigned> reduceQuantVector_(std::vector<unsigned> inVec);

        public:
            static RuntimeModel& getInstance(const mv::TargetDescriptor& td)
            {
                static RuntimeModel instance(td);
                return instance;
            }
            RuntimeModel(RuntimeModel const&) = delete;
            void operator=(RuntimeModel const&) = delete;

            // CONVERT METHODS (String to enums, enums to strings, enums mapping etc)
            static MVCNN::MemoryLocation convertAllocatorToMemoryLocale(const std::string& allocatorName,
                                                                        mv::Tensor::MemoryLocation& tensorLocation);
            static MVCNN::DType convertDtype(const DType& dtype);
            static MVCNN::DPULayerType convertTaskOp(const std::string& opName);
            static MVCNN::MPE_Mode convertMPEMode(MPE_Mode mpe);
            static MVCNN::PPELayerType convertPPELayerType(PPELayerTypeEnum ppe);
            static void alignTensor(mv::ComputationModel& cm, std::unique_ptr<MVCNN::TensorReferenceT>& tensorT, Tensor &tensor, const size_t dimension, bool padFinalOutput = false);
            static bool DMAinAfterDMAout(mv::ComputationModel& cm, std::unique_ptr<MVCNN::TensorReferenceT>& tensorT, Tensor &tensor, const size_t dimension, bool padFinalOutput = false);

            static std::unique_ptr<MVCNN::TensorReferenceT> buildTensorReferenceT(ComputationModel &cm, Element&, Data::TensorIterator t, const std::string& allocatorName = "");
            static std::unique_ptr<MVCNN::TensorReferenceT> buildTensorReferenceT(ComputationModel &cm, Element&, Data::TensorIterator t, unsigned clusterId, const std::string &allocatorName = "");
            static void updateTensorReferenceT(ComputationModel &cm, Element&, Data::TensorIterator s, Data::TensorIterator d, unsigned clusterId, std::unique_ptr<MVCNN::TensorReferenceT>& tensorT, const std::string &allocatorName = "");
            static std::unique_ptr<MVCNN::GraphNodeT> buildGraphNodeT(ComputationModel &cm, Element&, Data::OpListIterator op);
            static std::unique_ptr<MVCNN::SourceStructureT> buildSourceStructureT(ComputationModel &cm, Element& compilationDescriptor);
            static std::unique_ptr<MVCNN::SummaryHeaderT> buildSummaryHeaderT(ComputationModel& cm, Element& compilationDescriptor, std::unique_ptr<MVCNN::SummaryHeaderT> originalHeader);
            static std::unique_ptr<MVCNN::SummaryHeaderT> buildSummaryHeaderMetaInformations(ComputationModel& cm, mv::Element& compilationDescriptor);
            static std::unique_ptr<MVCNN::VersionT> buildVersionT(ComputationModel&, Element& compilationDescriptor);
            static std::unique_ptr<MVCNN::ResourcesT> buildResourcesT(ComputationModel&, Element& compilationDescriptor);
            std::unique_ptr<MVCNN::BinaryDataT> buildBinaryDataT(ComputationModel&, Element&, mv::Tensor& t, bool huffmanCompression);
            static std::vector<std::unique_ptr<MVCNN::TaskListT>> buildTaskListT(ComputationModel& cm, Element& compilationDescriptor);
            static std::vector<std::unique_ptr<MVCNN::BarrierT>> buildBarrierTable(ComputationModel& cm, Element& compilationDescriptor);
            static std::unique_ptr<MVCNN::BarrierReferenceT> buildBarrierReferenceT(ComputationModel& cm, Element& compilationDescription, BarrierDependencies dep);
            static std::unique_ptr<MVCNN::BarrierReferenceT> buildBarrierReferenceT();
            static std::unique_ptr<MVCNN::BarrierT> buildBarrierT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt);
            static std::vector<std::unique_ptr<MVCNN::TaskT>> buildTaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt);

            // TASKS
            static std::vector<std::unique_ptr<MVCNN::TaskT>> buildSpecificTaskUnion(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt);
            static std::vector<std::unique_ptr<MVCNN::TaskT>> buildMvTensorTaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt);
            static std::vector<std::unique_ptr<MVCNN::TaskT>> buildUPADMATaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt);
            static std::vector<std::unique_ptr<MVCNN::TaskT>> buildNNDMATaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt, std::string splitting);
            static std::vector<std::unique_ptr<MVCNN::TaskT>> buildNCE1TaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt);
            static std::vector<std::unique_ptr<MVCNN::TaskT>> buildNCE2TaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt, std::string splitting);
            static std::vector<std::unique_ptr<MVCNN::TaskT>> buildUPATask(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt);
            static std::vector<std::unique_ptr<MVCNN::TaskT>> buildControllerTaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt);
            static std::vector<std::unique_ptr<MVCNN::TaskT>> buildBarrierTaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt);

            // NCE2 TASK
            static std::unique_ptr<MVCNN::NCEInvariantFieldsT> buildNCEInvariantFieldsT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt, int numClusters);
            static std::unique_ptr<MVCNN::NCEInvariantFieldsT> buildNCEInvariantFieldsT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt);
            static std::vector<std::unique_ptr<MVCNN::NCEVariantFieldsT>> buildNCEVariantFieldsTVector(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt, unsigned numTask, std::string strategy);
            static std::unique_ptr<MVCNN::NCEVariantFieldsT> buildNCEVariantFieldsT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt, Workload workload, unsigned clusterId, std::string strategy);
            //PADDING HAVE TO BE DIFFERENT FUNCTIONS CAUSE OF THE LOGIC (PADS FOR CLUSTER ZERO WHEN SUBTENSORING OVER H->NO PAD DOWN)
            static void getWorkloadPadding(Control::OpListIterator opIt, Workload &workload);
            static void getWorkloadPadding(Control::OpListIterator opIt, Workload &workload, unsigned clusterId);
            static std::array<unsigned short, 4> getNewPadding(std::array<unsigned short, 4> padding, int clusterId, int numClusters);
            static std::array <unsigned short, 4> getPadding(Control::OpListIterator opIt, unsigned clusterId);
            static bool hardwareBugDepthwise(Control::OpListIterator opIt);
            static std::unique_ptr<MVCNN::PPETaskT> buildPPETaskT(ComputationModel& cm, Element& compilationDescriptor, Control::OpListIterator opIt);
            static std::unique_ptr<MVCNN::PPETaskT> buildPPETaskT();
            static std::unique_ptr<MVCNN::PPEFixedFunctionT> buildPPEFixedFunctionT(ComputationModel&, Element&, const PPEFixedFunction &ppeFixedFunction);
            static void updatePWLTaskT(std::unique_ptr<MVCNN::NCEInvariantFieldsT>& toBuild , Control::OpListIterator& opIt);
            static void adaptFakeSparsityIndex(std::unique_ptr<MVCNN::NCEInvariantFieldsT>& inv, Control::OpListIterator opIt, int clusterId);
            static void adaptFakeSparsityIndex(std::unique_ptr<MVCNN::NCEInvariantFieldsT>& inv, Control::OpListIterator opIt);

            // UPA Layer Task
            static MVCNN::UPALayerTaskT * buildUPAQuantizeTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPAROIPoolingTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPAPSROIPoolingTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPAProposalTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPASoftmaxTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPASigmoidTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPANormalizeTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPAPassthroughTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPADummyTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPAResampleTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPAReshapeTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPARegionYoloTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPAReorgYoloTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPAPermuteTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPAEltwiseFP16Task(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPAInterpTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPANormTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPADetectionOutputTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPAPriorboxTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPAArgmaxTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPATopKTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPACustomTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPADeconvTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPATileTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPACTCDecoderTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);
            static MVCNN::UPALayerTaskT * buildUPARefConvTask(ComputationModel& cm, Element &compilationDescriptor, Control::OpListIterator opIt);

            // UTILS
            static unsigned countProducerConsumerTasks(mv::ComputationModel& cm, mv::Control::OpListIterator opIt);

            void serialize(const std::string& path);
            void serialize();
            void deserialize(const std::string& path);
            void deserialize(char * buffer, int length);
            void buildGraphFile(ComputationModel& cm, Element& compilationDescriptor);
            void buildHeader(ComputationModel& cm, Element& compilationDescriptor);
            std::shared_ptr<std::vector<char>> getBlob();
            static void case1MC(unsigned numTasks, ComputationModel& cm, mv::DmaDirection direction, mv::Element &compilationDescriptor, bool padFinalOutput, bool dmaToDma, std::vector<std::unique_ptr<MVCNN::TaskT>>& toReturn, Data::TensorIterator src, Data::TensorIterator dst, std::uint8_t port, const std::string &srcAllocator = "", const std::string &dstAllocator = "");
            static void case2MC(unsigned numTasks, ComputationModel& cm, mv::DmaDirection direction, mv::Element &compilationDescriptor, bool padFinalOutput, bool dmaToDma, std::vector<std::unique_ptr<MVCNN::TaskT> > &toReturn, Data::TensorIterator src, Data::TensorIterator dst, std::uint8_t port, const std::string &srcAllocator = "", const std::string &dstAllocator = "");
    };
}

#endif
