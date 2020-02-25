#ifndef MV_TOOLS_GRAPH_COMPARATOR_
#define MV_TOOLS_GRAPH_COMPARATOR_


#include <fstream>
#include "include/mcm/logger/logger.hpp"
#include "include/mcm/utils/env_loader.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "schema/graphfile/graphfile_generated.h"
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/registry.h"
#include "flatbuffers/util.h"

namespace mv
{

    namespace tools
    {

        class GraphComparator
        {

            std::ifstream inStream_;
            char *data1Buffer_;
            char *data2Buffer_;
            std::vector<std::string> lastDiff_; 

            void loadGraphFile_(const std::string& path, char *dataBuffer, MVCNN::GraphFileT& graph);
            void loadGraphFile_(const char *dataBuffer, std::size_t length, MVCNN::GraphFileT& graph);

            template <class T>
            void compare_(const std::unique_ptr<T>& lhs, const std::unique_ptr<T>& rhs,
                std::vector<std::string>& diff, const std::string& label)
            {
                if (!lhs && !rhs)
                    return;
                if (!lhs || !rhs)
                {
                    diff.push_back(label);
                    return;
                }
                compare_(*lhs, *rhs, diff, label);
            }

            void compare_(const MVCNN::BarrierConfigurationTaskT& lhs, const MVCNN::BarrierConfigurationTaskT& rhs,
                std::vector<std::string>& diff, const std::string& label = "BarrierConfigurationTask");
            void compare_(const MVCNN::BarrierReferenceT& lhs, const MVCNN::BarrierReferenceT& rhs,
                std::vector<std::string>& diff, const std::string& label = "BarrierReference");
            void compare_(const MVCNN::BarrierT& lhs, const MVCNN::BarrierT& rhs,
                std::vector<std::string>& diff, const std::string& label = "Barrier");
            void compare_(const MVCNN::BinaryDataT& lhs, const MVCNN::BinaryDataT& rhs,
                std::vector<std::string>& diff, const std::string& label = "BinaryData");
            void compare_(const MVCNN::ControllerTaskT& lhs, const MVCNN::ControllerTaskT& rhs,
                std::vector<std::string>& diff, const std::string& label = "ControllerTask");
            void compare_(const MVCNN::Conv2DT& lhs, const MVCNN::Conv2DT& rhs,
                std::vector<std::string>& diff, const std::string& label = "Conv2DT");
            void compare_(const MVCNN::CustomT& lhs, const MVCNN::CustomT& rhs,
                std::vector<std::string>& diff, const std::string& label = "Custom");
            void compare_(const MVCNN::GraphFileT& lhs, const MVCNN::GraphFileT& rhs,
                std::vector<std::string>& diff, const std::string& label = "GraphFile");
            void compare_(const MVCNN::IndirectDataReferenceT& lhs, const MVCNN::IndirectDataReferenceT& rhs,
                std::vector<std::string>& diff, const std::string& label = "IndirectDataReference");
            void compare_(const MVCNN::GraphNodeT& lhs, const MVCNN::GraphNodeT& rhs,
                std::vector<std::string>& diff, const std::string& label = "Link");

            void compare_(const MVCNN::MvTensorTaskT& lhs, const MVCNN::MvTensorTaskT& rhs,
                std::vector<std::string>& diff, const std::string& label = "MvTensorTask");
            void compare_(const MVCNN::NCE1ConvT& lhs, const MVCNN::NCE1ConvT& rhs,
                std::vector<std::string>& diff, const std::string& label = "NCE1Conv");
            void compare_(const MVCNN::NCE1FCLT& lhs, const MVCNN::NCE1FCLT& rhs,
                std::vector<std::string>& diff, const std::string& label = "NCE1FCL");
            void compare_(const MVCNN::NCE1PoolT& lhs, const MVCNN::NCE1PoolT& rhs,
                std::vector<std::string>& diff, const std::string& label = "NCE1Pool");
            void compare_(const MVCNN::NCE1TaskT& lhs, const MVCNN::NCE1TaskT& rhs,
                std::vector<std::string>& diff, const std::string& label = "NCE1Task");
            void compare_(const MVCNN::NCE2TaskT& lhs, const MVCNN::NCE2TaskT& rhs,
                std::vector<std::string>& diff, const std::string& label = "NCE2Task");
            void compare_(const MVCNN::NCEInvariantFieldsT& lhs, const MVCNN::NCEInvariantFieldsT& rhs,
                std::vector<std::string>& diff, const std::string& label = "NCEInvariantFields");
            void compare_(const MVCNN::NCEVariantFieldsT& lhs, const MVCNN::NCEVariantFieldsT& rhs,
                std::vector<std::string>& diff, const std::string& label = "NCEVariantFields");
            void compare_(const MVCNN::NNDMATaskT& lhs, const MVCNN::NNDMATaskT& rhs,
                std::vector<std::string>& diff, const std::string& label = "NNDMATask");
            void compare_(const MVCNN::NNTensorTaskT& lhs, const MVCNN::NNTensorTaskT& rhs,
                std::vector<std::string>& diff, const std::string& label = "NNTensorTask");
            void compare_(const MVCNN::PassthroughT& lhs, const MVCNN::PassthroughT& rhs,
                std::vector<std::string>& diff, const std::string& label = "Passthrough");
            void compare_(const MVCNN::PoolingT& lhs, const MVCNN::PoolingT& rhs,
                std::vector<std::string>& diff, const std::string& label = "Pooling");
            void compare_(const MVCNN::PPEAssistT& lhs, const MVCNN::PPEAssistT& rhs,
                std::vector<std::string>& diff, const std::string& label = "PPEAssist"); 
            void compare_(const MVCNN::PPEConfigureT& lhs, const MVCNN::PPEConfigureT& rhs,
                std::vector<std::string>& diff, const std::string& label = "PPEConfigure");
            void compare_(const MVCNN::PPEFixedFunctionT& lhs, const MVCNN::PPEFixedFunctionT& rhs,
                std::vector<std::string>& diff, const std::string& label = "PPEFixedFunction");
            void compare_(const MVCNN::PPETaskT& lhs, const MVCNN::PPETaskT& rhs,
                std::vector<std::string>& diff, const std::string& label = "PPETask");
            void compare_(const MVCNN::ReLUT& lhs, const MVCNN::ReLUT& rhs,
                std::vector<std::string>& diff, const std::string& label = "ReLU");
            void compare_(const MVCNN::ResourcesT& lhs, const MVCNN::ResourcesT& rhs,
                std::vector<std::string>& diff, const std::string& label = "Resources");
            void compare_(const MVCNN::SourceStructureT& lhs, const MVCNN::SourceStructureT& rhs,
                std::vector<std::string>& diff, const std::string& label = "SourceStructure");
            void compare_(const MVCNN::SummaryHeaderT& lhs, const MVCNN::SummaryHeaderT& rhs,
                std::vector<std::string>& diff, const std::string& label = "SummaryHeader");
            void compare_(const MVCNN::TaskListT& lhs, const MVCNN::TaskListT& rhs,
                std::vector<std::string>& diff, const std::string& label = "TaskList");
            void compare_(const MVCNN::TaskT& lhs, const MVCNN::TaskT& rhs,
                std::vector<std::string>& diff, const std::string& label = "Task");
            void compare_(const MVCNN::TensorReferenceT& lhs, const MVCNN::TensorReferenceT& rhs,
                std::vector<std::string>& diff, const std::string& label = "TensorReference");
            void compare_(const MVCNN::TensorT& lhs, const MVCNN::TensorT& rhs,
                std::vector<std::string>& diff, const std::string& label = "Tensor");
            void compare_(const MVCNN::TimerTaskT& lhs, const MVCNN::TimerTaskT& rhs,
                std::vector<std::string>& diff, const std::string& label = "TimerTask");
            void compare_(const MVCNN::UPADMATaskT& lhs, const MVCNN::UPADMATaskT& rhs,
                std::vector<std::string>& diff, const std::string& label = "UPADMATask");
            void compare_(const MVCNN::VersionT& lhs, const MVCNN::VersionT& rhs,
                std::vector<std::string>& diff, const std::string& label = "Version");

        public:

            GraphComparator();
            ~GraphComparator();
            bool compare(const std::string& path1, const std::string& path2);
            bool compare(const char* dataBuffer1, std::size_t length1, const char* dataBuffer2,  std::size_t length2);
            bool compare(const MVCNN::GraphFileT& graph1, const MVCNN::GraphFileT& graph2);
            const std::vector<std::string>& lastDiff() const;
            MVCNN::GraphFileT loadGraphFile(const std::string& path, char* dataBuffer);

        };

    }

}

#endif // MV_TOOLS_GRAPH_COMPARATOR_
