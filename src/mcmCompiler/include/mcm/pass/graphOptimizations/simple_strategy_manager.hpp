#ifndef __SIMPLE_STRATEGY_MANAGER_HPP__
#define __SIMPLE_STRATEGY_MANAGER_HPP__

#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "include/mcm/pass/graphOptimizations/StrategyManager.hpp"


namespace mv
{

    namespace graphOptimizer
    {
        struct EnumClassHash final {
            template <typename E>
            size_t operator()(E t) const {
                return std::hash<int32_t>()(static_cast<int32_t>(t));
            }
        };

        class StrategyManagerSimple : public StrategyManager
        {
            public:
            StrategyManagerSimple(OpModel& model,mv::Element& passDesc, mv::TargetDescriptor& td);

            size_t totalClusters=4;
            size_t dpuPerCluster=5;
            std::string referenceDevice = "A0";
            bool globalEnableStreaming=true;
            bool globalEnablePipelining = true;
            bool globalEnablePrefetching = true;
            bool globalEnableWeightsSparsity=true;
            bool globalForceSpilling=false;
            mv::Target target = mv::Target::ma2490;
            double clusterMemory=917504;
            double cmxPipeLineWeightsOverhead=34816.0;
            enum class FailCause
            {
                Pass,
                cmxConcatDecision,
                MemorySize,
                StreamAndClusterComp,
                SpillHKSwitch,
                SOKNotAlign16,
                InputNotSpilled,
                OutputNotSpilled,
                StreamingNotSpilled,
                WorkloadLessKernelSOH,
                ChannelMjr1,
                ChannelMjr2,
                DWChannels,
                SOHheight,
                RequiresSparsity,
                RealSparseForFakeSparseOp,
                DilatedSOH,
                DWLargeStrideReplacementSOK,
                SpiltOverHWithStreamOverK,
                SpiltOverHWithStreamOverHInCMX,
                SparsityKSegmented,
                SparsitySpilling,
                DeConvSubConvSOKHeight,
                SpiltOverHForLayer79InACLNet,
                SpiltOverHForLayer97and113ModelE,
                SpiltOverHForConvModelF,
                SpiltOverKForConvModelF,
                SpiltOverHForFaceDetectionRetail0004,
                SplitOverHOverlappedWronglyComputed,
                SoftwareDeconvolutionSet,
                UpaHKSwitch
            };

            std::unordered_map<FailCause, std::string, :: mv::graphOptimizer::EnumClassHash> failure_causes = {
                {FailCause::Pass, "Pass"},
                {FailCause::cmxConcatDecision, "cmxConcatDecision"},
                {FailCause::MemorySize, "MemorySize"},
                {FailCause::StreamAndClusterComp, "Stream+ClusterComp"},
                {FailCause::SpillHKSwitch, "SpillHKSwitch"},
                {FailCause::SOKNotAlign16, "SOKNotAlign16"},
                {FailCause::InputNotSpilled, "InputNotSpilled"},
                {FailCause::OutputNotSpilled, "OutputNotSpilled"},
                {FailCause::StreamingNotSpilled, "StreamingNotSpilled"},
                {FailCause::WorkloadLessKernelSOH, "Workload<KernelSOH"},
                {FailCause::ChannelMjr1, "ChannelMjr1"},
                {FailCause::ChannelMjr2, "ChannelMjr2"},
                {FailCause::DWChannels, "DWChannels"},
                {FailCause::SOHheight, "SOHheight"},
                {FailCause::RequiresSparsity, "RequiresSparsity"},
                {FailCause::RealSparseForFakeSparseOp, "RealSparseForFakeSparseOp"},
                {FailCause::DilatedSOH, "DilatedSOH"},
                {FailCause::DWLargeStrideReplacementSOK, "DWLargeStrideReplacementSOK"},
                {FailCause::SpiltOverHWithStreamOverK, "SpiltOverHWithStreamOverK"},
                {FailCause::SpiltOverHWithStreamOverHInCMX, "SpiltOverHWithStreamOverHInCMX"},
                {FailCause::SparsityKSegmented, "SparsityKSegmented"},
                {FailCause::SparsitySpilling, "SparsitySpilling"},
                {FailCause::DeConvSubConvSOKHeight, "DeConvSubConvSOKHeight"},
                {FailCause::SpiltOverHForLayer79InACLNet, "SpiltOverHForLayer79InACLNet"},
                {FailCause::SpiltOverHForLayer97and113ModelE, "SpiltOverHForLayer97and113ModelE"},
                {FailCause::SpiltOverHForConvModelF, "SpiltOverHForConvModelF"},
                {FailCause::SpiltOverKForConvModelF, "SpiltOverKForConvModelF"},
                {FailCause::SpiltOverHForFaceDetectionRetail0004, "SpiltOverHForFaceDetectionRetail0004"},
                {FailCause::SplitOverHOverlappedWronglyComputed, "SplitOverHOverlappedWronglyComputed"},
                {FailCause::SoftwareDeconvolutionSet, "SoftwareDeconvolutionSet"},
                {FailCause::UpaHKSwitch, "UpaHKSwitch"}
            };

            bool requiresActivationSparsity(Op& op, std::string clustering);
            bool requiresWeightsSparsity(Op& op);
            bool requiresCompilerActivationSparsity(Op& op);
            bool requiresRealActivationSparsity(Op& op, std::string clustering);
            bool requiresFakeActivationSparsity(Op& op);
            bool decideWeightsSparsity(mv::Op op, float, float);
            bool decideWeightsPipelineable(mv::Op& op, StrategySet& strategy, bool parentSpilling);
            bool decideCMXable(mv::Op& op, bool input);

            std::vector<size_t> getStreamsOverH(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity,
                                    bool wSparsity, bool fSparsity, bool spilling, bool parentSpilling);
            unsigned getMinStreamOverH(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity,
                            bool wSparsity, bool fSparsity, bool spilling, bool parentSpilling);
            bool validateHStream(mv::Op& op, mv::Attribute clustering, std::size_t splits);
            std::vector<std::size_t> getStreamsOverK(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity,
                                    bool wSparsity, bool fSparsity, bool spilling, bool parentSpilling);
            unsigned getMinStreamOverK(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity,
                                    bool wSparsity, bool fSparsity, bool spilling, bool parentSpilling, bool pipelined = false);
            unsigned getNextStreamOverK(mv::Op& op, mv::Attribute clustering, size_t startSplit, bool spilling);
            bool validateKStream(mv::Op& op, mv::Attribute clustering, size_t split, bool spilling);
            std::vector<std::size_t> getStreamsOverC(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity,
                                    bool wSparsity, bool fSparsity, bool spilling);
            unsigned getMinStreamOverC(mv::Op& op, mv::Attribute clustering, Shape streams, bool iSparsity, bool oSparsity,
                                    bool wSparsity, bool fSparsity, bool spilling);
            std::vector<size_t> getMaxStreamOverK(mv::Op& op);

            bool createStrategyFromBool(mv::Op op, std::string name);
            std::vector<Attribute> createTFStrategyPoolFromBool(mv::Op op,std::string name);
            std::vector<mv::Attribute> createTStrategyPoolFromBool(mv::Op op,std::string name);
            std::vector<mv::Attribute> createStrategyPoolFromStrategySet(mv::Op op, std::string name);

            FailCause validateStrategy(mv::Op& op,StrategySet& strategy);
            bool isSOKCompatible(mv::Op& op);
            void generateStrategySetForLayer(mv::Op& op,std::vector<StrategySet>& strategyVec);

            int8_t checkInOutSizes(mv::Op& op, size_t input_gate);
            int8_t checkKernelSizes(mv::Op& op);
            int8_t checkStrideSizes(mv::Op& op);
            int8_t checkHWUnsupportedOp(mv::Op& op);
            
        };
    }
}

#endif //__STRATEGY_MANAGER_HPP__