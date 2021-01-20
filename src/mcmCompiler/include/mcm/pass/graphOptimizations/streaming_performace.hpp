#include "pass/lp_scheduler/pipeline_chains_transform.hpp"
#include "include/mcm/pass/graphOptimizations/strategy_utils.hpp"
#include "include/mcm/utils/custom_math.hpp"

namespace mv {
class StreamingPerformance {
public:

    StreamingPerformance(mv::ComputationModel& model, mv::OpModel& omodel);
    void evaluateStreamingOverKStrategies();
    
    
private:
    typedef mv::scheduler::Pipeline_Chains pipeline_chains_t;
    typedef typename pipeline_chains_t::chain_subgraph_t subgraph_t;
    
    mv::ComputationModel& model_;
    mv::OpModel& omodel_;
    std::shared_ptr<mv::Element> globalParams_;
    mv::scheduler::Pipeline_Chains pipelineChains_;
    std::list<subgraph_t> chainSubgraphs_;
    std::map<std::string, size_t> weightsPerClusterPerOp_;
    std::map<size_t, size_t> minWeightsPerClusterPerChain_;
    std::vector<mv::Element> streamingStrategyList_;
    std::vector<mv::Element> multiClusterStrategyList_;
    bool enableChannelMajorConv_;
    size_t nClusters_;

    std::map<size_t, size_t> calculateMininumWeightsSizePerClusterPerChain();
    size_t calculateperClusterWeightsSize(mv::Op& op, const mv::Attribute& clustering, bool weightsSparsity, const mv::Shape& streamConfig);
};
}  