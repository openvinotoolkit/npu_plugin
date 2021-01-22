#include "pass/lp_scheduler/pipeline_chains_transform.hpp"
#include "include/mcm/pass/graphOptimizations/strategy_utils.hpp"
#include "include/mcm/utils/custom_math.hpp"

namespace mv {
class StreamingPerformance {
public:
    StreamingPerformance(mv::ComputationModel& model, mv::OpModel& omodel);
    ~StreamingPerformance();
    void increaseStreamingOverKforPerformance();

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
    std::vector<mv::Element> tensorMemoryLocation_;
    std::vector<mv::Element> newStrategies_;
    const bool enableChannelMajorConv_;
    const size_t nClusters_;
    FILE* fptr_ = nullptr;
    const size_t minWeightsPerClusterPerChainConstant_ = 66560;  
    const std::map<std::string, size_t> minOutputChannels_ = {
            {"SplitOverK", 64}, {"Clustering", 16}, {"SplitOverH", 16}, {"HKSwitch", 16}};

    std::map<size_t, size_t> calculateMininumWeightsSizePerClusterPerChain();

    size_t calculateperClusterWeightsSize(mv::Op& op, const mv::Attribute& clustering, const bool weightsSparsity,
                                          const mv::Shape& streamConfig);

    std::tuple<std::vector<mv::Element>, mv::Attribute, bool> getGraphOptimizerAssignedStategies(const std::string opName);

    std::pair<size_t, double> calculatefullWeightsSizeForOpandOptimalKStreaming(const std::string multiclusterStrategy,
                                                                                const size_t weightsPerClusterforOp,
                                                                                const size_t minWeightsPerClusterPerChain,
                                                                                const bool isKStreaming,
                                                                                const int numberOfkStreams);

    void writeStatsToFile(const unsigned chainID, const std::string opName, const int kStreaming, const int hStreaming,
                          const std::string multiclusterStrategy,const size_t fullweightsSize, const size_t alignedFullOutputChannels,
                          const size_t weightsPerClusterPerOp, size_t minWeightsPerClusterPerChain,
                          const double optimalNumberOfKStreams,const double maxpossibleStreams,const double newKStreams);

    void evaluateGraphOptimizerAssignedKStreamingStrategies();
    void assignNewSrategies();
};
}  // namespace mv