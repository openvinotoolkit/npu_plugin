#ifndef __HEURSITIC_STRATEGY_HPP__
#define __HEURISTIC_STRATEGY_HPP__

#include <unordered_map>
#include <queue>
#include <stack>

using StrategySet       = std::unordered_map<std::string,mv::Attribute>;
using StrategyMap       = std::unordered_map<std::string, StrategySet>;

namespace mv
{
namespace graphOptimizer
{

class HeuristicGraphOptimizer : public LogSender 
{
    public:
    OpModel& model_;
    mv::Element& passDesc_;


    std::vector<std::shared_ptr<std::vector<StrategySet>>> allStrategies_;
    std::unordered_map<std::string, std::vector<StrategySet>> strategy_model_;
    std::unordered_map<std::string, StrategySet> bestStrategies_;
    std::size_t totalClusters_;
    std::size_t dpuPerCluster_;
    int clusterMemory_;
    double COST_MAX = numeric_limits<double>::infinity();
    // TODO for now, these are set to static values, works well enough for first pass
    // Next step to port in a simple model of individual task performance from GO or Archbench
    static constexpr double SPARSE_COST = 1.05;
    double SPILL_COST = 1.5;
    double SOH_HEURISTIC_MULTIPLIER = 1.0;
    double CMX_BANDWIDTH_;
    double DDR_BANDWIDTH_;
    double LATENCY_;
    double LATENCY_DDR_;
    double PIPELINE_STAGES;
    mv::Target target = mv::Target::ma2490;
    std::string referenceDevice_ = "A0";
    const std::unordered_map<std::string, double> clusteringStrategyCost = 
                                        {
                                            {"SplitOverHOverlapped", 0.95},
                                            {"SplitOverH", 1.0},
                                            {"HKSwitch", 1.05},
                                            {"SplitOverK", 4.0},
                                            {"Clustering", 10.0}
                                        };

    HeuristicGraphOptimizer(OpModel&, mv::Element&);
    StrategyMap& getChosenStrategies();
    std::string getLogID() const;
    void init(mv::TargetDescriptor& td);

    void assignMultiClusteringGreedy();
    void forceConnectedSOH();
    void abandonSOH(mv::Data::OpListIterator opIt, bool allowHK);
    bool addSpillsAtStrategyTransitions();
    void chooseRollbackOrSpill();
    void verifySpillStrategies(bool lockClusteringStrategy);
    void alignAndValidateSpecialOps();
    void serviceActivationSparsity();
    void costActivationSparsity();
    void increaseWeightsPipelining();
    bool findSparseOutput(mv::Data::OpListIterator opIt);
    bool findDenseOutput(mv::Data::OpListIterator opIt);
    bool findRealSparseInput(mv::Data::OpListIterator opIt, bool doAssignment);
    StrategySet assignStrategyCost(mv::Data::OpListIterator opIt, 
                                        std::vector<mv::graphOptimizer::StrategyManager::StrategySet>& opStrategies);
    double computeTime(mv::Data::OpListIterator opIt, StrategySet& strategySet);
    double dmaTime(mv::Data::OpListIterator opIt, StrategySet& strategySet);
    double outputDmaTime(mv::Data::OpListIterator opIt, StrategySet& strategySet, bool forceSpill);
    double inputDmaTime(mv::Data::OpListIterator opIt, StrategySet& strategySet);
    double averageWeightsDmaTime(mv::Data::OpListIterator opIt, StrategySet& strategySet);
    double averageOutputDmaTime(mv::Data::OpListIterator opIt, StrategySet& strategySet);
    bool resolveSpillingIncompatability(mv::Data::OpListIterator child, mv::Data::OpListIterator parent, bool lockClusteringStrategy);
    void processForSpillRemoval(mv::Data::OpListIterator opIt);
    double getMultiplier(mv::Data::OpListIterator opIt);
    bool canServiceActivationSparsity(mv::Data::OpListIterator opIt, StrategySet& strategy);
    bool requiresRealActivationSparsity(mv::Data::OpListIterator opIt, StrategySet& strategy);
    bool requiresCompilerActivationSparsity(mv::Data::OpListIterator opIt, StrategySet& strategy);
    bool requiresSparseInput(mv::Data::OpListIterator opIt, StrategySet& strategy);


    bool attemptToSpillOp(mv::Data::OpListIterator opIt, bool lockClustering);
    bool isKCompatible(mv::Data::OpListIterator opIt, bool allowHK = false);
    bool couldBeKCompatible(mv::Data::OpListIterator opIt);
    std::pair<double, StrategySet> findKCompatible(mv::Data::OpListIterator opIt, bool doAssignment, bool allowHK);
    double findHCompatible(mv::Data::OpListIterator opIt, bool doAssignment, bool allowHK);
    bool isHK(mv::Data::OpListIterator opIt);
    bool isRemoveableSpill(mv::Data::OpListIterator opIt);
    bool isCMXable(mv::Data::OpListIterator opIt, StrategySet& strategy, bool isInput);
    bool hasGreedySOK(mv::Data::OpListIterator opIt);
    bool isGreedyEligible(mv::Data::OpListIterator opIt);
    void doSingleRollback(mv::Data::OpListIterator opIt);
    bool checkMultipleInputOp(mv::Data::OpListIterator opIt);
    bool forceRollback(mv::Data::OpListIterator opIt);
    bool assignBestStrategyOfType(mv::Data::OpListIterator opIt, std::string clusteringStrategy);
    bool hasLayerWorkaroundAvoidPipeline(mv::Data::OpListIterator opIt, StrategySet& strategy);
    bool hasLayerWorkaroundAvoidStrategy(mv::Data::OpListIterator opIt, StrategySet& strategy);
    bool strategyChangeRequiresSpill(mv::Data::OpListIterator& opIt, mv::Data::OpListIterator& pIt);
    double findBestStrategyOfLocation(mv::Data::OpListIterator opIt, bool doAssignment, bool inputDDR, bool lockOutput, bool outputDDR, bool lockClustering, std::string clustering);
};

}
}
#endif //__HEURISTIC_STRATEGY_HPP__