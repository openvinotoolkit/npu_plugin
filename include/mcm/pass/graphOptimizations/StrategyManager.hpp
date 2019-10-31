#ifndef __STRATEGY_MANAGER_HPP__
#define __STRATEGY_MANAGER_HPP__

#include "include/mcm/op_model.hpp"
#include "include/mcm/base/exception/argument_error.hpp"
#include "math.h"
#include <unordered_set>
#include "tuple"
#include "limits"
#include <atomic>
#include <functional>

namespace mv {
namespace graphOptimizer  {

using namespace std;
using namespace std::placeholders;

class MetaEdge
{
private:
    using StrategySet       = unordered_map<string,Attribute>;
    using OptimizationGraph = graph<StrategySet&,MetaEdge>;
    using CriticalPathNodes = vector<OptimizationGraph::node_list_iterator>;

    static std::atomic<int> unique_ctr;

    double cost_;
    CriticalPathNodes criticalPath_;
    int id;

public:

    MetaEdge(double cost) :
        cost_(cost),
        criticalPath_(0)
    {
        id = (unique_ctr++);
    }

    //since current dijkstra works with edgeCostMapping, arithmetic operators will be on cost_ member
    //and relational operators will work with the unique id

    bool operator==(const MetaEdge& other);
    bool operator<(const MetaEdge& other);
    bool operator>(const MetaEdge& other);
    bool operator!=(const MetaEdge& other);

    double operator+(const double other);
    double operator+(const MetaEdge& other);

    void extend(const CriticalPathNodes& childCriPath);
    const CriticalPathNodes& criticalPaths() const;
    double cost() const;
};

class MetaGraph  {

    using StrategySet       = unordered_map<string,Attribute>;
    using OptimizationGraph = graph<StrategySet&,MetaEdge>;
    using CriticalPathNodes = vector<OptimizationGraph::node_list_iterator>;
    using CriticalEdges     = vector<OptimizationGraph::edge_list_iterator>;

public:

    //helper internal structs
    struct StrategySetPair
    {
        StrategySet* parent;
        StrategySet* child;
        StrategySetPair(StrategySet* first,StrategySet* second) : parent(first) , child(second) {} ;
        StrategySetPair() : parent(nullptr) , child(nullptr) {};
        void operator=(const StrategySetPair& other);
        void print() const;
    };

    struct StrategySetHash { size_t operator()(const StrategySetPair& val) const; };
    struct StrategytSetCompare { bool operator()(const StrategySetPair& lhs, const StrategySetPair& rhs) const; };
    struct costEdgeIteratorComp { bool operator()(const OptimizationGraph::edge_list_iterator lhs, const OptimizationGraph::edge_list_iterator rhs) const; };
    struct costNodeIteratorComp { bool operator()(const OptimizationGraph::node_list_iterator lhs, const OptimizationGraph::node_list_iterator rhs) const; };

    struct GraphLevel
    {
        Op* op;
        vector<OptimizationGraph::node_list_iterator> level;
        GraphLevel() : op(nullptr),level(0) {};
        GraphLevel(Op* opPtr) : op(opPtr),level(0) {};
    };

    struct CriticalPath
    {
        shared_ptr<CriticalPathNodes> nodes;
        double cost;
        CriticalPath(shared_ptr<CriticalPathNodes> criticalNodes,double pathCost) : nodes(criticalNodes),cost(pathCost) {};
        CriticalPath() : nodes(0), cost(-1) {};
    };

    //graphs will need to be uniquely identifieable
    static std::atomic<int> unique_ctr;

    OptimizationGraph internalGraph_;
    map<typename OptimizationGraph::edge_list_iterator, double, costEdgeIteratorComp> edgeCostMap;
    vector<GraphLevel> levels;
    vector<shared_ptr<vector<StrategySet>>> levelContainer_;
    unordered_map<StrategySetPair,CriticalPath,StrategySetHash,StrategytSetCompare> criticalPaths_;

    std::vector<shared_ptr<MetaGraph>> childMetaGraphs;

    unsigned firstLevelIdx_;
    unsigned lastLevelIdx_;

    string name;
    bool solved_;

public:

    MetaGraph() :
        internalGraph_(),
        edgeCostMap(),
        childMetaGraphs(0),
        firstLevelIdx_(0),
        lastLevelIdx_(0),
        levels(0),
        levelContainer_(0),
        name("unnamed"),
        solved_(false)
    {
    }

    void addNewLevel(Op& op,shared_ptr<vector<StrategySet>> newLevel,function<double(Op&,Op&,StrategySet&,StrategySet&)> cost);
    void solve();
    void fuseMeta(shared_ptr<MetaGraph> childGraph);
    shared_ptr<CriticalPath> getLowestCriticalPathExtended();
    void write(string dotFileLocation,bool skipInf);
};

class StrategyManager : public LogSender
{
public:
    using GlobalSetting     = unordered_map<string,Attribute>;
    using StrategySet       = unordered_map<string,Attribute>;
    using OptimizationGraph = graph<StrategySet&,MetaEdge>;
    using CriticalPathNodes = vector<OptimizationGraph::node_list_iterator>;
    using LayerStrategySet  = unordered_map<string,StrategySet>;
    using SubGraph = tuple<mv::Data::OpListIterator,mv::Data::OpListIterator,vector<mv::Data::OpListIterator>>;

    static constexpr auto inf_ = numeric_limits<double>::infinity();
    static std::atomic<int> unique_ctr;

    GlobalSetting globalConfig_;
    GlobalSetting globalStrategies_;
    LayerStrategySet layerStrategies_;

    OpModel& model_;
    mv::Element& passDesc_;

    string dotFileLocation;
    string jsonOutFileName;

    StrategyManager(OpModel& model,mv::Element& passDesc);

    //strategy management helper getters/setters
    Attribute& getStrategy(mv::Op op,string strategy);
    void setGlobalConfig(string& name,Attribute& config);
    void setGlobalStrategy(string& name, Attribute& strategy);
    const Attribute& getGlobalConfig(const string& name) const;
    const Attribute& getGlobalStrategy(const string& name) const;
    const StrategySet& getLayerStrategySet(const string& name) const;
    bool hasAttr(const GlobalSetting& map,const string& name) const;
    bool hasAttr(const LayerStrategySet& map,const string& name) const;
    std::string getLogID() const;

    //strategy management methods
    void updateValuesFromJSON();
    void updateDefaultValues();
    void printStrategy();
    std::vector<mv::Element> convertStreamingStrategyToElement(CriticalPathNodes& strategiesToConvert, std::shared_ptr<mv::Element> compDesc);
    std::vector<mv::Element> convertClusteringStrategyToElement(CriticalPathNodes& strategiesToConvert, std::shared_ptr<mv::Element> compDesc);
    std::vector<mv::Element> convertLocationStrategyToElement(CriticalPathNodes& strategiesToConvert);
    std::vector<mv::Element> convertSparsityStrategyToElement(CriticalPathNodes& strategiesToConvert);
    void saveStrategyToJsonFile(std::vector<mv::Element> &stategiesToSave,std::string jsonOutputFileName);
    void saveStrategyToCompilationDescriptor(vector<mv::Element> &stategiesToSave, std::shared_ptr<mv::Element> compDesc);
    void saveMetaStrategy(CriticalPathNodes& criticalPathNodes);

//    string strategyString(OptimizationGraphNode n);

    //Graph parsing methods
    void initLayerStrategySets();
    bool isLinearGraph(mv::Data::OpListIterator opBegin,mv::Data::OpListIterator opEnd,vector<mv::Data::OpListIterator> children);
    mv::Data::OpListIterator naiveLCA(vector<mv::Data::OpListIterator> children,mv::Data::OpListIterator opEnd);
    mv::Data::OpListIterator naiveLCA(mv::Data::OpListIterator nodeA,mv::Data::OpListIterator nodeB,mv::Data::OpListIterator opEnd);
    shared_ptr<vector<SubGraph>> extractSubgraphs(mv::Data::OpListIterator opBegin,mv::Data::OpListIterator opEnd,std::vector<mv::Data::OpListIterator> childIdx);
    std::shared_ptr<MetaGraph> linearGraphSolver(mv::Data::OpDFSIterator opBegin,mv::Data::OpDFSIterator opEnd,mv::Data::OpDFSIterator firstChild);
    std::shared_ptr<MetaGraph> recursiveGraphSolver(mv::Data::OpListIterator opBegin, mv::Data::OpListIterator opEnd);
    std::shared_ptr<MetaGraph> recursiveGraphSolver(mv::Data::OpListIterator opBegin, mv::Data::OpListIterator opEnd,std::vector<mv::Data::OpListIterator> childIdx);
    void graphParameterOptimizations();

    //template methods to be overwritten
    virtual void generateStrategySetForLayer(mv::Op& op,vector<StrategySet>& strategyVec);
    virtual double transitionCost(Op& parentOp,Op& childOp,StrategySet& parent,StrategySet& child);
    virtual ~StrategyManager() {};
};


}
}

#endif //__STRATEGY_MANAGER_HPP__
