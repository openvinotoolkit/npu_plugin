#ifndef INTERFERENCE_GRAPH_HPP_
#define INTERFERENCE_GRAPH_HPP_

#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include <unordered_set>

namespace mv
{
    struct TensorInterferenceGraphNode
    {
        std::string name;
        size_t weight;
        size_t neighborsWeight;
        size_t address;
        size_t height;
        bool isSelected;
        bool isColored;

        TensorInterferenceGraphNode() :
            name(""),
            weight(0),
            neighborsWeight(0),
            address(0),
            height(0),
            isSelected(false),
            isColored(false)
        {
        }

        TensorInterferenceGraphNode(std::string name_) :
            name(name_),
            weight(0),
            neighborsWeight(0),
            address(0),
            height(0),
            isSelected(false),
            isColored(false)
        {
        }

        TensorInterferenceGraphNode(const mv::TensorInterferenceGraphNode& rhs) {
            name = rhs.name;
            weight = rhs.weight;
            neighborsWeight = rhs.neighborsWeight;
            height = rhs.height;
            address = rhs.address;
            isSelected = rhs.isSelected;
            isColored = rhs.isColored;
        }

        TensorInterferenceGraphNode(mv::TensorInterferenceGraphNode&& rhs) noexcept : TensorInterferenceGraphNode{}
        {
            swap(rhs);
        }

        bool operator==(const mv::TensorInterferenceGraphNode& rhs) const
        {
            return (name == rhs.name);
        }

        TensorInterferenceGraphNode& operator=(TensorInterferenceGraphNode rhs)
        {
            swap(rhs);
            return *this;
        }

        void swap(TensorInterferenceGraphNode& rhs)
        {
            name.swap(rhs.name);
            std::swap(weight, rhs.weight);
            std::swap(height, rhs.height);
            std::swap(neighborsWeight, rhs.neighborsWeight);
            std::swap(address, rhs.address);
            std::swap(isColored, rhs.isColored);
            std::swap(isSelected, rhs.isSelected);
        }
        void print() const
        {
            std::cout << " name " << name << " address " << address << " weight "
                << weight << " neightborsWeight " << neighborsWeight << " height "
                << height << " isColored " << isColored << " isSelected " << isSelected << std::endl;
        }
    };
    class TensorInterferenceGraph;
    using TensorIteratorFilter = std::function<bool(const mv::Data::TensorIterator& t)>;
    using OpIteratorFilter = std::function<bool(mv::Data::OpListIterator& t)>;
    using SinkOpIteratorFilter = std::function<bool(mv::Data::OpListIterator& t)>;
    using OrderingStrategyFunc = std::function<std::vector<mv::TensorInterferenceGraphNode>(mv::TensorInterferenceGraph& g)>;

    class TensorInterferenceGraph : public mv::graph<TensorInterferenceGraphNode, int>
    {
        private:
            struct pair_hash
            {
                template <class T1, class T2>
                std::size_t operator() (const std::pair<T1, T2> &pair) const
                {
                    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
                }
            };
            std::unordered_map<std::string, std::string> topMasterMap_;
            std::unordered_set<std::pair<std::string, std::string>, pair_hash> cmTransitiveClosureSet_;
            std::unordered_map<std::string, node_list_iterator> nodeIteratorsMap_;

            std::string getTensorTopMaster_(const Data::TensorIterator& t, DataModel& dm);
            std::unordered_set<std::string> getTaskTopTensors_(const std::vector<Data::TensorIterator>& tensorList, ComputationModel& model,
                DataModel& dm, const TensorIteratorFilter& tensorFilter, bool isDMA);

            void cmTransitiveClosure_(mv::ComputationModel& model);
            void cmTransitiveClosureHelper_(mv::OpModel& om, mv::ControlModel& cm, std::string source, std::string target);
            bool checkNodesAreNeighbors_(TensorInterferenceGraph::node_list_iterator& n1, TensorInterferenceGraph::node_list_iterator& n2);
            bool checkNodesDontInterfere_(std::unordered_set<std::string>& sourceNodeNames, std::unordered_set<std::string>& sinkNodeNames);
            bool isTensorInTopNames_(const std::vector<Data::TensorIterator>& tensorList, DataModel& model, const std::string tensorName);
            void genIntereferenceGraph_(const mv::pass::PassEntry& pass, ComputationModel& model , const TensorIteratorFilter& tensorFilter,const OpIteratorFilter& taskFilter, const SinkOpIteratorFilter& sinkFilter, bool isDMA);
            std::set<std::string> getTensorNames_(ComputationModel& model, const TensorIteratorFilter& tensorFilter, const OpIteratorFilter& taskFilter, bool isDMA);
            void addWeightsToInterferenceGraph_(const mv::pass::PassEntry& pass, ComputationModel& model, std::size_t alignment);
            std::size_t  getNeighborsWeight_(std::string& node);
            void buildCompleteGraph_(std::set<std::string> tensorNames);
            bool checkIsCMXTensor_(const Data::TensorIterator tensorIt);

        public:
            TensorInterferenceGraph() : graph<mv::TensorInterferenceGraphNode, int>() {}
            TensorInterferenceGraph(const mv::pass::PassEntry& pass, ComputationModel& model, std::size_t alignment, const TensorIteratorFilter& tensorFilter = nullptr,
                const mv::OpIteratorFilter& taskFilter = nullptr, const SinkOpIteratorFilter& sinkFilter = nullptr, bool isCompleteTig = false, bool isDMA = false);

            TensorInterferenceGraph(const mv::TensorInterferenceGraph& g);
            void drawGraph(std::string outputFile);
            void printGraph(std::string name);
    };
}
#endif
