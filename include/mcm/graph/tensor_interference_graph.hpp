#ifndef INTERFERENCE_GRAPH_HPP_
#define INTERFERENCE_GRAPH_HPP_

#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

namespace mv
{
    struct TensorInterferenceGraphNode
    {
        std::string name;
        size_t weight;
        size_t neighborsWeight;
        size_t address;
        size_t height;
        bool isColored;

        TensorInterferenceGraphNode() :
            name(""),
            weight(0),
            neighborsWeight(0),
            address(0),
            height(0),
            isColored(false)
        {
        }

        TensorInterferenceGraphNode(std::string name_) :
            name(name_),
            weight(0),
            neighborsWeight(0),
            address(0),
            height(0),
            isColored(false)
        {
        }

        TensorInterferenceGraphNode(const mv::TensorInterferenceGraphNode& rhs) {
            name = rhs.name;
            weight = rhs.weight;
            neighborsWeight = rhs.neighborsWeight;
            height = rhs.height;
            address = rhs.address;
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
        }
        void print() const
        {
            std::cout << " name " << name << " address " << address << " weight "
                << weight << " neightborsWeight " << neighborsWeight << " height "
                << height << " isColored " << isColored << std::endl;
        }
    };
    class TensorInterferenceGraph;
    using TensorIteratorFilter = std::function<bool(const mv::Data::TensorIterator& t)>;
    using OpIteratorFilter = std::function<bool(mv::Data::OpListIterator& t)>;
    using OrderingStrategyFunc = std::function<std::vector<mv::TensorInterferenceGraphNode>(mv::TensorInterferenceGraph& g)>;

    class TensorInterferenceGraph : public mv::graph<TensorInterferenceGraphNode, int>
    {
        private:
            std::string getTensorTopMaster_(const Data::TensorIterator& t, ComputationModel& model);
            std::set<std::string> getTaskTopTensors_(const std::vector<Data::TensorIterator>& tensorList, ComputationModel& model,
                const TensorIteratorFilter& tensorFilter, bool isDMA);
            bool checkNodesAreNeighbors_(TensorInterferenceGraph::node_list_iterator& n1, TensorInterferenceGraph::node_list_iterator& n2);
            bool checkNodesDontInterfere_(mv::ComputationModel& model, const std::string& tensor1, const std::string& tensor2, std::set<std::string>& sourceNodeNames, std::set<std::string>& sinkNodeNames);
            bool isTensorInTopNames_(const std::vector<Data::TensorIterator>& tensorList, ComputationModel& model, const std::string tensorName);
            bool isSinkNode_(Data::OpListIterator& opIterator);
            void genIntereferenceGraph_(const mv::pass::PassEntry& pass, ComputationModel& model , const TensorIteratorFilter& tensorFilter,const OpIteratorFilter& taskFilter, bool isDMA);
            std::set<std::string> getTensorNames_(ComputationModel& model, const TensorIteratorFilter& tensorFilter, const OpIteratorFilter& taskFilter, bool isDMA);
            void addWeightsToInterferenceGraph_(const mv::pass::PassEntry& pass, ComputationModel& model, std::size_t alignment);
            std::size_t  getNeighborsWeight_(std::string& node);
            void buildCompleteGraph_(std::set<std::string> tensorNames);
            bool checkIsCMXTensor_(const Data::TensorIterator tensorIt);

        public:
            TensorInterferenceGraph() : graph<mv::TensorInterferenceGraphNode, int>() {}
            TensorInterferenceGraph(const mv::pass::PassEntry& pass, ComputationModel& model, std::size_t alignment, const TensorIteratorFilter& tensorFilter = nullptr,
                const mv::OpIteratorFilter& taskFilter = nullptr, bool isCompleteTig = false, bool isDMA = false);

            TensorInterferenceGraph(const mv::TensorInterferenceGraph& g);
            void drawGraph(std::string outputFile);
            void printGraph(std::string name);
    };
}
#endif