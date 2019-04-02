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

        TensorInterferenceGraphNode(std::string name_) : name(name_), weight(0), neighborsWeight(0), address(0), isColored(false) {

        }

        TensorInterferenceGraphNode(const mv::TensorInterferenceGraphNode& rhs) {
            name = rhs.name;
            weight = rhs.weight;
            neighborsWeight = rhs.neighborsWeight;
            address = rhs.address;
            isColored = rhs.isColored;
        }

        bool operator==(const mv::TensorInterferenceGraphNode& rhs) const
        {
            return (name == rhs.name);
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
                const TensorIteratorFilter& tensorFilter);
            bool checkNodesAreNeighbors_(TensorInterferenceGraph::node_list_iterator& n1, TensorInterferenceGraph::node_list_iterator& n2);
            bool checkNodeInterference_(ComputationModel& model, const std::string& tensor1, const std::string& tensor2);
            bool isTensorInTopNames_(const std::vector<Data::TensorIterator>& tensorList, ComputationModel& model, const std::string tensorName);
            bool isSinkNode_(Data::OpListIterator& opIterator);
            void genIntereferenceGraph_(ComputationModel& model , const TensorIteratorFilter& tensorFilter,const OpIteratorFilter& taskFilter);
            std::set<std::string> getTensorNames_(ComputationModel& model, const TensorIteratorFilter& tensorFilter, const OpIteratorFilter& taskFilter);
            void cleanupDMATensorNodes_();
            void addWeightsToInterferenceGraph_(ComputationModel& model, std::size_t alignment);
            std::size_t  getNeighborsWeight_(ComputationModel& model, std::string& node, std::size_t alignment);
            void buildCompleteGraph_(std::set<std::string> tensorNames);

        public:
            TensorInterferenceGraph() : graph<mv::TensorInterferenceGraphNode, int>() {}
            TensorInterferenceGraph(ComputationModel& model, std::size_t alignment, const TensorIteratorFilter& tensorFilter = nullptr,
                const mv::OpIteratorFilter& taskFilter = nullptr, bool isCompleteTig = false);

            TensorInterferenceGraph(const mv::TensorInterferenceGraph& g);
            void drawGraph(std::string outputFile);
            void printGraph(std::string name);
    };
}
#endif