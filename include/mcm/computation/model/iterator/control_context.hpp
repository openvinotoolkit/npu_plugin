#ifndef CONTROL_CONTEXT_HPP_
#define CONTROL_CONTEXT_HPP_

#include "include/mcm/graph/graph.hpp"
#include "include/mcm/computation/model/iterator/model_iterator.hpp"
#include "include/mcm/computation/resource/stage.hpp"

namespace mv
{

    namespace Control
    {

        using OpListIterator = IteratorDetail::OpIterator<computation_graph::second_graph, computation_graph::second_graph::node_list_iterator, ComputationOp, ControlFlow>;
        using OpReverseListIterator = IteratorDetail::OpIterator<computation_graph::second_graph, computation_graph::second_graph::node_reverse_list_iterator, ComputationOp, ControlFlow>;
        using OpDFSIterator = IteratorDetail::OpIterator<computation_graph::second_graph, computation_graph::second_graph::node_dfs_iterator, ComputationOp, ControlFlow>;
        using OpBFSIterator = IteratorDetail::OpIterator<computation_graph::second_graph, computation_graph::second_graph::node_bfs_iterator, ComputationOp, ControlFlow>;
        using OpChildIterator = IteratorDetail::OpIterator<computation_graph::second_graph, computation_graph::second_graph::node_child_iterator, ComputationOp, ControlFlow>;
        using OpParentIterator = IteratorDetail::OpIterator<computation_graph::second_graph, computation_graph::second_graph::node_child_iterator, ComputationOp, ControlFlow>;
        using OpSiblingIterator = IteratorDetail::OpIterator<computation_graph::second_graph, computation_graph::second_graph::node_sibling_iterator, ComputationOp, ControlFlow>;
        
        using FlowListIterator = IteratorDetail::FlowIterator<computation_graph::second_graph, computation_graph::second_graph::edge_list_iterator, ControlFlow, ComputationOp>;
        using FlowReverseListIterator = IteratorDetail::FlowIterator<computation_graph::second_graph, computation_graph::second_graph::edge_reverse_list_iterator, ControlFlow, ComputationOp>;
        using FlowDFSIterator = IteratorDetail::FlowIterator<computation_graph::second_graph, computation_graph::second_graph::edge_dfs_iterator, ControlFlow, ComputationOp>;
        using FlowBFSIterator = IteratorDetail::FlowIterator<computation_graph::second_graph, computation_graph::second_graph::edge_bfs_iterator, ControlFlow, ComputationOp>;
        using FlowChildIterator = IteratorDetail::FlowIterator<computation_graph::second_graph, computation_graph::second_graph::edge_child_iterator, ControlFlow, ComputationOp>;
        using FlowParentIterator = IteratorDetail::FlowIterator<computation_graph::second_graph, computation_graph::second_graph::edge_child_iterator, ControlFlow, ComputationOp>;
        using FlowSiblingIterator = IteratorDetail::FlowIterator<computation_graph::second_graph, computation_graph::second_graph::edge_sibling_iterator, ControlFlow, ComputationOp>;
        
        using StageIterator = IteratorDetail::ModelLinearIterator<map<unsigned_type, allocator::owner_ptr<ComputationStage>>::iterator, ComputationStage>;
        using StageMemberIterator = IteratorDetail::ModelLinearIterator<set<allocator::access_ptr<ComputationElement>, ComputationElement::ElementOrderComparator>::iterator, ComputationElement>;

    }

}

#endif // CONTROL_CONTEXT_HPP_