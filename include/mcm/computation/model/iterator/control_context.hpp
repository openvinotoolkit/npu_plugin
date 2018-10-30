#ifndef CONTROL_CONTEXT_HPP_
#define CONTROL_CONTEXT_HPP_

#include <memory>
#include <map>
#include <set>
#include <string>
#include "include/mcm/graph/conjoined_graph.hpp"
#include "include/mcm/computation/model/iterator/model_iterator.hpp"

namespace mv
{

    class Op;
    class DataFlow;
    class ControlFlow;
    class Stage;

    namespace Control
    {

        /*using OpListIterator = IteratorDetail::OpIterator<computation_graph::second_graph, computation_graph::second_graph::node_list_iterator, ComputationOp, ControlFlow>;
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
        using FlowSiblingIterator = IteratorDetail::FlowIterator<computation_graph::second_graph, computation_graph::second_graph::edge_sibling_iterator, ControlFlow, ComputationOp>;*/
        
        using OpListIterator = IteratorDetail::OpIterator<computation_graph::second_graph, computation_graph::second_graph::node_list_iterator, Op, ControlFlow>;
        using OpReverseListIterator = IteratorDetail::OpIterator<computation_graph::second_graph, computation_graph::second_graph::node_reverse_list_iterator, Op, ControlFlow>;
        using OpDFSIterator = IteratorDetail::OpIterator<computation_graph::second_graph, computation_graph::second_graph::node_dfs_iterator, Op, ControlFlow>;
        using OpBFSIterator = IteratorDetail::OpIterator<computation_graph::second_graph, computation_graph::second_graph::node_bfs_iterator, Op, ControlFlow>;
        using OpChildIterator = IteratorDetail::OpIterator<computation_graph::second_graph, computation_graph::second_graph::node_child_iterator, Op, ControlFlow>;
        using OpParentIterator = IteratorDetail::OpIterator<computation_graph::second_graph, computation_graph::second_graph::node_child_iterator, Op, ControlFlow>;
        using OpSiblingIterator = IteratorDetail::OpIterator<computation_graph::second_graph, computation_graph::second_graph::node_sibling_iterator, Op, ControlFlow>;
        
        using FlowListIterator = IteratorDetail::FlowIterator<computation_graph::second_graph, computation_graph::second_graph::edge_list_iterator, ControlFlow, Op>;
        using FlowReverseListIterator = IteratorDetail::FlowIterator<computation_graph::second_graph, computation_graph::second_graph::edge_reverse_list_iterator, ControlFlow, Op>;
        using FlowDFSIterator = IteratorDetail::FlowIterator<computation_graph::second_graph, computation_graph::second_graph::edge_dfs_iterator, ControlFlow, Op>;
        using FlowBFSIterator = IteratorDetail::FlowIterator<computation_graph::second_graph, computation_graph::second_graph::edge_bfs_iterator, ControlFlow, Op>;
        using FlowChildIterator = IteratorDetail::FlowIterator<computation_graph::second_graph, computation_graph::second_graph::edge_child_iterator, ControlFlow, Op>;
        using FlowParentIterator = IteratorDetail::FlowIterator<computation_graph::second_graph, computation_graph::second_graph::edge_child_iterator, ControlFlow, Op>;
        using FlowSiblingIterator = IteratorDetail::FlowIterator<computation_graph::second_graph, computation_graph::second_graph::edge_sibling_iterator, ControlFlow, Op>;

        using StageIterator = IteratorDetail::ModelValueIterator<std::map<std::size_t, std::shared_ptr<Stage>>::iterator, Stage>;

    }

}

#endif // CONTROL_CONTEXT_HPP_