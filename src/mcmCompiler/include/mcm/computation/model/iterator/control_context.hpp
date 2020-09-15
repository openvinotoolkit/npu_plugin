#ifndef CONTROL_CONTEXT_HPP_
#define CONTROL_CONTEXT_HPP_

#include <memory>
#include <map>
#include <set>
#include <string>
#include "include/mcm/graph/graph.hpp"
#include "include/mcm/computation/model/iterator/model_iterator.hpp"
#include "include/mcm/computation/flow/control_flow.hpp"
#include "include/mcm/computation/op/op.hpp"

namespace mv
{

    using controlGraph = graph<Op, ControlFlow>;

    namespace Control
    {
        
        using OpListIterator = IteratorDetail::OpIterator<controlGraph, controlGraph::node_list_iterator, Op, ControlFlow>;
        using OpReverseListIterator = IteratorDetail::OpIterator<controlGraph, controlGraph::node_reverse_list_iterator, Op, ControlFlow>;
        using OpDFSIterator = IteratorDetail::OpIterator<controlGraph, controlGraph::node_dfs_iterator, Op, ControlFlow>;
        using OpBFSIterator = IteratorDetail::OpIterator<controlGraph, controlGraph::node_bfs_iterator, Op, ControlFlow>;
        using OpChildIterator = IteratorDetail::OpIterator<controlGraph, controlGraph::node_child_iterator, Op, ControlFlow>;
        using OpParentIterator = IteratorDetail::OpIterator<controlGraph, controlGraph::node_parent_iterator, Op, ControlFlow>;
        using OpSiblingIterator = IteratorDetail::OpIterator<controlGraph, controlGraph::node_sibling_iterator, Op, ControlFlow>;
        
        using FlowListIterator = IteratorDetail::FlowIterator<controlGraph, controlGraph::edge_list_iterator, ControlFlow, Op>;
        using FlowReverseListIterator = IteratorDetail::FlowIterator<controlGraph, controlGraph::edge_reverse_list_iterator, ControlFlow, Op>;
        using FlowDFSIterator = IteratorDetail::FlowIterator<controlGraph, controlGraph::edge_dfs_iterator, ControlFlow, Op>;
        using FlowBFSIterator = IteratorDetail::FlowIterator<controlGraph, controlGraph::edge_bfs_iterator, ControlFlow, Op>;
        using FlowChildIterator = IteratorDetail::FlowIterator<controlGraph, controlGraph::edge_child_iterator, ControlFlow, Op>;
        using FlowParentIterator = IteratorDetail::FlowIterator<controlGraph, controlGraph::edge_child_iterator, ControlFlow, Op>;
        using FlowSiblingIterator = IteratorDetail::FlowIterator<controlGraph, controlGraph::edge_sibling_iterator, ControlFlow, Op>;

    }

}

#endif // CONTROL_CONTEXT_HPP_
