#ifndef DATA_CONTEXT_HPP_
#define DATA_CONTEXT_HPP_

#include "include/mcm/graph/graph.hpp"
#include "include/mcm/computation/model/iterator/model_iterator.hpp"
#include "include/mcm/computation/flow/data_flow.hpp"
#include "include/mcm/computation/op/op.hpp"

namespace mv
{  

    using dataGraph = graph<Op, DataFlow>;

    namespace Data
    {

        using OpListIterator = IteratorDetail::OpIterator<dataGraph, dataGraph::node_list_iterator, Op, DataFlow>;
        using OpReverseListIterator = IteratorDetail::OpIterator<dataGraph, dataGraph::node_reverse_list_iterator, Op, DataFlow>;
        using OpDFSIterator = IteratorDetail::OpIterator<dataGraph, dataGraph::node_dfs_iterator, Op, DataFlow>;
        using OpBFSIterator = IteratorDetail::OpIterator<dataGraph, dataGraph::node_bfs_iterator, Op, DataFlow>;
        using OpChildIterator = IteratorDetail::OpIterator<dataGraph, dataGraph::node_child_iterator, Op, DataFlow>;
        using OpParentIterator = IteratorDetail::OpIterator<dataGraph, dataGraph::node_parent_iterator, Op, DataFlow>;
        using OpSiblingIterator = IteratorDetail::OpIterator<dataGraph, dataGraph::node_sibling_iterator, Op, DataFlow>;
        
        using FlowListIterator = IteratorDetail::FlowIterator<dataGraph, dataGraph::edge_list_iterator, DataFlow, Op>;
        using FlowReverseListIterator = IteratorDetail::FlowIterator<dataGraph, dataGraph::edge_reverse_list_iterator, DataFlow, Op>;
        using FlowDFSIterator = IteratorDetail::FlowIterator<dataGraph, dataGraph::edge_dfs_iterator, DataFlow, Op>;
        using FlowBFSIterator = IteratorDetail::FlowIterator<dataGraph, dataGraph::edge_bfs_iterator, DataFlow, Op>;
        using FlowChildIterator = IteratorDetail::FlowIterator<dataGraph, dataGraph::edge_child_iterator, DataFlow, Op>;
        using FlowParentIterator = IteratorDetail::FlowIterator<dataGraph, dataGraph::edge_child_iterator, DataFlow, Op>;
        using FlowSiblingIterator = IteratorDetail::FlowIterator<dataGraph, dataGraph::edge_sibling_iterator, DataFlow, Op>;
        
    }

}

#endif // DATA_CONTEXT_HPP_
