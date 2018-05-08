#ifndef ITERATOR_HPP_
#define ITERATOR_HPP_

#include "include/fathom/graph/graph.hpp"
#include "include/fathom/computation/model/types.hpp"

namespace mv
{

    class OpListIterator : public computation_graph::node_list_iterator
    {

    public:

        OpListIterator(const computation_graph::node_list_iterator &other) :
        computation_graph::node_list_iterator(other)
        {

        }

        OpListIterator()
        {

        }

        ComputationOp* operator->() const
        {
            return (computation_graph::node_list_iterator::operator*()).operator->();
            //return &computation_graph::node_list_iterator::operator*();
        }

    };

}

#endif // ITERATOR_HPP_