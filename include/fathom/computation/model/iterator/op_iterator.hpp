#ifndef OP_ITERATOR_HPP_
#define OP_ITERATOR_HPP_

#include "include/fathom/graph/graph.hpp"
#include "include/fathom/computation/model/types.hpp"

namespace mv
{

    class OpListIterator : public computation_graph::first_graph::node_list_iterator
    {

    public:

        OpListIterator(const computation_graph::first_graph::node_list_iterator &other) :
        computation_graph::first_graph::node_list_iterator(other)
        {

        }

        OpListIterator()
        {

        }

        /*ComputationOp* operator->() const
        {
            return (computation_graph::node_list_iterator::operator*()).operator->();
        }*/

        ComputationOp& operator*() const
        {
            return *(computation_graph::first_graph::node_list_iterator::operator*());
        }

    };

}

#endif // OP_ITERATOR_HPP_