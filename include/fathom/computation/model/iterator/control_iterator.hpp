#ifndef CONTROL_ITERATOR_HPP_
#define CONTROL_ITERATOR_HPP_

#include "include/fathom/graph/graph.hpp"
#include "include/fathom/computation/model/types.hpp"

namespace mv
{

    class ControlListIterator : public computation_graph::second_graph::node_list_iterator
    {

    public:

        ControlListIterator(const computation_graph::second_graph::node_list_iterator &other) :
        computation_graph::second_graph::node_list_iterator(other)
        {

        }

        ControlListIterator()
        {

        }

        /*ComputationOp* operator->() const
        {
            return (computation_graph::node_list_iterator::operator*()).operator->();
        }*/

        ComputationOp& operator*() const
        {
            return *(computation_graph::second_graph::node_list_iterator::operator*());
        }
        
        ControlListIterator& operator=(const computation_graph::second_graph::node_list_iterator &other)
        {
            
            computation_graph::second_graph::node_list_iterator::operator=(other);
            return *this;
            
        }


    };

}

#endif // OP_ITERATOR_HPP_