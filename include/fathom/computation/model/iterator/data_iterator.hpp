#ifndef DATA_ITERATOR_HPP_
#define DATA_ITERATOR_HPP_

#include "include/fathom/graph/graph.hpp"
#include "include/fathom/computation/model/types.hpp"
#include "include/fathom/computation/flow/data.hpp"

namespace mv
{

    class DataListIterator : public computation_graph::first_graph::edge_list_iterator
    {

    public:

        DataListIterator(const computation_graph::first_graph::edge_list_iterator &other) :
        computation_graph::first_graph::edge_list_iterator(other)
        {

        }

        DataListIterator(const computation_graph::first_graph::edge_sibling_iterator &other) :
        computation_graph::first_graph::edge_list_iterator(other)
        {   
            
        }

        DataListIterator()
        {

        }

        /*DataFlow* operator->() const
        {
            return &(computation_graph::edge_list_iterator::operator*());
        }*/

        DataFlow& operator*() const
        {
            return computation_graph::first_graph::edge_list_iterator::operator*();
        }

    };

}

#endif // DATA_ITERATOR_HPP_