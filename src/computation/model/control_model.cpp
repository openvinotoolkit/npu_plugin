#include "include/fathom/computation/model/control_model.hpp"

mv::ControlModel::ControlModel(const ComputationModel &other) :
ComputationModel(other)
{

}


mv::ControlListIterator mv::ControlModel::getFirst()
{
   computation_graph::first_graph::node_list_iterator it = input_;
   return controlGraph_.node_find(*it);
}


mv::ControlListIterator mv::ControlModel::getLast()
{
    return lastOp_;
}


bool mv::ControlModel::isValid() const
{
    return ComputationModel::isValid();
}

mv::ControlListIterator mv::ControlModel::end()
{
    return controlGraph_.node_end();
}