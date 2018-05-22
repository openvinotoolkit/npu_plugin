#ifndef CONTROL_MODEL_HPP_
#define CONTROL_MODEL_HPP_

#include "include/fathom/computation/model/model.hpp"
#include "include/fathom/computation/model/iterator/control_context.hpp"
#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class ControlModel : public ComputationModel
    {

    public:

        ControlModel(const ComputationModel &ComputationModel);

        ControlContext::OpListIterator getFirst();
        ControlContext::OpListIterator getLast();
        ControlContext::OpListIterator opEnd();

        bool isValid() const;

    };

}

#endif // CONTROL_MODEL_HPP_