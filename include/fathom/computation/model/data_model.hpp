#ifndef DATA_MODEL_HPP_
#define DATA_MODEL_HPP_

#include "include/fathom/computation/model/model.hpp"
#include "include/fathom/computation/model/iterator/data_context.hpp"
#include "include/fathom/computation/op/computation_op.hpp"

namespace mv
{

    class DataModel : public ComputationModel
    {

    public:

        //DataModel(Logger::VerboseLevel verboseLevel = Logger::VerboseLevel::VerboseWarning, bool logTime = false);
        //DataModel(Logger &logger);
        //bool addAttr(OpListIterator &op, const string &name, const Attribute &attr);

        DataModel(const ComputationModel &ComputationModel);

        DataContext::FlowSiblingIterator getInput();
        DataContext::FlowSiblingIterator getOutput();
        DataContext::FlowListIterator flowEnd();

        bool isValid() const;

    };

}

#endif // DATA_MODEL_HPP_