#ifndef DATA_MODEL_HPP_
#define DATA_MODEL_HPP_

#include "include/mcm/computation/model/model.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/op/computation_op.hpp"

namespace mv
{

    class DataModel : public ComputationModel
    {

    public:

        //DataModel(Logger::VerboseLevel verboseLevel = Logger::VerboseLevel::VerboseWarning, bool logTime = false);
        //DataModel(Logger &logger);
        //bool addAttr(OpListIterator &op, const string &name, const Attribute &attr);

        DataModel(const ComputationModel &ComputationModel);

        Data::OpListIterator switchContext(Control::OpListIterator &other);

        Data::FlowSiblingIterator getInput();
        Data::FlowSiblingIterator getOutput();
        Data::FlowListIterator flowEnd();

        GroupContext::MemberIterator addGroupElement(Data::FlowListIterator &element, GroupContext::GroupIterator &group);
        bool removeGroupElement(Data::FlowListIterator &element, GroupContext::GroupIterator &group);
        using ComputationModel::addGroupElement;
        using ComputationModel::removeGroupElement;
        
        Data::TensorIterator findTensor(string name);

        bool addAllocator(const string &name, size_type maxSize);
        bool allocateTensor(const string &allocatorName, Control::StageIterator &stage, Data::TensorIterator &tensor);



    };

}

#endif // DATA_MODEL_HPP_