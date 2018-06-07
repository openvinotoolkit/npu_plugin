#ifndef DATA_MODEL_HPP_
#define DATA_MODEL_HPP_

#include "include/fathom/computation/model/model.hpp"
#include "include/fathom/computation/model/iterator/data_context.hpp"
#include "include/fathom/computation/model/iterator/tensor_context.hpp"
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

        DataContext::OpListIterator switchContext(ControlContext::OpListIterator &other);

        DataContext::FlowSiblingIterator getInput();
        DataContext::FlowSiblingIterator getOutput();
        DataContext::FlowListIterator flowEnd();

        GroupContext::MemberIterator addGroupElement(DataContext::FlowListIterator &element, GroupContext::GroupIterator &group);
        bool removeGroupElement(DataContext::FlowListIterator &element, GroupContext::GroupIterator &group);
        using ComputationModel::addGroupElement;
        using ComputationModel::removeGroupElement;
        
        TensorContext::UnpopulatedTensorIterator findUnpopulatedTensor(string name);

        bool addAllocator(const string &name, size_type maxSize);
        bool allocateTensor(const string &allocatorName, ControlContext::StageIterator &stage, TensorContext::PopulatedTensorIterator &tensor);
        bool allocateTensor(const string &allocatorName, ControlContext::StageIterator &stage, TensorContext::UnpopulatedTensorIterator &tensor);



    };

}

#endif // DATA_MODEL_HPP_