#ifndef DATA_MODEL_HPP_
#define DATA_MODEL_HPP_

#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/resource/memory_allocator.hpp"
#include "include/mcm/computation/op/computation_op.hpp"

namespace mv
{

    namespace Data
    {
        using BufferIterator = IteratorDetail::ModelValueIterator<std::map<TensorIterator, allocator::owner_ptr<MemoryAllocator::MemoryBuffer>, 
            MemoryAllocator::TensorIteratorComparator>::iterator, MemoryAllocator::MemoryBuffer>;
    }

    class DataModel : public ComputationModel
    {

    public:
    
        DataModel(const ComputationModel &ComputationModel);

        Data::OpListIterator switchContext(Control::OpListIterator other);

        Data::FlowSiblingIterator getInputFlow();
        Data::FlowSiblingIterator getOutputFlow();
        Data::FlowListIterator flowBegin();
        Data::FlowListIterator flowEnd();

        GroupContext::MemberIterator addGroupElement(Data::FlowListIterator &element, GroupContext::GroupIterator &group);
        bool removeGroupElement(Data::FlowListIterator &element, GroupContext::GroupIterator &group);
        using ComputationModel::addGroupElement;
        using ComputationModel::removeGroupElement;
        
        Data::TensorIterator findTensor(string name);
        unsigned tensorsCount() const;

        bool addAllocator(const string &name, std::size_t size, Order order);
        bool hasAllocator(const string& name);
        Data::BufferIterator allocateTensor(const string &allocatorName, Control::StageIterator &stage, Data::TensorIterator &tensor, mv::dynamic_vector<size_t> pad);
        bool deallocateTensor(const string &allocatorName, Control::StageIterator &stage, Data::TensorIterator &tensor);
        void deallocateAll(const string &allocatorName, Control::StageIterator &stage);
        Data::BufferIterator bufferBegin(const string &allocatorName, Control::StageIterator &stage);
        Data::BufferIterator bufferEnd(const string &allocatorName, Control::StageIterator &stage);
        Data::BufferIterator getBuffer(const string &allocatorName, Control::StageIterator &stage, Data::TensorIterator tensor);

        bool addAttr(Data::TensorIterator tensor, const string& name, const Attribute& attr);

    };

}

#endif // DATA_MODEL_HPP_
