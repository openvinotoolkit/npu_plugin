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
        using BufferIterator = IteratorDetail::ModelValueIterator<std::map<TensorIterator, std::shared_ptr<MemoryAllocator::MemoryBuffer>,
            MemoryAllocator::TensorIteratorComparator>::iterator, MemoryAllocator::MemoryBuffer>;
    }

    class DataModel : public ComputationModel
    {

    protected:

        virtual std::string getLogID_() const override;

    public:

        DataModel(const ComputationModel& ComputationModel);

        Data::OpListIterator switchContext(Control::OpListIterator other);

        Data::FlowSiblingIterator getInputFlow();
        Data::FlowSiblingIterator getOutputFlow();
        Data::FlowListIterator flowBegin();
        Data::FlowListIterator flowEnd();

        GroupContext::MemberIterator addGroupElement(Data::FlowListIterator& element, GroupContext::GroupIterator& group);
        bool removeGroupElement(Data::FlowListIterator& element, GroupContext::GroupIterator& group);
        using ComputationModel::addGroupElement;
        using ComputationModel::removeGroupElement;

        Data::TensorIterator defineTensor(const std::string& name, const Shape& shape, DType dType, Order order);
        Data::TensorIterator defineTensor(const std::string& name, const Shape& shape, DType dType, Order order, const std::vector<double>& data);
        bool undefineTensor(const std::string& name);
        Data::TensorIterator findTensor(const std::string& name);
        unsigned tensorsCount() const;

        bool addAllocator(const std::string& name, std::size_t size, Order order);
        bool hasAllocator(const std::string& name);
        Data::BufferIterator allocateTensor(const std::string& allocatorName, Control::StageIterator& stage, Data::TensorIterator& tensor, std::vector<size_t> pad);
        bool deallocateTensor(const std::string& allocatorName, Control::StageIterator& stage, Data::TensorIterator& tensor);
        void deallocateAll(const std::string& allocatorName, Control::StageIterator& stage);
        Data::BufferIterator bufferBegin(const std::string& allocatorName, Control::StageIterator& stage);
        Data::BufferIterator bufferEnd(const std::string& allocatorName, Control::StageIterator& stage);
        Data::BufferIterator getBuffer(const std::string& allocatorName, Control::StageIterator& stage, Data::TensorIterator tensor);

        bool addAttr(Data::TensorIterator tensor, const std::string& name, const Attribute& attr);

    };

}

#endif // DATA_MODEL_HPP_
