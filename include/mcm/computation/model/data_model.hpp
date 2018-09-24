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
        using BufferIterator = IteratorDetail::ModelLinearIterator<std::set<std::shared_ptr<MemoryAllocator::MemoryBuffer>,
            MemoryAllocator::BufferOrderComparator>::iterator, MemoryAllocator::MemoryBuffer>;
    }

    class DataModel : public ComputationModel
    {

    public:

        DataModel(ComputationModel& ComputationModel);

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

        bool addAllocator(const std::string& name, std::size_t size);
        bool hasAllocator(const std::string& name);
        Data::BufferIterator allocateTensor(const std::string& allocatorName, Control::StageIterator& stage, Data::TensorIterator& tensor);
        Data::BufferIterator allocateTensor(const std::string& allocatorName, Data::BufferIterator buffer, Data::TensorIterator tensor,
            const std::vector<std::size_t>& leftPadding, const std::vector<std::size_t>& rightPadding);
        void padLeft(const std::string& allocatorName, Data::BufferIterator buffer, const std::vector<std::size_t>& padding);
        void padRight(const std::string& allocatorName, Data::BufferIterator buffer, const std::vector<std::size_t>& padding);
        bool deallocateTensor(const std::string& allocatorName, Control::StageIterator& stage, Data::TensorIterator& tensor);
        void deallocateAll(const std::string& allocatorName, Control::StageIterator& stage);
        Data::BufferIterator bufferBegin(const std::string& allocatorName, Control::StageIterator& stage);
        Data::BufferIterator bufferEnd(const std::string& allocatorName, Control::StageIterator& stage);
        Data::BufferIterator getBuffer(const std::string& allocatorName, Control::StageIterator& stage, Data::TensorIterator tensor);

        virtual std::string getLogID() const override;

        bool deallocate(Data::TensorIterator tensor, std::size_t stageIdx);
        void deallocateAll(std::size_t stageIdx);

    };

}

#endif // DATA_MODEL_HPP_
