#ifndef DATA_MODEL_HPP_
#define DATA_MODEL_HPP_

#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/model/iterator/control_context.hpp"
#include "include/mcm/computation/resource/memory_allocator.hpp"

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
        virtual ~DataModel();

        Data::OpListIterator switchContext(Control::OpListIterator other);

        Data::FlowSiblingIterator getInputFlow();
        Data::FlowSiblingIterator getOutputFlow();
        Data::FlowListIterator flowBegin();
        Data::FlowListIterator flowEnd();

        void addGroupElement(Data::FlowListIterator element, GroupIterator group);
        void addGroupElement(Data::TensorIterator element, GroupIterator group);
        void removeGroupElement(Data::FlowListIterator element, GroupIterator group);
        void removeGroupElement(Data::TensorIterator element, GroupIterator group);
        using ComputationModel::addGroupElement;
        using ComputationModel::removeGroupElement;

        Data::TensorIterator defineTensor(const std::string& name, const Shape& shape, DType dType, Order order);
        Data::TensorIterator defineTensor(const std::string& name, const Shape& shape, DType dType, Order order, const std::vector<double>& data);
        Data::TensorIterator defineTensor(const std::string& name, const Shape& shape, DType dType, Order order, const std::vector<int64_t>& data);
        Data::TensorIterator defineTensor(const std::string& name, const Shape& shape, DType dType, Order order, const std::vector<mv::DataElement>& data);
        Data::TensorIterator defineTensor(const Tensor& tensor);
        Data::TensorIterator defineTensor(std::shared_ptr<Tensor> tensor);
        bool isTensorDefined(std::shared_ptr<Tensor> tensor) const;
        void undefineTensor(Data::TensorIterator tensor);
        void undefineTensor(const std::string& name);
        std::size_t tensorsCount() const;
        unsigned long long populatedTotalSize() const;
        unsigned long long unpopulatedTotalSize() const;

        bool addAllocator(const std::string& name, std::size_t size, std::size_t alignment);
        bool hasAllocator(const std::string& name);
        const MemoryAllocator& getAllocator(const std::string& allocatorName);
        Data::BufferIterator allocateTensor(const std::string& allocatorName, Control::StageIterator& stage, Data::TensorIterator& tensor);
        Data::BufferIterator allocateTensor(const std::string& allocatorName, Data::BufferIterator buffer, Data::TensorIterator tensor,
            const std::vector<std::size_t>& leftPadding, const std::vector<std::size_t>& rightPadding);
        Data::BufferIterator moveTensor(const std::string& allocatorName, Data::BufferIterator slaveBuffer, Data::BufferIterator masterBuffer,
            const std::vector<std::size_t>& leftPadding, const std::vector<std::size_t>& rightPadding, bool propagate_to_slaves=false);
        void padLeft(const std::string& allocatorName, Data::BufferIterator buffer, const std::vector<std::size_t>& padding);
        void padRight(const std::string& allocatorName, Data::BufferIterator buffer, const std::vector<std::size_t>& padding);
        bool deallocateTensor(const std::string& allocatorName, Control::StageIterator& stage, Data::TensorIterator& tensor);
        void deallocateAll(const std::string& allocatorName, Control::StageIterator& stage);
        Data::BufferIterator bufferBegin(const std::string& allocatorName, Control::StageIterator& stage);
        Data::BufferIterator bufferEnd(const std::string& allocatorName, Control::StageIterator& stage);
        Data::BufferIterator getBuffer(const std::string& allocatorName, Control::StageIterator& stage, Data::TensorIterator tensor);
        std::vector<Data::BufferIterator> buffers(Control::StageIterator& stage);

        bool iterable(const std::string& allocatorName, Control::StageIterator& stage);

        virtual std::string getLogID() const override;

        bool deallocate(Data::TensorIterator tensor, std::size_t stageIdx);
        void deallocateAll(std::size_t stageIdx);

    };

}

#endif // DATA_MODEL_HPP_
