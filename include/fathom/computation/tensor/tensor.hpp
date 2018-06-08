#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include "include/fathom/computation/tensor/shape.hpp"
#include "include/fathom/computation/model/element.hpp"

namespace mv
{
    
    class Tensor : public ComputationElement
    {

        allocator::owner_ptr<vector<float_type>> data_;

    public:

        Tensor(const string &name, const Shape &shape, DType dType, Order order) :
        ComputationElement(name)
        {
            logger_.log(Logger::MessageType::MessageDebug, "Defined tensor " + toString());
            addAttr("dType", AttrType::DTypeType, dType);
            addAttr("order", AttrType::OrderType, order);
            addAttr("shape", AttrType::ShapeType, shape);
            addAttr("populated", AttrType::BoolType, false);
        }

        Tensor(const string &name, const Shape &shape, DType dType, Order order, allocator::owner_ptr<vector<float_type>> data) :
        ComputationElement(name)
        {
            data_ = data;
            addAttr("dType", AttrType::DTypeType, dType);
            addAttr("order", AttrType::OrderType, order);
            addAttr("shape", AttrType::ShapeType, shape);
            addAttr("populated", AttrType::BoolType, true);
            logger_.log(Logger::MessageType::MessageDebug, "Defined tensor " + toString());
        }

        Tensor(const Tensor &other) :
        ComputationElement(other),
        data_(other.data_)
        {
            logger_.log(Logger::MessageType::MessageDebug, "Copied tensor " + toString());
        }

        Tensor() :
        ComputationElement("unknown_tensor")
        {
            logger_.log(Logger::MessageType::MessageWarning, "Defined unknown tensor");
            addAttr("dType", AttrType::DTypeType, DType::Unknown);
            addAttr("order", AttrType::OrderType, Order::Unknown);
            addAttr("shape", AttrType::ShapeType, Shape());
            addAttr("populated", AttrType::BoolType, false);
        }

        bool populate(float_type *data, size_type size, const Shape &shape, DType dType, Order order)
        {

            if (size != shape.totalSize())
            {
                logger_.log(Logger::MessageType::MessageError, "Unable to populate tensor - mismatch between input array size (" + 
                    Printable::toString(size) + ") and declared shape (" +shape.toString() + ")");
                return false;
            }
            //TODO failure handling
            data_->assign(data, size);
            getAttr("populated").setContent<bool>(true);
            return true;

        }

        bool unpopulate()
        {
            if (!getAttr("populated").getContent<bool>())
                return false;
            
            data_->clear();
            getAttr("populated").setContent<bool>(false);
            return true;

        }

        bool isPopulated() const
        {
            return getAttr("populated").getContent<bool>();
        }

        // TODO - Handle the case when tensor got deleted, by the reference is still in use
        vector<float_type> &getData()
        {
            if (!isPopulated())
                logger_.log(Logger::MessageType::MessageWarning, "Attempt of restoring data from an unpopulated tensor '" + name_ + "'");
            return *data_;
        }
        
        Shape getShape() const
        {
            return getAttr("shape").getContent<Shape>();
        }

        DType getDType() const
        {
            return getAttr("dType").getContent<DType>();
        }

        Order getOrder() const
        {
            return getAttr("order").getContent<Order>();
        }

        string toString() const
        {
            return "'" + name_ + "' " + ComputationElement::toString();
        }

    };

}

#endif // MODEL_TENSOR_HPP_