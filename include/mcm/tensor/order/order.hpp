#ifndef MV_TENSOR_ORDER_HPP_
#define MV_TENSOR_ORDER_HPP_

#include <string>
#include <unordered_map>
#include <functional>
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/tensor/order/order_registry.hpp"
#include "include/mcm/base/exception/order_error.hpp"

namespace mv
{
    class Order : public LogSender
    {

    private:

        static const std::unordered_map<std::size_t, std::string> rowMajorID;
        static const  std::unordered_map<std::size_t, std::string> rowMajorPlanarID;
        static const std::unordered_map<std::size_t, std::string> colMajorID;

        std::vector<std::size_t>& contVector_;
        Order(std::vector<std::size_t>& contVectorParam)
            :contVector_(contVectorParam)
        {

        }


    public:

        inline static std::string getRowMajorID(std::size_t dimension)
        {
            return rowMajorID.at(dimension);
        }

        inline static std::string getColMajorID(std::size_t dimension)
        {
            return colMajorID.at(dimension);
        }

        inline static std::string getRowMajorPlanarID(std::size_t dimension)
        {
            return rowMajorPlanarID.at(dimension);
        }

        bool isRowMajor();
        bool isColMajor();
        bool isRowMajorPlanar();
        bool isColMajorPlanar();
        bool isRowInterleaved();


        Order(const Order& other);
        Order& operator=(const Order& other);
        Order(const std::string& value)
           :Order([this, value]()->Order
            {

                if(!OrderRegistry::checkOrder(value))
                    throw OrderError(*this, "Invalid string passed for Order construction " + value);

                return Order(OrderRegistry::getContVector(value));
            }())
        {

        }

        bool operator!=(const Order& other) const;
        bool operator==(const Order& other) const;

        std::size_t subToInd(const Shape &s, const std::vector<std::size_t>& sub) const;
        std::vector<std::size_t> indToSub(const Shape &s, std::size_t idx) const;

        std::size_t operator[](std::size_t idx) const;
        std::size_t size() const;
        std::string toString() const;

        inline std::size_t firstContiguousDimensionIndex() const
        {
            return 0;
        }

        inline std::size_t lastContiguousDimensionIndex() const
        {
            return contVector_.size() - 1;
        }

        inline std::size_t nextContiguosDimensionIndex(std::size_t dim) const
        {
            return ++dim;
        }

        inline std::size_t prevContiguosDimensionIndex(std::size_t dim) const
        {
            return --dim;;
        }

        inline bool isFirstContiguousDimensionIndex(std::size_t dim) const
        {
            return dim == firstContiguousDimensionIndex();
        }

        inline bool isLastContiguousDimensionIndex(std::size_t dim) const
        {
            return dim == lastContiguousDimensionIndex();
        }

        std::string getLogID() const override;

    };

}

#endif // MV_TENSOR_ORDER_HPP_
