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

        static const std::unordered_map<std::size_t, std::string> rowMajorID_;
        static const std::unordered_map<std::size_t, std::string> rowMajorPlanarID_;
        static const std::unordered_map<std::size_t, std::string> colMajorID_;
        static const std::unordered_map<std::size_t, std::string> colMajorPlanarID_;
        static const std::unordered_map<std::size_t, std::string> ZMajorID_;

        std::vector<std::size_t> contVector_;
        std::string contVectorStr_;

        Order(const std::vector<std::size_t>& contVectorParam, const std::string& contVectorStrParam);

    public:

        inline static std::string getRowMajorID(std::size_t dimension)
        {
            return rowMajorID_.at(dimension);
        }

        inline static std::string getColMajorID(std::size_t dimension)
        {
            return colMajorID_.at(dimension);
        }

        inline static std::string getColMajorPlanarID(std::size_t dimension)
        {
            return colMajorPlanarID_.at(dimension);
        }

        inline static std::string getRowMajorPlanarID(std::size_t dimension)
        {
            return rowMajorPlanarID_.at(dimension);
        }

        inline static std::string getZMajorID(std::size_t dimension)
        {
            return ZMajorID_.at(dimension);
        }

        bool isRowMajor();
        bool isColMajor();
        bool isRowMajorPlanar();
        bool isColMajorPlanar();
        bool isRowInterleaved();
        bool isZMajor();

        Order(const std::string& value);
        Order(const Order& other);
        Order& operator=(const Order& other);

        bool operator!=(const Order& other) const;
        bool operator==(const Order& other) const;

        std::size_t subToInd(const Shape &s, const std::vector<std::size_t>& sub) const;
        std::vector<std::size_t> indToSub(const Shape &s, std::size_t idx) const;

        const std::vector<std::size_t>& getContiguityVector();

        // Strides computed in WORDS and bytes respectively
        std::vector<unsigned> computeWordStrides(const Shape &shape) const;
        std::vector<unsigned> computeByteStrides(const Shape &s, unsigned dataSize) const;
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
            return --dim;
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
