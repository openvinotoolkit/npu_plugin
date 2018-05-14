#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <cstdint>
#include <string>
#include <vector>
#include "include/fathom/graph/stl_allocator.hpp"
#include "include/fathom/graph/graph.hpp"

namespace mv
{
    /**
     * @brief Type for storing the size of the graph container (number of nodes).
     * It is guaranteed to be large enough to store size of any graph container.
     */
    typedef uint32_t size_type;
    static const uint32_t max_size = UINT32_MAX;

    typedef float float_type;
    typedef int32_t int_type;
    typedef uint32_t unsigned_type;

    typedef uint8_t byte_type;
    static const uint8_t max_byte = UINT8_MAX;

    static const byte_type max_ndims = 32;
    typedef uint16_t dim_type;

    typedef std::string string;

    template <typename T>
    using vector = std::vector<T>;

    typedef stl_allocator allocator;

    class ComputationOp;
    class DataFlow;

    using computation_graph = graph<allocator::owner_ptr<ComputationOp>, DataFlow, allocator, size_type>;

    enum class Order
    {
        NWHC
    };

    enum class DType
    {
        Float
    };

    enum class AttrType
    {
        
        UnknownType,
        ByteType,
        UnsingedType,
        IntegerType,
        FloatType,
        TensorType,
        DTypeType,
        OrderType,
        ShapeType,
        StringType

    };

}

#endif // TYPES_HPP_