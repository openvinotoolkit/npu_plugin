#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <assert.h>
#include "include/mcm/graph/stl_allocator.hpp"
#include "include/mcm/graph/conjoined_graph.hpp"
#include "include/mcm/graph/static_vector.hpp"

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

    static const byte_type max_ndims = 8;
    typedef uint16_t dim_type;

    typedef std::string string;

    typedef stl_allocator allocator;
    
    template <class T>
    using dynamic_vector = std::vector<T>;
    
    template <class T_key, class T_value>
    using map = std::map<T_key, T_value>;

    template <class T_value, class T_comparator>
    using set = allocator::set<T_value, T_comparator>;

    class ComputationOp;
    class DataFlow;
    class ControlFlow;

    using computation_graph = conjoined_graph<allocator::owner_ptr<ComputationOp>,
        allocator::owner_ptr<DataFlow>, allocator::owner_ptr<ControlFlow>, allocator, size_type>;

    class StdOutLogger;
    typedef StdOutLogger DefaultLogger;

    enum class Order
    {
        LastDimMajor,
        Unknown
    };

    enum class DType
    {
        Float,
        Unknown
    };

    enum class AttrType
    {
        
        UnknownType,
        ByteType,
        UnsignedType,
        IntegerType,
        FloatType,
        DTypeType,
        OrderType,
        ShapeType,
        StringType,
        BoolType,
        OpTypeType,
        FloatVec2DType,
        FloatVec3DType,
        FloatVec4DType,
        IntVec2DType,
        IntVec3DType,
        IntVec4DType,
        UnsignedVec2DType,
        UnsignedVec3DType,
        UnsignedVec4DType,
        FloatVecType

    };

    template <class T>
    struct Vector2D
    {
        T e0;
        T e1;
    };

    template <class T>
    struct Vector3D
    {
        T e0;
        T e1;
        T e2;
    };

    template <class T>
    struct Vector4D
    {
        T e0;
        T e1;
        T e2;
        T e3;
    };

    using FloatVector2D = Vector2D<float_type>;
    using FloatVector3D = Vector3D<float_type>;
    using FloatVector4D = Vector4D<float_type>;

    using IntVector2D = Vector2D<int_type>;
    using IntVector3D = Vector3D<int_type>;
    using IntVector4D = Vector4D<int_type>;

    using UnsignedVector2D = Vector2D<unsigned_type>;
    using UnsignedVector3D = Vector3D<unsigned_type>;
    using UnsignedVector4D = Vector4D<unsigned_type>;

}

#endif // TYPES_HPP_