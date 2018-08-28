#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <assert.h>
#include "include/mcm/graph/stl_allocator.hpp"
#include "include/mcm/graph/conjoined_graph.hpp"

namespace mv
{

    class ComputationOp;
    class DataFlow;
    class ControlFlow;

    using computation_graph = conjoined_graph<std::shared_ptr<ComputationOp>,
        std::shared_ptr<DataFlow>, std::shared_ptr<ControlFlow>, stl_allocator, std::size_t>;

    class StdOutLogger;
    typedef StdOutLogger DefaultLogger;

    enum class Order
    {
        ColumnMajor,
        ColumnMajorPlanar,
        RowMajor,
        RowMajorPlanar,
        TBDLayout,
        Unknown
    };

    const std::map<Order, std::string> orderStrings
    {
        {Order::ColumnMajor, "ColumnMajor"},
        {Order::ColumnMajorPlanar, "ColumnMajorPlanar"},
        {Order::RowMajor, "RowMajor"},
        {Order::RowMajorPlanar, "RowMajorPlanar"},
        {Order::TBDLayout, "TBDLayout"},
        {Order::Unknown, "Unknown"}
    };

    const std::map<std::string, Order> orderStringsReversed
    {
        {"ColumnMajor", Order::ColumnMajor},
        {"ColumnMajorPlanar", Order::ColumnMajorPlanar},
        {"RowMajor", Order::RowMajor},
        {"RowMajorPlanar", Order::RowMajorPlanar},
        {"TBDLayout", Order::TBDLayout},
        {"Unknown", Order::Unknown}
    };

    enum class DType
    {
        Float,
        Unknown
    };

    const std::map<DType, std::string> dtypeStrings
    {
        {DType::Float, "Float"},
        {DType::Unknown, "Unknown"}
    };

    const std::map<std::string, DType> dtypeStringsReversed
    {
        {"Float", DType::Float},
        {"Unknown", DType::Unknown}
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
        FloatVecType,
        StringVecType

    };

    const std::map<AttrType, std::string> attrTypeStrings
    {
        {AttrType::UnknownType, "unknown"},
        {AttrType::ByteType, "byte"},
        {AttrType::UnsignedType, "unsigned"},
        {AttrType::IntegerType, "int"},
        {AttrType::FloatType, "double"},
        {AttrType::DTypeType, "dtype"},
        {AttrType::OrderType, "order"},
        {AttrType::ShapeType, "shape"},
        {AttrType::StringType, "string"},
        {AttrType::BoolType, "bool"},
        {AttrType::OpTypeType, "operation"},
        {AttrType::FloatVec2DType, "doubleVec2D"},
        {AttrType::FloatVec3DType, "doubleVec3D"},
        {AttrType::FloatVec4DType, "doubleVec4D"},
        {AttrType::IntVec2DType, "intVec2D"},
        {AttrType::IntVec3DType, "intVec3D"},
        {AttrType::IntVec4DType, "intVec4D"},
        {AttrType::UnsignedVec2DType, "unsignedVec2D"},
        {AttrType::UnsignedVec3DType, "unsignedVec3D"},
        {AttrType::UnsignedVec4DType, "unsignedVec4D"},
        {AttrType::FloatVecType, "doubleVec"},
        {AttrType::StringVecType, "stringVec"}

    };


    //Reverse maps couldn't be construced on the fly due to non-constness
    const std::map<std::string, AttrType> attrTypeStringsReversed
    {
        {"unknown", AttrType::UnknownType},
        {"byte", AttrType::ByteType},
        {"unsigned", AttrType::UnsignedType},
        {"int", AttrType::IntegerType},
        {"double", AttrType::FloatType},
        {"dtype", AttrType::DTypeType},
        {"order", AttrType::OrderType},
        {"shape", AttrType::ShapeType},
        {"string", AttrType::StringType},
        {"bool", AttrType::BoolType},
        {"operation", AttrType::OpTypeType},
        {"doubleVec2D", AttrType::FloatVec2DType},
        {"doubleVec3D", AttrType::FloatVec3DType},
        {"doubleVec4D", AttrType::FloatVec4DType},
        {"intVec2D", AttrType::IntVec2DType},
        {"intVec3D", AttrType::IntVec3DType},
        {"intVec4D", AttrType::IntVec4DType},
        {"unsignedVec2D", AttrType::UnsignedVec2DType},
        {"unsignedVec3D", AttrType::UnsignedVec3DType},
        {"unsignedVec4D", AttrType::UnsignedVec4DType},
        {"doubleVec", AttrType::FloatVecType},
        {"stringVec", AttrType::StringVecType}
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

    using FloatVector2D = Vector2D<double>;
    using FloatVector3D = Vector3D<double>;
    using FloatVector4D = Vector4D<double>;

    using IntVector2D = Vector2D<int>;
    using IntVector3D = Vector3D<int>;
    using IntVector4D = Vector4D<int>;

    using UnsignedVector2D = Vector2D<unsigned>;
    using UnsignedVector3D = Vector3D<unsigned>;
    using UnsignedVector4D = Vector4D<unsigned>;

}

#endif // TYPES_HPP_
