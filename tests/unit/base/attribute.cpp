#include "gtest/gtest.h"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"
#include "include/mcm/tensor/order/order.hpp"
#include "include/mcm/tensor/shape.hpp"
#include <array>

TEST(attribute, def_double)
{

    double v1 = 1.0;
    mv::Attribute a1 = v1;
    ASSERT_EQ(a1.get<double>(), v1);

}

TEST(attribute, mod_double)
{
    double v1 = 1.0, v2 = 2.0;
    mv::Attribute a1 = v1;
    a1 = v2;
    ASSERT_EQ(a1.get<double>(), v2);
}

TEST(attribute, def_int)
{

    int v1 = 1;
    mv::Attribute a1 = v1;
    ASSERT_EQ(a1.get<int>(), v1);

}

TEST(attribute, mod_int)
{
    int v1 = 1, v2 = 2;
    mv::Attribute a1 = v1;
    a1 = v2;
    ASSERT_EQ(a1.get<int>(), v2);
}

TEST(attribute, def_bool)
{

    bool v1 = false;
    mv::Attribute a1 = v1;
    ASSERT_EQ(a1.get<bool>(), v1);

}

TEST(attribute, mod_bool)
{
    bool v1 = false, v2 = true;
    mv::Attribute a1 = v1;
    a1 = v2;
    ASSERT_EQ(a1.get<bool>(), v2);
}


TEST(attribute, def_unsigned_short)
{

    unsigned short v1 = 1;
    mv::Attribute a1 = v1;
    ASSERT_EQ(a1.get<unsigned short>(), v1);

}

TEST(attribute, mod_unsigned_short)
{
    unsigned short v1 = 1, v2 = 2;
    mv::Attribute a1 = v1;
    a1 = v2;
    ASSERT_EQ(a1.get<unsigned short>(), v2);
}

TEST(attribute, def_std_size_t)
{

    std::size_t v1 = 1;
    mv::Attribute a1 = v1;
    ASSERT_EQ(a1.get<std::size_t>(), v1);

}

TEST(attribute, mod_std_size_t)
{
    std::size_t v1 = 1, v2 = 2;
    mv::Attribute a1 = v1;
    a1 = v2;
    ASSERT_EQ(a1.get<std::size_t>(), v2);
}

TEST(attribute, def_std_string)
{

    std::string v1 = "str";
    mv::Attribute a1 = v1;
    ASSERT_EQ(a1.get<std::string>(), v1);

}

TEST(attribute, mod_std_string)
{
    std::string v1 = "str1", v2 = "str2";
    mv::Attribute a1 = v1;
    a1 = v2;
    ASSERT_EQ(a1.get<std::string>(), v2);
}

TEST(attribute, def_std_vector_std_size_t)
{

    std::vector<std::size_t> v1({1, 2, 3});
    mv::Attribute a1 = v1;
    ASSERT_EQ(a1.get<std::vector<std::size_t>>(), v1);

}

TEST(attribute, mod_std_vector_std_size_t)
{
    std::vector<std::size_t> v1({1, 2, 3}), v2({4, 5, 6});
    mv::Attribute a1 = v1;
    a1 = v2;
    ASSERT_EQ(a1.get<std::vector<std::size_t>>(), v2);
}

TEST(attribute, def_std_array_unsidged_short_2)
{

    std::array<unsigned short, 2> v1 = {1, 2};
    mv::Attribute a1 = v1;
    auto r = a1.get<std::array<unsigned short, 2>>();
    ASSERT_EQ(r, v1);

}

TEST(attribute, mod_std_array_unsidged_short_2)
{
    std::array<unsigned short, 2> v1 = {1, 2}, v2 = {3, 4};
    mv::Attribute a1 = v1;
    a1 = v2;
    auto r = a1.get<std::array<unsigned short, 2>>();
    ASSERT_EQ(r, v2);
}

TEST(attribute, def_std_array_unsidged_short_3)
{

    std::array<unsigned short, 3> v1 = {1, 2, 3};
    mv::Attribute a1 = v1;
    auto r = a1.get<std::array<unsigned short, 3>>();
    ASSERT_EQ(r, v1);

}

TEST(attribute, mod_std_array_unsidged_short_3)
{
    std::array<unsigned short, 3> v1 = {1, 2, 3}, v2 = {4, 5, 6};
    mv::Attribute a1 = v1;
    a1 = v2;
    auto r = a1.get<std::array<unsigned short, 3>>();
    ASSERT_EQ(r, v2);
}

TEST(attribute, def_std_array_unsidged_short_4)
{

    std::array<unsigned short, 4> v1 = {1, 2, 3, 4};
    mv::Attribute a1 = v1;
    auto r = a1.get<std::array<unsigned short, 4>>();
    ASSERT_EQ(r, v1);

}

TEST(attribute, mod_std_array_unsidged_short_4)
{
    std::array<unsigned short, 4> v1 = {1, 2, 3, 4}, v2 = {5, 6, 7, 8};
    mv::Attribute a1 = v1;
    a1 = v2;
    auto r = a1.get<std::array<unsigned short, 4>>();
    ASSERT_EQ(r, v2);
}


TEST(attribute, def_dtype)
{

    mv::DType v1(mv::DType("Float16"));
    mv::Attribute a1 = v1;
    ASSERT_EQ(a1.get<mv::DType>(), v1);

}

TEST(attribute, def_order)
{

    mv::Order v1(mv::Order("CHW"));
    mv::Attribute a1 = v1;
    ASSERT_EQ(a1.get<mv::Order>(), v1);

}

TEST(attribute, mod_order)
{

    mv::Order v1(mv::Order("CHW"));
    mv::Order v2(mv::Order("WHC"));
    mv::Attribute a1 = v1;
    a1 = v2;
    ASSERT_EQ(a1.get<mv::Order>(), v2);

}

TEST(attribute, def_shape)
{

    mv::Shape v1({1, 2, 3});
    mv::Attribute a1 = v1;
    ASSERT_EQ(a1.get<mv::Shape>(), v1);

}

TEST(attribute, mod_shape)
{

    mv::Shape v1({1, 2, 3}), v2({4, 5, 6});
    mv::Attribute a1 = v1;
    a1 = v2;
    ASSERT_EQ(a1.get<mv::Shape>(), v2);

}

TEST(attribute, reassign_type)
{

    double v1 = 1.0;
    int v2 = 1;

    mv::Attribute a1 = v1;
    ASSERT_EQ(a1.getTypeID(), typeid(double));
    ASSERT_EQ(a1.get<double>(), v1);
    a1 = v2;
    ASSERT_EQ(a1.getTypeID(), typeid(int));
    ASSERT_EQ(a1.get<int>(), v2);

}

TEST(attribute, get_failure)
{
    
    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Silent);

    double v1 = 1.0;
    mv::Attribute a1 = v1;
    ASSERT_ANY_THROW(a1.get<int>());

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Error);

}

class UnregisteredAttr
{

};

TEST(attribute, def_failure)
{

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Silent);

    UnregisteredAttr v1;
    ASSERT_ANY_THROW(mv::Attribute a1 = v1);

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Error);

}
