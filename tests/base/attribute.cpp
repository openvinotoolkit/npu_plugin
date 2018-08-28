#include "gtest/gtest.h"
#include "include/mcm/base/attribute.hpp"

TEST(attribute, definition)
{

    double value = 1.0f;
    mv::Attribute attr(mv::AttrType::FloatType, value);
    ASSERT_EQ(attr.getContent<double>(), value);

}

TEST(attribute, modification)
{
    double value = 1.0f, newValue = 2.0f;
    mv::Attribute attr(mv::AttrType::FloatType, value);
    attr.setContent(newValue);
    ASSERT_EQ(attr.getContent<double>(), newValue);
}