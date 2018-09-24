#include "gtest/gtest.h"
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/data_generator.hpp"

TEST(shape, definition)
{

    unsigned dims[] = {32, 32, 16, 8, 4};
    mv::Shape s({dims[0], dims[1], dims[2], dims[3], dims[4]});

    for (std::size_t i = 0; i < s.ndims(); ++i)
        ASSERT_EQ(s[i], dims[i]);

    for (std::size_t i = 0; i < s.ndims(); ++i)
        ASSERT_EQ(s[-(i + 1)], dims[s.ndims() - 1 - i]);

}

TEST(shape, manipulation)
{

    unsigned dims[] = {32, 32, 16, 8, 4};
    unsigned modifier[] = {2, 3, 4, 5, 6};
    mv::Shape s({dims[0], dims[1], dims[2], dims[3], dims[4]});

    for (unsigned i = 0; i < s.ndims(); ++i)
    {
        s[i] *= modifier[i];
        ASSERT_EQ(s[i], dims[i] * modifier[i]);
    }

}

TEST(shape, broadcasting_same_ndims)
{

    mv::Shape s1({32, 32, 16, 4});
    mv::Shape s2({32, 32, 16, 1});
    mv::Shape s3 = mv::Shape::broadcast(s1, s2);
    mv::Shape s4 = mv::Shape::broadcast(s2, s1);

    ASSERT_EQ(s1.ndims(), s3.ndims());
    ASSERT_EQ(s1.ndims(), s4.ndims());

    for (unsigned i = 0; i < s1.ndims(); ++i)
    {
        ASSERT_EQ(s3[i], s1[i]);
        ASSERT_EQ(s4[i], s1[i]);
    }

}

TEST(shape, broadcasting_diff_ndims)
{

    mv::Shape s1({32, 32, 16, 4});
    mv::Shape s2({32, 16, 1});
    mv::Shape s3 = mv::Shape::broadcast(s1, s2);
    mv::Shape s4 = mv::Shape::broadcast(s2, s1);

    ASSERT_EQ(s1.ndims(), s3.ndims());
    ASSERT_EQ(s1.ndims(), s4.ndims());

    for (unsigned i = 0; i < s1.ndims(); ++i)
    {
        ASSERT_EQ(s3[i], s1[i]);
        ASSERT_EQ(s4[i], s1[i]);
    }

}

TEST(shape, broadcasting_vector)
{

    mv::Shape s1({32, 32, 16});
    mv::Shape s2({16});
    mv::Shape s3 = mv::Shape::broadcast(s1, s2);
    mv::Shape s4 = mv::Shape::broadcast(s2, s1);

    ASSERT_EQ(s1.ndims(), s3.ndims());
    ASSERT_EQ(s1.ndims(), s4.ndims());

    for (unsigned i = 0; i < s1.ndims(); ++i)
    {
        ASSERT_EQ(s3[i], s1[i]);
        ASSERT_EQ(s4[i], s1[i]);
    }

}

TEST(shape, broadcasting_failure)
{

    mv::Shape s1({32, 32, 16});
    mv::Shape s2({16, 16});
    ASSERT_ANY_THROW(mv::Shape::broadcast(s1, s2));
    ASSERT_ANY_THROW(mv::Shape::broadcast(s2, s1));

}