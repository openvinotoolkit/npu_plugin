#include "gtest/gtest.h"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

TEST(computation_model, data_model_construction)
{

    mv::OpModel om;

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto input = om.input(mv::Shape(32, 32, 1), mv::DType::Float, mv::Order::NWHC);
    float *rawData = new float[27]
    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
     15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f};
    auto weights1 = om.constant(rawData, 27, mv::Shape(3, 3, 1, 3), mv::DType::Float, mv::Order::NWHC);
    auto conv1 = om.conv2D(input, weights1, {4, 4}, {1, 1, 1, 1});
    om.output(conv1);

    mv::DataModel dm(om);

    // Check if model is valid
    ASSERT_TRUE(dm.isValid());

}