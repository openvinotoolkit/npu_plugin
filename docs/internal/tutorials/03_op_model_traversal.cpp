/**
 * @brief Tutorial 
 * 
 * @file 03_op_model_traversal.cpp
 * @author Stanislaw Maciag
 * @date 2018-08-09
 */

#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

int main()
{

    /*
        In this example a two branched copModelputation graph will be used
    */
    mv::OpModel opModel(mv::Logger::VerboseLevel::VerboseInfo);
    auto input = opModel.input(mv::Shape(128, 128, 3), mv::DType::Float, mv::Order::ColumnMajor);

    mv::dynamic_vector<mv::float_type> conv1WeightsData = mv::utils::generateSequence<mv::float_type>(3u * 3u * 3u * 8u);
    mv::dynamic_vector<mv::float_type> conv2WeightsData = mv::utils::generateSequence<mv::float_type>(3u * 3u * 3u * 8u);
    mv::dynamic_vector<mv::float_type> conv3WeightsData = mv::utils::generateSequence<mv::float_type>(5u * 5u * 8u * 16u);
    mv::dynamic_vector<mv::float_type> conv4WeightsData = mv::utils::generateSequence<mv::float_type>(6u * 6u * 16u * 32u);

    auto conv1WeightsIt = opModel.constant(conv1WeightsData, mv::Shape(3, 3, 3, 8), mv::DType::Float, mv::Order::ColumnMajor);
    auto conv1It = opModel.conv2D(input, conv1WeightsIt, {2, 2}, {1, 1, 1, 1});
    auto pool1It = opModel.maxpool2D(conv1It, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto conv2WeightsIt = opModel.constant(conv2WeightsData, mv::Shape(3, 3, 3, 8), mv::DType::Float, mv::Order::ColumnMajor);
    auto conv2It = opModel.conv2D(input, conv2WeightsIt, {2, 2}, {1, 1, 1, 1});
    auto pool2It = opModel.maxpool2D(conv2It, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto addIt = opModel.add(pool1It, pool2It);
    auto conv3WeightsIt = opModel.constant(conv3WeightsData, mv::Shape(5, 5, 8, 16), mv::DType::Float, mv::Order::ColumnMajor);
    auto conv3It = opModel.conv2D(addIt, conv3WeightsIt, {2, 2}, {2, 2, 2, 2});
    auto pool3It = opModel.maxpool2D(conv3It, {5, 5}, {3, 3}, {2, 2, 2, 2});
    auto conv4WeightsIt = opModel.constant(conv4WeightsData, mv::Shape(6, 6, 16, 32), mv::DType::Float, mv::Order::ColumnMajor);
    auto conv4It = opModel.conv2D(pool3It, conv4WeightsIt, {1, 1}, {0, 0, 0, 0});
    opModel.output(conv4It);

    /*
        The traversal of a copModelputation model using an OpModel view usually starts by obtaining an iterator to an input operation.
        This can be done using getInput() method.
    */
    mv::Data::OpListIterator inputOp = opModel.getInput();

    /*
        An iterator for an operation can also be obtained using getSourceOp() method that takes a tensor iterator as an argument. Each
        tensor has a source operation, so input operation iterator can be obtained using input tensor iterator defined eariler.
    */
    inputOp = opModel.getSourceOp(input);

    /*
        
    */

    return 0;
}