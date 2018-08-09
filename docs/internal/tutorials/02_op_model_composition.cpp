/**
 * @brief Tutorial on using an OpModel (operation context view) to compose a computation model
 * 
 * The OpModel (defined in include/mcm/computation/model/op_model.hpp) view implements the CompositionalModel interface
 * (defined in include/mcm/api/compositional_model.hpp) which is commonly referred as the Composition API. Through this API, 
 * all methods that allows to create operations of particular types are exposed for the outside use, so typically that will 
 * be the first bridge interface for an application using mcmCompiler engine.
 * 
 * @file 02_op_model.cpp
 * @author Stanislaw Maciag
 * @date 2018-08-09
 */

#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

int main()
{

    mv::OpModel opModel(mv::Logger::VerboseLevel::VerboseInfo);

    /*
        CompositionalModel is an abstract type by its own. Therefore, in case of necessity of limiting the exposure
        of computation model to CompositionalModel, the reference type has to be applied.
    */
    mv::CompositionalModel& compModel(opModel);

    /*
        Using CompositionalModel (or plain OpModel) the structure of a nerual network (or other computation) can be composed.
        The first operation that has to be defined is always an input operation. Input will require the definition of 
        its shape, data type, and data order (layout).
    */
    auto input = compModel.input(mv::Shape(128, 128, 3), mv::DType::Float, mv::Order::ColumnMajor);

    /*
        In order to handle operations that use known, constant numeric values as parameters (like most of the convolutions), 
        the constant operation was introduced. The consant operation takes the same set of arguments as an input, but
        additionally it requires a vector of float values to be passed. The length of this vector has to match the total
        size (product of all dimensions) of a shape argument. Below the data vector is being initialized as a sequence of
        integer numbers starting from zero, singular increment.
    */
    mv::dynamic_vector<mv::float_type> weights1Data = mv::utils::generateSequence<mv::float_type>(3u * 3u * 3u * 8u);
    mv::Shape weights1Shape(3, 3, 3, 8);

    if (weights1Data.size() == weights1Shape.totalSize())
        std::cout << "A vector of lenght " << weights1Data.size() << " can be used to populate a tensor of shape "
            << weights1Shape.toString() << std::endl;
    
    /*
        Almost all operations (except an output operation) return a tensor (or, in the future, tensors),
        referenced by a tensor iterator, thus the call from CompositionalModel will return an object of type Data::TensorIterator.
    */
    mv::Data::TensorIterator weights1 = compModel.constant(weights1Data, mv::Shape(3, 3, 3, 8), mv::DType::Float, mv::Order::ColumnMajor);

    /*
        The composition continues the way that tensors already obtained (staring from the input) are used to as inputs for other operations.
        Some operations have their requirements about shapes of input tensors.
        Conv2D requires weights tensor (the second input) to have a shape of (k1, k2, n_in, n_out), where:
            - k1, k2 - horizontal and vertical size of kernel
            - n_in - the third dimension of input data tensor
            - n_out - the number of output channels (depth)
        Moreover conv2D will accept only 3 dimensional tensor as an input data (the first input).
    */
    auto conv1 = compModel.conv2D(input, weights1, {2, 2}, {1, 1, 1, 1});
    auto pool1 = compModel.maxpool2D(conv1, {3, 3}, {2, 2}, {1, 1, 1, 1});
    mv::dynamic_vector<mv::float_type> weights2Data = mv::utils::generateSequence<mv::float_type>(5u * 5u * 8u * 16u);
    auto weights2 = compModel.constant(weights2Data, mv::Shape(5, 5, 8, 16), mv::DType::Float, mv::Order::ColumnMajor);
    auto conv2 = compModel.conv2D(pool1, weights2, {2, 2}, {2, 2, 2, 2});
    auto pool2 = compModel.maxpool2D(conv2, {5, 5}, {4, 4}, {2, 2, 2, 2});
    mv::dynamic_vector<mv::float_type> weights3Data = mv::utils::generateSequence<mv::float_type>(4u * 4u * 16u * 32u);
    auto weights3 = compModel.constant(weights3Data, mv::Shape(4, 4, 16, 32), mv::DType::Float, mv::Order::ColumnMajor);
    auto conv3 = compModel.conv2D(pool2, weights3, {1, 1}, {0, 0, 0, 0});
    auto output = compModel.output(conv3);

    return 0;

}