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
        In this example a two branched computation graph will be used
    */
    mv::OpModel opModel;
    auto input = opModel.input({128, 128, 3}, mv::DType("Float16"), mv::OrderType::ColumnMajor);

    std::vector<double> conv1WeightsData = mv::utils::generateSequence<double>(3u * 3u * 3u * 8u);
    std::vector<double> conv2WeightsData = mv::utils::generateSequence<double>(3u * 3u * 3u * 8u);
    std::vector<double> conv3WeightsData = mv::utils::generateSequence<double>(5u * 5u * 8u * 16u);
    std::vector<double> conv4WeightsData = mv::utils::generateSequence<double>(6u * 6u * 16u * 32u);

    auto conv1WeightsIt = opModel.constant(conv1WeightsData, {3, 3, 3, 8}, mv::DType("Float16"), mv::OrderType::ColumnMajor);
    auto conv1It = opModel.conv2D(input, conv1WeightsIt, {2, 2}, {1, 1, 1, 1});
    auto pool1It = opModel.maxpool2D(conv1It, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto conv2WeightsIt = opModel.constant(conv2WeightsData, {3, 3, 3, 8}, mv::DType("Float16"), mv::OrderType::ColumnMajor);
    auto conv2It = opModel.conv2D(input, conv2WeightsIt, {2, 2}, {1, 1, 1, 1});
    auto pool2It = opModel.maxpool2D(conv2It, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto addIt = opModel.add(pool1It, pool2It);
    auto conv3WeightsIt = opModel.constant(conv3WeightsData, {5, 5, 8, 16}, mv::DType("Float16"), mv::OrderType::ColumnMajor);
    auto conv3It = opModel.conv2D(addIt, conv3WeightsIt, {2, 2}, {2, 2, 2, 2});
    auto pool3It = opModel.maxpool2D(conv3It, {5, 5}, {3, 3}, {2, 2, 2, 2});
    auto conv4WeightsIt = opModel.constant(conv4WeightsData, {6, 6, 16, 32}, mv::DType("Float16"), mv::OrderType::ColumnMajor);
    auto conv4It = opModel.conv2D(pool3It, conv4WeightsIt, {1, 1}, {0, 0, 0, 0});
    opModel.output(conv4It);

    /*
        The traversal of a computation model using an OpModel view usually starts by obtaining an iterator to an input operation.
        This can be done using getInput() method.
    */
    mv::Data::OpListIterator inputOp = opModel.getInput();

    /*
        An iterator for an operation can also be obtained using getSourceOp() method that takes a tensor iterator as an argument. Each
        tensor has a source operation, so input operation iterator can be obtained using input tensor iterator defined eariler. That method
        of conversion between tensor iterator and operation iterator will work for any tensor.
    */
    inputOp = opModel.getSourceOp(input);

    /*
        An operation iterator can be used for a traversal of computation graph. Calling its preincrement operator "++it" will advance
        the referred operation to the next one, based on a traversal method defined by the type of an iterator.
        In OpModel itarators from the Data:: namespace are used. It indicates the fact, that in this view data flows are treated as
        relation definitions (edges) between operations (nodes).
        The first available type of the operation iterator - Data::OpListIterator - actualy ignores the whole built graph stucture (ignores 
        data flows) and uses the list of all defined operations stored internally. The order of this list is defined by the order of addition
        to the computation model. The Data::OpListIterator is the only iterator that allows to access every operation always, even in the case of
        disjoint (invalid) model (and potentially in the case of model with multiple inputs in the future).
    */
    for (mv::Data::OpListIterator it = inputOp; it != opModel.opEnd(); ++it)
        std::cout << it->getName() << std::endl;


    return 0;
}