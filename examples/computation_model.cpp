#include <iostream>
#include "include/fathom/computation/model/model.hpp"

int main()
{

    mv::ComputationModel cm(mv::Logger::VerboseLevel::VerboseDebug);

    auto inIt = cm.input(mv::Shape(1, 32, 32, 1), mv::DType::Float, mv::Order::NWHC);

    mv::vector<mv::float_type> weightsData =
    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
     15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f};

    mv::ConstantTensor weights(mv::Shape(1, 3, 3, 3), mv::DType::Float, mv::Order::NWHC, weightsData);
    auto convIt = cm.convolutional(inIt, weights, 2, 2);
    auto outIt = cm.output(convIt);

    auto attr = outIt->getAttr<mv::Shape>("outputShape");
    std::cout << attr.getContent<mv::Shape>().toString() << std::endl;

    return 0;

}