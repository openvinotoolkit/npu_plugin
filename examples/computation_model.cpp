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
    auto convIt = cm.conv2D(inIt, weights, 2, 2);
    auto outIt = cm.output(convIt);

    auto attr = outIt->getAttr("outputShape");
    std::cout << "Op '" << outIt->getName() << "' attribute 'outputShape' content: " <<  attr.getContent<mv::Shape>().toString() << std::endl;
    std::cout << "Op '" << outIt->getName() << "' attribute 'outputShape' type: " <<  mv::Printable::toString(outIt->getAttrType("outputShape")) << std::endl;

    convIt->addAttr("customAttr", mv::Attribute(mv::AttrType::IntegerType, 10));
    convIt->addAttr("customAttr", mv::Attribute(mv::AttrType::IntegerType, 10));

    cm.addAttr(inIt, "customAttr", mv::Attribute(mv::AttrType::UnsingedType, 1U));

    std::cout << "Op '" << inIt->getName() << "' - number of attributes: " << inIt->attrsCount() << std::endl;
    std::cout << "Op '" << convIt->getName() << "' - number of attributes: " << convIt->attrsCount() << std::endl;
    std::cout << "Op '" << outIt->getName() << "' - number of attributes: " << outIt->attrsCount() << std::endl;

    return 0;

}