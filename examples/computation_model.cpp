#include <iostream>
#include "include/fathom/computation/model/op_model.hpp"
#include "include/fathom/computation/model/data_model.hpp"

int main()
{

    mv::OpModel om(mv::Logger::VerboseLevel::VerboseInfo);
    auto inIt = om.input(mv::Shape(1, 32, 32, 1), mv::DType::Float, mv::Order::NWHC);

    mv::vector<mv::float_type> weightsData =
    {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
     15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f};

    mv::ConstantTensor weights(mv::Shape(1, 3, 3, 3), mv::DType::Float, mv::Order::NWHC, weightsData);
    auto convIt = om.conv2D(inIt, weights, 2, 2, 1, 1);
    auto outIt = om.output(convIt);

    auto msgType = mv::Logger::MessageType::MessageInfo;

    auto attr = (*outIt).getAttr("outputShape");
    om.logger().log(msgType, "Op '" + (*outIt).getName() + "' attribute 'outputShape' content: " + attr.getContent<mv::Shape>().toString());
    om.logger().log(msgType, "Op '" + (*outIt).getName() + "' attribute 'outputShape' type: " +  mv::Printable::toString((*outIt).getAttrType("outputShape")));

    om.addAttr(convIt, "customAttr", mv::Attribute(mv::AttrType::IntegerType, 10));
    om.addAttr(inIt, "customAttr", mv::Attribute(mv::AttrType::UnsingedType, 1U));

    om.logger().log(msgType, "Op '" + (*inIt).getName() + "' - number of attributes: " + mv::Printable::toString((*inIt).attrsCount()));
    om.logger().log(msgType, "Op '" + (*convIt).getName() + "' - number of attributes: " + mv::Printable::toString((*convIt).attrsCount()));
    om.logger().log(msgType, "Op '" + (*outIt).getName() + "' - number of attributes: " + mv::Printable::toString((*outIt).attrsCount()));

    mv::DataModel dm(om);

    dm.logger().log(msgType, "Input op: " + (*om.getInput()).getName());
    dm.logger().log(msgType, "Input tensor (output tensor of the input op): " + (*dm.getInput()).getTensor().getName());
    dm.logger().log(msgType, "Output op: " + (*om.getOutput()).getName());
    dm.logger().log(msgType, "Output tensor (input tensor of the output op): " + (*dm.getOutput()).getTensor().getName());

    return 0;

}