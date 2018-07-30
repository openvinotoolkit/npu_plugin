#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

int main()
{

    mv::OpModel om(mv::Logger::VerboseLevel::VerboseInfo);
    auto input = om.input(mv::Shape(128, 128, 3), mv::DType::Float, mv::Order::ColumnMajor);
    mv::dynamic_vector<mv::float_type> weights1Data = mv::utils::generateSequence<mv::float_type>(3u * 3u * 3u * 8u);
    mv::dynamic_vector<mv::float_type> weights2Data = mv::utils::generateSequence<mv::float_type>(5u * 5u * 8u * 16u);
    mv::dynamic_vector<mv::float_type> weights3Data = mv::utils::generateSequence<mv::float_type>(4u * 4u * 16u * 32u);

    auto weights1 = om.constant(weights1Data, mv::Shape(3, 3, 3, 8), mv::DType::Float, mv::Order::ColumnMajor);
    auto conv1 = om.conv2D(input, weights1, {2, 2}, {1, 1, 1, 1});
    auto pool1 = om.maxpool2D(conv1, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto weights2 = om.constant(weights2Data, mv::Shape(5, 5, 8, 16), mv::DType::Float, mv::Order::ColumnMajor);
    auto conv2 = om.conv2D(pool1, weights2, {2, 2}, {2, 2, 2, 2});
    auto pool2 = om.maxpool2D(conv2, {5, 5}, {4, 4}, {2, 2, 2, 2});
    auto weights3 = om.constant(weights3Data, mv::Shape(4, 4, 16, 32), mv::DType::Float, mv::Order::ColumnMajor);
    auto conv3 = om.conv2D(pool2, weights3, {1, 1}, {0, 0, 0, 0});
    auto output = om.output(conv3);

    auto msgType = mv::Logger::MessageType::MessageInfo;

    auto attr = output->getAttr("shape");
    om.logger().log(msgType, "Tensor '" + output->getName() + "' attribute 'shape' content: " + attr.getContent<mv::Shape>().toString());
    om.logger().log(msgType, "Tensor '" + output->getName() + "' attribute 'shape' type: " +  mv::Printable::toString(output->getAttrType("shape")));

    om.addAttr(om.getSourceOp(conv1), "customAttr", mv::Attribute(mv::AttrType::IntegerType, 10));
    om.addAttr(om.getSourceOp(input), "customAttr", mv::Attribute(mv::AttrType::UnsignedType, 1U));

    om.logger().log(msgType, "Tensor '" + input->getName() + "' - number of attributes: " + mv::Printable::toString(input->attrsCount()));
    om.logger().log(msgType, "Tensor '" + conv1->getName() + "' - number of attributes: " + mv::Printable::toString(conv1->attrsCount()));
    om.logger().log(msgType, "Tensor '" + output->getName() + "' - number of attributes: " + mv::Printable::toString(output->attrsCount()));

    mv::DataModel dm(om);

    dm.logger().log(msgType, "Input op: " + om.getInput()->getName());
    dm.logger().log(msgType, "Input tensor (output tensor of the input op): " + dm.getInputFlow()->getTensor()->getName());
    dm.logger().log(msgType, "Output op: " + om.getOutput()->getName());
    dm.logger().log(msgType, "Output tensor (input tensor of the output op): " + dm.getOutputFlow()->getTensor()->getName());

    mv::ControlModel cm(om);

    mv::size_type i = 0;
    for (mv::Control::OpDFSIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
    {
        cm.logger().log(msgType, "Op " + mv::Printable::toString(i) + ": " + it->getName());
        ++i;
    }

}