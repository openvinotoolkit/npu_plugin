#include "include/fathom/computation/model/op_model.hpp"
#include "include/fathom/computation/model/data_model.hpp"
#include "include/fathom/computation/model/control_model.hpp"
#include "include/fathom/computation/utils/data_generator.hpp"

int main()
{

    mv::OpModel om(mv::Logger::VerboseLevel::VerboseInfo);
    auto inIt = om.input(mv::Shape(1, 128, 128, 3), mv::DType::Float, mv::Order::NWHC);

    mv::vector<mv::float_type> conv1WeightsData = mv::utils::generateSequence<mv::float_type>(3u * 3u * 3u * 8u);
    mv::vector<mv::float_type> conv2WeightsData = mv::utils::generateSequence<mv::float_type>(5u * 5u * 8u * 16u);
    mv::vector<mv::float_type> conv3WeightsData = mv::utils::generateSequence<mv::float_type>(4u * 4u * 16u * 32u);

    auto conv1It = om.conv(inIt, mv::ConstantTensor(mv::Shape(3, 3, 3, 8), mv::DType::Float, mv::Order::NWHC, conv1WeightsData), 2, 2, 1, 1);
    auto pool1It = om.maxpool(conv1It, mv::Shape(3, 3), 2, 2, 1, 1);
    auto conv2It = om.conv(pool1It, mv::ConstantTensor(mv::Shape(5, 5, 8, 16), mv::DType::Float, mv::Order::NWHC, conv2WeightsData), 2, 2, 2, 2, "customName");
    auto pool2It = om.maxpool(conv2It, mv::Shape(5, 5), 4, 4, 2, 2);
    auto conv3It = om.conv(pool2It, mv::ConstantTensor(mv::Shape(4, 4, 16, 32), mv::DType::Float, mv::Order::NWHC, conv3WeightsData), 1, 1, 0, 0);
    auto outIt = om.output(conv3It);

    auto msgType = mv::Logger::MessageType::MessageInfo;

    auto attr = (*outIt).getAttr("outputShape");
    om.logger().log(msgType, "Op '" + (*outIt).getName() + "' attribute 'outputShape' content: " + attr.getContent<mv::Shape>().toString());
    om.logger().log(msgType, "Op '" + (*outIt).getName() + "' attribute 'outputShape' type: " +  mv::Printable::toString((*outIt).getAttrType("outputShape")));

    om.addAttr(conv1It, "customAttr", mv::Attribute(mv::AttrType::IntegerType, 10));
    om.addAttr(inIt, "customAttr", mv::Attribute(mv::AttrType::UnsingedType, 1U));

    om.logger().log(msgType, "Op '" + (*inIt).getName() + "' - number of attributes: " + mv::Printable::toString((*inIt).attrsCount()));
    om.logger().log(msgType, "Op '" + (*conv1It).getName() + "' - number of attributes: " + mv::Printable::toString((*conv1It).attrsCount()));
    om.logger().log(msgType, "Op '" + (*outIt).getName() + "' - number of attributes: " + mv::Printable::toString((*outIt).attrsCount()));

    mv::DataModel dm(om);

    dm.logger().log(msgType, "Input op: " + (*om.getInput()).getName());
    dm.logger().log(msgType, "Input tensor (output tensor of the input op): " + (*dm.getInput()).getTensor().getName());
    dm.logger().log(msgType, "Output op: " + (*om.getOutput()).getName());
    dm.logger().log(msgType, "Output tensor (input tensor of the output op): " + (*dm.getOutput()).getTensor().getName());

    mv::ControlModel cm(om);

    cm.logger().log(msgType, "First op: " + (*cm.getFirst()).getName());
    cm.logger().log(msgType, "Last op: " + (*cm.getLast()).getName());

    mv::size_type i = 0;
    for (auto it = cm.getFirst(); it != cm.end(); ++it)
    {
        cm.logger().log(msgType, "Op " + mv::Printable::toString(i) + ": " + (*it).getName());
        ++i;
    }

    return 0;

}