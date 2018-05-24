#include "include/fathom/computation/model/op_model.hpp"
#include "include/fathom/computation/model/data_model.hpp"
#include "include/fathom/computation/model/control_model.hpp"
#include "include/fathom/computation/utils/data_generator.hpp"
#include "include/fathom/deployer/std_ostream.hpp"
#include "include/fathom/pass/deploy/dot_pass.hpp"

int main()
{

    mv::OpModel om(mv::Logger::VerboseLevel::VerboseInfo);
    auto inIt = om.input(mv::Shape(1, 128, 128, 3), mv::DType::Float, mv::Order::NWHC);

    mv::vector<mv::float_type> conv1WeightsData = mv::utils::generateSequence<mv::float_type>(3u * 3u * 3u * 8u);
    mv::vector<mv::float_type> conv2WeightsData = mv::utils::generateSequence<mv::float_type>(3u * 3u * 3u * 8u);
    mv::vector<mv::float_type> conv3WeightsData = mv::utils::generateSequence<mv::float_type>(5u * 5u * 16u * 32u);
    mv::vector<mv::float_type> conv4WeightsData = mv::utils::generateSequence<mv::float_type>(6u * 6u * 32u * 64u);

    auto conv1It = om.conv(inIt, mv::ConstantTensor(mv::Shape(3, 3, 3, 8), mv::DType::Float, mv::Order::NWHC, conv1WeightsData), 2, 2, 1, 1);
    auto pool1It = om.maxpool(conv1It, mv::Shape(3, 3), 2, 2, 1, 1);
    auto conv2It = om.conv(inIt, mv::ConstantTensor(mv::Shape(3, 3, 3, 8), mv::DType::Float, mv::Order::NWHC, conv1WeightsData), 2, 2, 1, 1);
    auto pool2It = om.maxpool(conv2It, mv::Shape(3, 3), 2, 2, 1, 1);
    auto concat1It = om.concat(pool1It, pool2It);
    auto conv3It = om.conv(concat1It, mv::ConstantTensor(mv::Shape(5, 5, 16, 32), mv::DType::Float, mv::Order::NWHC, conv3WeightsData), 2, 2, 2, 2);
    auto pool3It = om.maxpool(conv3It, mv::Shape(5, 5), 3, 3, 2, 2);
    auto conv4It = om.conv(pool3It, mv::ConstantTensor(mv::Shape(6, 6, 32, 64), mv::DType::Float, mv::Order::NWHC, conv4WeightsData), 1, 1, 0, 0);
    auto outIt = om.output(conv4It);

    auto msgType = mv::Logger::MessageType::MessageInfo;
    mv::DataModel dm(om);

    dm.logger().log(msgType, "Input op: " + om.getInput()->getName());
    dm.logger().log(msgType, "Input tensor (output tensor of the input op): " + dm.getInput()->getTensor().getName());
    dm.logger().log(msgType, "Output op: " + om.getOutput()->getName());
    dm.logger().log(msgType, "Output tensor (input tensor of the output op): " + dm.getOutput()->getTensor().getName());

    mv::ControlModel cm(om);

    cm.logger().log(msgType, "First op: " + cm.getFirst()->getName());
    cm.logger().log(msgType, "Last op: " + cm.getLast()->getName());

    mv::size_type i = 0;
    for (auto it = cm.getFirst(); it != cm.opEnd(); ++it)
    {
        cm.logger().log(msgType, "Op " + mv::Printable::toString(i) + ": " + it->getName());
        ++i;
    }

    mv::StdOStream ostream;
    mv::pass::DotPass dotPass(cm.logger(), ostream, mv::pass::DotPass::OutputScope::ControlModel, mv::pass::DotPass::ContentLevel::ContentFull);

    dotPass.run(cm);    

    return 0;

}