#include "include/fathom/computation/model/op_model.hpp"
#include "include/fathom/computation/model/data_model.hpp"
#include "include/fathom/computation/model/control_model.hpp"
#include "include/fathom/computation/utils/data_generator.hpp"
#include "include/fathom/deployer/std_ostream.hpp"
#include "include/fathom/pass/deploy/dot_pass.hpp"

int main()
{

    mv::OpModel om(mv::Logger::VerboseLevel::VerboseInfo);
    auto inIt = om.input(mv::Shape(128, 128, 3), mv::DType::Float, mv::Order::NWHC);

    mv::dynamic_vector<mv::float_type> conv1WeightsData = mv::utils::generateSequence<mv::float_type>(3u * 3u * 3u * 8u);
    mv::dynamic_vector<mv::float_type> conv2WeightsData = mv::utils::generateSequence<mv::float_type>(3u * 3u * 3u * 8u);
    mv::dynamic_vector<mv::float_type> conv3WeightsData = mv::utils::generateSequence<mv::float_type>(5u * 5u * 16u * 32u);
    mv::dynamic_vector<mv::float_type> conv4WeightsData = mv::utils::generateSequence<mv::float_type>(6u * 6u * 32u * 64u);

    auto conv1WeightsIt = om.constant(conv1WeightsData, mv::Shape(3, 3, 3, 8), mv::DType::Float, mv::Order::NWHC);
    auto conv1It = om.conv2D(inIt->getOutput(0), conv1WeightsIt->getOutput(0), {2, 2}, {1, 1, 1, 1});
    auto pool1It = om.maxpool2D(conv1It->getOutput(0), {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto conv2WeightsIt = om.constant(conv2WeightsData, mv::Shape(3, 3, 3, 8), mv::DType::Float, mv::Order::NWHC);
    auto conv2It = om.conv2D(inIt->getOutput(0), conv2WeightsIt->getOutput(0), {2, 2}, {1, 1, 1, 1});
    auto pool2It = om.maxpool2D(conv2It->getOutput(0), {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto concat1It = om.concat(pool1It->getOutput(0), pool2It->getOutput(0));
    auto conv3WeightsIt = om.constant(conv3WeightsData, mv::Shape(5, 5, 16, 32), mv::DType::Float, mv::Order::NWHC);
    auto conv3It = om.conv2D(concat1It->getOutput(0), conv3WeightsIt->getOutput(0), {2, 2}, {2, 2, 2, 2});
    auto pool3It = om.maxpool2D(conv3It->getOutput(0), {5, 5}, {3, 3}, {2, 2, 2, 2});
    auto conv4WeightsIt = om.constant(conv4WeightsData, mv::Shape(6, 6, 32, 64), mv::DType::Float, mv::Order::NWHC);
    auto conv4It = om.conv2D(pool3It->getOutput(0), conv4WeightsIt->getOutput(0), {1, 1}, {0, 0, 0, 0});
    auto outIt = om.output(conv4It->getOutput(0));

    auto msgType = mv::Logger::MessageType::MessageInfo;
    mv::DataModel dm(om);

    dm.logger().log(msgType, "Input op: " + om.getInput()->getName());
    dm.logger().log(msgType, "Input tensor (output tensor of the input op): " + dm.getInput()->getTensor()->getName());
    dm.logger().log(msgType, "Output op: " + om.getOutput()->getName());
    dm.logger().log(msgType, "Output tensor (input tensor of the output op): " + dm.getOutput()->getTensor()->getName());

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
    mv::pass::DotPass dotPass(cm.logger(), ostream, mv::pass::DotPass::OutputScope::OpControlModel, mv::pass::DotPass::ContentLevel::ContentFull);

    dotPass.run(cm);    

    return 0;

}