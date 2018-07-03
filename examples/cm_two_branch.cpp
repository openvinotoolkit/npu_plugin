#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/base/stream/fstd_ostream.hpp"
#include "include/mcm/pass/deploy/generate_dot.hpp"

int main()
{

    mv::OpModel om(mv::Logger::VerboseLevel::VerboseInfo);
    auto input = om.input(mv::Shape(128, 128, 3), mv::DType::Float, mv::Order::LastDimMajor);

    mv::dynamic_vector<mv::float_type> conv1WeightsData = mv::utils::generateSequence<mv::float_type>(3u * 3u * 3u * 8u);
    mv::dynamic_vector<mv::float_type> conv2WeightsData = mv::utils::generateSequence<mv::float_type>(3u * 3u * 3u * 8u);
    mv::dynamic_vector<mv::float_type> conv3WeightsData = mv::utils::generateSequence<mv::float_type>(5u * 5u * 16u * 32u);
    mv::dynamic_vector<mv::float_type> conv4WeightsData = mv::utils::generateSequence<mv::float_type>(6u * 6u * 32u * 64u);

    auto conv1WeightsIt = om.constant(conv1WeightsData, mv::Shape(3, 3, 3, 8), mv::DType::Float, mv::Order::LastDimMajor);
    auto conv1It = om.conv2D(input, conv1WeightsIt, {2, 2}, {1, 1, 1, 1});
    auto pool1It = om.maxpool2D(conv1It, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto conv2WeightsIt = om.constant(conv2WeightsData, mv::Shape(3, 3, 3, 8), mv::DType::Float, mv::Order::LastDimMajor);
    auto conv2It = om.conv2D(input, conv2WeightsIt, {2, 2}, {1, 1, 1, 1});
    auto pool2It = om.maxpool2D(conv2It, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto concat1It = om.concat(pool1It, pool2It);
    auto conv3WeightsIt = om.constant(conv3WeightsData, mv::Shape(5, 5, 16, 32), mv::DType::Float, mv::Order::LastDimMajor);
    auto conv3It = om.conv2D(concat1It, conv3WeightsIt, {2, 2}, {2, 2, 2, 2});
    auto pool3It = om.maxpool2D(conv3It, {5, 5}, {3, 3}, {2, 2, 2, 2});
    auto conv4WeightsIt = om.constant(conv4WeightsData, mv::Shape(6, 6, 32, 64), mv::DType::Float, mv::Order::LastDimMajor);
    auto conv4It = om.conv2D(pool3It, conv4WeightsIt, {1, 1}, {0, 0, 0, 0});
    om.output(conv4It);

    auto msgType = mv::Logger::MessageType::MessageInfo;
    mv::DataModel dm(om);

    dm.logger().log(msgType, "Input op: " + om.getInput()->getName());
    dm.logger().log(msgType, "Input tensor (output tensor of the input op): " + dm.getInputFlow()->getTensor()->getName());
    dm.logger().log(msgType, "Output op: " + om.getOutput()->getName());
    dm.logger().log(msgType, "Output tensor (input tensor of the output op): " + dm.getOutputFlow()->getTensor()->getName());

    mv::ControlModel cm(om);

    cm.logger().log(msgType, "First op: " + cm.getFirst()->getName());
    cm.logger().log(msgType, "Last op: " + cm.getLast()->getName());

    mv::size_type i = 0;
    for (auto it = cm.getFirst(); it != cm.opEnd(); ++it)
    {
        cm.logger().log(msgType, "Op " + mv::Printable::toString(i) + ": " + it->getName());
        ++i;
    }

    mv::FStdOStream ostream("om.dot");
    mv::pass::GenerateDot generateOMDot(ostream, mv::pass::GenerateDot::OutputScope::OpModel, mv::pass::GenerateDot::ContentLevel::ContentFull);
    bool dotResult = generateOMDot.run(om);    
    if (dotResult)
        system("dot -Tsvg om.dot -o om.svg");

    ostream.setFileName("cm.dot");
    mv::pass::GenerateDot generateCMDot(ostream, mv::pass::GenerateDot::OutputScope::ControlModel, mv::pass::GenerateDot::ContentLevel::ContentFull);
    dotResult = generateCMDot.run(cm);    
    if (dotResult)
        system("dot -Tsvg cm.dot -o cm.svg");

    ostream.setFileName("dm.dot");
    mv::pass::GenerateDot generateDMDot(ostream, mv::pass::GenerateDot::OutputScope::DataModel, mv::pass::GenerateDot::ContentLevel::ContentFull);
    dotResult = generateDMDot.run(dm);    
    if (dotResult)
        system("dot -Tsvg dm.dot -o dm.svg");

    return 0;

}