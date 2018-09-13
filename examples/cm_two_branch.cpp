#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/utils/data_generator.hpp"

int main()
{

    mv::OpModel om;
    auto input = om.input({128, 128, 3}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);

    std::vector<double> conv1WeightsData = mv::utils::generateSequence<double>(3u * 3u * 3u * 8u);
    std::vector<double> conv2WeightsData = mv::utils::generateSequence<double>(3u * 3u * 3u * 8u);
    std::vector<double> conv3WeightsData = mv::utils::generateSequence<double>(5u * 5u * 16u * 32u);
    std::vector<double> conv4WeightsData = mv::utils::generateSequence<double>(6u * 6u * 32u * 64u);

    auto conv1WeightsIt = om.constant(conv1WeightsData, {3, 3, 3, 8}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    auto conv1It = om.conv2D(input, conv1WeightsIt, {2, 2}, {1, 1, 1, 1});
    auto pool1It = om.maxpool2D(conv1It, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto conv2WeightsIt = om.constant(conv2WeightsData, {3, 3, 3, 8}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    auto conv2It = om.conv2D(input, conv2WeightsIt, {2, 2}, {1, 1, 1, 1});
    auto pool2It = om.maxpool2D(conv2It, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto concat1It = om.concat(pool1It, pool2It);
    auto conv3WeightsIt = om.constant(conv3WeightsData, {5, 5, 16, 32}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    auto conv3It = om.conv2D(concat1It, conv3WeightsIt, {2, 2}, {2, 2, 2, 2});
    auto pool3It = om.maxpool2D(conv3It, {5, 5}, {3, 3}, {2, 2, 2, 2});
    auto conv4WeightsIt = om.constant(conv4WeightsData, {6, 6, 32, 64}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    auto conv4It = om.conv2D(pool3It, conv4WeightsIt, {1, 1}, {0, 0, 0, 0});
    om.output(conv4It);

    auto msgType = mv::Logger::MessageType::MessageInfo;
    mv::DataModel dm(om);

    mv::Logger::instance().log(msgType, "cm_two_branch", "Input op: " + om.getInput()->getName());
    mv::Logger::instance().log(msgType, "cm_two_branch", "Input tensor (output tensor of the input op): " + dm.getInputFlow()->getTensor()->getName());
    mv::Logger::instance().log(msgType, "cm_two_branch", "Output op: " + om.getOutput()->getName());
    mv::Logger::instance().log(msgType, "cm_two_branch", "Output tensor (input tensor of the output op): " + dm.getOutputFlow()->getTensor()->getName());

    mv::ControlModel cm(om);

    std::size_t i = 0;
    for (auto it = cm.getFirst(); it != cm.opEnd(); ++it)
    {
        mv::Logger::instance().log(msgType, "cm_two_branch", "Op " + mv::Printable::toString(i) + ": " + it->getName());
        ++i;
    }
    
    return 0;

}