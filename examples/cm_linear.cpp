#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

int main()
{

    mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);

    // Define blank computation model (op view)
    mv::OpModel om("Model1");

    // Initialize weights data
    std::vector<double> weights1Data = mv::utils::generateSequence<double>(3u * 3u * 3u * 8u);
    std::vector<double> weights2Data = mv::utils::generateSequence<double>(5u * 5u * 8u * 16u);
    std::vector<double> weights3Data = mv::utils::generateSequence<double>(4u * 4u * 16u * 32u);

    // Compose model - use Composition API to create ops and obtain tensors
    auto input = om.input({128, 128, 3}, mv::DTypeType::Float16, mv::Order("CHW"));
    auto weights1 = om.constant(weights1Data, {3, 3, 3, 8}, mv::DTypeType::Float16, mv::Order("NCHW"));
    auto conv1 = om.conv(input, weights1, {2, 2}, {1, 1, 1, 1});
    auto pool1 = om.maxPool(conv1, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto weights2 = om.constant(weights2Data, {5, 5, 8, 16}, mv::DTypeType::Float16, mv::Order("NCHW"));
    auto conv2 = om.conv(pool1, weights2, {2, 2}, {2, 2, 2, 2});
    auto pool2 = om.maxPool(conv2, {5, 5}, {4, 4}, {2, 2, 2, 2});
    auto weights3 = om.constant(weights3Data, {4, 4, 16, 32}, mv::DTypeType::Float16, mv::Order("NCHW"));
    auto conv3 = om.conv(pool2, weights3, {1, 1}, {0, 0, 0, 0});
    om.output(conv3);

    // Obtain ops from tensors and add them to groups
    auto pool1Op = om.getSourceOp(pool1);
    auto pool2Op = om.getSourceOp(pool2);
    auto group1It = om.addGroup("pools");
    om.addGroupElement(pool1Op, group1It);
    om.addGroupElement(pool2Op, group1It);

    auto group2It = om.addGroup("convs");
    auto conv1Op = om.getSourceOp(conv1);
    auto conv2Op = om.getSourceOp(conv2);
    auto conv3Op = om.getSourceOp(conv3);
    om.addGroupElement(conv1Op, group2It);
    om.addGroupElement(conv2Op, group2It);
    om.addGroupElement(conv3Op, group2It);

    // Add groups to another group
    auto group3It = om.addGroup("ops");
    om.addGroupElement(group1It, group3It);
    om.addGroupElement(group2It, group3It);

    // Add ops that are already in some group to another group
    auto group4It = om.addGroup("first");
    om.addGroupElement(conv1Op, group4It);
    om.addGroupElement(pool1Op, group4It);

    mv::ControlModel cm(om);

    std::size_t i = 0;
    for (mv::Control::OpDFSIterator it = cm.getFirst(); it != cm.opEnd(); ++it)
    {
        cm.log(mv::Logger::MessageType::Info, "Op " + std::to_string(i) + ": " + it->getName());
        ++i;
    }

}
