#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/deployer/cstd_ostream.hpp"
#include "include/mcm/pass/deploy/dot_pass.hpp"

int main()
{
    // Define blank computation model (op view)
    mv::OpModel om(mv::Logger::VerboseLevel::VerboseInfo);
    
    // Initialize weights data
    mv::dynamic_vector<mv::float_type> weights1Data = mv::utils::generateSequence<mv::float_type>(3u * 3u * 3u * 8u);
    mv::dynamic_vector<mv::float_type> weights2Data = mv::utils::generateSequence<mv::float_type>(5u * 5u * 8u * 16u);
    mv::dynamic_vector<mv::float_type> weights3Data = mv::utils::generateSequence<mv::float_type>(4u * 4u * 16u * 32u);

    // Compose model - use Composition API to create ops and obtain tensors 
    auto input = om.input(mv::Shape(128, 128, 3), mv::DType::Float, mv::Order::NWHC);
    auto weights1 = om.constant(weights1Data, mv::Shape(3, 3, 3, 8), mv::DType::Float, mv::Order::NWHC);
    auto conv1 = om.conv2D(input, weights1, {2, 2}, {1, 1, 1, 1});
    auto pool1 = om.maxpool2D(conv1, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto weights2 = om.constant(weights2Data, mv::Shape(5, 5, 8, 16), mv::DType::Float, mv::Order::NWHC);
    auto conv2 = om.conv2D(pool1, weights2, {2, 2}, {2, 2, 2, 2});
    auto pool2 = om.maxpool2D(conv2, {5, 5}, {4, 4}, {2, 2, 2, 2});
    auto weights3 = om.constant(weights3Data, mv::Shape(4, 4, 16, 32), mv::DType::Float, mv::Order::NWHC);
    auto conv3 = om.conv2D(pool2, weights3, {1, 1}, {0, 0, 0, 0});
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

    for (auto it = om.groupBegin(); it != om.groupEnd(); ++it)
    {

        std::cout << it->getName() << std::endl;

        for (auto it2 = om.memberBegin(it); it2 != om.memberEnd(it); ++it2)
        {
            std::cout << it2->getName() << std::endl;
        }

    }

    std::cout << conv1->toString() << std::endl;

    /*mv::CStdOStream ostream;
    mv::pass::DotPass dotPass(om.logger(), ostream, mv::pass::DotPass::OutputScope::OpControlModel, mv::pass::DotPass::ContentLevel::ContentFull);
    dotPass.run(om);*/

    std::cout << group1It->toString() << std::endl;

    mv::ControlModel cm(om);

    /*auto stage0It = cm.addStage();
    auto stage1It = cm.addStage();
    auto stage2It = cm.addStage();
    auto stage3It = cm.addStage();
    auto stage4It = cm.addStage();
    auto stage5It = cm.addStage();
    auto stage6It = cm.addStage();

    cm.addToStage(stage0It, inIt);
    cm.addToStage(stage1It, conv1);
    cm.addToStage(stage2It, pool1);
    cm.addToStage(stage3It, conv2);
    cm.addToStage(stage4It, pool2);
    cm.addToStage(stage5It, conv3);
    cm.addToStage(stage6It, outIt);*/

    for (auto itStage = cm.stageBegin(); itStage != cm.stageEnd(); ++itStage)
    {
        std::cout << itStage->getName() << ":";
        for (auto it = cm.stageMemberBegin(itStage); it != cm.stageMemberEnd(itStage); ++it)
            std::cout << " " << it->getName();
        std::cout << std::endl;
    }

    //cm.removeStage(stage5It);

    mv::DataModel dm(cm);

    auto stageIt = cm.stageBegin();

    dm.addAllocator("BSS", 1048576);
    dm.allocateTensor("BSS", stageIt, om.getSourceOp(conv1).leftmostInput()->getTensor());
    dm.allocateTensor("BSS", stageIt, om.getSourceOp(conv1).leftmostOutput()->getTensor());

    ++stageIt;

    dm.allocateTensor("BSS", stageIt, om.getSourceOp(pool1).leftmostInput()->getTensor());

    return 0;

}