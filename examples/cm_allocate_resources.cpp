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
    mv::dynamic_vector<mv::float_type> conv2WeightsData = mv::utils::generateSequence<mv::float_type>(5u * 5u * 8u * 16u);
    mv::dynamic_vector<mv::float_type> conv3WeightsData = mv::utils::generateSequence<mv::float_type>(4u * 4u * 16u * 32u);

    auto conv1WeightsIt = om.constant(conv1WeightsData, mv::Shape(3, 3, 3, 8), mv::DType::Float, mv::Order::NWHC);
    auto conv1It = om.conv2D(inIt->getOutput(0), conv1WeightsIt->getOutput(0), {2, 2}, {1, 1, 1, 1});
    auto pool1It = om.maxpool2D(conv1It->getOutput(0), {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto conv2WeightsIt = om.constant(conv2WeightsData, mv::Shape(5, 5, 8, 16), mv::DType::Float, mv::Order::NWHC);
    auto conv2It = om.conv2D(pool1It->getOutput(0), conv2WeightsIt->getOutput(0), {2, 2}, {2, 2, 2, 2});
    auto pool2It = om.maxpool2D(conv2It->getOutput(0), {5, 5}, {4, 4}, {2, 2, 2, 2});
    auto conv3WeightsIt = om.constant(conv3WeightsData, mv::Shape(4, 4, 16, 32), mv::DType::Float, mv::Order::NWHC);
    auto conv3It = om.conv2D(pool2It->getOutput(0), conv3WeightsIt->getOutput(0), {1, 1}, {0, 0, 0, 0});
    auto outIt = om.output(conv3It->getOutput(0));

    auto group1It = om.addGroup("pools");
    om.addGroupElement(pool1It, group1It);
    om.addGroupElement(pool2It, group1It);

    auto group2It = om.addGroup("convs");
    om.addGroupElement(conv1It, group2It);
    om.addGroupElement(conv2It, group2It);
    om.addGroupElement(conv3It, group2It);

   
    auto group3It = om.addGroup("ops");
    om.addGroupElement(group1It, group3It);
    om.addGroupElement(group2It, group3It);

    auto group4It = om.addGroup("first");
    om.addGroupElement(conv1It, group4It);
    om.addGroupElement(pool1It, group4It);

    for (auto it = om.groupBegin(); it != om.groupEnd(); ++it)
    {

        std::cout << it->getName() << std::endl;

        for (auto it2 = om.memberBegin(it); it2 != om.memberEnd(it); ++it2)
        {
            std::cout << it2->getName() << std::endl;
        }

    }

    std::cout << conv1It->toString() << std::endl;

    /*mv::StdOStream ostream;
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
    cm.addToStage(stage1It, conv1It);
    cm.addToStage(stage2It, pool1It);
    cm.addToStage(stage3It, conv2It);
    cm.addToStage(stage4It, pool2It);
    cm.addToStage(stage5It, conv3It);
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
    dm.allocateTensor("BSS", stageIt, conv1It.leftmostInput()->getTensor());
    dm.allocateTensor("BSS", stageIt, conv1It.leftmostOutput()->getTensor());

    ++stageIt;

    dm.allocateTensor("BSS", stageIt, pool1It.leftmostInput()->getTensor());

    return 0;

}