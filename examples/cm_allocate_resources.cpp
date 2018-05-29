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
    mv::vector<mv::float_type> conv2WeightsData = mv::utils::generateSequence<mv::float_type>(5u * 5u * 8u * 16u);
    mv::vector<mv::float_type> conv3WeightsData = mv::utils::generateSequence<mv::float_type>(4u * 4u * 16u * 32u);

    auto conv1It = om.conv(inIt, mv::ConstantTensor(mv::Shape(3, 3, 3, 8), mv::DType::Float, mv::Order::NWHC, conv1WeightsData), 2, 2, 1, 1);
    auto pool1It = om.maxpool(conv1It, mv::Shape(3, 3), 2, 2, 1, 1);
    auto conv2It = om.conv(pool1It, mv::ConstantTensor(mv::Shape(5, 5, 8, 16), mv::DType::Float, mv::Order::NWHC, conv2WeightsData), 2, 2, 2, 2);
    auto pool2It = om.maxpool(conv2It, mv::Shape(5, 5), 4, 4, 2, 2);
    auto conv3It = om.conv(pool2It, mv::ConstantTensor(mv::Shape(4, 4, 16, 32), mv::DType::Float, mv::Order::NWHC, conv3WeightsData), 1, 1, 0, 0);
    auto outIt = om.output(conv3It);

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

    return 0;

}