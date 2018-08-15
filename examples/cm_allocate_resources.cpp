#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

int main()
{
    // Define the primary compilation unit
    mv::CompilationUnit unit(mv::Logger::VerboseLevel::VerboseInfo);

    // Obtain compositional model from the compilation unit
    mv::CompositionalModel& cm = unit.model();
    
    // Initialize weights data
    mv::dynamic_vector<mv::float_type> weights1Data = mv::utils::generateSequence<mv::float_type>(3u * 3u * 3u * 8u);
    mv::dynamic_vector<mv::float_type> weights2Data = mv::utils::generateSequence<mv::float_type>(5u * 5u * 8u * 16u);
    mv::dynamic_vector<mv::float_type> weights3Data = mv::utils::generateSequence<mv::float_type>(4u * 4u * 16u * 32u);

    // Compose model - use Composition API to create ops and obtain tensors 
    auto input = cm.input(mv::Shape(128, 128, 3), mv::DType::Float, mv::Order::ColumnMajor);
    auto weights1 = cm.constant(weights1Data, mv::Shape(3, 3, 3, 8), mv::DType::Float, mv::Order::ColumnMajor);
    auto conv1 = cm.conv2D(input, weights1, {2, 2}, {1, 1, 1, 1});
    auto pool1 = cm.maxpool2D(conv1, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto weights2 = cm.constant(weights2Data, mv::Shape(5, 5, 8, 16), mv::DType::Float, mv::Order::ColumnMajor);
    auto conv2 = cm.conv2D(pool1, weights2, {2, 2}, {2, 2, 2, 2});
    auto pool2 = cm.maxpool2D(conv2, {5, 5}, {4, 4}, {2, 2, 2, 2});
    auto weights3 = cm.constant(weights3Data, mv::Shape(4, 4, 16, 32), mv::DType::Float, mv::Order::ColumnMajor);
    auto conv3 = cm.conv2D(pool2, weights3, {1, 1}, {0, 0, 0, 0});
    cm.output(conv3); 

    // Load target descriptor for the selected target to the compilation unit
    std::string targetDescPath = std::getenv("MCM_HOME") + std::string("/config/target/ma2480.json");
    unit.loadTargetDescriptor(mv::Target::ma2480);

    // Define the manadatory arguments for passes using compilation descriptor obtained from the compilation unit
    // Output DOT - file name (base)
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string("allocate_resouces.dot");
    // Output DOT - scope of visualization - executable operations, data flow, control flow
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("ExecOpControlModel");
    // Output DOT - content included in the visualization - full content
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    // Output DOT - HTML-like flag - enable HTML-like formatting
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    // Output BLOB - file name of the output binary
    unit.compilationDescriptor()["GenerateBlob"]["output"] = std::string("allocate_resouces.blob");

    // Initialize compilation 
    unit.initialize();

    // Run all passes
    auto result = unit.run();

    // Obtain ops from tensors and add them to groups
    //auto pool1Op = cm.getSourceOp(pool1);
    //auto pool2Op = cm.getSourceOp(pool2);

    /*auto group1It = om.addGroup("pools");
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

    mv::DataModel dm(om);

    for (auto it = om.groupBegin(); it != om.groupEnd(); ++it)
    {

        std::cout << it->getName() << std::endl;

        for (auto it2 = om.memberBegin(it); it2 != om.memberEnd(it); ++it2)
        {
            std::cout << it2->getName() << std::endl;
        }

    }

    std::cout << conv1->toString() << std::endl;*/

    /*mv::CStdOStream ostream;
    mv::pass::DotPass dotPass(om.logger(), ostream, mv::pass::DotPass::OutputScope::OpControlModel, mv::pass::DotPass::ContentLevel::ContentFull);
    dotPass.run(om);*/

    //std::cout << group1It->toString() << std::endl;

    //mv::ControlModel cm(om);

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

    /*for (auto itStage = cm.stageBegin(); itStage != cm.stageEnd(); ++itStage)
    {
        std::cout << itStage->getName() << ":";
        for (auto it = cm.stageMemberBegin(itStage); it != cm.stageMemberEnd(itStage); ++it)
            std::cout << " " << it->getName();
        std::cout << std::endl;
    }*/

    //cm.removeStage(stage5It);

    //mv::DataModel dm(cm);

    /*auto stageIt = cm.stageBegin();

    dm.addAllocator("BSS", 1048576);
    dm.allocateTensor("BSS", stageIt, om.getSourceOp(conv1).leftmostInput()->getTensor());
    dm.allocateTensor("BSS", stageIt, om.getSourceOp(conv1).leftmostOutput()->getTensor());

    ++stageIt;

    dm.allocateTensor("BSS", stageIt, om.getSourceOp(pool1).leftmostInput()->getTensor());*/

    return 0;

}
