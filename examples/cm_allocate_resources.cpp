#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

int main()
{
    // Define the primary compilation unit
    mv::CompilationUnit unit("model1");

    // Obtain compositional model from the compilation unit
    mv::OpModel& om = unit.model();
    // Initialize weights data
    std::vector<double> weights1Data = mv::utils::generateSequence<double>(3u * 3u * 3u * 8u);
    std::vector<double> weights2Data = mv::utils::generateSequence<double>(5u * 5u * 8u * 16u);
    std::vector<double> weights3Data = mv::utils::generateSequence<double>(4u * 4u * 16u * 32u);

    // Compose model - use Composition API to create ops and obtain tensors
    auto input = om.input({128, 128, 3}, mv::DType("Float16"), mv::Order("CHW"));
    auto weights1 = om.constant(weights1Data, {3, 3, 3, 8}, mv::DType("Float16"), mv::Order("NCHW"));
    auto conv1 = om.conv(input, weights1, {2, 2}, {1, 1, 1, 1}, 1);
    auto pool1 = om.maxPool(conv1, {3, 3}, {2, 2}, {1, 1, 1, 1});
    auto weights2 = om.constant(weights2Data, {5, 5, 8, 16}, mv::DType("Float16"), mv::Order("NCHW"));
    auto conv2 = om.conv(pool1, weights2, {2, 2}, {2, 2, 2, 2}, 1);
    auto pool2 = om.maxPool(conv2, {5, 5}, {4, 4}, {2, 2, 2, 2});
    auto weights3 = om.constant(weights3Data, {4, 4, 16, 32}, mv::DType("Float16"), mv::Order("NCHW"));
    auto conv3 = om.conv(pool2, weights3, {1, 1}, {0, 0, 0, 0}, 1);
    om.output(conv3);

    // Load target descriptor for the selected target to the compilation unit
    //std::string targetDescPath = std::getenv("MCM_HOME") + std::string("/config/target/ma2480.json");
    //unit.loadTargetDescriptor(mv::Target::ma2480);

    // Load target descriptor for the selected target to the compilation unit
    if (!unit.loadTargetDescriptor(mv::Target::ma2480))
    	exit(1);

    unit.loadDefaultCompilationDescriptor();
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "allocate_resources.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);

    compDesc.setPassArg("GenerateDot", "output", std::string("allocate_resources.dot"));
    compDesc.setPassArg("GenerateDot", "scope", std::string("OpControlModel"));
    compDesc.setPassArg("GenerateDot", "content", std::string("full"));
    compDesc.setPassArg("GenerateDot", "html", true);

    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    // Initialize compilation
    unit.initialize();

    // Run all passes
    auto result = unit.run();

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

    auto stage1It = cm.addStage();
    auto stage2It = cm.addStage();
    auto stage3It = cm.addStage();
    auto stage4It = cm.addStage();
    auto stage5It = cm.addStage();

    cm.addToStage(stage1It, conv1Op);
    cm.addToStage(stage2It, pool1Op);
    cm.addToStage(stage3It, conv2Op);
    cm.addToStage(stage4It, pool2Op);
    cm.addToStage(stage5It, conv3Op);

    cm.removeStage(stage5It);

    return 0;

}
