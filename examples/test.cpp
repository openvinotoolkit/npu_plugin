#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

int main()
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt6 = test_cm.input({24, 24, 20}, mv::DTypeType::Float16, mv::OrderType::RowMajor);
    // define first convolution  3D conv
    std::vector<double> weightsData61 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000, 0.010);
    auto weightsIt61 = test_cm.constant(weightsData61, {5, 5, 3, 1}, mv::DTypeType::Float16, mv::OrderType::ColumnMajorPlanar);   // kh, kw, ins, outs
    auto convIt61 = test_cm.conv2D(inIt6, weightsIt61, {2, 2}, {0, 0, 0, 0});
    // define first maxpool
    auto maxpoolIt61 = test_cm.maxpool2D(convIt61,{5,5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData62 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 1.000, 0.010);
    auto weightsIt62 = test_cm.constant(weightsData62, {3, 3, 1, 1}, mv::DTypeType::Float16, mv::OrderType::ColumnMajorPlanar);   // kh, kw, ins, outs
    auto convIt62 = test_cm.conv2D(maxpoolIt61, weightsIt62, {1, 1}, {0, 0, 0, 0});
    // define second maxpool
    auto maxpoolIt62 = test_cm.maxpool2D(convIt62,{3,3}, {2, 2}, {1, 1, 1, 1});
    // define output
    auto outIt6 = test_cm.output(maxpoolIt62);


    std::string blobName = "test_conv_06.blob";
    unit.compilationDescriptor()["GenerateBlob"]["output"] = blobName;
    unit.compilationDescriptor()["MarkHardwareConvolution"]["disableHardware"] = true;
    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();
    unit.passManager().disablePass(mv::PassGenre::Validation);
    unit.passManager().disablePass(mv::PassGenre::Serialization);
    unit.passManager().enablePass(mv::PassGenre::Serialization, "GenerateBlob");

    auto compOutput = unit.run();


}