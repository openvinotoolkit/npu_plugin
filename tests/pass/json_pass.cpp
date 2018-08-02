#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/tensor/math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"

TEST(jsonpass, case1)
{
    // Verbose level - change to mv::Logger::VerboseLevel::VerboseSilent to disable log messages
    mv::Logger::VerboseLevel verboseLevel = mv::Logger::VerboseLevel::VerboseInfo;

    // Define the primary compilation unit
    mv::CompilationUnit unit(verboseLevel);

    // Obtain a compositional model from the compilation unit
    mv::OpModel om = unit.model();

    auto input = om.input(mv::Shape(64, 64, 16), mv::DType::Float, mv::Order::ColumnMajor);
    mv::dynamic_vector<mv::float_type> weightsData = mv::utils::generateSequence<mv::float_type>(3 * 3 * 16 * 32);
    auto weights = om.constant(weightsData, mv::Shape(3, 3, 16, 32), mv::DType::Float, mv::Order::ColumnMajor, "weights");
    auto conv = om.conv2D(input, weights, {1, 1}, {1, 1, 1, 1});
    auto convOp = om.getSourceOp(conv);
    mv::dynamic_vector<mv::float_type> scalesData = mv::utils::generateSequence<mv::float_type>(32);
    auto scales = om.constant(scalesData, mv::Shape(32), mv::DType::Float, mv::Order::ColumnMajor, "biases");
    auto scale = om.scale(conv, scales);
    om.output(scale);
    
    // Load target descriptor for the selected target to the compilation unit
    std::string targetDescPath = std::getenv("MCM_HOME") + std::string("/config/target/ma2480.json");
    unit.targetDescriptor().load(targetDescPath);

    // Define the manadatory arguments for passes using compilation descriptor obtained from the compilation unit
    // Output DOT - file name (base)
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string("example.dot");
    // Output DOT - scope of visualization - executable operations, data flow, control flow
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("ExecOpControlModel");
    // Output DOT - content included in the visualization - full content
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    // Output DOT - HTML-like flag - enable HTML-like formatting
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    // Output BLOB - file name of the output binary
    unit.compilationDescriptor()["GenerateJson"]["output"] = std::string("example.json");
    unit.compilationDescriptor()["GenerateBlob"]["output"] = std::string("example.blob");

    // Initialize compilation
    unit.initialize();
    //unit.passManager().disablePass(mv::PassGenre::Serialization);

    // Run all passes
    auto result = unit.run();
}

TEST(jsonpass, case2)
{
    // Verbose level - change to mv::Logger::VerboseLevel::VerboseSilent to disable log messages
    mv::Logger::VerboseLevel verboseLevel = mv::Logger::VerboseLevel::VerboseInfo;

    // Define the primary compilation unit
    mv::CompilationUnit unit(verboseLevel);

    unit.loadModelFromJson("example.json");

    // Obtain a compositional model from the compilation unit
    mv::OpModel om = unit.model();

    // Load target descriptor for the selected target to the compilation unit
    std::string targetDescPath = std::getenv("MCM_HOME") + std::string("/config/target/ma2480.json");
    unit.targetDescriptor().load(targetDescPath);

    // Define the manadatory arguments for passes using compilation descriptor obtained from the compilation unit
    // Output DOT - file name (base)
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string("example2.dot");
    // Output DOT - scope of visualization - executable operations, data flow, control flow
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("ExecOpControlModel");
    // Output DOT - content included in the visualization - full content
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    // Output DOT - HTML-like flag - enable HTML-like formatting
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    // Output BLOB - file name of the output binary
    unit.compilationDescriptor()["GenerateJson"]["output"] = std::string("example2.json");
    unit.compilationDescriptor()["GenerateBlob"]["output"] = std::string("example2.blob");

    // Initialize compilation
    unit.initialize();
    //unit.passManager().disablePass(mv::PassGenre::Serialization);

    // Run all passes
    auto result = unit.run();
}

