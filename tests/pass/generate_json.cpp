#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"

TEST(generate_json, case1)
{

    // Test files names
    /*std::string originalBlob = "generate_json_case1_original.blob";
    std::string restoredBlob = "generate_json_case1_restored.blob";
    std::string json = "generate_json_case1.json";

    // Compose model, save it to a JSON file and compile it to a blob
    mv::CompilationUnit unit1;
    mv::OpModel om = unit1.model();
    auto input = om.input(mv::Shape(64, 64, 16), mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(3 * 3 * 16 * 32);
    auto weights = om.constant(weightsData, mv::Shape(3, 3, 16, 32), mv::DType("Float16"), mv::Order("CHW"), "weights");
    auto conv = om.conv(input, weights, {1, 1}, {1, 1, 1, 1});
    auto convOp = om.getSourceOp(conv);
    std::vector<double> scalesData = mv::utils::generateSequence<double>(32);
    auto scales = om.constant(scalesData, mv::Shape(32), mv::DType("Float16"), mv::Order("CHW"), "biases");
    auto scale = om.scale(conv, scales);
    om.output(scale);
    
    unit1.loadTargetDescriptor(mv::Target::ma2480);
    unit1.compilationDescriptor()["GenerateJSON"]["output"] = json;
    unit1.compilationDescriptor()["GenerateBlob"]["output"] = originalBlob;
    unit1.initialize();
    unit1.passManager().disablePass();
    unit1.passManager().enablePass(mv::PassGenre::Serialization, "GenerateJSON");
    unit1.passManager().enablePass(mv::PassGenre::Serialization, "GenerateBlob");
    auto result1 = unit1.run();

    // Restore model from the JSON file generated previously, compile it to a blob
    mv::CompilationUnit unit2;

    unit2.loadModelFromJson(json);
    mv::OpModel om2 = unit2.model();

    unit2.loadTargetDescriptor(mv::Target::ma2480);
    unit2.compilationDescriptor()["GenerateBlob"]["output"] = restoredBlob;
    unit2.initialize();
    unit2.passManager().disablePass();
    unit2.passManager().enablePass(mv::PassGenre::Serialization, "GenerateBlob");
    auto result2 = unit2.run();

    // Compare original and restored blob
    EXPECT_EQ(result1["passes"].last()["blobSize"].get<long long>(), 
        result2["passes"].last()["blobSize"].get<long long>()) << "ERROR: wrong blob size";

    std::string command = "diff \"" + originalBlob + "\" \"" + restoredBlob + "\"";
    EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";*/

}
