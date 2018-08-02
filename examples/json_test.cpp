#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/tensor/math.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/pass/pass_registry.hpp"

int main()
{

    mv::CompilationUnit unit1;
    mv::OpModel om = unit1.model();
    auto input = om.input(mv::Shape(64, 64, 16), mv::DType::Float, mv::Order::ColumnMajor);
    mv::dynamic_vector<mv::float_type> weightsData = mv::utils::generateSequence<mv::float_type>(3 * 3 * 16 * 32);
    auto weights = om.constant(weightsData, mv::Shape(3, 3, 16, 32), mv::DType::Float, mv::Order::ColumnMajor, "weights");
    auto conv = om.conv2D(input, weights, {1, 1}, {1, 1, 1, 1});
    auto convOp = om.getSourceOp(conv);
    mv::dynamic_vector<mv::float_type> scalesData = mv::utils::generateSequence<mv::float_type>(32);
    auto scales = om.constant(scalesData, mv::Shape(32), mv::DType::Float, mv::Order::ColumnMajor, "biases");
    auto scale = om.scale(conv, scales);
    om.output(scale);
    
    unit1.loadTargetDescriptor(mv::Target::ma2480);
    unit1.compilationDescriptor()["GenerateJSON"]["output"] = std::string("example.json");
    unit1.compilationDescriptor()["GenerateBlob"]["output"] = std::string("example.blob");
    unit1.initialize();
    unit1.passManager().disablePass();
    unit1.passManager().enablePass(mv::PassGenre::Serialization, "GenerateJSON");
    unit1.passManager().enablePass(mv::PassGenre::Serialization, "GenerateBlob");
    auto result = unit1.run();

    mv::CompilationUnit unit2;

    unit2.loadModelFromJson("example.json");
    mv::OpModel om2 = unit2.model();

    unit2.loadTargetDescriptor(mv::Target::ma2480);
    unit2.compilationDescriptor()["GenerateBlob"]["output"] = std::string("example2.blob");
    unit2.initialize();
    unit2.passManager().disablePass();
    unit2.passManager().enablePass(mv::PassGenre::Serialization, "GenerateBlob");
    auto result2 = unit2.run();

}