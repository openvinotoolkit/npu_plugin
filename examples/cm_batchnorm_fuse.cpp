#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/compiler/compilation_unit.hpp"

int main()
{

    mv::CompilationUnit unit(mv::Logger::VerboseLevel::VerboseInfo);
    mv::CompositionalModel& cm = unit.model();

    auto input = cm.input(mv::Shape(224, 224, 3), mv::DType::Float, mv::Order::ColumnMajor);
    mv::dynamic_vector<mv::float_type> weightsData = mv::utils::generateSequence<mv::float_type>(3 * 3 * 3 * 3);
    auto weights = cm.constant(weightsData, mv::Shape(3, 3, 3, 3), mv::DType::Float, mv::Order::ColumnMajor, "weights");
    auto conv = cm.conv2D(input, weights, {1, 1}, {1, 1, 1, 1});
    mv::dynamic_vector<mv::float_type> meanData = mv::utils::generateSequence<mv::float_type>(conv->getShape().totalSize());
    mv::dynamic_vector<mv::float_type> varianceData = mv::utils::generateSequence<mv::float_type>(conv->getShape().totalSize());
    mv::dynamic_vector<mv::float_type> offsetData = mv::utils::generateSequence<mv::float_type>(conv->getShape().totalSize());
    mv::dynamic_vector<mv::float_type> scaleData = mv::utils::generateSequence<mv::float_type>(conv->getShape().totalSize());
    auto bnmean = cm.constant(meanData, conv->getShape(), mv::DType::Float, mv::Order::ColumnMajor, "mean");
    auto bnvariance = cm.constant(varianceData, conv->getShape(), mv::DType::Float, mv::Order::ColumnMajor, "variance");
    auto bnoffset = cm.constant(offsetData, conv->getShape(), mv::DType::Float, mv::Order::ColumnMajor, "offset");
    auto bnscale = cm.constant(scaleData, conv->getShape(), mv::DType::Float, mv::Order::ColumnMajor, "scale");
    auto batchnorm = cm.batchNorm(conv, bnmean, bnvariance, bnoffset, bnscale, 1e-6);
    cm.output(batchnorm);

    std::string targetDescPath = std::getenv("MCM_HOME") + std::string("/config/target/ma2480.json");
    unit.targetDescriptor().load(targetDescPath);
    unit.passManager().disablePass();
    unit.passManager().enablePass(mv::PassGenre::Adaptation, "FuseBatchNorm");
    unit.passManager().enablePass(mv::PassGenre::Validation, "GenerateDot");
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string("cm_batchnorm_fuse.dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("ExecOpControlModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    
    unit.initialize();
    unit.run();

    //system("dot -Tsvg cm_batchnorm_fuse.dot -o cm_batchnorm_fuse.svg");
    //system("dot -Tsvg cm_batchnorm_fuse_adapt.dot -o cm_batchnorm_fuse_adapt.svg");

    return 0;

}