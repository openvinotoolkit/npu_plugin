#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/deployer/fstd_ostream.hpp"
#include "include/mcm/pass/deploy/generate_dot.hpp"
#include "include/mcm/pass/transform/fuse_batch_norm.hpp"

int main()
{

    mv::OpModel om(mv::Logger::VerboseLevel::VerboseInfo);
    auto input = om.input(mv::Shape(224, 224, 3), mv::DType::Float, mv::Order::LastDimMajor);
    mv::dynamic_vector<mv::float_type> weightsData = mv::utils::generateSequence<mv::float_type>(3 * 3 * 3 * 3);
    auto weights = om.constant(weightsData, mv::Shape(3, 3, 3, 3), mv::DType::Float, mv::Order::LastDimMajor, "weights");
    auto conv = om.conv2D(input, weights, {1, 1}, {1, 1, 1, 1});
    mv::dynamic_vector<mv::float_type> meanData = mv::utils::generateSequence<mv::float_type>(conv->getShape().totalSize());
    mv::dynamic_vector<mv::float_type> varianceData = mv::utils::generateSequence<mv::float_type>(conv->getShape().totalSize());
    mv::dynamic_vector<mv::float_type> offsetData = mv::utils::generateSequence<mv::float_type>(conv->getShape().totalSize());
    mv::dynamic_vector<mv::float_type> scaleData = mv::utils::generateSequence<mv::float_type>(conv->getShape().totalSize());
    auto bnmean = om.constant(meanData, conv->getShape(), mv::DType::Float, mv::Order::LastDimMajor, "mean");
    auto bnvariance = om.constant(varianceData, conv->getShape(), mv::DType::Float, mv::Order::LastDimMajor, "variance");
    auto bnoffset = om.constant(offsetData, conv->getShape(), mv::DType::Float, mv::Order::LastDimMajor, "offset");
    auto bnscale = om.constant(scaleData, conv->getShape(), mv::DType::Float, mv::Order::LastDimMajor, "scale");
    auto batchnorm = om.batchNorm(conv, bnmean, bnvariance, bnoffset, bnscale, 1e-6);
    om.output(batchnorm);

    mv::FStdOStream ostream("cm1.dot");
    mv::pass::GenerateDot generateDot(ostream, mv::pass::GenerateDot::OutputScope::ControlModel, mv::pass::GenerateDot::ContentLevel::ContentFull);
    bool dotResult = generateDot.run(om);    
    if (dotResult)
        system("dot -Tsvg cm1.dot -o cm1.svg");

    mv::pass::FuseBatchNorm fuseBatchNorm;
    fuseBatchNorm.run(om);
    
    ostream.setFileName("cm2.dot");
    dotResult = generateDot.run(om);    
    if (dotResult)
        system("dot -Tsvg cm2.dot -o cm2.svg");

    return 0;

}