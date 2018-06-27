#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/deployer/fstd_ostream.hpp"
#include "include/mcm/pass/deploy/generate_dot.hpp"
#include "include/mcm/pass/transform/fuse_bias.hpp"

int main()
{

    mv::OpModel om(mv::Logger::VerboseLevel::VerboseInfo);
    auto input = om.input(mv::Shape(64, 64, 16), mv::DType::Float, mv::Order::LastDimMajor);
    mv::dynamic_vector<mv::float_type> weightsData = mv::utils::generateSequence<mv::float_type>(3 * 3 * 16 * 32);
    auto weights = om.constant(weightsData, mv::Shape(3, 3, 16, 32), mv::DType::Float, mv::Order::LastDimMajor, "weights");
    auto conv = om.conv2D(input, weights, {1, 1}, {1, 1, 1, 1});
    mv::dynamic_vector<mv::float_type> biasesData = mv::utils::generateSequence<mv::float_type>(32);
    auto biases = om.constant(biasesData, mv::Shape(32), mv::DType::Float, mv::Order::LastDimMajor, "biases");
    auto bias = om.bias(conv,biases);
    om.output(bias);

    mv::FStdOStream ostream("cm1.dot");
    mv::pass::GenerateDot generateDot(ostream, mv::pass::GenerateDot::OutputScope::OpControlModel, mv::pass::GenerateDot::ContentLevel::ContentFull);
    bool dotResult = generateDot.run(om);    
    if (dotResult)
        system("dot -Tsvg cm1.dot -o cm1.svg");

    mv::pass::FuseBias fuseBias;
    fuseBias.run(om);
    
    ostream.setFileName("cm2.dot");
    dotResult = generateDot.run(om);    
    if (dotResult)
        system("dot -Tsvg cm2.dot -o cm2.svg");

    return 0;

}