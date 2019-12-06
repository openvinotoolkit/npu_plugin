#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{
    double inf = std::numeric_limits<double>::infinity();
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();


    auto input0 = om.input({16,16,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4),  {{0},{1},{-inf},{inf}}, "input#170");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (16*128*3*3, 1, 0);

    // per channel scale
    const std::vector<int64_t> zp(128, 0);
    std::vector<double> scale(128);
    for (std::size_t i = 0; i < 128; i++)
    {
        if (i%2==0)
        {
            scale[i] = 1;
        }
        else
        {
            scale[i] = 0.5;
        }
    }
    const std::vector<double> infv(128, -inf);

    const std::vector<double> infvp(128, inf);
    auto weights0 = om.constantInt(weightsData0,{3,3,16,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {zp,scale,infv,infvp});
    auto conv0 = om.conv(input0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("UInt8"),{{0},{1},{-inf},{inf}} , "conv");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (128, 3, 0);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{128}, mv::DType("Int32"), mv::Order::getColMajorID(1), {{},{},{},{}});
    auto bias_c0 = om.bias(conv0, biasWeights0, mv::DType("Default"), {{0},{1},{-inf},{inf}}, "conv:bias");
    auto relu = om.relu(bias_c0,  mv::DType("Default"), {{0},{1},{-inf},{inf}}, "relu");
    om.output(relu);

    std::string path = std::getenv("MCM_HOME");
    std::string compDescPath = path + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
    std::cout << " DONE " << std::endl;
}