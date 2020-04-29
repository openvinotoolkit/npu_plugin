#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("DepthwiseChannelScalingModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({16,16,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#3");


    std::vector<double> scale(16);
    for (std::size_t i = 0; i < 16; i++)
    {
        if (i%2==0)
        {
            scale[i] = 0.0015868041664361954;
        }
        else
        {
            scale[i] = 0.0015868041664361954/2;
        }
    }

    // Load weights from file
    std::vector<int64_t> d_weightsData0(1*1*16*1);
    std::string Weights_filename(mv::utils::projectRootPath() + "/tests/layer/depthwise_channel_scaling/depthwise_channel_scaling.in2");
    std::ifstream w_file;
    w_file.open(Weights_filename, std::fstream::in | std::fstream::binary);
    w_file.read((char*)(d_weightsData0.data()), 16 * sizeof(uint64_t));

    // Load biases from file
    std::vector<int64_t> biasd_WeightsData0(16);
    std::string Biases_filename(mv::utils::projectRootPath() + "/tests/layer/depthwise_channel_scaling/depthwise_channel_scaling.in3");
    std::ifstream b_file;
    b_file.open(Biases_filename, std::fstream::in | std::fstream::binary);
    b_file.read((char*)(biasd_WeightsData0.data()), 16 * sizeof(uint64_t));

    auto d_weights0 = om.constantInt(d_weightsData0,{1,1,16,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{145},{scale},{-0.2301538735628128},{0.17448118329048157}}, "dwconv0#0_weights#1");
    auto depthConv0 = om.depthwiseConv(input0, d_weights0, {1, 1}, {0, 0, 0, 0}, 1, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}}, "dwconv0#4");

    auto biasdWeights0 = om.constantInt(biasd_WeightsData0,{16}, mv::DType("Int32"), mv::Order::getColMajorID(1), {{0},{1.2445522770576645e-05},{-inf},{inf}}, "dwconv0#0_bias#2");
    auto bias_cd0 = om.bias(depthConv0, biasdWeights0, mv::DType("UInt8"), {{128},{0.007843137718737125},{-1.003921627998352},{0.9960784316062927}});

    om.output(bias_cd0);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
