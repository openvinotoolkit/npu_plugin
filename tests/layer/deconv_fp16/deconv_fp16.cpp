#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main()
{

    mv::CompilationUnit unit("Deconv");
    mv::OpModel& om = unit.model();
    static const auto inf = std::numeric_limits<double>::infinity();

    auto data_0 = om.input({30,23,1024,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4) /*NHWC*/,  {{0},{0.00392156862745098},{0},{1},{0},{1}}, "input");

    mv::Shape kernel = mv::Shape({1,1,1024,1024});
    std::vector<int64_t> weightsData0(kernel.totalSize(), 1);

    auto weights0 = om.constantInt(weightsData0,kernel, mv::DType("UInt8"), mv::Order("NCHW"), {{0},{0.00392156862745098},{-1.000000000000000},{1.000000000000000}}, "weights_conv");
    auto conv0 = om.conv(data_0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1, mv::DType("Default"),{{0},{1},{-inf},{inf},{0},{1}} , "conv");

    //Create weights as all ones
    mv::Shape deconvWeightsShape({2, 2, 1024, 512});
    size_t deconvWeightsSize = deconvWeightsShape[0] * deconvWeightsShape[1] * deconvWeightsShape[2] * deconvWeightsShape[3];

    std::vector<double> doubleWeights(deconvWeightsSize, 1.0);
    auto weights1 = om.constant(doubleWeights, deconvWeightsShape, mv::DType("Float32"),
                                   mv::Order("NCHW"), {{0},{1.},{},{}}, "deconv_weights");

    auto deconv = om.deconv(conv0, weights1, {2, 2}, {0, 0, 0, 0}, 1, 1, false, mv::DType("Float16"), {{0},{1.0},{},{}}, "deconv_upscaling");

    const auto input_min = om.constant({0.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "input_min");
    const auto input_max = om.constant({4096.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "input_max");
    const auto output_min = om.constant({0.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "output_min");
    const auto output_max = om.constant({4096.000000}, {1}, mv::DType("Float32"), mv::Order("W"), {{0},{1.000000000000000},{-inf},{inf},{0},{1}}, "output_max");
    auto fakeQuant = om.fakeQuantize(deconv, input_min, input_max, output_min, output_max, 256, "fakeQuantize");
    
    om.output(fakeQuant, mv::DType("Float16"));

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}