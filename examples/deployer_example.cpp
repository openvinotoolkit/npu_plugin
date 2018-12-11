#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"
#include "include/mcm/utils/hardware_tests.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/deployer/configuration.hpp"
#include "include/mcm/utils/deployer/executor.hpp"
#include "include/mcm/utils/deployer/deployer_utils.hpp"

#include <iostream>
#include <fstream>
// Create both RAM and file blobs
using namespace mv;
using namespace exe;
int main()
{
    mv::Logger::setVerboseLevel(VerboseLevel::Info);

    //Logger::logFilter({std::regex("OpModel")}, true);
 // Define the primary compilation unit
    CompilationUnit unit("gold11Model");
    OpModel& cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt6 = cm.input({64, 64, 3}, DTypeType::Float16, Order("WHC"));
    // define first convolution  3D conv
    std::vector<double> weightsData61 = utils::generateSequence(5u * 5u * 3u * 1u, 0.000, 0.010);
    auto weightsIt61 = cm.constant(weightsData61, {5, 5, 3, 1}, DTypeType::Float16, Order("NCHW"));   // kh, kw, ins, outs
    auto convIt61 = cm.conv(inIt6, weightsIt61, {2, 2}, {0, 0, 0, 0}, 1);

    std::vector<double> biasesData = { 64444.0 };
    auto biases = cm.constant(biasesData, {1}, DTypeType::Float16, Order("W"), "biases");
    auto bias1 = cm.bias(convIt61, biases);
    // define first maxpool
    auto maxpoolIt61 = cm.maxPool(bias1,{5,5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData62 = utils::generateSequence(3u * 3u * 1u * 1u, 65504.0, 0.000);
    auto weightsIt62 = cm.constant(weightsData62, {3, 3, 1, 1}, DTypeType::Float16, Order("NCHW"));   // kh, kw, ins, outs
    auto convIt62 = cm.conv(maxpoolIt61, weightsIt62, {1, 1}, {0, 0, 0, 0}, 1);

    // define scale
    std::vector<double> scalesData = { 6550.0 };
    auto scales = cm.constant(scalesData, {1}, DTypeType::Float16, Order("W"), "scales");
    auto scaleIt62 = cm.scale(convIt62, scales);
    // define output
    auto outIt6 = cm.output(scaleIt62);

    std::string blobName = "test_scale_11.blob";
    unit.compilationDescriptor()["GenerateBlob"]["fileName"] = blobName;
    unit.compilationDescriptor()["GenerateBlob"]["enableFileOutput"] = true;
    unit.compilationDescriptor()["GenerateBlob"]["enableRAMOutput"] = true;
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string("blob_eltwise_multiply.dot");
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpControlModel");
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    unit.compilationDescriptor()["GenerateCaffe"]["outputPrototxt"] = std::string("cppExampleprototxt.prototxt");
    unit.compilationDescriptor()["GenerateCaffe"]["outputCaffeModel"] = std::string("cppExampleweights.caffemodel");
    unit.compilationDescriptor()["MarkHardwareOperations"]["disableHardware"] = true;

    unit.loadTargetDescriptor(Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();
    try {
        //Create Configuration
        Configuration config(cm.getBinaryBuffer());
        Executor exec;
        Order order("NHWC");
        Shape shape({64, 64 ,3 ,1});

        Tensor inputTensor = mv::exe::dep_utils::getInputData(InputMode::ALL_ZERO, order, shape);
        Tensor res = exec.execute(config, inputTensor);

        std::cout << "res Order " << res.getOrder().toString() << std::endl;
        std::cout << "res Shape " << res.getShape().toString() << std::endl;
        std::cout << "ndims " << res.getShape().ndims() << std::endl;
        std::cout << "totalSize " << res.getShape().totalSize() << std::endl;
    }
    catch (...)
    {

    }

    //for (unsigned int i=0; i < res.getShape().totalSize(); i++)
    //    if (res(i) != 0)
    //        std::cout << "res[" << i << "] = " << res(i) << std::endl;
}