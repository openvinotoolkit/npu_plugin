/**
 * @brief Example presenting composition and compilation of the ResNet50 CNN
 * 
 * In this example ResNet50 model is composed using MCMCompiler's Composition API. Then
 * the compilation for the target device MA2480 is initialized and compilation passes scheduled by 
 * the target descriptor are executed. Included GenerateDot pass will generate *.dot files
 * that visualize the computation model at the end of each accomplished compilation phase.
 * Included GenerateBlob pass will serialize the model to a binary deployable to the target device.
 * 
 * Notes:
 * - This implementation of ResNet50 uses fused batch norm representation - batch norm is expressed
 * as a sequence of scale and bias
 * - This implementation of ResNet50 is aligned with Caffe - batch norm is followed by scale and bias
 * - Weights and other model parameters are initialized as sequences of numbers starting with 0
 * 
 * @file cm_resnet50.cpp
 * @author Stanislaw Maciag
 * @date 2018-07-19
 */

#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"

/**
 * @brief Helper function creates a simple conv2D+In place relu block block
 * 
 * @param model Master compositional model
 * @param input Tensor that is an input data for the conv2D
 * @param kernelShape Shape of conv2D kernel
 * @param stride Stride of conv2D
 * @param padding Padding of conv2D
 * @return mv::Data::TensorIterator Iterator referencing the created conv2D output
 */
mv::Data::TensorIterator convReluBlock(mv::CompositionalModel& model, mv::Data::TensorIterator input, mv::Shape kernelShape, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding)
{
    std::vector<double> weightsData = mv::utils::generateSequence<double>(kernelShape.totalSize());
    auto weights = model.constant(weightsData, kernelShape, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    auto conv = model.conv2D(input, weights, stride, padding);
    return model.relu(conv);
}

/**
 * @brief Helper function creates a simple conv2D+In place relu block block
 *
 * @param model Master compositional model
 * @param input Tensor that is an input data for the conv2D
 * @param kernelShape Shape of conv2D kernel
 * @param stride Stride of conv2D
 * @param padding Padding of conv2D
 * @return mv::Data::TensorIterator Iterator referencing the created conv2D output
 */
mv::Data::TensorIterator convReluBlock(mv::CompositionalModel& model, mv::Data::TensorIterator input, mv::Shape kernelShape, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, unsigned output_channels)
{
    //Shape for weights is (kernel_height, kernel_width, number of input channel, number of output channels)
    mv::Shape actual_kernel_shape({kernelShape[0], kernelShape[1], input->getShape()[2], output_channels});
    return convReluBlock(model, input, actual_kernel_shape, stride, padding);
}

/**
 * @brief Helper function creates a simple conv2D+In place relu block block
 *
 * @param model Master compositional model
 * @param input Tensor that is an input data for the conv2D
 * @param kernelShape Shape of conv2D kernel
 * @param stride Stride of conv2D
 * @param padding Padding of conv2D
 * @return mv::Data::TensorIterator Iterator referencing the created conv2D output
 */
mv::Data::TensorIterator squeezeNetBlock(mv::CompositionalModel& model, mv::Data::TensorIterator input, unsigned output_channel_squeeze1x1, unsigned output_channel_expand1x1)
{
    mv::Shape squeeze1x1_kernel_shape({1, 1});
    std::array<unsigned short, 2> squeeze1x1_stride = {1, 1};
    std::array<unsigned short, 4> squeeze1x1_padding = {0, 0, 0, 0};

    mv::Shape expand1x1_kernel_shape({1, 1});
    std::array<unsigned short, 2> expand1x1_stride = {1, 1};
    std::array<unsigned short, 4> expand1x1_padding = {0, 0, 0, 0};

    mv::Shape expand3x3_kernel_shape({3, 3});
    std::array<unsigned short, 2> expand3x3_stride = {1, 1};
    std::array<unsigned short, 4> expand3x3_padding = {1, 1, 1, 1};

    auto squeeze1x1 = convReluBlock(model, input, squeeze1x1_kernel_shape, squeeze1x1_stride, squeeze1x1_padding, output_channel_squeeze1x1);
    auto expand1x1 = convReluBlock(model, squeeze1x1, expand1x1_kernel_shape, expand1x1_stride, expand1x1_padding, output_channel_expand1x1);
    auto expand3x3 = convReluBlock(model, squeeze1x1, expand3x3_kernel_shape, expand3x3_stride, expand3x3_padding, output_channel_expand1x1);

    return model.concat(expand1x1, expand3x3);
}


int main()
{

    // Define the primary compilation unit
    mv::CompilationUnit unit("Squeezenet");

    std::string descPath = mv::utils::projectRootPath() + "/config/compilation/squeezenet.json.json";
    std::ifstream compDescFile(descPath);
    if (compDescFile.good())
    {
        std::cout << "DECLARING COMPILATION UNIT with descriptor json filename: " << descPath << std::endl;
        unit.loadCompilationDescriptor(descPath);
    }

    // Obtain a compositional model from the compilation unit
    mv::CompositionalModel& cm = unit.model();

    // Compose the model for Squeezenet

    //From start until first squeezenet block
    auto input = cm.input({227, 227, 3}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    auto block1 = convReluBlock(cm, input, {3,3}, {2,2}, {0,0,0,0}, 64);
    auto pool1 = cm.maxpool2D(block1, {3, 3}, {2, 2}, {0,0,0,0});

    //Two squeezeblocks
    auto block2 = squeezeNetBlock(cm, pool1, 16, 64);
    auto block3 = squeezeNetBlock(cm, block2, 16, 64);

    //One pooling
    auto pool3 = cm.maxpool2D(block3, {3, 3}, {2, 2}, {0,0,0,0});

    //Two squeezeblocks
    auto block4 = squeezeNetBlock(cm, pool3, 32, 128);
    auto block5 = squeezeNetBlock(cm, block4, 32, 128);

    //One pooling
    auto pool5 = cm.maxpool2D(block5, {3, 3}, {2, 2}, {0,0,0,0});

    //Four squeezeblocks
    auto block6 = squeezeNetBlock(cm, pool5, 48, 192);
    auto block7 = squeezeNetBlock(cm, block6, 48, 192);
    auto block8 = squeezeNetBlock(cm, block7, 64, 256);
    auto block9 = squeezeNetBlock(cm, block8, 64, 256);

    //Final convolution
    auto conv10 = convReluBlock(cm, block9, {1,1}, {1,1}, {0,0,0,0}, 1000);
    //Final pooling
    auto pool10 = cm.avgpool2D(conv10, {conv10->getShape()[0], conv10->getShape()[1]}, {1, 1}, {0,0,0,0}); //Global pooling?
    //Softmax
    auto prob = cm.softmax(pool10);

    cm.output(prob);

    // Load target descriptor for the selected target to the compilation unit
    if (!unit.loadTargetDescriptor(mv::Target::ma2480))
        exit(1);

    // Define the manadatory arguments for passes using compilation descriptor obtained from the compilation unit
    // Output DOT - file name (base)
    unit.compilationDescriptor()["GenerateDot"]["output"] = std::string("squeezenet.dot");
    // Output DOT - scope of visualization - executable operations, data flow, control flow
    unit.compilationDescriptor()["GenerateDot"]["scope"] = std::string("ExecOpControlModel");
    // Output DOT - content included in the visualization - full content
    unit.compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    // Output DOT - HTML-like flag - enable HTML-like formatting
    unit.compilationDescriptor()["GenerateDot"]["html"] = true;
    // Output BLOB - file name of the output binary
    unit.compilationDescriptor()["GenerateBlob"]["output"] = std::string("squeezenet.blob");
    //unit.compilationDescriptor()["GenerateJSON"]["output"] = std::string("resnet50.json");

    // Initialize compilation 
    unit.initialize();
    //unit.passManager().disablePass(mv::PassGenre::Serialization);

    // Run all passes
    auto result = unit.run();

    // Obtain generated binary size from the compilation output
    //std::cout << "BLOB size: " << result["passes"].last()["blobSize"].get<long long>() << std::endl;
    std::cout << result.stringifyPretty() << std::endl;

    // Uncomment for an easy generation of SVG images for DOT output files (requires dot package)
    system("dot -Tsvg squeezenet.dot -o squeezenet.svg");
    system("dot -Tsvg squeezenet_adapt.dot -o squeezenet_adapt.svg");
    system("dot -Tsvg squeezenet_final.dot -o squeezenet_final.svg");

    return 0;

}
