#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"

TEST(populate_buffer_map, single_input)
{
    
    // Compilation unit must be first
    mv::CompilationUnit unit("BufferMap_test");
    mv::OpModel& om = unit.model();
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    
    // Test parameters definition
    mv::Shape inputShape({224,224,3,1});
    mv::Order inputOrder = mv::Order::getZMajorID(4);
    mv::DType inputDType("UInt8");
    std::string inputName = "input#170";
    double inf = std::numeric_limits<double>::infinity();

    // If anything fails during the composition/compilation, test itself is considered invalid
    try 
    {
        
        unit.loadCompilationDescriptor(compDescPath);
        unit.loadTargetDescriptor(mv::Target::ma2490);

        auto input0 = om.input(inputShape, inputDType, inputOrder, {{128},{0.007843137718737125},{-1.0},{1.0}}, inputName);

        std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3*3*3*48);
        auto weights0 = om.constantInt(weightsData0,{3,3,3,48}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{143},{0.04871978983283043},{-6.9027419090271},{5.472084999084473}}, "MobilenetV2/Conv/Relu6#0_weights#1");
        auto conv0 = om.conv(input0, weights0, {2, 2}, {0, 1, 0, 1}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/Conv/Relu6#171");

        std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (48);
        auto biasWeights0 = om.constantInt(biasWeightsData0,{48}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00038211600622162223},{-inf},{inf}}, "MobilenetV2/Conv/Relu6#0_bias#2");
        auto bias_c0 = om.bias(conv0, biasWeights0, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});

        om.output(bias_c0);
        unit.initialize();
        unit.run();
    } 
    catch(const std::exception& e) {
        FAIL() << "Test definition invalid - " << e.what();
    }
    
    ASSERT_EQ(unit.getBufferMap().getInputCount(), 1UL);
    mv::BufferEntry inBuffer("", mv::BufferType::Scratch, mv::Order("WH"), {1, 1}, mv::DType("Default"));
    ASSERT_NO_THROW(inBuffer = unit.getBufferMap().getInput()[0]);
    ASSERT_EQ(inBuffer.getName(), inputName);
    ASSERT_EQ(inBuffer.getBufferType(), mv::BufferType::Input);
    ASSERT_EQ(inBuffer.getSize(), inputShape.totalSize());
    ASSERT_EQ(inBuffer.getOrder(), inputOrder);
    ASSERT_EQ(inBuffer.getDType(), inputDType);
    ASSERT_EQ(inBuffer.getShape(), inputShape);

}

TEST(populate_buffer_map, single_output)
{
    
    // Compilation unit must be first
    mv::CompilationUnit unit("BufferMap_test");
    mv::OpModel& om = unit.model();
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    
    // Test parameters definition
    mv::Shape inputShape({224,224,3,1});
    mv::Order inputOrder = mv::Order::getZMajorID(4);
    mv::DType inputDType("UInt8");
    
    std::string outputName = "output_0";
    // input after conv2d - 3x3 kernel, no of filters 48, padding (0,1,0,1), stride (1,1)
    mv::Shape outputShape({112, 112, 48, 1});
    mv::Order outputOrder = inputOrder;
    mv::DType outputDType = inputDType;

    double inf = std::numeric_limits<double>::infinity();

    // If anything fails during the composition/compilation, test itself is considered invalid
    try 
    {
        
        unit.loadCompilationDescriptor(compDescPath);
        unit.loadTargetDescriptor(mv::Target::ma2490);

        auto input0 = om.input(inputShape, inputDType, inputOrder, {{128},{0.007843137718737125},{-1.0},{1.0}});

        std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3*3*3*48);
        auto weights0 = om.constantInt(weightsData0,{3,3,3,48}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{143},{0.04871978983283043},{-6.9027419090271},{5.472084999084473}}, "MobilenetV2/Conv/Relu6#0_weights#1");
        auto conv0 = om.conv(input0, weights0, {2, 2}, {0, 1, 0, 1}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/Conv/Relu6#171");

        std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (48);
        auto biasWeights0 = om.constantInt(biasWeightsData0,{48}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00038211600622162223},{-inf},{inf}}, "MobilenetV2/Conv/Relu6#0_bias#2");
        auto bias_c0 = om.bias(conv0, biasWeights0, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});
        om.output(bias_c0, mv::DType("Default"), {{}, {}, {}, {}}, true, outputName);
        unit.initialize();
        unit.run();
    } 
    catch(const std::exception& e) {
        FAIL() << "Test definition invalid - " << e.what();
    }
    
    ASSERT_EQ(unit.getBufferMap().getOutputCount(), 1UL);
    mv::BufferEntry outBuffer("", mv::BufferType::Scratch, mv::Order("WH"), {1, 1}, mv::DType("Default"));
    ASSERT_NO_THROW(outBuffer = unit.getBufferMap().getOutput()[0]);
    ASSERT_EQ(outBuffer.getName(), outputName);
    ASSERT_EQ(outBuffer.getBufferType(), mv::BufferType::Output);
    ASSERT_EQ(outBuffer.getSize(), outputShape.totalSize());
    ASSERT_EQ(outBuffer.getOrder(), outputOrder);
    ASSERT_EQ(outBuffer.getDType(), outputDType);
    ASSERT_EQ(outBuffer.getShape(), outputShape);

}

TEST(populate_buffer_map, multiple_outputs)
{
    
    // Compilation unit must be first
    mv::CompilationUnit unit("BufferMap_test");
    mv::OpModel& om = unit.model();
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    
    // Test parameters definition
    mv::Shape inputShape({64, 64, 3, 1});
    mv::Order inputOrder = mv::Order::getZMajorID(4);
    mv::DType inputDType("UInt8");
    
    std::string output1Name = "output_0";
    // input0 after:
    // 1) conv2d - 3x3 kernel, no of filters 32, padding (0,0,0,0), stride (2,2)
    // 2) conv2d - 3x3 kernel, no of filters 16, padding (0,0,0,0), stride (2,2)
    mv::Shape output1Shape({15, 15, 16, 1});
    mv::Order output1Order = inputOrder;
    mv::DType output1DType = inputDType;

    std::string output2Name = "output_1";
    // input0 after conv2d - 5x5 kernel, no of filters 16, padding (0,0,0,0), stride (2,2)
    // 1) conv2d - 3x3 kernel, no of filters 32, padding (0,0,0,0), stride (2,2)
    // 2) conv2d - 5x5 kernel, no of filters 16, padding (0,0,0,0), stride (2,2)
    mv::Shape output2Shape({14, 14, 16, 1});
    mv::Order output2Order = inputOrder;
    mv::DType output2DType = inputDType;

    double inf = std::numeric_limits<double>::infinity();

    // If anything fails during the composition/compilation, test itself is considered invalid
    try 
    {

        unit.loadCompilationDescriptor(compDescPath);
        unit.loadTargetDescriptor(mv::Target::ma2490);

        auto input_6_0 = om.input(inputShape, inputDType, inputOrder, {{128},{0.007843137718737},{-1.000000000000000},{1.000000000000000},{0},{1}}, "input");
        auto conv1_0_weights_1_0_data = mv::utils::generateSequence<int64_t>(3 * 3 * 3 * 32);
        auto conv1_0_weights_1_0 = om.constantInt(conv1_0_weights_1_0_data, {3, 3, 3, 32}, mv::DType("UInt8"), mv::Order("NCHW"), {{105},{0.002647720742971},{-0.279308497905731},{0.395860284566879},{0},{1}}, "conv1_weights");
        auto conv1_7_0 = om.conv(input_6_0, conv1_0_weights_1_0, {2, 2}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859369},{0.000000000000000},{1.000000000000000},{0},{1}}, "conv1");
        
        auto conv1_0_bias_2weights_0_data = mv::utils::generateSequence<int64_t>(32);
        auto conv1_0_bias_2weights_0 = om.constantInt(conv1_0_bias_2weights_0_data, {32}, mv::DType("UInt8"), mv::Order("W"), {{0},{0.000020766436137},{-inf},{inf},{0},{1}}, "conv1_bias_weights");
        auto conv1_0_bias_2_0 = om.bias(conv1_7_0, conv1_0_bias_2weights_0, mv::DType("UInt8"), {{0},{0.000020766436137},{-inf},{inf},{0},{1}}, "conv1_bias");

        // Parallel branch 1
        auto conv1_1_conv1_3_weights_4_0_data = mv::utils::generateSequence<int64_t>(3 * 3 * 32 * 16);
        auto conv1_1_conv1_3_weights_4_0 = om.constantInt(conv1_1_conv1_3_weights_4_0_data, {3, 3, 32, 16}, mv::DType("UInt8"), mv::Order("NCHW"), {{119},{0.002745436970145},{-0.326061517000198},{0.374024897813797},{0},{1}}, "conv2a_weights");
        auto conv1_1_conv1_8_0 = om.conv(conv1_0_bias_2_0, conv1_1_conv1_3_weights_4_0, {2, 2}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859369},{0.000000000000000},{1.000000000000000},{0},{1}}, "conv2a");
        auto conv1_1_conv1_3_bias_5weights_0_data = mv::utils::generateSequence<int64_t>(16);
        auto conv1_1_conv1_3_bias_5weights_0 = om.constantInt(conv1_1_conv1_3_bias_5weights_0_data, {16}, mv::DType("UInt8"), mv::Order("W"), {{0},{0.000010766419109},{-inf},{inf},{0},{1}}, "conv2a_bias_weights");
        auto conv1_1_conv1_3_bias_5_0 = om.bias(conv1_1_conv1_8_0, conv1_1_conv1_3_bias_5weights_0, mv::DType("UInt8"), {{0},{0.000010766419109},{-inf},{inf},{0},{1}}, "conv2a_bias");

        // Parallel branch 2 with different size
        auto conv1_1_conv1_3_weights_4_1_data = mv::utils::generateSequence<int64_t>(5 * 5 * 32 * 16);
        auto conv1_1_conv1_3_weights_4_1 = om.constantInt(conv1_1_conv1_3_weights_4_1_data, {5, 5, 32, 16}, mv::DType("UInt8"), mv::Order("NCHW"), {{121},{0.003013054607436},{-0.365644007921219},{0.402684897184372},{0},{1}}, "conv2b_weights");
        auto conv1_1_conv1_8_1 = om.conv(conv1_0_bias_2_0, conv1_1_conv1_3_weights_4_1, {2, 2}, {0, 0, 0, 0}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859369},{0.000000000000000},{1.000000000000000},{0},{1}}, "conv2b");
        auto conv1_1_conv1_3_bias_5weights_1_data = mv::utils::generateSequence<int64_t>(16);
        auto conv1_1_conv1_3_bias_5weights_1 = om.constantInt(conv1_1_conv1_3_bias_5weights_1_data, {16}, mv::DType("UInt8"), mv::Order("W"), {{0},{0.000011815900507},{-inf},{inf},{0},{1}}, "conv2b_bias_weights");
        auto conv1_1_conv1_3_bias_5_1 = om.bias(conv1_1_conv1_8_1, conv1_1_conv1_3_bias_5weights_1, mv::DType("UInt8"), {{0},{0.000011815900507},{-inf},{inf},{0},{1}}, "conv2b_bias");

        auto output1 = om.output(conv1_1_conv1_3_bias_5_0, mv::DType("Default"), {{},{},{},{}}, true, output1Name);
        auto output2 = om.output(conv1_1_conv1_3_bias_5_1, mv::DType("Default"), {{},{},{},{}}, true, output2Name);
        unit.initialize();
        unit.run();
    } 
    catch(const std::exception& e) {
        FAIL() << "Test definition invalid - " << e.what();
    }
    
    ASSERT_EQ(unit.getBufferMap().getOutputCount(), 2UL);
    mv::BufferEntry out1Buffer("", mv::BufferType::Scratch, mv::Order("WH"), {1, 1}, mv::DType("Default"));
    ASSERT_NO_THROW(out1Buffer = unit.getBufferMap().getOutput()[0]);
    mv::BufferEntry out2Buffer("", mv::BufferType::Scratch, mv::Order("WH"), {1, 1}, mv::DType("Default"));
    ASSERT_NO_THROW(out2Buffer = unit.getBufferMap().getOutput()[1]);
    ASSERT_EQ(out1Buffer.getName(), output1Name);
    ASSERT_EQ(out1Buffer.getBufferType(), mv::BufferType::Output);
    ASSERT_EQ(out1Buffer.getSize(), output1Shape.totalSize());
    ASSERT_EQ(out1Buffer.getOrder(), output1Order);
    ASSERT_EQ(out1Buffer.getDType(), output1DType);
    ASSERT_EQ(out1Buffer.getShape(), output1Shape);
    ASSERT_EQ(out2Buffer.getName(), output2Name);
    ASSERT_EQ(out2Buffer.getBufferType(), mv::BufferType::Output);
    ASSERT_EQ(out2Buffer.getSize(), output2Shape.totalSize());
    ASSERT_EQ(out2Buffer.getOrder(), output2Order);
    ASSERT_EQ(out2Buffer.getDType(), output2DType);
    ASSERT_EQ(out2Buffer.getShape(), output2Shape);

}


TEST(parse_buffer_map, single_input)
{
    
    // Compilation unit must be first
    std::string modelName = "BufferMap_test";
    mv::CompilationUnit unit(modelName);
    mv::OpModel& om = unit.model();
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    
    // Test parameters definition
    mv::Shape inputShape({224,224,3,1});
    mv::Order inputOrder = mv::Order::getZMajorID(4);
    mv::DType inputDType("UInt8");
    std::string inputName = "input#170";

    std::string outputName = "output_0";
    // input after conv2d - 3x3 kernel, no of filters 48, padding (0,1,0,1), stride (1,1)
    mv::Shape outputShape({112, 112, 48, 1});
    mv::Order outputOrder = inputOrder;
    mv::DType outputDType = inputDType;

    double inf = std::numeric_limits<double>::infinity();

    // If anything fails during the composition/compilation, test itself is considered invalid
    try 
    {
        
        unit.loadCompilationDescriptor(compDescPath);
        unit.loadTargetDescriptor(mv::Target::ma2490);

        auto input0 = om.input(inputShape, inputDType, inputOrder, {{128},{0.007843137718737125},{-1.0},{1.0}}, inputName);

        std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3*3*3*48);
        auto weights0 = om.constantInt(weightsData0,{3,3,3,48}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{143},{0.04871978983283043},{-6.9027419090271},{5.472084999084473}}, "MobilenetV2/Conv/Relu6#0_weights#1");
        auto conv0 = om.conv(input0, weights0, {2, 2}, {0, 1, 0, 1}, 1, 1, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}}, "MobilenetV2/Conv/Relu6#171");

        std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (48);
        auto biasWeights0 = om.constantInt(biasWeightsData0,{48}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00038211600622162223},{-inf},{inf}}, "MobilenetV2/Conv/Relu6#0_bias#2");
        auto bias_c0 = om.bias(conv0, biasWeights0, mv::DType("UInt8"), {{0},{0.023528477177023888},{0.0},{5.999761581420898}});
        om.output(bias_c0, mv::DType("Default"), {{}, {}, {}, {}}, true, outputName);
        unit.initialize();
        unit.run();
    } 
    catch(const std::exception& e) {
        FAIL() << "Test definition invalid - " << e.what();
    }
    
    auto inputBuffer = *unit.getBufferMap().getInput();
    auto outputBuffer = *unit.getBufferMap().getOutput();
    auto scratchBuffer = *unit.getBufferMap().getScratch();

    std::vector<char> blob = *unit.getBlob();
    unit.reset();

    ASSERT_EQ(unit.getBufferMap().getInputCount(), 0U);
    ASSERT_EQ(unit.getBufferMap().getOutputCount(), 0U);
    ASSERT_EQ(unit.getBufferMap().getScratchCount(), 0U);
    ASSERT_EQ(unit.getBufferMap().getProfilingCount(), 0U);
    ASSERT_EQ(unit.getBlob()->size(), 0U);

    mv::CompilationUnit unit2(blob.data(), blob.size(),
        mv::TargetDescriptor(mv::utils::projectRootPath() + "/config/target/release_kmb.json"));

    char name[256];
    unit2.getName(name, 256);
    ASSERT_EQ(std::string(name), modelName);

    ASSERT_EQ(unit2.getBufferMap().getInputCount(), 1U);
    ASSERT_EQ(unit2.getBufferMap().getOutputCount(), 1U);
    ASSERT_EQ(unit2.getBufferMap().getScratchCount(), 1U);
    
    mv::BufferEntry parsedOutputBuffer("", mv::BufferType::Scratch, mv::Order("WH"), {1, 1}, mv::DType("Default"));
    ASSERT_NO_THROW(parsedOutputBuffer = unit2.getBufferMap().getOutput()[0]);
    ASSERT_EQ(parsedOutputBuffer.getName(), outputName);
    ASSERT_EQ(parsedOutputBuffer.getBufferType(), mv::BufferType::Output);
    ASSERT_EQ(parsedOutputBuffer.getSize(), outputShape.totalSize());
    ASSERT_EQ(parsedOutputBuffer.getOrder(), outputOrder);
    ASSERT_EQ(parsedOutputBuffer.getDType(), outputDType);
    ASSERT_EQ(parsedOutputBuffer, outputBuffer);  

    mv::BufferEntry parsedInputBuffer("", mv::BufferType::Scratch, mv::Order("WH"), {1, 1}, mv::DType("Default"));
    ASSERT_NO_THROW(parsedInputBuffer = unit2.getBufferMap().getInput()[0]);
    ASSERT_EQ(parsedInputBuffer.getName(), inputName);
    ASSERT_EQ(parsedInputBuffer.getBufferType(), mv::BufferType::Input);
    ASSERT_EQ(parsedInputBuffer.getSize(), inputShape.totalSize());
    ASSERT_EQ(parsedInputBuffer.getOrder(), inputOrder);
    ASSERT_EQ(parsedInputBuffer.getDType(), inputDType);
    ASSERT_EQ(parsedInputBuffer, inputBuffer);  

    mv::BufferEntry parsedSratchBuffer("", mv::BufferType::Input, mv::Order("WH"), {1, 1}, mv::DType("Default"));
    ASSERT_NO_THROW(parsedSratchBuffer = unit2.getBufferMap().getScratch()[0]);
    ASSERT_EQ(parsedSratchBuffer.getName(), std::string("Scratch"));
    ASSERT_EQ(parsedSratchBuffer, scratchBuffer);  


}