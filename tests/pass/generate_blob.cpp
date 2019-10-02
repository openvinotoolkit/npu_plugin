#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/custom_math.hpp"

#include "include/mcm/utils/hardware_tests.hpp"
#include "include/mcm/op_model.hpp"
#include <iostream>
#include <fstream>

const std::string g_CompilationDescPath = "/config/compilation/release_ma2480.json";

mv::Data::TensorIterator convBatchNormBlock(mv::CompositionalModel& model, mv::Data::TensorIterator input,  mv::Shape kernelShape, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding)
{
    std::vector<double> weightsData = mv::utils::generateSequence<double>(kernelShape.totalSize());
    auto weights = model.constant(weightsData, kernelShape, mv::DType("Float16"), mv::Order("NHWC"));
    auto conv = model.conv(input, weights, stride, padding, 1);

    // For debugging purpose weights are initialized as sequences of numbers, to be replaced with actual weights
    std::vector<double> meanData = mv::utils::generateSequence<double>(conv->getShape()[mv::KERNEL_INPUT_CHANNELS]);
    std::vector<double> varianceData = mv::utils::generateSequence<double>(conv->getShape()[mv::KERNEL_INPUT_CHANNELS]);
    std::vector<double> offsetData = mv::utils::generateSequence<double>(conv->getShape()[mv::KERNEL_INPUT_CHANNELS]);
    std::vector<double> scaleData = mv::utils::generateSequence<double>(conv->getShape()[mv::KERNEL_INPUT_CHANNELS]);
    auto bnmean = model.constant(meanData, {conv->getShape()[mv::KERNEL_INPUT_CHANNELS]}, mv::DType("Float16"), mv::Order("W"));
    auto bnvariance = model.constant(varianceData, {conv->getShape()[mv::KERNEL_INPUT_CHANNELS]}, mv::DType("Float16"), mv::Order("W"));
    auto bnoffset = model.constant(offsetData, {conv->getShape()[mv::KERNEL_INPUT_CHANNELS]}, mv::DType("Float16"), mv::Order("W"));
    auto bnscale = model.constant(scaleData, {conv->getShape()[mv::KERNEL_INPUT_CHANNELS]}, mv::DType("Float16"), mv::Order("W"));
    return model.batchNormalization(conv, bnmean, bnvariance, bnoffset, bnscale, 1e-6);
}

TEST (mv_num_convert, fp32_to_fp16)
{
   EXPECT_EQ(mv::fp32_to_fp16(1.0f),0x3c00 );
   EXPECT_EQ(mv::fp32_to_fp16(2.0f),0x4000 );
   EXPECT_EQ(mv::fp32_to_fp16(9.0f),0x4880 );
   EXPECT_EQ(mv::fp32_to_fp16(15.0f),0x4b80 );
   EXPECT_EQ(mv::fp32_to_fp16(27.0f),0x4ec0 );
   EXPECT_EQ(mv::fp32_to_fp16(1.0009765625f),0x3c01 );
   EXPECT_EQ(mv::fp32_to_fp16(-2.0f),0xc000 );
   EXPECT_EQ(mv::fp32_to_fp16(65504.0f),0x7bff );
   EXPECT_EQ(mv::fp32_to_fp16(0.0000610352f),0x0400 );
   EXPECT_EQ(mv::fp32_to_fp16(0.0000609756f),0x03ff );
   EXPECT_EQ(mv::fp32_to_fp16(0.0000000596046f),0x0001 );
   EXPECT_EQ(mv::fp32_to_fp16(0.0f),0x0000 );
   EXPECT_EQ(mv::fp32_to_fp16(0.333251953125f),0x3555 );
}

TEST (mv_num_convert, fp64_to_fp16)
{
   EXPECT_EQ(mv::fp32_to_fp16(1.0),0x3c00 );
   EXPECT_EQ(mv::fp32_to_fp16(2.0),0x4000 );
   EXPECT_EQ(mv::fp32_to_fp16(9.0),0x4880 );
   EXPECT_EQ(mv::fp32_to_fp16(15.0),0x4b80 );
   EXPECT_EQ(mv::fp32_to_fp16(27.0),0x4ec0 );
   EXPECT_EQ(mv::fp32_to_fp16(1.0009765625),0x3c01 );
   EXPECT_EQ(mv::fp32_to_fp16(-2.0),0xc000 );
   EXPECT_EQ(mv::fp32_to_fp16(65504.0),0x7bff );
   EXPECT_EQ(mv::fp32_to_fp16(0.0000610352),0x0400 );
   EXPECT_EQ(mv::fp32_to_fp16(0.0000609756),0x03ff );
   EXPECT_EQ(mv::fp32_to_fp16(0.0000000596046),0x0001 );
   EXPECT_EQ(mv::fp32_to_fp16(0.0),0x0000 );
   EXPECT_EQ(mv::fp32_to_fp16(0.333251953125),0x3555 );
}

// test 01 : 1 2d convolution
TEST (generate_blob, DISABLED_blob_output_conv_01)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto input1 = test_cm.input({32, 32, 1, 1}, mv::DType("Float16"), mv::Order("NWHC"));
    std::vector<double> weights1Data({ 0.1111, 0.1121, 0.1131, 0.1141, 0.1151, 0.1161, 0.1171, 0.1181, 0.1191});
    auto weights1 = test_cm.constant(weights1Data, {3, 3, 1, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    auto conv1 = test_cm.conv(input1, weights1, {4, 4}, {0, 0, 0, 0}, 1);
    auto output1 = test_cm.output(conv1);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    std::string blobName = "test_conv_01.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();
    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (444LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    std::string goldBlobPath = mv::utils::projectRootPath() + std::string("/tests/data/gold_01.blob");
    std::string command = "diff \"" + blobName + "\" \"" + goldBlobPath + "\"";
    EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";

}

// test 02 : 1 2d convolution, add input z dimension (c=3)
TEST (generate_blob, DISABLED_blob_output_conv_02)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto input2 = test_cm.input({32, 32, 3, 1}, mv::DType("Float16"), mv::Order("NWHC"));   //N WH C
    std::vector<double> weightsData2 = mv::utils::generateSequence<double>(3u * 3u * 3u * 3u, 0.101, 0.001);

    auto weights2 = test_cm.constant(weightsData2, {3, 3, 3, 3}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, kN, C
    auto conv2 = test_cm.conv(input2, weights2, {4, 4}, {0, 0, 0, 0}, 1);   // input tensor, wieghts tensor, stridex, stridey, padx, pady

    auto output2 = test_cm.output(conv2);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "test_conv_02.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (572LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    std::string goldBlobPath = mv::utils::projectRootPath() + std::string("/tests/data/gold_02.blob");
    std::string command = "diff \"" + blobName + "\" \"" + goldBlobPath + "\"";
    EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";

}

// test 03 : 1 2d convolution, change input=256x256  stride=2
TEST (generate_blob, DISABLED_blob_output_conv_03)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto input3 = test_cm.input({256, 256, 3, 1}, mv::DType("Float16"), mv::Order("NWHC"));   //N WH C

    std::vector<double> weightsData3 = mv::utils::generateSequence(3u * 3u * 3u * 3u, 0.101, 0.001);

    auto weights3 = test_cm.constant(weightsData3, {3, 3, 3, 3}, mv::DType("Float16"), mv::Order("NCHW"));
    auto conv3 = test_cm.conv(input3, weights3, {2, 2}, {0, 0, 0, 0}, 1);   // input tensor, wieghts tensor, stridex, stridey, padx, pady

    auto output3 = test_cm.output(conv3);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "test_conv_03.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (572LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    std::string goldBlobPath = mv::utils::projectRootPath() + std::string("/tests/data/gold_03.blob");
    std::string command = "diff \"" + blobName + "\" \"" + goldBlobPath + "\"";
    EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";

}

// test 04 : 1 2d convolution, change kernel to 5x5
TEST (generate_blob, DISABLED_blob_output_conv_04)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto input4 = test_cm.input({256, 256, 3, 1}, mv::DType("Float16"), mv::Order("NWHC"));   //N WH C
    std::vector<double> weightsData4 = mv::utils::generateSequence(5u * 5u * 3u * 3u, 0.101, 0.001);

    auto weights4 = test_cm.constant(weightsData4, {5, 5, 3, 3}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, kN, C
    auto conv4 = test_cm.conv(input4, weights4, {2, 2}, {0, 0, 0, 0}, 1);   // input tensor, wieghts tensor, stridex, stridey, padx, pady

    auto output4 = test_cm.output(conv4);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "test_conv_04.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (892LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    std::string goldBlobPath = mv::utils::projectRootPath() + std::string("/tests/data/gold_04.blob");
    std::string command = "diff \"" + blobName + "\" \"" + goldBlobPath + "\"";
    EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";

}

// test 05 : 2 successive 3x3 convolutions (blur->edge filters)
TEST (generate_blob, DISABLED_blob_blur_edge_05)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 greyscale 256x256 image
    auto input5 = test_cm.input({256, 256, 1, 1}, mv::DType("Float16"), mv::Order("NWHC"));

    std::vector<double> blurKData({ 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 });
    std::vector<double> edgeKData({ 65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0 });
    auto bweights = test_cm.constant(blurKData, {3, 3, 1, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    auto eweights = test_cm.constant(edgeKData, {3, 3, 1, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    auto conv1 = test_cm.conv(input5, bweights, {1, 1}, {0, 0, 0, 0}, 1);
    auto conv2 = test_cm.conv(conv1, eweights, {1, 1}, {0, 0, 0, 0}, 1);

    auto output = test_cm.output(conv2);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "test_conv_05.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (740LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    std::string goldBlobPath = mv::utils::projectRootPath() + std::string("/tests/data/gold_05.blob");
    std::string command = "diff \"" + blobName + "\" \"" + goldBlobPath + "\"";
    EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";

}

// test 06 : conv1->maxpool1->conv2->maxpool2
TEST (generate_blob, DISABLED_blob_4_ops)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt6 = test_cm.input({64, 64, 3, 1}, mv::DType("Float16"), mv::Order("NWHC"));
    // define first convolution  3D conv
    std::vector<double> weightsData61 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000, 0.010);
    auto weightsIt61 = test_cm.constant(weightsData61, {5, 5, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt61 = test_cm.conv(inIt6, weightsIt61, {2, 2}, {0, 0, 0, 0}, 1);

    // define first maxpool
    auto maxpoolIt61 = test_cm.maxPool(convIt61,{5,5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData62 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 1.000, 0.010);
    auto weightsIt62 = test_cm.constant(weightsData62, {3, 3, 1, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt62 = test_cm.conv(maxpoolIt61, weightsIt62, {1, 1}, {0, 0, 0, 0}, 1);

    // define second maxpool
    auto maxpoolIt62 = test_cm.maxPool(convIt62,{3,3}, {2, 2}, {1, 1, 1, 1});
    // define output
    auto outIt6 = test_cm.output(maxpoolIt62);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "test_conv_06.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (1156LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    std::string goldBlobPath = mv::utils::projectRootPath() + std::string("/tests/data/gold_06.blob");
    std::string command = "diff \"" + blobName + "\" \"" + goldBlobPath + "\"";
    EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";

}

/*
 test 07 :               /-conv1->maxpool1->conv2->maxpool2-\
                  input-<                                    >-elementwise_add->output
                         \-conva->avgpoola->convb->avgpoolb-/
*/
TEST (generate_blob, DISABLED_blob_eltwise_add)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt7 = test_cm.input({64, 64, 3, 1}, mv::DType("Float16"), mv::Order("NWHC"));
    auto maxpoolIt11= test_cm.maxPool(inIt7,{1,1}, {1, 1}, {0,0,0,0});
    // define first convolution
    std::vector<double> weightsData71 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.100, 0.010);
    auto weightsIt71 = test_cm.constant(weightsData71, {5, 5, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt71 = test_cm.conv(maxpoolIt11, weightsIt71, {2, 2}, {0, 0, 0, 0}, 1);

    // define first avgpool
    auto avgpoolIt71 = test_cm.averagePool(convIt71,{5,5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData72 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 10.000, 0.010);
    auto weightsIt72 = test_cm.constant(weightsData72, {3, 3, 1, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt72 = test_cm.conv(avgpoolIt71, weightsIt72, {1, 1}, {0, 0, 0, 0}, 1);

    // define second avgpool
    auto avgpoolIt72 = test_cm.averagePool(convIt72,{3,3}, {2, 2}, {1, 1, 1, 1});
    // define first convolution branch a
    std::vector<double> weightsData7a = mv::utils::generateSequence(5u * 5u * 3u * 1u, 1.000, 0.010);
    auto weightsIt7a = test_cm.constant(weightsData7a, {5, 5, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt7a = test_cm.conv(inIt7, weightsIt7a, {2, 2}, {0, 0, 0, 0}, 1);

    // define first maxpool branch a
    auto maxpoolIt7a = test_cm.maxPool(convIt7a,{5,5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData7b = mv::utils::generateSequence(3u * 3u * 1u * 1u, 20.000, 0.010);
    auto weightsIt7b = test_cm.constant(weightsData7b, {3, 3, 1, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt7b = test_cm.conv(maxpoolIt7a, weightsIt7b, {1, 1}, {0, 0, 0, 0}, 1);

    // define second maxpool
    auto maxpoolIt7b = test_cm.maxPool(convIt7b,{3,3}, {2, 2}, {1, 1, 1, 1});
    // define elementwise sum
    auto eltwiseIt7 = test_cm.add({maxpoolIt7b,avgpoolIt72});
    // define output
    auto outIt7 = test_cm.output(eltwiseIt7);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "test_add_07.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (2468LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    std::string goldBlobPath = mv::utils::projectRootPath() + std::string("/tests/data/gold_07.blob");
    std::string command = "diff \"" + blobName + "\" \"" + goldBlobPath + "\"";
    EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";

}


/*
 test 08 :              /-conv1->maxpool1->conv2->maxpool2-\
                  input-<                                    >-elementwise_multiply->output
                         \-conva->avgpoola->convb->avgpoolb-/
*/
TEST (generate_blob, DISABLED_blob_eltwise_multiply)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt7 = test_cm.input({64, 64, 3, 1}, mv::DType("Float16"), mv::Order("NWHC"));
    auto maxpoolIt11= test_cm.maxPool(inIt7,{1,1}, {1, 1}, {0,0,0,0});
    // define first convolution
    std::vector<double> weightsData71 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.100, 0.010);
    auto weightsIt71 = test_cm.constant(weightsData71, {5, 5, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt71 = test_cm.conv(maxpoolIt11, weightsIt71, {2, 2}, {0, 0, 0, 0}, 1);

    // define first avgpool
    auto avgpoolIt71 = test_cm.averagePool(convIt71,{5,5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData72 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 6550.0, 0.000);
    auto weightsIt72 = test_cm.constant(weightsData72, {3, 3, 1, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt72 = test_cm.conv(avgpoolIt71, weightsIt72, {1, 1}, {0, 0, 0, 0}, 1);

    // define second avgpool
    auto avgpoolIt72 = test_cm.averagePool(convIt72,{3,3}, {2, 2}, {1, 1, 1, 1});
    // define first convolution branch a
    std::vector<double> weightsData7a = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000, 0.010);
    auto weightsIt7a = test_cm.constant(weightsData7a, {5, 5, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt7a = test_cm.conv(inIt7, weightsIt7a, {2, 2}, {0, 0, 0, 0}, 1);

    // define first maxpool branch a
    auto maxpoolIt7a = test_cm.maxPool(convIt7a,{5,5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData7b = mv::utils::generateSequence(3u * 3u * 1u * 1u, 65504.0, 0.000);
    auto weightsIt7b = test_cm.constant(weightsData7b, {3, 3, 1, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt7b = test_cm.conv(maxpoolIt7a, weightsIt7b, {1, 1}, {0, 0, 0, 0}, 1);

    // define second maxpool
    auto maxpoolIt7b = test_cm.maxPool(convIt7b,{3,3}, {2, 2}, {1, 1, 1, 1});
    // define elementwise sum
    auto eltwiseIt7 = test_cm.multiply({maxpoolIt7b,avgpoolIt72});
    // define output
    auto outIt7 = test_cm.output(eltwiseIt7);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "test_multiply_08.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (2468LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    std::string goldBlobPath = mv::utils::projectRootPath() + std::string("/tests/data/gold_08.blob");
    std::string command = "diff \"" + blobName + "\" \"" + goldBlobPath + "\"";
    EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";

}

/*
 test 09 :              /-conv1->maxpool1->conv2->maxpool2-\
                  input-<                                    >-elementwise_add->softmax->output
                         \-conva->avgpoola->convb->avgpoolb-/
*/
TEST (generate_blob, DISABLED_blob_softmax)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt7 = test_cm.input({64, 64, 3, 1}, mv::DType("Float16"), mv::Order("NWHC"));
    // define first convolution
    std::vector<double> weightsData71 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.100, 0.010);
    auto weightsIt71 = test_cm.constant(weightsData71, {5, 5, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt71 = test_cm.conv(inIt7, weightsIt71, {2, 2}, {0, 0, 0, 0}, 1);

    // define first avgpool
    auto avgpoolIt71 = test_cm.averagePool(convIt71, {5, 5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData72 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 6550.0, 0.000);
    auto weightsIt72 = test_cm.constant(weightsData72, {3, 3, 1, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt72 = test_cm.conv(avgpoolIt71, weightsIt72, {1, 1}, {0, 0, 0, 0}, 1);

    // define second avgpool
    auto avgpoolIt72 = test_cm.averagePool(convIt72,{3,3}, {2, 2}, {1, 1, 1, 1});
    // define first convolution branch a
    std::vector<double> weightsData7a = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000, 0.010);
    auto weightsIt7a = test_cm.constant(weightsData7a, {5, 5, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt7a = test_cm.conv(inIt7, weightsIt7a, {2, 2}, {0, 0, 0, 0}, 1);

    // define first maxpool branch a
    auto maxpoolIt7a = test_cm.maxPool(convIt7a,{5,5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData7b = mv::utils::generateSequence(3u * 3u * 1u * 1u, 65504.0, 0.000);
    auto weightsIt7b = test_cm.constant(weightsData7b, {3, 3, 1, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt7b = test_cm.conv(maxpoolIt7a, weightsIt7b, {1, 1}, {0, 0, 0, 0}, 1);

    // define second maxpool
    auto maxpoolIt7b = test_cm.maxPool(convIt7b, {3,3}, {2, 2}, {1, 1, 1, 1});
    // define elementwise sum
    auto eltwiseIt7 = test_cm.add({maxpoolIt7b,avgpoolIt72});
    auto softIt7 = test_cm.softmax(eltwiseIt7);
    // define output
    auto outIt7 = test_cm.output(softIt7);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "test_softmax_09.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (2452LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    std::string goldBlobPath = mv::utils::projectRootPath() + std::string("/tests/data/gold_09.blob");
    std::string command = "diff \"" + blobName + "\" \"" + goldBlobPath + "\"";
    EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";

}

// test 10 : conv1(+bias)->maxpool1->conv2(+relu)->maxpool2
TEST (generate_blob, DISABLED_blob_convbias_convrelu)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt6 = test_cm.input({64, 64, 3, 1}, mv::DType("Float16"), mv::Order("NWHC"));
    // define first convolution  3D conv
    std::vector<double> weightsData61 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000, 0.010);
    auto weightsIt61 = test_cm.constant(weightsData61, {5, 5, 3, 1}, mv::DType("Float16"), mv::Order("NWHC"));   // kh, kw, ins, outs
    auto convIt61 = test_cm.conv(inIt6, weightsIt61, {2, 2}, {0, 0, 0, 0}, 1);

    std::vector<double> biasesData = { 64444.0 };
    auto biases = test_cm.constant(biasesData, {1}, mv::DType("Float16"), mv::Order("W"),{{},{},{},{}}, "biases");
    auto bias1 = test_cm.bias(convIt61, biases);
    // define first maxpool
    auto maxpoolIt61 = test_cm.maxPool(bias1,{5,5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData62 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 65504.0, 0.000);
    auto weightsIt62 = test_cm.constant(weightsData62, {3, 3, 1, 1}, mv::DType("Float16"), mv::Order("NWHC"));   // kh, kw, ins, outs
    auto convIt62 = test_cm.conv(maxpoolIt61, weightsIt62, {1, 1}, {0, 0, 0, 0}, 1);

    std::vector<double> meanData = mv::utils::generateSequence<double>(convIt62->getShape().totalSize());
    std::vector<double> varianceData = mv::utils::generateSequence<double>(convIt62->getShape().totalSize());
    std::vector<double> offsetData = mv::utils::generateSequence<double>(convIt62->getShape().totalSize());
    std::vector<double> scaleData = mv::utils::generateSequence<double>(convIt62->getShape().totalSize());
    auto bnmean = test_cm.constant(meanData, convIt62->getShape(), mv::DType("Float16"), mv::Order("NCHW"),{{},{},{},{}}, "mean");
    auto bnvariance = test_cm.constant(varianceData, convIt62->getShape(), mv::DType("Float16"), mv::Order("NCHW"),{{},{},{},{}}, "variance");
    auto bnoffset = test_cm.constant(offsetData, convIt62->getShape(), mv::DType("Float16"), mv::Order("NCHW"),{{},{},{},{}}, "offset");
    auto bnscale = test_cm.constant(scaleData, convIt62->getShape(), mv::DType("Float16"), mv::Order("NCHW"), {{},{},{},{}},"scale");
    auto batchnorm = test_cm.batchNormalization(convIt62, bnmean, bnvariance, bnoffset, bnscale, 1e-6);
    auto reluIt62 = test_cm.relu(batchnorm);
    // define second maxpool
    auto maxpoolIt62 = test_cm.maxPool(reluIt62,{3,3}, {2, 2}, {1, 1, 1, 1});
    // define output
    auto outIt6 = test_cm.output(maxpoolIt62);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "test_relu_10.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (1940LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    std::string goldBlobPath = mv::utils::projectRootPath() + std::string("/tests/data/gold_10.blob");
    std::string command = "diff \"" + blobName + "\" \"" + goldBlobPath + "\"";
    //EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";

}

// test 09 : conv1(+bias)->maxpool1->conv2(+relu)->maxpool2->scale
TEST (generate_blob, DISABLED_blob_scale)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt6 = test_cm.input({64, 64, 3, 1}, mv::DType("Float16"), mv::Order("NWHC"));
    // define first convolution  3D conv
    std::vector<double> weightsData61 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000, 0.010);
    auto weightsIt61 = test_cm.constant(weightsData61, {5, 5, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt61 = test_cm.conv(inIt6, weightsIt61, {2, 2}, {0, 0, 0, 0}, 1);

    std::vector<double> biasesData = { 64444.0 };
    auto biases = test_cm.constant(biasesData, {1}, mv::DType("Float16"), mv::Order("W"),{{},{},{},{}}, "biases");
    auto bias1 = test_cm.bias(convIt61, biases);
    // define first maxpool
    auto maxpoolIt61 = test_cm.maxPool(bias1,{5,5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData62 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 65504.0, 0.000);
    auto weightsIt62 = test_cm.constant(weightsData62, {3, 3, 1, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt62 = test_cm.conv(maxpoolIt61, weightsIt62, {1, 1}, {0, 0, 0, 0}, 1);

    // define scale
    std::vector<double> scalesData = { 6550.0 };
    auto scales = test_cm.constant(scalesData, {1}, mv::DType("Float16"), mv::Order("W"),{{},{},{},{}}, "scales");
    auto scaleIt62 = test_cm.scale(convIt62, scales);
    // define output
    auto outIt6 = test_cm.output(scaleIt62);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "test_scale_11.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (1076LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    std::string goldBlobPath = mv::utils::projectRootPath() + std::string("/tests/data/gold_11.blob");
    std::string command = "diff \"" + blobName + "\" \"" + goldBlobPath + "\"";
    EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";

}

TEST (generate_blob_WDDM, blob_maxpool1)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input
    auto inIt = test_cm.input({64, 64, 3, 1}, mv::DType("Float16"), mv::Order("NWHC"));

    // define first maxpool
    auto maxpoolIt = test_cm.maxPool(inIt,{5,5}, {3, 3}, {2, 2, 2, 2});

    // define output
    auto output = test_cm.output(maxpoolIt);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "test_maxpool1";
    mv::Attribute blobNameAttr(blobName + "_sw.blob");
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);

    unit.initialize();
    auto compOutput = unit.run();
    EXPECT_EQ (292LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";

}

TEST (generate_blob_WDDM, blob_maxpool2)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input
    auto inIt = test_cm.input({65, 65, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));

    // define first maxpool
    auto maxpoolIt = test_cm.maxPool(inIt,{5,5}, {3, 3}, {0, 0, 0, 0});

    // define output
    auto output = test_cm.output(maxpoolIt);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "test_maxpool2.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);

    unit.initialize();
    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (292LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";
}

TEST (generate_blob_WDDM, blob_maxpool3)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input
    auto inIt = test_cm.input({65, 65, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));  // {w,h,c}

    // define first maxpool
    auto maxpoolIt = test_cm.maxPool(inIt, {7,5}, {2, 2}, {1, 1, 2, 2});   // w,w,h,h

    // define output
    auto output = test_cm.output(maxpoolIt);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "test_maxpool3.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (292LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";
}

TEST (generate_blob_WDDM, blob_maxpool4)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input
    auto inIt = test_cm.input({1920, 1080, 4, 1}, mv::DType("Float16"), mv::Order("NCHW"));  // {w,h,c}

    // define first maxpool
    auto maxpoolIt = test_cm.maxPool(inIt, {8,8}, {8, 8}, {0, 0, 0, 0});   // w,w,h,h

    // define output
    auto output = test_cm.output(maxpoolIt);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "test_maxpool4.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (292LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";
}

TEST (generate_blob_WDDM, blob_avgpool1)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input
    auto inIt = test_cm.input({256, 100, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));  // {w,h,c}

    // define first maxpool
    auto avgpoolIt = test_cm.averagePool(inIt, {4,4}, {4, 4}, {0,0,0,0});   // w,w,h,h

    // define output
    auto output = test_cm.output(avgpoolIt);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "avgpool1";
    mv::Attribute blobNameAttr("test_"+blobName+".blob");
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (292LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";
}

TEST (generate_blob_WDDM, blob_avgpool2)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input
    auto inIt = test_cm.input({255, 99, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));  // {w,h,c}

    // define first maxpool
    auto avgpoolIt = test_cm.averagePool(inIt, {4,2}, {3, 3}, {2, 2, 1, 1});   // w,w,h,h

    // define output
    auto output = test_cm.output(avgpoolIt);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "avgpool2";
    mv::Attribute blobNameAttr("test_"+blobName+".blob");
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (292LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";
}

TEST (generate_blob_WDDM, blob_conv1)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    auto input1 = test_cm.input({225, 225, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weights1Data = mv::utils::generateSequence<double>(3*3*3);
    auto weights1 = test_cm.constant(weights1Data, {3, 3, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
//    auto weights1 = test_cm.constant(weights1Data, {3, 3, 3, 1}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = test_cm.conv(input1, weights1, {2, 2}, {0, 0, 0, 0}, 1);
    auto output = test_cm.output(conv1);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "wddm_conv1";
    mv::Attribute blobNameAttr("test_"+blobName+".blob");
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (444LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";
}

// test 10 : conv->leakyRel
TEST (generate_blob_WDDM, DISABLED_blob_leakyRelu)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt6 = test_cm.input({64, 64, 3, 1}, mv::DType("Float16"), mv::Order("NWHC"));
    // define first convolution  3D conv
    std::vector<double> weightsData61 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000, 0.010);
    auto weightsIt61 = test_cm.constant(weightsData61, {5, 5, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt61 = test_cm.conv(inIt6, weightsIt61, {2, 2}, {0, 0, 0, 0}, 1);

    auto leakyRelu = test_cm.leakyRelu(convIt61,1.0);
    // define output
    auto output = test_cm.output(leakyRelu);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "test_leakyrelu.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (700LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";
}

// test 10 : conv->elu
TEST (generate_blob_WDDM, DISABLED_blob_elu)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt6 = test_cm.input({64, 64, 3, 1}, mv::DType("Float16"), mv::Order("NWHC"));
    // define first convolution  3D conv
    std::vector<double> weightsData61 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000, 0.010);
    auto weightsIt61 = test_cm.constant(weightsData61, {5, 5, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt61 = test_cm.conv(inIt6, weightsIt61, {2, 2}, {0, 0, 0, 0}, 1);

    auto elu = test_cm.elu(convIt61,1);
    // define output
    auto output = test_cm.output(elu);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "test_elu.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (700LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";
}

TEST (generate_blob_WDDM, DISABLED_blob_sigmoid)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt6 = test_cm.input({64, 64, 3, 1}, mv::DType("Float16"), mv::Order("NWHC"));
    // define first convolution  3D conv
    std::vector<double> weightsData61 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000, 0.010);
    auto weightsIt61 = test_cm.constant(weightsData61, {5, 5, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt61 = test_cm.conv(inIt6, weightsIt61, {2, 2}, {0, 0, 0, 0}, 1);

    auto sigmoid = test_cm.sigmoid(convIt61);
    // define output
    auto output = test_cm.output(sigmoid);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "test_sigmoid.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (684LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";
}

TEST (generate_blob_WDDM, DISABLED_blob_tanh)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt6 = test_cm.input({64, 64, 3, 1}, mv::DType("Float16"), mv::Order("NWHC"));
    // define first convolution  3D conv
    std::vector<double> weightsData61 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000, 0.010);
    auto weightsIt61 = test_cm.constant(weightsData61, {5, 5, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt61 = test_cm.conv(inIt6, weightsIt61, {2, 2}, {0, 0, 0, 0}, 1);

    auto tanh = test_cm.tanh(convIt61);
    // define output
    auto output = test_cm.output(tanh);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "test_tanh.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (684LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";
}

TEST (generate_blob_WDDM, DISABLED_blob_lrn)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt6 = test_cm.input({64, 64, 3, 1}, mv::DType("Float16"), mv::Order("NWHC"));
    // define first convolution  3D conv
    mv::Shape kernelShape = {5, 5, 3, 1};
    std::vector<double> weightsData = mv::utils::generateSequence<double>(kernelShape.totalSize());
    auto weightsIt = test_cm.constant(weightsData, {5, 5, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));   // kh, kw, ins, outs
    auto convIt = test_cm.conv(inIt6, weightsIt, {2, 2}, {0, 0, 0, 0}, 1);

    auto lrn = test_cm.localResponseNormalization(convIt,1,5);
    // define output
    auto output = test_cm.output(lrn);

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "test_lrn.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();
    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (700LL, compOutput.get<std::vector<mv::Element>>("passes").back().get<int64_t>("blobSize")) << "ERROR: wrong blob size";
}

// Create both RAM and file blobs
TEST (generate_blob, DISABLED_runtime_binary_RAM_FILE)
{

    // Define the primary compilation unit
    mv::CompilationUnit unit("RAMtest1");

    // Obtain compositional model from the compilation unit
    mv::OpModel& cm = unit.model();

    // Compose the model for ResNet18
    auto input = cm.input({224, 224, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    auto conv1 = convBatchNormBlock(cm, input, {7, 7, 3, 64}, {2, 2}, {3, 3, 3, 3});
    conv1 = cm.relu(conv1);
    auto pool1 = cm.maxPool(conv1, {3, 3}, {2, 2}, {1, 1, 1, 1});
    cm.output(pool1);

    // Load target descriptor for the selected target to the compilation unit
    EXPECT_EQ (true, unit.loadTargetDescriptor(mv::Target::ma2480)) << "ERROR: cannot load target descriptor";

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "RAMtest1.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", true);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    // Initialize compilation
    unit.initialize();

    // test get name and size methods
    cm.getBinaryBuffer()->getBuffer("testName1",4000) ;
    EXPECT_EQ ("testName1", cm.getBinaryBuffer()->getBinaryName()) << "ERROR: Unable to read name of RuntimeBinary";
    EXPECT_EQ (4000, cm.getBinaryBuffer()->getBufferSize() ) << "ERROR: incorrect size of RuntimeBinary";

    // test handling multiple calls to allocate data_ buffer
    cm.getBinaryBuffer()->getBuffer("testName2",2000) ;
    EXPECT_EQ ("testName2", cm.getBinaryBuffer()->getBinaryName()) << "ERROR: Unable to read name of RuntimeBinary";
    EXPECT_EQ (2000, cm.getBinaryBuffer()->getBufferSize() ) << "ERROR: incorrect size of RuntimeBinary";

    // Run all passes
    unit.initialize();
    unit.run();
    cm.getBinaryBuffer()->dumpBuffer("final_RAM1.blob") ;

    char* dataBuffer = cm.getBinaryBuffer()->getDataPointer() ;
    char magicNumber = dataBuffer[35];
    EXPECT_EQ (0x22, magicNumber) << "ERROR: wrong data read from runtimeBinary data buffer ";

    // compare blob file contents to RAM blob
    std::string RAMBlobPath = mv::utils::projectRootPath() + std::string("/build/tests/final_RAM1.blob");
    std::string BlobPath = mv::utils::projectRootPath() + std::string("/build/tests/RAMtest1.blob");
    std::string command = "diff \"" + BlobPath + "\" \"" + RAMBlobPath + "\"";
    EXPECT_EQ (0, system(command.c_str())) << "ERROR: RAM and file blobs do not match ";

    // check blob sizes
    std::ifstream p_file(RAMBlobPath, std::ios::in | std::ios::binary);
    ASSERT_EQ(p_file.is_open(),true) << "ERROR: RAM blob dump file does not exist ";
    p_file.seekg(0, std::ios::end);
    auto RAMBlobSize = p_file.tellg();
    p_file.close();

    EXPECT_GT (RAMBlobSize, 19000) << "ERROR: wrong RAM blob size ";

}

// Create RAM blob but not file blob
TEST (generate_blob, DISABLED_runtime_binary_RAM)
{

    // Define the primary compilation unit
    mv::CompilationUnit unit("RAMtest2");

    // Obtain compositional model from the compilation unit
    mv::OpModel& cm = unit.model();

    // Compose the model for ResNet18
    auto input = cm.input({224, 224, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    auto conv1 = convBatchNormBlock(cm, input, {7, 7, 3, 64}, {2, 2}, {3, 3, 3, 3});
    conv1 = cm.relu(conv1);
    auto pool1 = cm.maxPool(conv1, {3, 3}, {2, 2}, {1, 1, 1, 1});
    cm.output(pool1);

    // Load target descriptor for the selected target to the compilation unit
    EXPECT_EQ (true, unit.loadTargetDescriptor(mv::Target::ma2480)) << "ERROR: cannot load target descriptor";

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "RAMtest2.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", false);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", true);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    // Initialize compilation
    unit.initialize();
    unit.run();
    cm.getBinaryBuffer()->dumpBuffer("final_RAM2.blob") ;

    std::cout << "in RTB test: after dump" << std::endl;

    std::string RAMBlobPath = mv::utils::projectRootPath() + std::string("/build/tests/final_RAM2.blob");
    std::string BlobPath = mv::utils::projectRootPath() + std::string("/build/tests/RAMtest2.blob");

    // check blob sizes
    std::ifstream p_file(RAMBlobPath, std::ios::in | std::ios::binary);
    ASSERT_EQ(p_file.is_open(),true) << "ERROR: RAM blob dump file does not exist ";
    p_file.seekg(0, std::ios::end);
    auto RAMBlobSize = p_file.tellg();
    p_file.close();

    EXPECT_GT (RAMBlobSize, 19000) << "ERROR: wrong RAM blob size ";

    // file blob should not exist
    std::ifstream b_file(BlobPath, std::ios::in | std::ios::binary);
    if (b_file.is_open())
    {
        EXPECT_EQ(0,1) << "ERROR: blob file RAMtest2.blob exists.";
        b_file.close();
    }

}

// Create file blob but not RAM blob
TEST (generate_blob, DISABLED_runtime_binary_FILE)
{

    // Define the primary compilation unit
    mv::CompilationUnit unit("RAMtest3");

    // Obtain compositional model from the compilation unit
    mv::OpModel& cm = unit.model();

    // Compose the model for ResNet18
    auto input = cm.input({224, 224, 3, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    auto conv1 = convBatchNormBlock(cm, input, {7, 7, 3, 64}, {2, 2}, {3, 3, 3, 3});
    conv1 = cm.relu(conv1);
    auto pool1 = cm.maxPool(conv1, {3, 3}, {2, 2}, {1, 1, 1, 1});
    cm.output(pool1);

    // Load target descriptor for the selected target to the compilation unit
    EXPECT_EQ (true, unit.loadTargetDescriptor(mv::Target::ma2480)) << "ERROR: cannot load target descriptor";

    std::string path = mv::utils::projectRootPath() + g_CompilationDescPath;
    unit.loadCompilationDescriptor(path);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();

    std::string blobName = "RAMtest3.blob";
    mv::Attribute blobNameAttr(blobName);
    compDesc.setPassArg("GenerateBlob", "fileName", blobName);
    compDesc.setPassArg("GenerateBlob", "enableFileOutput", true);
    compDesc.setPassArg("GenerateBlob", "enableRAMOutput", false);
    compDesc.setPassArg("MarkHardwareOperations", "disableHardware", true);

    // Initialize compilation
    unit.initialize();
    unit.run();
    cm.getBinaryBuffer()->dumpBuffer("final_RAM3.blob") ;

    std::string RAMBlobPath = mv::utils::projectRootPath() + std::string("/build/tests/final_RAM3.blob");
    std::string BlobPath = mv::utils::projectRootPath() + std::string("/build/tests/RAMtest3.blob");

    // check blob sizes
    std::ifstream p_file(RAMBlobPath, std::ios::in | std::ios::binary);
    ASSERT_EQ(p_file.is_open(),true) << "ERROR: RAM blob dump file does not exist ";
    p_file.seekg(0, std::ios::end);
    auto RAMBlobSize = p_file.tellg();
    p_file.close();

    std::ifstream q_file(BlobPath, std::ios::in | std::ios::binary);
    ASSERT_EQ(q_file.is_open(),true) << "ERROR: blob file does not exist ";
    q_file.seekg(0, std::ios::end);
    auto BlobSize = q_file.tellg();
    q_file.close();

    EXPECT_EQ (RAMBlobSize, 0) << "ERROR: wrong RAM blob size ";
    EXPECT_GT (BlobSize, 19000) << "ERROR: wrong blob file size ";

}
