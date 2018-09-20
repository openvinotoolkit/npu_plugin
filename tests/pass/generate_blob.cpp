#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"

TEST (mv_num_convert, fp32_to_fp16)
{
   mv_num_convert cvtr ;
   EXPECT_EQ(cvtr.fp32_to_fp16(1.0),0x3c00 );
   EXPECT_EQ(cvtr.fp32_to_fp16(1.0009765625),0x3c01 );
   EXPECT_EQ(cvtr.fp32_to_fp16(-2.0),0xc000 );
   EXPECT_EQ(cvtr.fp32_to_fp16(65504.0),0x7bff );
   EXPECT_EQ(cvtr.fp32_to_fp16(0.0000610352),0x0400 );
   EXPECT_EQ(cvtr.fp32_to_fp16(0.0000609756),0x03ff );
   EXPECT_EQ(cvtr.fp32_to_fp16(0.0000000596046),0x0001 );
   EXPECT_EQ(cvtr.fp32_to_fp16(0.0),0x0000 );
   EXPECT_EQ(cvtr.fp32_to_fp16(0.333251953125),0x3555 );
}

// test 01 : 1 2d convolution
TEST (generate_blob, blob_output_conv_01)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto input1 = test_cm.input({32, 32, 1}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    std::vector<double> weights1Data({ 0.1111, 0.1121, 0.1131, 0.1141, 0.1151, 0.1161, 0.1171, 0.1181, 0.1191});
    auto weights1 = test_cm.constant(weights1Data, {3, 3, 1, 1}, mv::DTypeType::Float16, mv::OrderType::ColumnMajor);
    weights1->setOrder(mv::OrderType::ColumnMajor);
    auto conv1 = test_cm.conv2D(input1, weights1, {4, 4}, {0, 0, 0, 0});
    auto output1 = test_cm.output(conv1);

    std::string blobName = "test_conv_01.blob";
    unit.compilationDescriptor()["GenerateBlob"]["output"] = blobName;
    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();
    unit.passManager().disablePass(mv::PassGenre::Validation);
    unit.passManager().disablePass(mv::PassGenre::Serialization);
    unit.passManager().enablePass(mv::PassGenre::Serialization, "GenerateBlob");
    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (444LL, compOutput["passes"].last()["blobSize"].get<long long>()) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    std::string goldBlobPath = mv::utils::projectRootPath() + std::string("/tests/data/gold_01.blob");
    std::string command = "diff \"" + blobName + "\" \"" + goldBlobPath + "\"";
    EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";

}

// test 02 : 1 2d convolution, add input z dimension (c=3)
TEST (generate_blob, blob_output_conv_02)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto input2 = test_cm.input({32, 32, 3}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   //N WH C
    std::vector<double> weightsData2 = mv::utils::generateSequence<double>(3u * 3u * 3u * 3u, 0.101, 0.001);

    auto weights2 = test_cm.constant(weightsData2, {3, 3, 3, 3}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, kN, C
    auto conv2 = test_cm.conv2D(input2, weights2, {4, 4}, {0, 0, 0, 0});   // input tensor, wieghts tensor, stridex, stridey, padx, pady
    auto output2 = test_cm.output(conv2);

    std::string blobName = "test_conv_02.blob";
    unit.compilationDescriptor()["GenerateBlob"]["output"] = blobName;
    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();
    unit.passManager().disablePass(mv::PassGenre::Validation);
    unit.passManager().disablePass(mv::PassGenre::Serialization);
    unit.passManager().enablePass(mv::PassGenre::Serialization, "GenerateBlob");

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (572LL, compOutput["passes"].last()["blobSize"].get<long long>()) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    std::string goldBlobPath = mv::utils::projectRootPath() + std::string("/tests/data/gold_02.blob");
    std::string command = "diff \"" + blobName + "\" \"" + goldBlobPath + "\"";
    EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";

}

// test 03 : 1 2d convolution, change input=256x256  stride=2
TEST (generate_blob, blob_output_conv_03)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto input3 = test_cm.input({256, 256, 3}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   //N WH C

    std::vector<double> weightsData3 = mv::utils::generateSequence(3u * 3u * 3u * 3u, 0.101, 0.001);

    auto weights3 = test_cm.constant(weightsData3, {3, 3, 3, 3}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    auto conv3 = test_cm.conv2D(input3, weights3, {2, 2}, {0, 0, 0, 0});   // input tensor, wieghts tensor, stridex, stridey, padx, pady
    auto output3 = test_cm.output(conv3);

    std::string blobName = "test_conv_03.blob";
    unit.compilationDescriptor()["GenerateBlob"]["output"] = blobName;
    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();
    unit.passManager().disablePass(mv::PassGenre::Validation);
    unit.passManager().disablePass(mv::PassGenre::Serialization);
    unit.passManager().enablePass(mv::PassGenre::Serialization, "GenerateBlob");

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (572LL, compOutput["passes"].last()["blobSize"].get<long long>()) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    std::string goldBlobPath = mv::utils::projectRootPath() + std::string("/tests/data/gold_03.blob");
    std::string command = "diff \"" + blobName + "\" \"" + goldBlobPath + "\"";
    EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";

}

// test 04 : 1 2d convolution, change kernel to 5x5
TEST (generate_blob, blob_output_conv_04)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto input4 = test_cm.input({256, 256, 3}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   //N WH C
    std::vector<double> weightsData4 = mv::utils::generateSequence(5u * 5u * 3u * 3u, 0.101, 0.001);

    auto weights4 = test_cm.constant(weightsData4, {5, 5, 3, 3}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, kN, C
    auto conv4 = test_cm.conv2D(input4, weights4, {2, 2}, {0, 0, 0, 0});   // input tensor, wieghts tensor, stridex, stridey, padx, pady
    auto output4 = test_cm.output(conv4);
    
    std::string blobName = "test_conv_04.blob";
    unit.compilationDescriptor()["GenerateBlob"]["output"] = blobName;
    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();
    unit.passManager().disablePass(mv::PassGenre::Validation);
    unit.passManager().disablePass(mv::PassGenre::Serialization);
    unit.passManager().enablePass(mv::PassGenre::Serialization, "GenerateBlob");

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (892LL, compOutput["passes"].last()["blobSize"].get<long long>()) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    std::string goldBlobPath = mv::utils::projectRootPath() + std::string("/tests/data/gold_04.blob");
    std::string command = "diff \"" + blobName + "\" \"" + goldBlobPath + "\"";
    EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";

}

// test 05 : 2 successive 3x3 convolutions (blur->edge filters)
TEST (generate_blob, blob_blur_edge_05)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 greyscale 256x256 image
    auto input5 = test_cm.input({256, 256, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);

    std::vector<double> blurKData({ 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 });
    std::vector<double> edgeKData({ 65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0 });
    auto bweights = test_cm.constant(blurKData, {3, 3, 1, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    auto eweights = test_cm.constant(edgeKData, {3, 3, 1, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    auto conv1 = test_cm.conv2D(input5, bweights, {1, 1}, {0, 0, 0, 0});
    auto conv2 = test_cm.conv2D(conv1, eweights, {1, 1}, {0, 0, 0, 0});
    auto output = test_cm.output(conv2);

    std::string blobName = "test_conv_05.blob";
    unit.compilationDescriptor()["GenerateBlob"]["output"] = blobName;
    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();
    unit.passManager().disablePass(mv::PassGenre::Validation);
    unit.passManager().disablePass(mv::PassGenre::Serialization);
    unit.passManager().enablePass(mv::PassGenre::Serialization, "GenerateBlob");

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (740LL, compOutput["passes"].last()["blobSize"].get<long long>()) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    std::string goldBlobPath = mv::utils::projectRootPath() + std::string("/tests/data/gold_05.blob");
    std::string command = "diff \"" + blobName + "\" \"" + goldBlobPath + "\"";
    EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";

}

// test 06 : conv1->maxpool1->conv2->maxpool2
TEST (generate_blob, blob_4_ops)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt6 = test_cm.input({64, 64, 3}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    // define first convolution  3D conv
    std::vector<double> weightsData61 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000, 0.010);
    auto weightsIt61 = test_cm.constant(weightsData61, {5, 5, 3, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt61 = test_cm.conv2D(inIt6, weightsIt61, {2, 2}, {0, 0, 0, 0});
    // define first maxpool
    auto maxpoolIt61 = test_cm.maxpool2D(convIt61,{5,5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData62 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 65504.0, 0.000);
    auto weightsIt62 = test_cm.constant(weightsData62, {3, 3, 1, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt62 = test_cm.conv2D(maxpoolIt61, weightsIt62, {1, 1}, {0, 0, 0, 0});
    // define second maxpool
    auto maxpoolIt62 = test_cm.maxpool2D(convIt62,{3,3}, {2, 2}, {1, 1, 1, 1});
    // define output
    auto outIt6 = test_cm.output(maxpoolIt62);


    std::string blobName = "test_conv_06.blob";
    unit.compilationDescriptor()["GenerateBlob"]["output"] = blobName;
    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();
    unit.passManager().disablePass(mv::PassGenre::Validation);
    unit.passManager().disablePass(mv::PassGenre::Serialization);
    unit.passManager().enablePass(mv::PassGenre::Serialization, "GenerateBlob");

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (1156LL, compOutput["passes"].last()["blobSize"].get<long long>()) << "ERROR: wrong blob size";

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
TEST (generate_blob, blob_eltwise_add)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt7 = test_cm.input({64, 64, 3}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    auto maxpoolIt11= test_cm.maxpool2D(inIt7,{1,1}, {1, 1}, {0,0,0,0});
    // define first convolution
    std::vector<double> weightsData71 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.100, 0.010);
    auto weightsIt71 = test_cm.constant(weightsData71, {5, 5, 3, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt71 = test_cm.conv2D(maxpoolIt11, weightsIt71, {2, 2}, {0, 0, 0, 0});
    // define first avgpool
    auto avgpoolIt71 = test_cm.avgpool2D(convIt71,{5,5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData72 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 6550.0, 0.000);
    auto weightsIt72 = test_cm.constant(weightsData72, {3, 3, 1, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt72 = test_cm.conv2D(avgpoolIt71, weightsIt72, {1, 1}, {0, 0, 0, 0});
    // define second avgpool
    auto avgpoolIt72 = test_cm.avgpool2D(convIt72,{3,3}, {2, 2}, {1, 1, 1, 1});
    // define first convolution branch a
    std::vector<double> weightsData7a = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000, 0.010);
    auto weightsIt7a = test_cm.constant(weightsData7a, {5, 5, 3, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt7a = test_cm.conv2D(inIt7, weightsIt7a, {2, 2}, {0, 0, 0, 0});
    // define first maxpool branch a
    auto maxpoolIt7a = test_cm.maxpool2D(convIt7a,{5,5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData7b = mv::utils::generateSequence(3u * 3u * 1u * 1u, 65504.0, 0.000);
    auto weightsIt7b = test_cm.constant(weightsData7b, {3, 3, 1, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt7b = test_cm.conv2D(maxpoolIt7a, weightsIt7b, {1, 1}, {0, 0, 0, 0});
    // define second maxpool
    auto maxpoolIt7b = test_cm.maxpool2D(convIt7b,{3,3}, {2, 2}, {1, 1, 1, 1});
    // define elementwise sum
    auto eltwiseIt7 = test_cm.add(maxpoolIt7b,avgpoolIt72);
    // define output
    auto outIt7 = test_cm.output(eltwiseIt7);

    std::string blobName = "test_add_07.blob";
    unit.compilationDescriptor()["GenerateBlob"]["output"] = blobName;
    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();
    unit.passManager().disablePass(mv::PassGenre::Validation);
    unit.passManager().disablePass(mv::PassGenre::Serialization);
    unit.passManager().enablePass(mv::PassGenre::Serialization, "GenerateBlob");

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (2468LL, compOutput["passes"].last()["blobSize"].get<long long>()) << "ERROR: wrong blob size";

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
TEST (generate_blob, blob_eltwise_multiply)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt7 = test_cm.input({64, 64, 3}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    auto maxpoolIt11= test_cm.maxpool2D(inIt7,{1,1}, {1, 1}, {0,0,0,0});
    // define first convolution
    std::vector<double> weightsData71 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.100, 0.010);
    auto weightsIt71 = test_cm.constant(weightsData71, {5, 5, 3, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt71 = test_cm.conv2D(maxpoolIt11, weightsIt71, {2, 2}, {0, 0, 0, 0});
    // define first avgpool
    auto avgpoolIt71 = test_cm.avgpool2D(convIt71,{5,5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData72 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 6550.0, 0.000);
    auto weightsIt72 = test_cm.constant(weightsData72, {3, 3, 1, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt72 = test_cm.conv2D(avgpoolIt71, weightsIt72, {1, 1}, {0, 0, 0, 0});
    // define second avgpool
    auto avgpoolIt72 = test_cm.avgpool2D(convIt72,{3,3}, {2, 2}, {1, 1, 1, 1});
    // define first convolution branch a
    std::vector<double> weightsData7a = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000, 0.010);
    auto weightsIt7a = test_cm.constant(weightsData7a, {5, 5, 3, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt7a = test_cm.conv2D(inIt7, weightsIt7a, {2, 2}, {0, 0, 0, 0});
    // define first maxpool branch a
    auto maxpoolIt7a = test_cm.maxpool2D(convIt7a,{5,5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData7b = mv::utils::generateSequence(3u * 3u * 1u * 1u, 65504.0, 0.000);
    auto weightsIt7b = test_cm.constant(weightsData7b, {3, 3, 1, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt7b = test_cm.conv2D(maxpoolIt7a, weightsIt7b, {1, 1}, {0, 0, 0, 0});
    // define second maxpool
    auto maxpoolIt7b = test_cm.maxpool2D(convIt7b,{3,3}, {2, 2}, {1, 1, 1, 1});
    // define elementwise sum
    auto eltwiseIt7 = test_cm.multiply(maxpoolIt7b,avgpoolIt72);
    // define output
    auto outIt7 = test_cm.output(eltwiseIt7);

    std::string blobName = "test_multiply_08.blob";
    unit.compilationDescriptor()["GenerateBlob"]["output"] = blobName;
    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();
    unit.passManager().disablePass(mv::PassGenre::Validation);
    unit.passManager().disablePass(mv::PassGenre::Serialization);
    unit.passManager().enablePass(mv::PassGenre::Serialization, "GenerateBlob");

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (2468LL, compOutput["passes"].last()["blobSize"].get<long long>()) << "ERROR: wrong blob size";

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
TEST (generate_blob, blob_softmax)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt7 = test_cm.input({64, 64, 3}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    // define first convolution
    std::vector<double> weightsData71 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.100, 0.010);
    auto weightsIt71 = test_cm.constant(weightsData71, {5, 5, 3, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt71 = test_cm.conv2D(inIt7, weightsIt71, {2, 2}, {0, 0, 0, 0});
    // define first avgpool
    auto avgpoolIt71 = test_cm.avgpool2D(convIt71, {5, 5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData72 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 6550.0, 0.000);
    auto weightsIt72 = test_cm.constant(weightsData72, {3, 3, 1, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt72 = test_cm.conv2D(avgpoolIt71, weightsIt72, {1, 1}, {0, 0, 0, 0});
    // define second avgpool
    auto avgpoolIt72 = test_cm.avgpool2D(convIt72,{3,3}, {2, 2}, {1, 1, 1, 1});
    // define first convolution branch a
    std::vector<double> weightsData7a = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000, 0.010);
    auto weightsIt7a = test_cm.constant(weightsData7a, {5, 5, 3, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt7a = test_cm.conv2D(inIt7, weightsIt7a, {2, 2}, {0, 0, 0, 0});
    // define first maxpool branch a
    auto maxpoolIt7a = test_cm.maxpool2D(convIt7a,{5,5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData7b = mv::utils::generateSequence(3u * 3u * 1u * 1u, 65504.0, 0.000);
    auto weightsIt7b = test_cm.constant(weightsData7b, {3, 3, 1, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt7b = test_cm.conv2D(maxpoolIt7a, weightsIt7b, {1, 1}, {0, 0, 0, 0});
    // define second maxpool
    auto maxpoolIt7b = test_cm.maxpool2D(convIt7b, {3,3}, {2, 2}, {1, 1, 1, 1});
    // define elementwise sum
    auto eltwiseIt7 = test_cm.add(maxpoolIt7b,avgpoolIt72);
    auto softIt7 = test_cm.softmax(eltwiseIt7);
    // define output
    auto outIt7 = test_cm.output(softIt7);

    std::string blobName = "test_softmax_09.blob";
    unit.compilationDescriptor()["GenerateBlob"]["output"] = blobName;
    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();
    unit.passManager().disablePass(mv::PassGenre::Validation);
    unit.passManager().disablePass(mv::PassGenre::Serialization);
    unit.passManager().enablePass(mv::PassGenre::Serialization, "GenerateBlob");

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (2452LL, compOutput["passes"].last()["blobSize"].get<long long>()) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    std::string goldBlobPath = mv::utils::projectRootPath() + std::string("/tests/data/gold_09.blob");
    std::string command = "diff \"" + blobName + "\" \"" + goldBlobPath + "\"";
    EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";

}

// test 10 : conv1(+bias)->maxpool1->conv2(+relu)->maxpool2
TEST (generate_blob, blob_convbias_convrelu)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt6 = test_cm.input({64, 64, 3}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    // define first convolution  3D conv
    std::vector<double> weightsData61 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000, 0.010);
    auto weightsIt61 = test_cm.constant(weightsData61, {5, 5, 3, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt61 = test_cm.conv2D(inIt6, weightsIt61, {2, 2}, {0, 0, 0, 0});
    std::vector<double> biasesData = { 64444.0 };
    auto biases = test_cm.constant(biasesData, {1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar, "biases");
    auto bias1 = test_cm.bias(convIt61, biases);
    // define first maxpool
    auto maxpoolIt61 = test_cm.maxpool2D(bias1,{5,5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData62 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 65504.0, 0.000);
    auto weightsIt62 = test_cm.constant(weightsData62, {3, 3, 1, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt62 = test_cm.conv2D(maxpoolIt61, weightsIt62, {1, 1}, {0, 0, 0, 0});
    std::vector<double> meanData = mv::utils::generateSequence<double>(convIt62->getShape().totalSize());
    std::vector<double> varianceData = mv::utils::generateSequence<double>(convIt62->getShape().totalSize());
    std::vector<double> offsetData = mv::utils::generateSequence<double>(convIt62->getShape().totalSize());
    std::vector<double> scaleData = mv::utils::generateSequence<double>(convIt62->getShape().totalSize());
    auto bnmean = test_cm.constant(meanData, convIt62->getShape(), mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar, "mean");
    auto bnvariance = test_cm.constant(varianceData, convIt62->getShape(), mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar, "variance");
    auto bnoffset = test_cm.constant(offsetData, convIt62->getShape(), mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar, "offset");
    auto bnscale = test_cm.constant(scaleData, convIt62->getShape(), mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar, "scale");
    auto batchnorm = test_cm.batchNorm(convIt62, bnmean, bnvariance, bnoffset, bnscale, 1e-6);
    auto reluIt62 = test_cm.relu(batchnorm);
    // define second maxpool
    auto maxpoolIt62 = test_cm.maxpool2D(reluIt62,{3,3}, {2, 2}, {1, 1, 1, 1});
    // define output
    auto outIt6 = test_cm.output(maxpoolIt62);

    std::string blobName = "test_relu_10.blob";
    unit.compilationDescriptor()["GenerateBlob"]["output"] = blobName;
    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();
    unit.passManager().disablePass(mv::PassGenre::Validation);
    unit.passManager().disablePass(mv::PassGenre::Serialization);
    unit.passManager().enablePass(mv::PassGenre::Serialization, "GenerateBlob");

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (1948LL, compOutput["passes"].last()["blobSize"].get<long long>()) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    std::string goldBlobPath = mv::utils::projectRootPath() + std::string("/tests/data/gold_10.blob");
    std::string command = "diff \"" + blobName + "\" \"" + goldBlobPath + "\"";
    //EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";

}

// test 09 : conv1(+bias)->maxpool1->conv2(+relu)->maxpool2->scale
TEST (generate_blob, blob_scale)
{

    mv::CompilationUnit unit("testModel");
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt6 = test_cm.input({64, 64, 3}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);
    // define first convolution  3D conv
    std::vector<double> weightsData61 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000, 0.010);
    auto weightsIt61 = test_cm.constant(weightsData61, {5, 5, 3, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt61 = test_cm.conv2D(inIt6, weightsIt61, {2, 2}, {0, 0, 0, 0});
    std::vector<double> biasesData = { 64444.0 };
    auto biases = test_cm.constant(biasesData, {1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar, "biases");
    auto bias1 = test_cm.bias(convIt61, biases);
    // define first maxpool
    auto maxpoolIt61 = test_cm.maxpool2D(bias1,{5,5}, {3, 3}, {1, 1, 1, 1});
    // define second convolution
    std::vector<double> weightsData62 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 65504.0, 0.000);
    auto weightsIt62 = test_cm.constant(weightsData62, {3, 3, 1, 1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt62 = test_cm.conv2D(maxpoolIt61, weightsIt62, {1, 1}, {0, 0, 0, 0});
    // define scale
    std::vector<double> scalesData = { 6550.0f };
    auto scales = test_cm.constant(scalesData, {1}, mv::DTypeType::Float16, mv::OrderType::RowMajorPlanar, "scales");
    auto scaleIt62 = test_cm.scale(convIt62, scales);
    // define output
    auto outIt6 = test_cm.output(scaleIt62);

    std::string blobName = "test_scale_11.blob";
    unit.compilationDescriptor()["GenerateBlob"]["output"] = blobName;
    unit.loadTargetDescriptor(mv::Target::ma2480);
    unit.initialize();
    unit.passManager().disablePass(mv::PassGenre::Validation);
    unit.passManager().disablePass(mv::PassGenre::Serialization);
    unit.passManager().enablePass(mv::PassGenre::Serialization, "GenerateBlob");

    auto compOutput = unit.run();

    // compare filesize written to expected
    EXPECT_EQ (1084LL, compOutput["passes"].last()["blobSize"].get<long long>()) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    std::string goldBlobPath = mv::utils::projectRootPath() + std::string("/tests/data/gold_11.blob");
    std::string command = "diff \"" + blobName + "\" \"" + goldBlobPath + "\"";
    EXPECT_EQ (0, system(command.c_str())) << "ERROR: generated blob file contents do not match expected";

}
