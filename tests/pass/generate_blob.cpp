#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/utils/serializer/Fp16Convert.h"

static mv::Logger::VerboseLevel logger_level = mv::Logger::VerboseLevel::VerboseSilent;

TEST (mv_num_convert, fp32_to_fp16)
{
   mv_num_convert cvtr ;
   EXPECT_EQ(cvtr.fp32_to_fp16(1.0f),0x3c00 );
   EXPECT_EQ(cvtr.fp32_to_fp16(1.0009765625f),0x3c01 );
   EXPECT_EQ(cvtr.fp32_to_fp16(-2.0f),0xc000 );
   EXPECT_EQ(cvtr.fp32_to_fp16(65504.0f),0x7bff );
   EXPECT_EQ(cvtr.fp32_to_fp16(0.0000610352f),0x0400 );
   EXPECT_EQ(cvtr.fp32_to_fp16(0.0000609756f),0x03ff );
   EXPECT_EQ(cvtr.fp32_to_fp16(0.0000000596046f),0x0001 );
   EXPECT_EQ(cvtr.fp32_to_fp16(0.0f),0x0000 );
   EXPECT_EQ(cvtr.fp32_to_fp16(0.333251953125f),0x3555 );
}

// test 01 : 1 2d convolution
TEST (generate_blob, blob_output_conv_01)
{

    mv::CompilationUnit unit(logger_level);
    mv::CompositionalModel& test_cm = unit.model();

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto input1 = test_cm.input(mv::Shape(32, 32, 1), mv::DType::Float, mv::Order::ColumnMajor);
    mv::dynamic_vector<mv::float_type> weights1Data({ 0.1111f, 0.1121f, 0.1131f, 0.1141f, 0.1151f, 0.1161f, 0.1171f, 0.1181f, 0.1191f});
    auto weights1 = test_cm.constant(weights1Data, mv::Shape(3, 3, 1, 1), mv::DType::Float, mv::Order::ColumnMajor);
    auto conv1 = test_cm.conv2D(input1, weights1, {4, 4}, {0, 0, 0, 0});
    auto output1 = test_cm.output(conv1);

    // Check output shape
    EXPECT_EQ(output1->getShape(), mv::Shape(8, 8, 1));

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

    mv::CompilationUnit unit(logger_level);
    mv::CompositionalModel& test_cm = unit.model();

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto input2 = test_cm.input(mv::Shape(32, 32, 3), mv::DType::Float, mv::Order::RowMajorPlanar);   //N WH C
    mv::dynamic_vector<mv::float_type> weightsData2 = mv::utils::generateSequence<mv::float_type>(3u * 3u * 3u * 3u, 0.101f, 0.001f);

    auto weights2 = test_cm.constant(weightsData2, mv::Shape(3, 3, 3, 3), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, kN, C
    auto conv2 = test_cm.conv2D(input2, weights2, {4, 4}, {0, 0, 0, 0});   // input tensor, wieghts tensor, stridex, stridey, padx, pady
    auto output2 = test_cm.output(conv2);

    // Check output shape
    EXPECT_EQ(output2->getShape(), mv::Shape(8, 8, 3));   // x, y, c

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

    mv::CompilationUnit unit(logger_level);
    mv::CompositionalModel& test_cm = unit.model();

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto input3 = test_cm.input(mv::Shape(256, 256, 3), mv::DType::Float, mv::Order::RowMajorPlanar);   //N WH C

    mv::dynamic_vector<mv::float_type> weightsData3 = mv::utils::generateSequence(3u * 3u * 3u * 3u, 0.101f, 0.001f);

    auto weights3 = test_cm.constant(weightsData3, mv::Shape(3, 3, 3, 3), mv::DType::Float, mv::Order::RowMajorPlanar);
    auto conv3 = test_cm.conv2D(input3, weights3, {2, 2}, {0, 0, 0, 0});   // input tensor, wieghts tensor, stridex, stridey, padx, pady
    auto output3 = test_cm.output(conv3);

    // Check output shape
    EXPECT_EQ(output3->getShape(), mv::Shape(127, 127, 3));   // x, y, c

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

    mv::CompilationUnit unit(logger_level);
    mv::CompositionalModel& test_cm = unit.model();

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto input4 = test_cm.input(mv::Shape(256, 256, 3), mv::DType::Float, mv::Order::RowMajorPlanar);   //N WH C
    mv::dynamic_vector<mv::float_type> weightsData4 = mv::utils::generateSequence(5u * 5u * 3u * 3u, 0.101f, 0.001f);

    auto weights4 = test_cm.constant(weightsData4, mv::Shape(5, 5, 3, 3), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, kN, C
    auto conv4 = test_cm.conv2D(input4, weights4, {2, 2}, {0, 0, 0, 0});   // input tensor, wieghts tensor, stridex, stridey, padx, pady
    auto output4 = test_cm.output(conv4);

    // Check output shape
    EXPECT_EQ(output4->getShape(), mv::Shape(126, 126, 3));   // x, y, c

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

    mv::CompilationUnit unit(logger_level);
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 greyscale 256x256 image
    auto input5 = test_cm.input(mv::Shape(256, 256, 1), mv::DType::Float, mv::Order::RowMajorPlanar);

    mv::dynamic_vector<mv::float_type> blurKData({ 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 });
    mv::dynamic_vector<mv::float_type> edgeKData({ 65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0 });
    auto bweights = test_cm.constant(blurKData, mv::Shape(3, 3, 1, 1), mv::DType::Float, mv::Order::RowMajorPlanar);
    auto eweights = test_cm.constant(edgeKData, mv::Shape(3, 3, 1, 1), mv::DType::Float, mv::Order::RowMajorPlanar);
    auto conv1 = test_cm.conv2D(input5, bweights, {1, 1}, {0, 0, 0, 0});
    auto conv2 = test_cm.conv2D(conv1, eweights, {1, 1}, {0, 0, 0, 0});
    auto output = test_cm.output(conv2);

    // Check output shape
    EXPECT_EQ(output->getShape(), mv::Shape(252, 252, 1));

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

    mv::CompilationUnit unit(logger_level);
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt6 = test_cm.input(mv::Shape(64, 64, 3), mv::DType::Float, mv::Order::RowMajorPlanar);

    // define first convolution  3D conv

    mv::dynamic_vector<mv::float_type> weightsData61 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000f, 0.010f);
    auto weightsIt61 = test_cm.constant(weightsData61, mv::Shape(5, 5, 3, 1), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, ins, outs
    EXPECT_EQ(weightsIt61->getShape()[0], 5);
    EXPECT_EQ(weightsIt61->getShape()[1], 5);
    EXPECT_EQ(weightsIt61->getShape()[2], 3);
    EXPECT_EQ(weightsIt61->getShape()[3], 1);
    auto convIt61 = test_cm.conv2D(inIt6, weightsIt61, {2, 2}, {0, 0, 0, 0});

    // define first maxpool
    auto maxpoolIt61 = test_cm.maxpool2D(convIt61,{5,5}, {3, 3}, {1, 1, 1, 1});

    // define second convolution
    mv::dynamic_vector<mv::float_type> weightsData62 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 65504.0f, 0.000f);
    auto weightsIt62 = test_cm.constant(weightsData62, mv::Shape(3, 3, 1, 1), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt62 = test_cm.conv2D(maxpoolIt61, weightsIt62, {1, 1}, {0, 0, 0, 0});

    // define second maxpool
    auto maxpoolIt62 = test_cm.maxpool2D(convIt62,{3,3}, {2, 2}, {1, 1, 1, 1});

    // define output
    auto outIt6 = test_cm.output(maxpoolIt62);

    // Check if model is valid
    EXPECT_TRUE(test_cm.isValid());

    // Check output shapes of each layer
    EXPECT_EQ(inIt6->getShape()[0], 64);     // X dim  input
    EXPECT_EQ(inIt6->getShape()[1], 64);     // X dim
    EXPECT_EQ(inIt6->getShape()[2], 3);      // Z dim

    EXPECT_EQ(convIt61->getShape()[0], 30);  // X dim  conv 1
    EXPECT_EQ(convIt61->getShape()[1], 30);  // X dim
    EXPECT_EQ(convIt61->getShape()[2], 1);   // Z dim

    EXPECT_EQ(maxpoolIt61->getShape()[0], 10);  // X dim  maxpool 1
    EXPECT_EQ(maxpoolIt61->getShape()[1], 10);  // X dim
    EXPECT_EQ(maxpoolIt61->getShape()[2], 1);   // Z dim

    EXPECT_EQ(convIt62->getShape()[0], 8);      // X dim  conv 2
    EXPECT_EQ(convIt62->getShape()[1], 8);      // X dim
    EXPECT_EQ(convIt62->getShape()[2], 1);      // Z dim

    EXPECT_EQ(maxpoolIt62->getShape()[0], 4);   // X dim  maxpool 1
    EXPECT_EQ(maxpoolIt62->getShape()[1], 4);   // X dim
    EXPECT_EQ(maxpoolIt62->getShape()[2], 1);   // Z dim

    EXPECT_EQ(outIt6->getShape()[0], 4);   // X dim  output

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

    mv::CompilationUnit unit(logger_level);
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt7 = test_cm.input(mv::Shape(64, 64, 3), mv::DType::Float, mv::Order::RowMajorPlanar);
    auto maxpoolIt11= test_cm.maxpool2D(inIt7,{1,1}, {1, 1}, {0,0,0,0});

    // define first convolution
    mv::dynamic_vector<mv::float_type> weightsData71 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.100f, 0.010f);
    auto weightsIt71 = test_cm.constant(weightsData71, mv::Shape(5, 5, 3, 1), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, ins, outs
    EXPECT_EQ(weightsIt71->getShape()[0], 5);
    EXPECT_EQ(weightsIt71->getShape()[1], 5);
    EXPECT_EQ(weightsIt71->getShape()[2], 3);
    EXPECT_EQ(weightsIt71->getShape()[3], 1);
    auto convIt71 = test_cm.conv2D(maxpoolIt11, weightsIt71, {2, 2}, {0, 0, 0, 0});

    // define first avgpool
    auto avgpoolIt71 = test_cm.avgpool2D(convIt71,{5,5}, {3, 3}, {1, 1, 1, 1});

    // define second convolution
    mv::dynamic_vector<mv::float_type> weightsData72 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 6550.0f, 0.000f);
    auto weightsIt72 = test_cm.constant(weightsData72, mv::Shape(3, 3, 1, 1), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt72 = test_cm.conv2D(avgpoolIt71, weightsIt72, {1, 1}, {0, 0, 0, 0});

    // define second avgpool
    auto avgpoolIt72 = test_cm.avgpool2D(convIt72,{3,3}, {2, 2}, {1, 1, 1, 1});

    // define first convolution branch a
    mv::dynamic_vector<mv::float_type> weightsData7a = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000f, 0.010f);
    auto weightsIt7a = test_cm.constant(weightsData7a, mv::Shape(5, 5, 3, 1), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, ins, outs
    EXPECT_EQ(weightsIt7a->getShape()[0], 5);
    EXPECT_EQ(weightsIt7a->getShape()[1], 5);
    EXPECT_EQ(weightsIt7a->getShape()[2], 3);
    EXPECT_EQ(weightsIt7a->getShape()[3], 1);
    auto convIt7a = test_cm.conv2D(inIt7, weightsIt7a, {2, 2}, {0, 0, 0, 0});

    // define first maxpool branch a
    auto maxpoolIt7a = test_cm.maxpool2D(convIt7a,{5,5}, {3, 3}, {1, 1, 1, 1});

    // define second convolution
    mv::dynamic_vector<mv::float_type> weightsData7b = mv::utils::generateSequence(3u * 3u * 1u * 1u, 65504.0f, 0.000f);
    auto weightsIt7b = test_cm.constant(weightsData7b, mv::Shape(3, 3, 1, 1), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt7b = test_cm.conv2D(maxpoolIt7a, weightsIt7b, {1, 1}, {0, 0, 0, 0});

    // define second maxpool
    auto maxpoolIt7b = test_cm.maxpool2D(convIt7b,{3,3}, {2, 2}, {1, 1, 1, 1});

    // define elementwise sum
    auto eltwiseIt7 = test_cm.add(maxpoolIt7b,avgpoolIt72);

    // define output
    auto outIt7 = test_cm.output(eltwiseIt7);

    // Check if model is valid
    EXPECT_TRUE(test_cm.isValid()) << "INVALID MODEL" ;

    // Check output shapes of each layer
    EXPECT_EQ(inIt7->getShape()[0], 64);     // X dim  input
    EXPECT_EQ(inIt7->getShape()[1], 64);     // X dim
    EXPECT_EQ(inIt7->getShape()[2], 3);      // Z dim

    EXPECT_EQ(convIt71->getShape()[0], 30);  // X dim  conv 1
    EXPECT_EQ(convIt71->getShape()[1], 30);  // X dim
    EXPECT_EQ(convIt71->getShape()[2], 1);   // Z dim

    EXPECT_EQ(avgpoolIt71->getShape()[0], 10);  // X dim  maxpool 1
    EXPECT_EQ(avgpoolIt71->getShape()[1], 10);  // X dim
    EXPECT_EQ(avgpoolIt71->getShape()[2], 1);   // Z dim

    EXPECT_EQ(convIt7b->getShape()[0], 8);      // X dim  conv 2
    EXPECT_EQ(convIt7b->getShape()[1], 8);      // X dim
    EXPECT_EQ(convIt7b->getShape()[2], 1);      // Z dim

    EXPECT_EQ(maxpoolIt7b->getShape()[0], 4);   // X dim  maxpool 1
    EXPECT_EQ(maxpoolIt7b->getShape()[1], 4);   // X dim
    EXPECT_EQ(maxpoolIt7b->getShape()[2], 1);   // Z dim

    EXPECT_EQ(outIt7->getShape()[0], 4);   // X dim  output

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

    mv::CompilationUnit unit(logger_level);
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt7 = test_cm.input(mv::Shape(64, 64, 3), mv::DType::Float, mv::Order::RowMajorPlanar);
    auto maxpoolIt11= test_cm.maxpool2D(inIt7,{1,1}, {1, 1}, {0,0,0,0});

    // define first convolution
    mv::dynamic_vector<mv::float_type> weightsData71 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.100f, 0.010f);
    auto weightsIt71 = test_cm.constant(weightsData71, mv::Shape(5, 5, 3, 1), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, ins, outs
    EXPECT_EQ(weightsIt71->getShape()[0], 5);
    EXPECT_EQ(weightsIt71->getShape()[1], 5);
    EXPECT_EQ(weightsIt71->getShape()[2], 3);
    EXPECT_EQ(weightsIt71->getShape()[3], 1);
    auto convIt71 = test_cm.conv2D(maxpoolIt11, weightsIt71, {2, 2}, {0, 0, 0, 0});

    // define first avgpool
    auto avgpoolIt71 = test_cm.avgpool2D(convIt71,{5,5}, {3, 3}, {1, 1, 1, 1});

    // define second convolution
    mv::dynamic_vector<mv::float_type> weightsData72 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 6550.0f, 0.000f);
    auto weightsIt72 = test_cm.constant(weightsData72, mv::Shape(3, 3, 1, 1), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt72 = test_cm.conv2D(avgpoolIt71, weightsIt72, {1, 1}, {0, 0, 0, 0});

    // define second avgpool

    auto avgpoolIt72 = test_cm.avgpool2D(convIt72,{3,3}, {2, 2}, {1, 1, 1, 1});

    // define first convolution branch a
    mv::dynamic_vector<mv::float_type> weightsData7a = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000f, 0.010f);
    auto weightsIt7a = test_cm.constant(weightsData7a, mv::Shape(5, 5, 3, 1), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, ins, outs
    EXPECT_EQ(weightsIt7a->getShape()[0], 5);
    EXPECT_EQ(weightsIt7a->getShape()[1], 5);
    EXPECT_EQ(weightsIt7a->getShape()[2], 3);
    EXPECT_EQ(weightsIt7a->getShape()[3], 1);
    auto convIt7a = test_cm.conv2D(inIt7, weightsIt7a, {2, 2}, {0, 0, 0, 0});

    // define first maxpool branch a
    auto maxpoolIt7a = test_cm.maxpool2D(convIt7a,{5,5}, {3, 3}, {1, 1, 1, 1});

    // define second convolution
    mv::dynamic_vector<mv::float_type> weightsData7b = mv::utils::generateSequence(3u * 3u * 1u * 1u, 65504.0f, 0.000f);
    auto weightsIt7b = test_cm.constant(weightsData7b, mv::Shape(3, 3, 1, 1), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt7b = test_cm.conv2D(maxpoolIt7a, weightsIt7b, {1, 1}, {0, 0, 0, 0});

    // define second maxpool
    auto maxpoolIt7b = test_cm.maxpool2D(convIt7b,{3,3}, {2, 2}, {1, 1, 1, 1});

    // define elementwise sum
    auto eltwiseIt7 = test_cm.multiply(maxpoolIt7b,avgpoolIt72);

    // define output
    auto outIt7 = test_cm.output(eltwiseIt7);

    // Check if model is valid
    EXPECT_TRUE(test_cm.isValid()) << "INVALID MODEL" ;

    // Check output shapes of each layer
    EXPECT_EQ(inIt7->getShape()[0], 64);     // X dim  input
    EXPECT_EQ(inIt7->getShape()[1], 64);     // X dim
    EXPECT_EQ(inIt7->getShape()[2], 3);      // Z dim

    EXPECT_EQ(convIt71->getShape()[0], 30);  // X dim  conv 1
    EXPECT_EQ(convIt71->getShape()[1], 30);  // X dim
    EXPECT_EQ(convIt71->getShape()[2], 1);   // Z dim

    EXPECT_EQ(avgpoolIt71->getShape()[0], 10);  // X dim  maxpool 1
    EXPECT_EQ(avgpoolIt71->getShape()[1], 10);  // X dim
    EXPECT_EQ(avgpoolIt71->getShape()[2], 1);   // Z dim

    EXPECT_EQ(convIt7b->getShape()[0], 8);      // X dim  conv 2
    EXPECT_EQ(convIt7b->getShape()[1], 8);      // X dim
    EXPECT_EQ(convIt7b->getShape()[2], 1);      // Z dim

    EXPECT_EQ(maxpoolIt7b->getShape()[0], 4);   // X dim  maxpool 1
    EXPECT_EQ(maxpoolIt7b->getShape()[1], 4);   // X dim
    EXPECT_EQ(maxpoolIt7b->getShape()[2], 1);   // Z dim

    EXPECT_EQ(outIt7->getShape()[0], 4);   // X dim  output

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

    mv::CompilationUnit unit(logger_level);
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt7 = test_cm.input(mv::Shape(64, 64, 3), mv::DType::Float, mv::Order::RowMajorPlanar);

    // define first convolution
    mv::dynamic_vector<mv::float_type> weightsData71 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.100f, 0.010f);
    auto weightsIt71 = test_cm.constant(weightsData71, mv::Shape(5, 5, 3, 1), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, ins, outs
    EXPECT_EQ(weightsIt71->getShape()[0], 5);
    EXPECT_EQ(weightsIt71->getShape()[1], 5);
    EXPECT_EQ(weightsIt71->getShape()[2], 3);
    EXPECT_EQ(weightsIt71->getShape()[3], 1);
    auto convIt71 = test_cm.conv2D(inIt7, weightsIt71, {2, 2}, {0, 0, 0, 0});

    // define first avgpool
    auto avgpoolIt71 = test_cm.avgpool2D(convIt71,{5,5}, {3, 3}, {1, 1, 1, 1});

    // define second convolution
    mv::dynamic_vector<mv::float_type> weightsData72 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 6550.0f, 0.000f);
    auto weightsIt72 = test_cm.constant(weightsData72, mv::Shape(3, 3, 1, 1), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt72 = test_cm.conv2D(avgpoolIt71, weightsIt72, {1, 1}, {0, 0, 0, 0});

    // define second avgpool

    auto avgpoolIt72 = test_cm.avgpool2D(convIt72,{3,3}, {2, 2}, {1, 1, 1, 1});

    // define first convolution branch a
    mv::dynamic_vector<mv::float_type> weightsData7a = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000f, 0.010f);
    auto weightsIt7a = test_cm.constant(weightsData7a, mv::Shape(5, 5, 3, 1), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, ins, outs
    EXPECT_EQ(weightsIt7a->getShape()[0], 5);
    EXPECT_EQ(weightsIt7a->getShape()[1], 5);
    EXPECT_EQ(weightsIt7a->getShape()[2], 3);
    EXPECT_EQ(weightsIt7a->getShape()[3], 1);
    auto convIt7a = test_cm.conv2D(inIt7, weightsIt7a, {2, 2}, {0, 0, 0, 0});

    // define first maxpool branch a
    auto maxpoolIt7a = test_cm.maxpool2D(convIt7a,{5,5}, {3, 3}, {1, 1, 1, 1});

    // define second convolution
    mv::dynamic_vector<mv::float_type> weightsData7b = mv::utils::generateSequence(3u * 3u * 1u * 1u, 65504.0f, 0.000f);
    auto weightsIt7b = test_cm.constant(weightsData7b, mv::Shape(3, 3, 1, 1), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt7b = test_cm.conv2D(maxpoolIt7a, weightsIt7b, {1, 1}, {0, 0, 0, 0});

    // define second maxpool

    auto maxpoolIt7b = test_cm.maxpool2D(convIt7b,{3,3}, {2, 2}, {1, 1, 1, 1});

    // define elementwise sum
    auto eltwiseIt7 = test_cm.add(maxpoolIt7b,avgpoolIt72);

    auto softIt7 = test_cm.softmax(eltwiseIt7);

    // define output
    auto outIt7 = test_cm.output(softIt7);

    // Check if model is valid
    EXPECT_TRUE(test_cm.isValid()) << "INVALID MODEL" ;

    // Check output shapes of each layer
    EXPECT_EQ(inIt7->getShape()[0], 64);     // X dim  input
    EXPECT_EQ(inIt7->getShape()[1], 64);     // X dim
    EXPECT_EQ(inIt7->getShape()[2], 3);      // Z dim

    EXPECT_EQ(convIt71->getShape()[0], 30);  // X dim  conv 1
    EXPECT_EQ(convIt71->getShape()[1], 30);  // X dim
    EXPECT_EQ(convIt71->getShape()[2], 1);   // Z dim

    EXPECT_EQ(avgpoolIt71->getShape()[0], 10);  // X dim  maxpool 1
    EXPECT_EQ(avgpoolIt71->getShape()[1], 10);  // X dim
    EXPECT_EQ(avgpoolIt71->getShape()[2], 1);   // Z dim

    EXPECT_EQ(convIt7b->getShape()[0], 8);      // X dim  conv 2
    EXPECT_EQ(convIt7b->getShape()[1], 8);      // X dim
    EXPECT_EQ(convIt7b->getShape()[2], 1);      // Z dim

    EXPECT_EQ(maxpoolIt7b->getShape()[0], 4);   // X dim  maxpool 1
    EXPECT_EQ(maxpoolIt7b->getShape()[1], 4);   // X dim
    EXPECT_EQ(maxpoolIt7b->getShape()[2], 1);   // Z dim

    EXPECT_EQ(outIt7->getShape()[0], 4);   // X dim  output

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

    mv::CompilationUnit unit(logger_level);
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt6 = test_cm.input(mv::Shape(64, 64, 3), mv::DType::Float, mv::Order::RowMajorPlanar);

    // define first convolution  3D conv

    mv::dynamic_vector<mv::float_type> weightsData61 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000f, 0.010f);
    auto weightsIt61 = test_cm.constant(weightsData61, mv::Shape(5, 5, 3, 1), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, ins, outs
    EXPECT_EQ(weightsIt61->getShape()[0], 5);
    EXPECT_EQ(weightsIt61->getShape()[1], 5);
    EXPECT_EQ(weightsIt61->getShape()[2], 3);
    EXPECT_EQ(weightsIt61->getShape()[3], 1);
    auto convIt61 = test_cm.conv2D(inIt6, weightsIt61, {2, 2}, {0, 0, 0, 0});

    mv::dynamic_vector<mv::float_type> biasesData = { 64444.0 };
    auto biases = test_cm.constant(biasesData, mv::Shape(1), mv::DType::Float, mv::Order::RowMajorPlanar, "biases");
    auto bias1 = test_cm.bias(convIt61, biases);

    // define first maxpool
    auto maxpoolIt61 = test_cm.maxpool2D(bias1,{5,5}, {3, 3}, {1, 1, 1, 1});

    // define second convolution
    mv::dynamic_vector<mv::float_type> weightsData62 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 65504.0f, 0.000f);
    auto weightsIt62 = test_cm.constant(weightsData62, mv::Shape(3, 3, 1, 1), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt62 = test_cm.conv2D(maxpoolIt61, weightsIt62, {1, 1}, {0, 0, 0, 0});

    mv::dynamic_vector<mv::float_type> meanData = mv::utils::generateSequence<mv::float_type>(convIt62->getShape().totalSize());
    mv::dynamic_vector<mv::float_type> varianceData = mv::utils::generateSequence<mv::float_type>(convIt62->getShape().totalSize());
    mv::dynamic_vector<mv::float_type> offsetData = mv::utils::generateSequence<mv::float_type>(convIt62->getShape().totalSize());
    mv::dynamic_vector<mv::float_type> scaleData = mv::utils::generateSequence<mv::float_type>(convIt62->getShape().totalSize());

    auto bnmean = test_cm.constant(meanData, convIt62->getShape(), mv::DType::Float, mv::Order::RowMajorPlanar, "mean");
    auto bnvariance = test_cm.constant(varianceData, convIt62->getShape(), mv::DType::Float, mv::Order::RowMajorPlanar, "variance");
    auto bnoffset = test_cm.constant(offsetData, convIt62->getShape(), mv::DType::Float, mv::Order::RowMajorPlanar, "offset");
    auto bnscale = test_cm.constant(scaleData, convIt62->getShape(), mv::DType::Float, mv::Order::RowMajorPlanar, "scale");
    auto batchnorm = test_cm.batchNorm(convIt62, bnmean, bnvariance, bnoffset, bnscale, 1e-6);
    auto reluIt62 = test_cm.relu(batchnorm);

    // define second maxpool
    auto maxpoolIt62 = test_cm.maxpool2D(reluIt62,{3,3}, {2, 2}, {1, 1, 1, 1});

    // define output
    auto outIt6 = test_cm.output(maxpoolIt62);

    // Check if model is valid
    EXPECT_TRUE(test_cm.isValid());

    // Check output shapes of each layer
    EXPECT_EQ(inIt6->getShape()[0], 64);     // X dim  input
    EXPECT_EQ(inIt6->getShape()[1], 64);     // X dim
    EXPECT_EQ(inIt6->getShape()[2], 3);      // Z dim

    EXPECT_EQ(convIt61->getShape()[0], 30);  // X dim  conv 1
    EXPECT_EQ(convIt61->getShape()[1], 30);  // X dim
    EXPECT_EQ(convIt61->getShape()[2], 1);   // Z dim

    EXPECT_EQ(maxpoolIt61->getShape()[0], 10);  // X dim  maxpool 1
    EXPECT_EQ(maxpoolIt61->getShape()[1], 10);  // X dim
    EXPECT_EQ(maxpoolIt61->getShape()[2], 1);   // Z dim

    EXPECT_EQ(convIt62->getShape()[0], 8);      // X dim  conv 2
    EXPECT_EQ(convIt62->getShape()[1], 8);      // X dim
    EXPECT_EQ(convIt62->getShape()[2], 1);      // Z dim

    EXPECT_EQ(maxpoolIt62->getShape()[0], 4);   // X dim  maxpool 1
    EXPECT_EQ(maxpoolIt62->getShape()[1], 4);   // X dim
    EXPECT_EQ(maxpoolIt62->getShape()[2], 1);   // Z dim

    EXPECT_EQ(outIt6->getShape()[0], 4);   // X dim  output

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

    mv::CompilationUnit unit(logger_level);
    mv::CompositionalModel& test_cm = unit.model();

    // Define input as 1 64x64x3 image
    auto inIt6 = test_cm.input(mv::Shape(64, 64, 3), mv::DType::Float, mv::Order::RowMajorPlanar);

    // define first convolution  3D conv

    mv::dynamic_vector<mv::float_type> weightsData61 = mv::utils::generateSequence(5u * 5u * 3u * 1u, 0.000f, 0.010f);
    auto weightsIt61 = test_cm.constant(weightsData61, mv::Shape(5, 5, 3, 1), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, ins, outs
    EXPECT_EQ(weightsIt61->getShape()[0], 5);
    EXPECT_EQ(weightsIt61->getShape()[1], 5);
    EXPECT_EQ(weightsIt61->getShape()[2], 3);
    EXPECT_EQ(weightsIt61->getShape()[3], 1);
    auto convIt61 = test_cm.conv2D(inIt6, weightsIt61, {2, 2}, {0, 0, 0, 0});

    mv::dynamic_vector<mv::float_type> biasesData = { 64444.0 };
    auto biases = test_cm.constant(biasesData, mv::Shape(1), mv::DType::Float, mv::Order::RowMajorPlanar, "biases");
    auto bias1 = test_cm.bias(convIt61, biases);

    // define first maxpool
    auto maxpoolIt61 = test_cm.maxpool2D(bias1,{5,5}, {3, 3}, {1, 1, 1, 1});

    // define second convolution
    mv::dynamic_vector<mv::float_type> weightsData62 = mv::utils::generateSequence(3u * 3u * 1u * 1u, 65504.0f, 0.000f);
    auto weightsIt62 = test_cm.constant(weightsData62, mv::Shape(3, 3, 1, 1), mv::DType::Float, mv::Order::RowMajorPlanar);   // kh, kw, ins, outs
    auto convIt62 = test_cm.conv2D(maxpoolIt61, weightsIt62, {1, 1}, {0, 0, 0, 0});
//    auto reluIt62 = test_cm.relu(convIt62);

    // define second maxpool
//    auto maxpoolIt62 = test_cm.maxpool2D(reluIt62,{3,3}, {2, 2}, {1, 1, 1, 1});

    // define scale
    mv::dynamic_vector<mv::float_type> scalesData = { 6550.0f };
    auto scales = test_cm.constant(scalesData, mv::Shape(1), mv::DType::Float, mv::Order::RowMajorPlanar, "scales");
    auto scaleIt62 = test_cm.scale(convIt62, scales);

    // define output
    auto outIt6 = test_cm.output(scaleIt62);

    // Check if model is valid
    EXPECT_TRUE(test_cm.isValid());

    // Check output shapes of each layer
    EXPECT_EQ(inIt6->getShape()[0], 64);     // X dim  input
    EXPECT_EQ(inIt6->getShape()[1], 64);     // X dim
    EXPECT_EQ(inIt6->getShape()[2], 3);      // Z dim

    EXPECT_EQ(convIt61->getShape()[0], 30);  // X dim  conv 1
    EXPECT_EQ(convIt61->getShape()[1], 30);  // X dim
    EXPECT_EQ(convIt61->getShape()[2], 1);   // Z dim

    EXPECT_EQ(maxpoolIt61->getShape()[0], 10);  // X dim  maxpool 1
    EXPECT_EQ(maxpoolIt61->getShape()[1], 10);  // X dim
    EXPECT_EQ(maxpoolIt61->getShape()[2], 1);   // Z dim

    EXPECT_EQ(convIt62->getShape()[0], 8);      // X dim  conv 2
    EXPECT_EQ(convIt62->getShape()[1], 8);      // X dim
    EXPECT_EQ(convIt62->getShape()[2], 1);      // Z dim

    EXPECT_EQ(outIt6->getShape()[0], 8);   // X dim  output

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
