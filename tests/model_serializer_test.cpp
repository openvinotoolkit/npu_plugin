#include "gtest/gtest.h"
#include "include/mcm/computation/model/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/deployer/serializer.hpp"
#include "include/mcm/deployer/Fp16Convert.h"
#include "include/mcm/utils/data_generator.hpp"

/*
// return full path of this executable 
char* get_exe_path()
{ 
    const int PATH_MAX = 250 ;
    char path_arr[PATH_MAX];
    ssize_t path_len = ::readlink("/proc/self/exe", path_arr, sizeof(path_arr));
    char *exe_path = new char[path_len];
    memcpy(exe_path, path_arr, path_len);
    return exe_path;
    //std::cout << "Running tests from " << exe_path << std::endl;
    //std::cout << "copied " << path_len << " chars" << std::endl;
}
*/

TEST (model_serializer, convert_fp32_to_fp16)
{
   EXPECT_EQ(f32Tof16(1.0f),0x3c00 );
   EXPECT_EQ(f32Tof16(1.0009765625f),0x3c01 );
   EXPECT_EQ(f32Tof16(-2.0f),0xc000 );
   EXPECT_EQ(f32Tof16(65504.0f),0x7bff );
   EXPECT_EQ(f32Tof16(0.0000610352f),0x0400 );
//  these 2 subnormal cases return zero in Movidius implementation
   EXPECT_EQ(f32Tof16(0.0000609756f),0x0000 );
   EXPECT_EQ(f32Tof16(0.0000000596046f),0x0000 );
//   EXPECT_EQ(f32Tof16(0.0000609756f),0x03ff );
//   EXPECT_EQ(f32Tof16(0.0000000596046f),0x0001 );
   EXPECT_EQ(f32Tof16(0.0f),0x0000 );
   EXPECT_EQ(f32Tof16(0.333251953125f),0x3555 );
}

// test 01 : 1 2d convolution
TEST (model_serializer, blob_output_conv_01) 
{
    // define test compute model: 1 convolution 
    mv::OpModel test_cm ;

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto input1 = test_cm.input(mv::Shape(32, 32, 1), mv::DType::Float, mv::Order::NWHC);
    mv::float_type rawData[] =
    { 0.1111f, 0.1121f, 0.1131f, 0.1141f, 0.1151f, 0.1161f, 0.1171f, 0.1181f, 0.1191f};
    mv::dynamic_vector<mv::float_type> weights1Data(rawData);
    auto weights1 = test_cm.constant(weights1Data, mv::Shape(3, 3, 1, 1), mv::DType::Float, mv::Order::NWHC);
    auto conv1 = test_cm.conv2D(input1, weights1, {4, 4}, {0, 0, 0, 0});
    auto output1 = test_cm.output(conv1);

    // Check output shape
    EXPECT_EQ(output1->getShape(), mv::Shape(8, 8, 1));

    mv::ControlModel cm(test_cm);

    // declare serializer as blob
    mv::Serializer gs(mv::mvblob_mode);

    // serialize compute model to file
    uint64_t filesize = gs.serialize(cm, "test_conv_01.blob");

    // compare filesize written to expected
    EXPECT_EQ (692, filesize) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    
    const char *command1 = "cp ../../tests/data/gold_01.blob .";
    system(command1);
    const char *command2 = "diff test_conv_01.blob gold_01.blob";
    EXPECT_EQ (0, system(command2)) << "ERROR: generated blob file contents do not match expected";

}


// test 02 : 1 2d convolution, add input z dimension (c=3)
TEST (model_serializer, blob_output_conv_02)
{
    mv::OpModel test_cm2 ;

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto input2 = test_cm2.input(mv::Shape(32, 32, 3), mv::DType::Float, mv::Order::NWHC);   //N WH C   
    mv::dynamic_vector<mv::float_type> weightsData2 = mv::utils::generateSequence<mv::float_type>(3u * 3u * 3u * 3u, 0.101f, 0.001f);

    auto weights2 = test_cm2.constant(weightsData2, mv::Shape(3, 3, 3, 3), mv::DType::Float, mv::Order::NWHC);   // kh, kw, kN, C
    auto conv2 = test_cm2.conv2D(input2, weights2, {4, 4}, {0, 0, 0, 0});   // input tensor, wieghts tensor, stridex, stridey, padx, pady
    auto output2 = test_cm2.output(conv2);

    // Check output shape
    EXPECT_EQ(output2->getShape(), mv::Shape(8, 8, 3));   // x, y, c

    mv::ControlModel cm2(test_cm2);

    // declare serializer as blob
    mv::Serializer gs2(mv::mvblob_mode);

    // serialize compute model to file
    uint64_t filesize2 = gs2.serialize(cm2, "test_conv_02.blob");

    // compare filesize written to expected
    EXPECT_EQ (948, filesize2) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    const char *command1 = "cp ../../tests/data/gold_02.blob .";
    system(command1);

    const char *command2 = "diff test_conv_02.blob gold_02.blob";
    EXPECT_EQ (0, system(command2)) << "ERROR: generated blob file contents do not match expected";

}

// test 03 : 1 2d convolution, change input=256x256  stride=2
TEST (model_serializer, blob_output_conv_03)
{
    mv::OpModel test_cm3(mv::Logger::VerboseLevel::VerboseWarning) ;

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto input3 = test_cm3.input(mv::Shape(256, 256, 3), mv::DType::Float, mv::Order::NWHC);   //N WH C

    mv::dynamic_vector<mv::float_type> weightsData3 = mv::utils::generateSequence(3u * 3u * 3u * 3u, 0.101f, 0.001f);

    auto weights3 = test_cm3.constant(weightsData3, mv::Shape(3, 3, 3, 3), mv::DType::Float, mv::Order::NWHC);
    auto conv3 = test_cm3.conv2D(input3, weights3, {2, 2}, {0, 0, 0, 0});   // input tensor, wieghts tensor, stridex, stridey, padx, pady
    auto output3 = test_cm3.output(conv3);

    // Check output shape
    EXPECT_EQ(output3->getShape(), mv::Shape(127, 127, 3));   // x, y, c

    // declare serializer as blob
    mv::Serializer gs3(mv::mvblob_mode);
    
    // serialize compute model to file
    mv::ControlModel cm3(test_cm3);
    uint64_t filesize3 = gs3.serialize(cm3, "test_conv_03.blob");

    // compare filesize written to expected
    EXPECT_EQ (948, filesize3) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    const char *command1 = "cp ../../tests/data/gold_03.blob .";
    system(command1);

    const char *command2 = "diff test_conv_03.blob gold_03.blob";
    EXPECT_EQ (0, system(command2)) << "ERROR: generated blob file contents do not match expected";

}

// test 04 : 1 2d convolution, change kernel to 5x5
TEST (model_serializer, blob_output_conv_04)
{
    mv::OpModel test_cm4(mv::Logger::VerboseLevel::VerboseWarning) ;

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto input4 = test_cm4.input(mv::Shape(256, 256, 3), mv::DType::Float, mv::Order::NWHC);   //N WH C
    mv::dynamic_vector<mv::float_type> weightsData4 = mv::utils::generateSequence(5u * 5u * 3u * 3u, 0.101f, 0.001f);

    auto weights4 = test_cm4.constant(weightsData4, mv::Shape(5, 5, 3, 3), mv::DType::Float, mv::Order::NWHC);   // kh, kw, kN, C
    auto conv4 = test_cm4.conv2D(input4, weights4, {2, 2}, {0, 0, 0, 0});   // input tensor, wieghts tensor, stridex, stridey, padx, pady
    auto output4 = test_cm4.output(conv4);

    // Check output shape
    EXPECT_EQ(output4->getShape(), mv::Shape(126, 126, 3));   // x, y, c

    // declare serializer as blob
    mv::Serializer gs4(mv::mvblob_mode);

    // serialize compute model to file
    mv::ControlModel cm4(test_cm4);
    uint64_t filesize4 = gs4.serialize(cm4, "test_conv_04.blob");

    // compare filesize written to expected
    EXPECT_EQ (1716, filesize4) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    const char *command1 = "cp ../../tests/data/gold_04.blob .";
    system(command1);

    const char *command2 = "diff test_conv_04.blob gold_04.blob";
    EXPECT_EQ (0, system(command2)) << "ERROR: generated blob file contents do not match expected";

}

// test 05 : 2 successive 3x3 convolutions (blur->edge filters)
TEST (model_serializer, blob_blur_edge_05)
{
    // define test compute model: 1 convolution 
    mv::OpModel test_cm5 ;

    // Define input as 1 greyscale 256x256 image
    auto input5 = test_cm5.input(mv::Shape(256, 256, 1), mv::DType::Float, mv::Order::NWHC);

    mv::float_type k1rawData[] = { 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2 };
    mv::dynamic_vector<mv::float_type> blurKData(k1rawData);
    mv::float_type k2rawData[] = { 65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0,65504.0 };
    mv::dynamic_vector<mv::float_type> edgeKData(k2rawData);
    auto bweights = test_cm5.constant(blurKData, mv::Shape(3, 3, 1, 1), mv::DType::Float, mv::Order::NWHC);
    auto eweights = test_cm5.constant(edgeKData, mv::Shape(3, 3, 1, 1), mv::DType::Float, mv::Order::NWHC);
    auto conv1 = test_cm5.conv2D(input5, bweights, {1, 1}, {0, 0, 0, 0});
    auto conv2 = test_cm5.conv2D(conv1, eweights, {1, 1}, {0, 0, 0, 0});
    auto output = test_cm5.output(conv2);

    // Check output shape
    EXPECT_EQ(output->getShape(), mv::Shape(252, 252, 1));

    mv::ControlModel cm5(test_cm5);

    // declare serializer as blob
    mv::Serializer gs5(mv::mvblob_mode);

    // serialize compute model to file
    uint64_t filesize5 = gs5.serialize(cm5, "test_conv_05.blob");

    // compare filesize written to expected
    EXPECT_EQ (1252, filesize5) << "ERROR: wrong blob size";

    // compare blob file contents to blob previously generated with mvNCCheck
    const char *command1 = "cp ../../tests/data/gold_05.blob .";
    EXPECT_EQ (0, system(command1)) << "ERROR: unable to copy file gold_05.blob to current folder";
    const char *command2 = "diff test_conv_05.blob gold_05.blob";
    EXPECT_EQ (0, system(command2)) << "ERROR: generated blob file contents do not match expected";

}
