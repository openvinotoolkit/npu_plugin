#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"

TEST(runtime_serialize, root_access)
{
    const std::string FILEPATH("/root/blob.bin");

    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({16, 16, 15, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{},{},{},{}}, "input#3");

    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*15*15);
    auto weights = om.constant(weightsData, {1, 1, 15, 15}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});

    std::vector<double> weightsData1 = mv::utils::generateSequence<double>(1*1*15*15);
    auto weights1 = om.constant(weightsData1, {1, 1, 15, 15}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(conv, weights1, {1, 1}, {0, 0, 0, 0});

    om.output(conv1);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("GenerateBlobKmb", "output", FILEPATH);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    
    try {
        unit.initialize();
        unit.run();
        FAIL() << "Expected mv::Runtime File location invalid not thrown";
    } 
    catch(mv::ArgumentError const & err) {
        ASSERT_TRUE(true);
    }
    catch(const std::exception& e) {
        FAIL() << "Different Exception occurrs" << e.what();
    }
}

TEST(runtime_serialize, higher_access)
{
    const std::string FILEPATH("../b/c/blob.bin");

    //Setup Comp Descriptor
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({16, 16, 15, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{},{},{},{}}, "input#3");

    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*15*15);
    auto weights = om.constant(weightsData, {1, 1, 15, 15}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});

    std::vector<double> weightsData1 = mv::utils::generateSequence<double>(1*1*15*15);
    auto weights1 = om.constant(weightsData1, {1, 1, 15, 15}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(conv, weights1, {1, 1}, {0, 0, 0, 0});

    om.output(conv1);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("GenerateBlobKmb", "output", FILEPATH);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    
    try {
        unit.initialize();
        unit.run();
        FAIL() << "Expected mv::Runtime File location invalid not thrown";
    } 
    catch(mv::ArgumentError const & err) {
        ASSERT_TRUE(true);
    }
    catch(const std::exception& e) {
        FAIL() << "Different Exception occurrs" << e.what();
    }
}

TEST(runtime_serialize, no_folders)
{
    const std::string FILEPATH("blob.bin");

    //Setup Comp Descriptor
    //Setup Comp Descriptor
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({16, 16, 15, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{},{},{},{}}, "input#3");

    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*15*15);
    auto weights = om.constant(weightsData, {1, 1, 15, 15}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});

    std::vector<double> weightsData1 = mv::utils::generateSequence<double>(1*1*15*15);
    auto weights1 = om.constant(weightsData1, {1, 1, 15, 15}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(conv, weights1, {1, 1}, {0, 0, 0, 0});

    om.output(conv1);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("GenerateBlobKmb", "output", FILEPATH);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();    
    unit.run();
    
    // test file exists
    std::ifstream infile(FILEPATH);
    if(infile.good())
        ASSERT_TRUE(true);
    else
        FAIL() << "Blob file not found";
}


TEST(runtime_serialize, 3_folders_notexist)
{
    const std::string FILEPATH("a/b/c/blob.bin");

    //Setup Comp Descriptor
    //Setup Comp Descriptor
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({16, 16, 15, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{},{},{},{}}, "input#3");

    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*15*15);
    auto weights = om.constant(weightsData, {1, 1, 15, 15}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});

    std::vector<double> weightsData1 = mv::utils::generateSequence<double>(1*1*15*15);
    auto weights1 = om.constant(weightsData1, {1, 1, 15, 15}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(conv, weights1, {1, 1}, {0, 0, 0, 0});

    om.output(conv1);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("GenerateBlobKmb", "output", FILEPATH);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
    
    // test file exists
    std::ifstream infile(FILEPATH);
    if(infile.good())
        ASSERT_TRUE(true);
    else
        FAIL() << "Blob file not found";
}

TEST(runtime_serialize, 3_folders_some_exist)
{
    // create some of the dir in advance
    const std::string FILEPATH("a/b/c/blob.bin");
    mkdir("a", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir("a/b", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    //Setup Comp Descriptor
    //Setup Comp Descriptor
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    auto input = om.input({16, 16, 15, 1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{},{},{},{}}, "input#3");

    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*15*15);
    auto weights = om.constant(weightsData, {1, 1, 15, 15}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});

    std::vector<double> weightsData1 = mv::utils::generateSequence<double>(1*1*15*15);
    auto weights1 = om.constant(weightsData1, {1, 1, 15, 15}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv1 = om.conv(conv, weights1, {1, 1}, {0, 0, 0, 0});

    om.output(conv1);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::CompilationDescriptor &compDesc = unit.compilationDescriptor();
    compDesc.setPassArg("GenerateBlobKmb", "output", FILEPATH);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
    
    // test file exists
    std::ifstream infile(FILEPATH);
    if(infile.good())
        ASSERT_TRUE(true);
    else
        FAIL() << "Blob file not found";
}

