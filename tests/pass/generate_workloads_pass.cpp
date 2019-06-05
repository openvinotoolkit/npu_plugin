#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/target/keembay/workloads.hpp"

/** Creates a Workloads instance*/
 mv::Workloads GenerateTestWorkloads_modelA1(mv::Data::TensorIterator& inputTensor, mv::MPE_Mode mode);
mv::Workloads GenerateTestWorkloads_modelA2(mv::Data::TensorIterator& inputTensor, mv::MPE_Mode mode);

mv::Workloads GenerateTestWorkloads_modelB1(mv::Data::TensorIterator& inputTensor, mv::MPE_Mode mode);
mv::Workloads GenerateTestWorkloads_modelB2(mv::Data::TensorIterator& inputTensor, mv::MPE_Mode mode);
mv::Workloads GenerateTestWorkloads_modelB4(mv::Data::TensorIterator& inputTensor, mv::MPE_Mode mode); 


/* TEST(generate_workloads_pass, costfunction_criticalpathA1)
{
    // Tests critical path, 1 workload (A)
    mv::OpModel om("testModel");
    auto input = om.input({56, 56, 64}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*64*64);
    auto weights = om.constant(weightsData, {1, 1, 64, 64}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});
    om.output(conv);

    mv::DataModel dm(om);
    auto resData = dm.getTensor("Conv_0:0");

    std::vector<mv::Data::TensorIterator> vectorTensors = {resData};
    mv::Workloads workloads = GenerateTestWorkloads_modelA1(resData, mv::MPE_Mode::Matrix);
    
    mv::CostFunctions costFunction = mv::CostFunctions::CriticalPath;
    workloads.generateExecutionCycles(vectorTensors, 1, costFunction);
    std::vector<float> results = workloads.getExecutionCycles();
    ASSERT_EQ(results[0], 784.0);
    ASSERT_EQ(results[1], 784.0);
}

TEST(generate_workloads_pass, costfunction_criticalpathA2)
{
    // Tests critical path, 2 workloads
    mv::OpModel om("testModel");
    auto input = om.input({56, 56, 64, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*64*64);
    auto weights = om.constant(weightsData, {1, 1, 64, 64}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});
    om.output(conv);

    mv::DataModel dm(om);
    auto resData = dm.getTensor("Conv_0:0");

    std::vector<mv::Data::TensorIterator> vectorTensors = {resData};
    mv::Workloads workloads = GenerateTestWorkloads_modelA2(resData, mv::MPE_Mode::Matrix);
    mv::CostFunctions costFunction = mv::CostFunctions::CriticalPath;

    workloads.generateExecutionCycles(vectorTensors, 1, costFunction);
    std::vector<float> results = workloads.getExecutionCycles();
    ASSERT_EQ(results[0], 784.0); //?
    ASSERT_EQ(results[1], 784.0); //?

    workloads.generateExecutionCycles(vectorTensors, 2, costFunction);
    results = workloads.getExecutionCycles();
    ASSERT_EQ(results[0], 784.0);
    ASSERT_EQ(results[1], 784.0);
}

TEST(generate_workloads_pass, execycles_workloadB1)
{
    // Tests 1 workload (B)
    mv::OpModel om("testModel");
    auto input = om.input({56, 56, 64, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*64*64);
    auto weights = om.constant(weightsData, {1, 1, 64, 64}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});
    om.output(conv);

    mv::DataModel dm(om);
    auto resData = dm.getTensor("Conv_0:0");

    std::vector<mv::Data::TensorIterator> vectorTensors = {resData};
    mv::Workloads workloads = GenerateTestWorkloads_modelB1(resData, mv::MPE_Mode::Matrix);
    
    mv::CostFunctions costFunction = mv::CostFunctions::CriticalPath;
    workloads.generateExecutionCycles(vectorTensors, 1, costFunction);
    std::vector<float> results = workloads.getExecutionCycles();
    ASSERT_EQ(results[0], 224.0);
    ASSERT_EQ(results[1], 224.0);

    costFunction = mv::CostFunctions::Balanced;
    workloads.generateExecutionCycles(vectorTensors, 1, costFunction);
    results = workloads.getExecutionCycles();
    ASSERT_EQ(results[0], -1.0);
    ASSERT_EQ(results[1], -1.0);

    costFunction = mv::CostFunctions::MinMaxWorkloads;
    workloads.generateExecutionCycles(vectorTensors, 1, costFunction);
    results = workloads.getExecutionCycles();
    ASSERT_EQ(results[0], 224.0);
    ASSERT_EQ(results[1], 448.0);
}

TEST(generate_workloads_pass, execycles_workloadB2)
{
    // Tests 2 workloads (B)
    mv::OpModel om("testModel");
    auto input = om.input({56, 56, 64, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*64*64);
    auto weights = om.constant(weightsData, {1, 1, 64, 64}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});
    om.output(conv);

    mv::DataModel dm(om);
    auto resData = dm.getTensor("Conv_0:0");

    std::vector<mv::Data::TensorIterator> vectorTensors = {resData};
    mv::Workloads workloads = GenerateTestWorkloads_modelB2(resData, mv::MPE_Mode::Matrix);
    
    mv::CostFunctions costFunction = mv::CostFunctions::CriticalPath;
    workloads.generateExecutionCycles(vectorTensors, 1, costFunction);
    std::vector<float> results = workloads.getExecutionCycles();
    ASSERT_EQ(results[0], 224.0);
    ASSERT_EQ(results[1], 224.0);

    costFunction = mv::CostFunctions::Balanced;
    workloads.generateExecutionCycles(vectorTensors, 1, costFunction);
    results = workloads.getExecutionCycles();
    ASSERT_EQ(results[0], -1.0);
    ASSERT_EQ(results[1], -1.0);

    costFunction = mv::CostFunctions::MinMaxWorkloads;
    workloads.generateExecutionCycles(vectorTensors, 1, costFunction);
    results = workloads.getExecutionCycles();
    ASSERT_EQ(results[0], 224.0);
    ASSERT_EQ(results[1], 336.0);
}

TEST(generate_workloads_pass, execycles_workloadB4)
{
    // Tests 2 workloads (B)
    mv::OpModel om("testModel");
    auto input = om.input({56, 56, 64, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*64*64);
    auto weights = om.constant(weightsData, {1, 1, 64, 64}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});
    om.output(conv);

    mv::DataModel dm(om);
    auto resData = dm.getTensor("Conv_0:0");

    std::vector<mv::Data::TensorIterator> vectorTensors = {resData};
    mv::Workloads workloads = GenerateTestWorkloads_modelB4(resData, mv::MPE_Mode::Matrix);
    
    mv::CostFunctions costFunction = mv::CostFunctions::CriticalPath;
    workloads.generateExecutionCycles(vectorTensors, 1, costFunction);
    std::vector<float> results = workloads.getExecutionCycles();
    ASSERT_EQ(results[0], INFINITY);
    ASSERT_EQ(results[1], INFINITY);

    costFunction = mv::CostFunctions::Balanced;
    workloads.generateExecutionCycles(vectorTensors, 1, costFunction);
    results = workloads.getExecutionCycles();
    ASSERT_EQ(results[0], 0.0);
    ASSERT_EQ(results[1], 0.0);

    costFunction = mv::CostFunctions::MinMaxWorkloads;
    workloads.generateExecutionCycles(vectorTensors, 1, costFunction);
    results = workloads.getExecutionCycles();
    ASSERT_EQ(results[0], INFINITY);
    ASSERT_EQ(results[1], INFINITY);
}

TEST(generate_workloads_pass, execycles_workloadB1_vector)
{
    // Tests 1 workload (B)
    mv::OpModel om("testModel");
    auto input = om.input({56, 56, 64, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*64*64);
    auto weights = om.constant(weightsData, {1, 1, 64, 64}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});
    om.output(conv);

    mv::DataModel dm(om);
    auto resData = dm.getTensor("Conv_0:0");

    std::vector<mv::Data::TensorIterator> vectorTensors = {resData};
    mv::Workloads workloads = GenerateTestWorkloads_modelB1(resData, mv::MPE_Mode::Vector);
    
    mv::CostFunctions costFunction = mv::CostFunctions::CriticalPath;
    workloads.generateExecutionCycles(vectorTensors, 1, costFunction);
    std::vector<float> results = workloads.getExecutionCycles();
    ASSERT_EQ(results[0], 224.0);
    ASSERT_EQ(results[1], 224.0);

    costFunction = mv::CostFunctions::Balanced;
    workloads.generateExecutionCycles(vectorTensors, 1, costFunction);
    results = workloads.getExecutionCycles();
    ASSERT_EQ(results[0], -1.0);
    ASSERT_EQ(results[1], -1.0);

    costFunction = mv::CostFunctions::MinMaxWorkloads;
    workloads.generateExecutionCycles(vectorTensors, 1, costFunction);
    results = workloads.getExecutionCycles();
    ASSERT_EQ(results[0], 224.0);
    ASSERT_EQ(results[1], 448.0);
}
 */


/* TEST(generate_workloads_pass, costfunction_greedyA)
{
    int nDPUxCluster = 4;
    std::vector<float> workloadCosts {112.0, 112.0, 112.0, 56.0, 112.0, 112.0, 112.0, 56.0};
    float result = mv::Workloads::greedyTaskAssignment(nDPUxCluster, workloadCosts);

    ASSERT_EQ(result, 224);
}

TEST(generate_workloads_pass, costfunction_greedyB)
{
    int nDPUxCluster = 4;
    std::vector<float> workloadCosts {196.0, 196.0, 196.0, 196.0};
    float result = mv::Workloads::greedyTaskAssignment(nDPUxCluster, workloadCosts);

    ASSERT_EQ(result, 196);
} */

/* TEST(generate_workloads_pass, validate_methodB1)
{
    // Validates 1 workload (B)
    mv::OpModel om("testModel");
    auto input = om.input({56, 56, 64, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*64*64);
    auto weights = om.constant(weightsData, {1, 1, 64, 64}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});
    om.output(conv);

    mv::DataModel dm(om);
    auto resData = dm.getTensor("Conv_0:0");

    std::vector<mv::Data::TensorIterator> vectorTensors = {resData};
    mv::Workloads workloads = GenerateTestWorkloads_modelB1(resData, mv::MPE_Mode::Matrix);

    bool result = workloads.validateWorkloads(vectorTensors);
    ASSERT_TRUE(result);
}

TEST(generate_workloads_pass, validateB2)
{
    // Validates 2 workloads (B)
    mv::OpModel om("testModel");
    auto input = om.input({56, 56, 64, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*64*64);
    auto weights = om.constant(weightsData, {1, 1, 64, 64}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});
    om.output(conv);

    mv::DataModel dm(om);
    auto resData = dm.getTensor("Conv_0:0");

    std::vector<mv::Data::TensorIterator> vectorTensors = {resData};
    mv::Workloads workloads = GenerateTestWorkloads_modelB2(resData, mv::MPE_Mode::Matrix);

    bool result = workloads.validateWorkloads(vectorTensors);
    ASSERT_TRUE(result);
} */

// TEST(generate_workloads_pass, ReadTensorSplitAlgorithms)
// {
//     //Setup Comp Descriptor
//     mv::CompilationUnit unit("testModel");
//     unit.loadTargetDescriptor(mv::Target::ma2490);
//     std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
//     unit.loadCompilationDescriptor(compDescPath);
//     mv::Element testPassDesc("GenerateWorkloads");
//     testPassDesc.set("TensorSplitAlgorithms", std::string("Metis,Rectangle,Z-Tiling"));

//     //Setup Workloads object
//     mv::OpModel om("testModel");
//     auto input = om.input({56, 56, 64, 1}, mv::DType("Float16"), mv::Order("NCHW"));
//     std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*64*64);
//     auto weights = om.constant(weightsData, {1, 1, 64, 64}, mv::DType("Float16"), mv::Order("NCWH"));
//     auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});
//     om.output(conv);

//     mv::DataModel dm(om);
//     auto resData = dm.getTensor("Conv_0:0");

//     std::vector<mv::Data::TensorIterator> vectorTensors = {resData};
//     mv::Workloads workloads = GenerateTestWorkloads_modelB1(resData, mv::MPE_Mode::Matrix);

//     std::vector<std::string>actual_results = workloads.getTensorSplitAlgorithms(testPassDesc);
    
//     ASSERT_EQ(actual_results[0], "Metis");
//     ASSERT_EQ(actual_results[1], "Rectangle");
//     ASSERT_EQ(actual_results[2], "Z-Tiling");
// }

// TEST(generate_workloads_pass, ReadTensorSplitAlgorithmsDefault)
// {
//     //Setup Comp Descriptor
//     mv::CompilationUnit unit("testModel");
//     unit.loadTargetDescriptor(mv::Target::ma2490);
//     std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
//     unit.loadCompilationDescriptor(compDescPath);
//     mv::Element testPassDesc("GenerateWorkloads");
//     testPassDesc.set("TensorSplitAlgorithms", std::string(""));

//     //Setup Workloads object
//     mv::OpModel om("testModel");
//     auto input = om.input({56, 56, 64, 1}, mv::DType("Float16"), mv::Order("NCHW"));
//     std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*64*64);
//     auto weights = om.constant(weightsData, {1, 1, 64, 64}, mv::DType("Float16"), mv::Order("NCWH"));
//     auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});
//     om.output(conv);

//     mv::DataModel dm(om);
//     auto resData = dm.getTensor("Conv_0:0");

//     std::vector<mv::Data::TensorIterator> vectorTensors = {resData};
//     mv::Workloads workloads = GenerateTestWorkloads_modelB1(resData, mv::MPE_Mode::Matrix);

//     std::vector<std::string>actual_results = workloads.getTensorSplitAlgorithms(testPassDesc);
    
//     ASSERT_EQ(actual_results[0], "Metis");
//     ASSERT_EQ(actual_results[1], "Rectangle");
//     ASSERT_EQ(actual_results[2], "Z-Tiling");
// }

// TEST(generate_workloads_pass, ReadTensorSplitAlgorithmsOne)
// {
//     //Setup Comp Descriptor
//     mv::CompilationUnit unit("testModel");
//     unit.loadTargetDescriptor(mv::Target::ma2490);
//     std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
//     unit.loadCompilationDescriptor(compDescPath);
//     mv::Element testPassDesc("GenerateWorkloads");
//     testPassDesc.set("TensorSplitAlgorithms", std::string("Rectangle"));

//     //Setup Workloads object
//     mv::OpModel om("testModel");
//     auto input = om.input({56, 56, 64, 1}, mv::DType("Float16"), mv::Order("NCHW"));
//     std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*64*64);
//     auto weights = om.constant(weightsData, {1, 1, 64, 64}, mv::DType("Float16"), mv::Order("NCWH"));
//     auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});
//     om.output(conv);

//     mv::DataModel dm(om);
//     auto resData = dm.getTensor("Conv_0:0");

//     std::vector<mv::Data::TensorIterator> vectorTensors = {resData};
//     mv::Workloads workloads = GenerateTestWorkloads_modelB1(resData, mv::MPE_Mode::Matrix);

//     std::vector<std::string>actual_results = workloads.getTensorSplitAlgorithms(testPassDesc);
    
//     ASSERT_EQ(actual_results.size(), 1);
//     ASSERT_EQ(actual_results[0], "Rectangle");
// }

TEST(generate_workloads_pass, ReadCostFunctions)
{
    //Setup Comp Descriptor
    mv::CompilationUnit unit("testModel");
    unit.loadTargetDescriptor(mv::Target::ma2490);
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::Element testPassDesc("GenerateWorkloads");
    testPassDesc.set("costfunction", std::string("criticalpath"));

    //Setup Workloads object
    mv::OpModel om("testModel");
    auto input = om.input({56, 56, 64, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*64*64);
    auto weights = om.constant(weightsData, {1, 1, 64, 64}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});
    om.output(conv);

    mv::DataModel dm(om);
    auto resData = dm.getTensor("Conv_0:0");

    std::vector<mv::Data::TensorIterator> vectorTensors = {resData};
    mv::Workloads workloads = GenerateTestWorkloads_modelB1(resData, mv::MPE_Mode::Matrix);

    mv::CostFunctions actual_result = workloads.getCostFunction(testPassDesc);
    
    ASSERT_EQ(actual_result, mv::CostFunctions::CriticalPath);
}

TEST(generate_workloads_pass, ReadCostFunctionParse)
{
    //Setup Comp Descriptor
    mv::CompilationUnit unit("testModel");
    unit.loadTargetDescriptor(mv::Target::ma2490);
    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490.json";
    unit.loadCompilationDescriptor(compDescPath);
    mv::Element testPassDesc("GenerateWorkloads");
    testPassDesc.set("costfunction", std::string("not recognized"));

    //Setup Workloads object
    mv::OpModel om("testModel");
    auto input = om.input({56, 56, 64, 1}, mv::DType("Float16"), mv::Order("NCHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*64*64);
    auto weights = om.constant(weightsData, {1, 1, 64, 64}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});
    om.output(conv);

    mv::DataModel dm(om);
    auto resData = dm.getTensor("Conv_0:0");

    std::vector<mv::Data::TensorIterator> vectorTensors = {resData};
    mv::Workloads workloads = GenerateTestWorkloads_modelB1(resData, mv::MPE_Mode::Matrix);

    mv::CostFunctions actual_result = workloads.getCostFunction(testPassDesc);
    
    // Not recognised should return default value of "Balanced"
    ASSERT_EQ(actual_result, mv::CostFunctions::Balanced);
}

/* >>>>>>>>>>>>>>>>>>>> helper functions to create workloads <<<<<<<<<<<<<<<<<<<<<<<< */

/** Creates a 1 x Workloads instance*/
mv::Workloads GenerateTestWorkloads_modelA1(mv::Data::TensorIterator& inputTensor, mv::MPE_Mode mode)
{
    mv::DPUMode dpu_mode = {4,4};
    if (mode == mv::Vector)
    {
        dpu_mode = {1, 16};
    }
    mv::Workloads workloads("ModelA", inputTensor->getShape() , dpu_mode);
    
    //0
    mv::Workload workload;
    workload.workloadID = 0;
    workload.clusterID = 0; 
    workload.padTop = 0;    
    workload.padBottom = 0; 
    workload.padLeft = 0;   
    workload.padRight = 0;  
    workload.MPEMode = mode;
    workload.MinX = 0;
    workload.MinY = 0;
    workload.MinZ = 0;      
    workload.MaxX = 56;
    workload.MaxY = 56;
    workload.MaxZ = 64;     
    workloads.addWorkload(workload);

    return workloads;
}

/** Creates a 2 x Workloads instance*/
mv::Workloads GenerateTestWorkloads_modelA2(mv::Data::TensorIterator& inputTensor, mv::MPE_Mode mode)
{
    mv::DPUMode dpu_mode = {4,4};
    if (mode == mv::Vector)
    {
        dpu_mode = {1, 16};
    }
    mv::Workloads workloads("ModelA", inputTensor->getShape() , dpu_mode);

    //0
    mv::Workload workload;
    workload.workloadID = 0;
    workload.clusterID = 0; 
    workload.padTop = 0;    
    workload.padBottom = 0; 
    workload.padLeft = 0;   
    workload.padRight = 0;  
    workload.MPEMode = mode;
    workload.MinX = 0;
    workload.MinY = 0;
    workload.MinZ = 0;      
    workload.MaxX = 28;
    workload.MaxY = 56;
    workload.MaxZ = 64;     
    workloads.addWorkload(workload);

    //1
    mv::Workload workload1;
    workload1.workloadID = 1;
    workload1.clusterID = 0; 
    workload1.padTop = 0;    
    workload1.padBottom = 0; 
    workload1.padLeft = 0;   
    workload1.padRight = 0;  
    workload1.MPEMode = mode;
    workload1.MinX = 28;
    workload1.MinY = 0;
    workload1.MinZ = 0; 
    workload1.MaxX = 56;
    workload1.MaxY = 56;
    workload1.MaxZ = 64;
    workloads.addWorkload(workload1);

    return workloads;
}

/** Creates a 4 x Workloads instance*/
mv::Workloads GenerateTestWorkloads_modelA4(mv::Data::TensorIterator& inputTensor, mv::MPE_Mode mode)
{
    mv::DPUMode dpu_mode = {4,4};
    if (mode == mv::Vector)
    {
        dpu_mode = {1, 16};
    }
    mv::Workloads workloads("ModelA", inputTensor->getShape() , dpu_mode);
    
    //0
    mv::Workload workload0;
    workload0.workloadID = 0;
    workload0.clusterID = 0; 
    workload0.padTop = 0;    
    workload0.padBottom = 0; 
    workload0.padLeft = 0;   
    workload0.padRight = 0;  
    workload0.MPEMode = mode;
    workload0.MinX = 0;
    workload0.MinY = 28;
    workload0.MinZ = 0;      
    workload0.MaxX = 28;
    workload0.MaxY = 32;
    workload0.MaxZ = 64;
    workloads.addWorkload(workload0);

    //1
    mv::Workload workload1;
    workload1.workloadID = 1;
    workload1.clusterID = 0; 
    workload1.padTop = 0;    
    workload1.padBottom = 0; 
    workload1.padLeft = 0;   
    workload1.padRight = 0;  
    workload1.MPEMode = mode;
    workload1.MinX = 28;
    workload1.MinY = 0;
    workload1.MinZ = 0;      
    workload1.MaxX = 56;
    workload1.MaxY = 28;
    workload1.MaxZ = 64;     
    workloads.addWorkload(workload1);

    //2
    mv::Workload workload2;
    workload2.workloadID = 2;
    workload2.clusterID = 0; 
    workload2.padTop = 0;    
    workload2.padBottom = 0; 
    workload2.padLeft = 0;   
    workload2.padRight = 0;  
    workload2.MPEMode = mode;
    workload2.MinX = 16;
    workload2.MinY = 32;
    workload2.MinZ = 0;      
    workload2.MaxX = 28;
    workload2.MaxY = 56;
    workload2.MaxZ = 64;     
    workloads.addWorkload(workload2);

    //3
    mv::Workload workload3;
    workload3.workloadID = 3;
    workload3.clusterID = 0; 
    workload3.padTop = 0;    
    workload3.padBottom = 0; 
    workload3.padLeft = 0;   
    workload3.padRight = 0;  
    workload3.MPEMode = mode;
    workload3.MinX = 28;
    workload3.MinY = 28;
    workload3.MinZ = 0;      
    workload3.MaxX = 56;
    workload3.MaxY = 56;
    workload3.MaxZ = 64;     
    workloads.addWorkload(workload3);

    return workloads;
}

/** Creates a 1 Workloads instance (model B)*/
mv::Workloads GenerateTestWorkloads_modelB1(mv::Data::TensorIterator& inputTensor, mv::MPE_Mode mode)
{
    mv::DPUMode dpu_mode = {4,4};
    if (mode == mv::Vector)
    {
        dpu_mode = {1, 16};
    }
    mv::Workloads workloads("ModelB", inputTensor->getShape() , dpu_mode);
    
    //0
    mv::Workload workload;
    workload.workloadID = 0;
    workload.clusterID = 0;           
    workload.padTop = 0;              
    workload.padBottom = 0;           
    workload.padLeft = 0;             
    workload.padRight = 0;            
    workload.MPEMode = mode;
    workload.MinX = 0;
    workload.MinY = 0;
    workload.MinZ = 0; 
    workload.MaxX = 56;
    workload.MaxY = 14;
    workload.MaxZ = 64; 
    
    workloads.addWorkload(workload); 
    return workloads;
}

/** Creates a 2 Workloads instance (model B)*/
mv::Workloads GenerateTestWorkloads_modelB2(mv::Data::TensorIterator& inputTensor, mv::MPE_Mode mode)
{
    mv::DPUMode dpu_mode = {4,4};
    if (mode == mv::Vector)
    {
        dpu_mode = {1, 16};
    }
    mv::Workloads workloads("ModelB", inputTensor->getShape() , dpu_mode);
    
    //0
    mv::Workload workload;
    workload.workloadID = 0;
    workload.clusterID = 0;           
    workload.padTop = 0;              
    workload.padBottom = 0;           
    workload.padLeft = 0;             
    workload.padRight = 0;            
    workload.MPEMode = mode;
    workload.MinX = 0;
    workload.MinY = 0;
    workload.MinZ = 0; 
    workload.MaxX = 28;
    workload.MaxY = 14;
    workload.MaxZ = 64; 
    workloads.addWorkload(workload); 

    //1
    mv::Workload workload1;
    workload1.workloadID = 1;
    workload1.clusterID = 0;           
    workload1.padTop = 0;              
    workload1.padBottom = 0;           
    workload1.padLeft = 0;             
    workload1.padRight = 0;            
    workload1.MPEMode = mode;
    workload1.MinX = 28;
    workload1.MinY = 0;
    workload1.MinZ = 0; 
    workload1.MaxX = 56;
    workload1.MaxY = 14;
    workload1.MaxZ = 64; 
    workloads.addWorkload(workload1);

    return workloads;
}

/** Creates a 4 Workloads instance (model B)*/
mv::Workloads GenerateTestWorkloads_modelB4(mv::Data::TensorIterator& inputTensor, mv::MPE_Mode mode)
{
    mv::DPUMode dpu_mode = {4,4};
    if (mode == mv::Vector)
    {
        dpu_mode = {1, 16};
    }
    mv::Workloads workloads("ModelB", inputTensor->getShape() , dpu_mode);
    
    //0
    mv::Workload workload0;
    workload0.workloadID = 0;
    workload0.clusterID = 0;           
    workload0.padTop = 0;              
    workload0.padBottom = 0;           
    workload0.padLeft = 0;             
    workload0.padRight = 0;            
    workload0.MPEMode = mv::Matrix;    //Matrix is MPE Mode (4,4)
    workload0.MinX = 0;
    workload0.MinY = 8;
    workload0.MinZ = 0; 
    workload0.MaxX = 16;
    workload0.MaxY = 14;
    workload0.MaxZ = 64; 
    workloads.addWorkload(workload0);

    //1
    mv::Workload workload1;
    workload1.workloadID = 1;
    workload1.clusterID = 0;           
    workload1.padTop = 0;              
    workload1.padBottom = 0;           
    workload1.padLeft = 0;             
    workload1.padRight = 0;            
    workload1.MPEMode = mv::Matrix;    //Matrix is MPE Mode (4,4)
    workload1.MinX = 16;
    workload1.MinY = 12;
    workload1.MinZ = 0; 
    workload1.MaxX = 28;
    workload1.MaxY = 14;
    workload1.MaxZ = 64; 
    workloads.addWorkload(workload1);

    //2
    mv::Workload workload2;
    workload2.workloadID = 2;
    workload2.clusterID = 0;           
    workload2.padTop = 0;              
    workload2.padBottom = 0;           
    workload2.padLeft = 0;             
    workload2.padRight = 0;            
    workload2.MPEMode = mv::Matrix;    //Matrix is MPE Mode (4,4)
    workload2.MinX = 28;
    workload2.MinY = 8;
    workload2.MinZ = 0; 
    workload2.MaxX = 44;
    workload2.MaxY = 14;
    workload2.MaxZ = 64; 
    workloads.addWorkload(workload2);

    //3
    mv::Workload workload3;
    workload3.workloadID = 3;
    workload3.clusterID = 0;           
    workload3.padTop = 0;              
    workload3.padBottom = 0;           
    workload3.padLeft = 0;             
    workload3.padRight = 0;            
    workload3.MPEMode = mv::Matrix;    //Matrix is MPE Mode (4,4)
    workload3.MinX = 44;
    workload3.MinY = 12;
    workload3.MinZ = 0; 
    workload3.MaxX = 56;
    workload3.MaxY = 14;
    workload3.MaxZ = 64; 
    workloads.addWorkload(workload3);

    return workloads;
}
