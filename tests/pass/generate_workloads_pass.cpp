#include "gtest/gtest.h"
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/target/keembay/workloads.hpp"
//#include "src/pass/finalization/generate_workloads_pass.cpp"

/** Creates a Workloads instance*/
mv::Workloads GenerateTestWorkloads_modelA();
mv::Data::TensorIterator GetTestTensor_modelA();
mv::Workloads GenerateTestWorkloads_modelB();
mv::Data::TensorIterator GetTestTensor_modelB();


TEST(generate_workloads_pass, generateWorkloadsFcn_Greedy)
{
    mv::CompilationUnit unit("testModel");
    mv::OpModel& om = unit.model();

    /*Working*/
    auto input = om.input({16, 16, 15}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*15*15);
    auto weights = om.constant(weightsData, {1, 1, 15, 15}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});
    om.output(conv);

    mv::Element dummyPassDesc("GenerateWorkloads");
    dummyPassDesc.set("costfunction", std::string("greedy"));

    mv::TargetDescriptor dummyTargDesc;
    mv::json::Object compOutput;
    mv::pass::PassRegistry::instance().find("GenerateWorkloads")->run(om, dummyTargDesc, dummyPassDesc, compOutput);

    ASSERT_TRUE(true);
}

/*
TODO: Cannot test internal methods of a Pass 
causes a "Duplicated Registry entry" when the *pass.cpp file included

TEST(generate_workloads_pass, costfunction_balancedA)
{
    int nDPUxCluster = 2;
    mv::Data::TensorIterator outputTensor = GetTestTensor_modelA();
    std::vector<mv::Data::TensorIterator> vectorTensors = {outputTensor};
    mv::Workloads workloads = GenerateTestWorkloads_modelA();
    CostFunctions costFunction = CostFunctions::Balanced;

    std::vector<float> results = getExecutionCycles(vectorTensors, workloads, nDPUxCluster, {4,4}, costFunction);

    ASSERT_EQ(results[0], -1);
    ASSERT_EQ(results[1], -1);
}

TEST(generate_workloads_pass, costfunction_balancedB)
{
    int nDPUxCluster = 4;
    mv::Data::TensorIterator outputTensor = GetTestTensor_modelB();
    std::vector<mv::Data::TensorIterator> vectorTensors = {outputTensor};
    mv::Workloads workloads = GenerateTestWorkloads_modelB();
    CostFunctions costFunction = CostFunctions::Balanced;

    std::vector<float> results = getExecutionCycles(vectorTensors, workloads, nDPUxCluster, {4,4}, costFunction);

    ASSERT_EQ(results[0], -1);
    ASSERT_EQ(results[1], -1);
}

TEST(generate_workloads_pass, costfunction_criticalpathA)
{
    int nDPUxCluster = 2;
    mv::Data::TensorIterator outputTensor = GetTestTensor_modelA();
    std::vector<mv::Data::TensorIterator> vectorTensors = {outputTensor};
    mv::Workloads workloads = GenerateTestWorkloads_modelA();
    CostFunctions costFunction = CostFunctions::CriticalPath;

    std::vector<float> results = getExecutionCycles(vectorTensors, workloads, nDPUxCluster, {4,4}, costFunction);

    ASSERT_EQ(results[0], 8);
    ASSERT_EQ(results[1], 8);
}


TEST(generate_workloads_pass, costfunction_criticalpathB)
{
    int nDPUxCluster = 4;
    mv::Data::TensorIterator outputTensor = GetTestTensor_modelB();
    std::vector<mv::Data::TensorIterator> vectorTensors = {outputTensor};
    mv::Workloads workloads = GenerateTestWorkloads_modelB();
    CostFunctions costFunction = CostFunctions::CriticalPath;

    std::vector<float> results = getExecutionCycles(vectorTensors, workloads, nDPUxCluster, {4,4}, costFunction);

    ASSERT_EQ(results[0], 392);
    ASSERT_EQ(results[1], 392);
}

TEST(generate_workloads_pass, costfunction_minmaxA)
{
    int nDPUxCluster = 2;
    mv::Data::TensorIterator outputTensor = GetTestTensor_modelA();
    std::vector<mv::Data::TensorIterator> vectorTensors = {outputTensor};
    mv::Workloads workloads = GenerateTestWorkloads_modelA();
    CostFunctions costFunction = CostFunctions::MinMaxWorkloads;

    std::vector<float> results = getExecutionCycles(vectorTensors, workloads, nDPUxCluster, {4,4}, costFunction);

    ASSERT_EQ(results[0], 4);
    ASSERT_EQ(results[1], 8);
}

TEST(generate_workloads_pass, costfunction_minmaxB)
{
    int nDPUxCluster = 4;
    mv::Data::TensorIterator outputTensor = GetTestTensor_modelB();
    std::vector<mv::Data::TensorIterator> vectorTensors = {outputTensor};
    mv::Workloads workloads = GenerateTestWorkloads_modelB();
    CostFunctions costFunction = CostFunctions::MinMaxWorkloads;

    std::vector<float> results = getExecutionCycles(vectorTensors, workloads, nDPUxCluster, {4,4}, costFunction);

    ASSERT_EQ(results[0], 196);
    ASSERT_EQ(results[1], 392);
}

TEST(generate_workloads_pass, costfunction_greedyA)
{
    int nDPUxCluster = 4;
    std::vector<float> workloadCosts {112.0, 112.0, 112.0, 56.0, 112.0, 112.0, 112.0, 56.0};
    float result = greedyTaskAssignment(nDPUxCluster, workloadCosts);

    ASSERT_EQ(result, 224);
}

TEST(generate_workloads_pass, costfunction_greedyB)
{
    int nDPUxCluster = 4;
    std::vector<float> workloadCosts {196.0, 196.0, 196.0, 196.0};
    float result = greedyTaskAssignment(nDPUxCluster, workloadCosts);

    ASSERT_EQ(result, 196);
}

TEST(generate_workloads_pass, validate_methodA)
{
    int nDPUxCluster = 2;
    mv::Data::TensorIterator outputTensor = GetTestTensor_modelA();
    std::vector<mv::Data::TensorIterator> vectorTensors = {outputTensor};
    mv::Workloads workloads = GenerateTestWorkloads_modelA();

    bool result = validateWorkloads(vectorTensors, workloads);
    ASSERT_TRUE(result);
}

TEST(generate_workloads_pass, validate_methodB)
{
    int nDPUxCluster = 2;
    mv::Data::TensorIterator outputTensor = GetTestTensor_modelB();
    std::vector<mv::Data::TensorIterator> vectorTensors {outputTensor};
    mv::Workloads workloads = GenerateTestWorkloads_modelB();

    bool result = validateWorkloads(vectorTensors, workloads);
    ASSERT_TRUE(result);
}
*/

/** Creates a Tensor for testing*/
mv::Data::TensorIterator GetTestTensor_modelA()
{
    mv::OpModel om("testModel");
    auto input = om.input({16, 16, 15}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(1*1*16*15);
    auto weights = om.constant(weightsData, {1, 1, 15, 16}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});
    om.output(conv);

    mv::DataModel dm(om);
    auto resData = dm.getTensor("DPU_Conv_0");

    return resData;
}

/** Creates a Tensor for testing*/
mv::Data::TensorIterator GetTestTensor_modelB()
{
    mv::OpModel om("testModel");
    auto input = om.input({56, 56, 3}, mv::DType("Float16"), mv::Order("CHW"));
    std::vector<double> weightsData = mv::utils::generateSequence<double>(64*3*3*3);
    auto weights = om.constant(weightsData, {64, 3, 3, 3}, mv::DType("Float16"), mv::Order("NCWH"));
    auto conv = om.conv(input, weights, {1, 1}, {0, 0, 0, 0});
    om.output(conv);

    mv::DataModel dm(om);
    auto resData = dm.getTensor("DPU_Conv_0");

    return resData;
}

/** Creates a Workloads instance*/
mv::Workloads GenerateTestWorkloads_modelA()
{
    mv::Workloads workloads("test");
    
    //0
    workloads.getWorkloads().push_back(mv::Workload()); 
    workloads.getWorkloads()[0].workloadID = 0;
    workloads.getWorkloads()[0].clusterID = 0;           //WW09 deliverbale is 1 cluster
    workloads.getWorkloads()[0].padTop = 0;              //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[0].padBottom = 0;           //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[0].padLeft = 0;             //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[0].padRight = 0;            //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[0].MPEMode = mv::Matrix;    //Matrix is MPE Mode (4,4)
    workloads.getWorkloads()[0].MinX = 0;
    workloads.getWorkloads()[0].MinY = 0;
    workloads.getWorkloads()[0].MinZ = 0;                //WW09 deliverbale is less than 16 channels
    workloads.getWorkloads()[0].MaxX = 7;
    workloads.getWorkloads()[0].MaxY = 7;
    workloads.getWorkloads()[0].MaxZ = 15;               //WW09 deliverbale is less than 16 channels

    //1
    workloads.getWorkloads().push_back(mv::Workload()); 
    workloads.getWorkloads()[1].workloadID = 1;
    workloads.getWorkloads()[1].clusterID = 0;           //WW09 deliverbale is 1 cluster
    workloads.getWorkloads()[1].padTop = 0;              //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[1].padBottom = 0;           //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[1].padLeft = 0;             //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[1].padRight = 0;            //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[1].MPEMode = mv::Matrix;    //Matrix is MPE Mode (4,4)
    workloads.getWorkloads()[1].MinX = 8;
    workloads.getWorkloads()[1].MinY = 0;
    workloads.getWorkloads()[1].MinZ = 0;                //WW09 deliverbale is less than 16 channels
    workloads.getWorkloads()[1].MaxX = 15;
    workloads.getWorkloads()[1].MaxY = 7;
    workloads.getWorkloads()[1].MaxZ = 15;               //WW09 deliverbale is less than 16 channels

    //2
    workloads.getWorkloads().push_back(mv::Workload()); 
    workloads.getWorkloads()[2].workloadID = 2;
    workloads.getWorkloads()[2].clusterID = 0;           //WW09 deliverbale is 1 cluster
    workloads.getWorkloads()[2].padTop = 0;              //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[2].padBottom = 0;           //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[2].padLeft = 0;             //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[2].padRight = 0;            //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[2].MPEMode = mv::Matrix;    //Matrix is MPE Mode (4,4)
    workloads.getWorkloads()[2].MinX = 8;
    workloads.getWorkloads()[2].MinY = 8;
    workloads.getWorkloads()[2].MinZ = 0;                //WW09 deliverbale is less than 16 channels
    workloads.getWorkloads()[2].MaxX = 15;
    workloads.getWorkloads()[2].MaxY = 15;
    workloads.getWorkloads()[2].MaxZ = 15;               //WW09 deliverbale is less than 16 channels

    //3
    workloads.getWorkloads().push_back(mv::Workload()); 
    workloads.getWorkloads()[3].workloadID = 3;
    workloads.getWorkloads()[3].clusterID = 0;           //WW09 deliverbale is 1 cluster
    workloads.getWorkloads()[3].padTop = 0;              //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[3].padBottom = 0;           //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[3].padLeft = 0;             //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[3].padRight = 0;            //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[3].MPEMode = mv::Matrix;    //Matrix is MPE Mode (4,4)
    workloads.getWorkloads()[3].MinX = 0;
    workloads.getWorkloads()[3].MinY = 8;
    workloads.getWorkloads()[3].MinZ = 0;                //WW09 deliverbale is less than 16 channels
    workloads.getWorkloads()[3].MaxX = 7;
    workloads.getWorkloads()[3].MaxY = 15;
    workloads.getWorkloads()[3].MaxZ = 15;               //WW09 deliverbale is less than 16 channels

    return workloads;
}

/** Creates a Workloads instance*/
mv::Workloads GenerateTestWorkloads_modelB()
{
    mv::Workloads workloads("test");
    
    //0
    workloads.getWorkloads().push_back(mv::Workload()); 
    workloads.getWorkloads()[0].workloadID = 0;
    workloads.getWorkloads()[0].clusterID = 0;           //WW09 deliverbale is 1 cluster
    workloads.getWorkloads()[0].padTop = 0;              //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[0].padBottom = 0;           //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[0].padLeft = 0;             //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[0].padRight = 0;            //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[0].MPEMode = mv::Matrix;    //Matrix is MPE Mode (4,4)
    workloads.getWorkloads()[0].MinX = 0;
    workloads.getWorkloads()[0].MinY = 28;
    workloads.getWorkloads()[0].MinZ = 0;                //WW09 deliverbale is less than 16 channels
    workloads.getWorkloads()[0].MaxX = 28;
    workloads.getWorkloads()[0].MaxY = 32;
    workloads.getWorkloads()[0].MaxZ = 64;               //WW09 deliverbale is less than 16 channels

    //1
    workloads.getWorkloads().push_back(mv::Workload()); 
    workloads.getWorkloads()[1].workloadID = 1;
    workloads.getWorkloads()[1].clusterID = 0;           //WW09 deliverbale is 1 cluster
    workloads.getWorkloads()[1].padTop = 0;              //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[1].padBottom = 0;           //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[1].padLeft = 0;             //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[1].padRight = 0;            //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[1].MPEMode = mv::Matrix;    //Matrix is MPE Mode (4,4)
    workloads.getWorkloads()[1].MinX = 28;
    workloads.getWorkloads()[1].MinY = 0;
    workloads.getWorkloads()[1].MinZ = 0;                //WW09 deliverbale is less than 16 channels
    workloads.getWorkloads()[1].MaxX = 56;
    workloads.getWorkloads()[1].MaxY = 28;
    workloads.getWorkloads()[1].MaxZ = 64;               //WW09 deliverbale is less than 16 channels

    //2
    workloads.getWorkloads().push_back(mv::Workload()); 
    workloads.getWorkloads()[2].workloadID = 2;
    workloads.getWorkloads()[2].clusterID = 0;           //WW09 deliverbale is 1 cluster
    workloads.getWorkloads()[2].padTop = 0;              //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[2].padBottom = 0;           //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[2].padLeft = 0;             //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[2].padRight = 0;            //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[2].MPEMode = mv::Matrix;    //Matrix is MPE Mode (4,4)
    workloads.getWorkloads()[2].MinX = 16;
    workloads.getWorkloads()[2].MinY = 32;
    workloads.getWorkloads()[2].MinZ = 0;                //WW09 deliverbale is less than 16 channels
    workloads.getWorkloads()[2].MaxX = 28;
    workloads.getWorkloads()[2].MaxY = 56;
    workloads.getWorkloads()[2].MaxZ = 64;               //WW09 deliverbale is less than 16 channels

    //3
    workloads.getWorkloads().push_back(mv::Workload()); 
    workloads.getWorkloads()[3].workloadID = 3;
    workloads.getWorkloads()[3].clusterID = 0;           //WW09 deliverbale is 1 cluster
    workloads.getWorkloads()[3].padTop = 0;              //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[3].padBottom = 0;           //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[3].padLeft = 0;             //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[3].padRight = 0;            //These are zero in PoC compiler - relevant after WW09
    workloads.getWorkloads()[3].MPEMode = mv::Matrix;    //Matrix is MPE Mode (4,4)
    workloads.getWorkloads()[3].MinX = 28;
    workloads.getWorkloads()[3].MinY = 28;
    workloads.getWorkloads()[3].MinZ = 0;                //WW09 deliverbale is less than 16 channels
    workloads.getWorkloads()[3].MaxX = 56;
    workloads.getWorkloads()[3].MaxY = 56;
    workloads.getWorkloads()[3].MaxZ = 64;               //WW09 deliverbale is less than 16 channels

    return workloads;
}

