#include "gtest/gtest.h"
#include "mcm/computation/resource/nce1.hpp"
#include "tests/include/MCMtest.hpp"
#include <iostream>
#include <string>

/* This file contains automated testing of these 3 networks
 *
 *  (1)  Input
 *         |
 *        Conv
 *
/*  (2)
 *      Input
 *       |
 *      Pool
 *      |  |
 *    conv conv
 *      |  |
 *     Concat
 *        |
 *     Softmax
 *
 *  (3)
 *
 *      Input
 *       |
 *      Pool
 *      |  |
 *    conv conv
 *      |  |
 *     Eltwise
 *        |
 *     Softmax
 *
 */
TEST (nce1, DISABLE_HWconv_op_parameters)
{

	/*input channel dimensions*/
	std::vector<int> inputChannels{15, 16, 17}; //Squeezenet
	std::vector<int>::iterator inputChannelsIt;

	/*kernel dimensions*/
	std::vector<int> kernelDimensions{1,3,7}; //Squeezenet
	std::vector<int>::iterator kernelDimensionsIt;

	/*output channels dimensions*/
	std::vector<int> outputChannels{16, 32, 48, 64, 96, 128, 192, 256, 1000}; //Squeezenet
	std::vector<int>::iterator outputChannelsIt;

	/*Stride dimensions*/
	std::vector<int> stride{1, 2}; //Squeezenet
	std::vector<int>::iterator strideIt;

	/*Pad  dimensions*/
	std::vector<int> pad{1, 2}; //Squeezenet
	std::vector<int>::iterator padIt;

	MCMtest test("HWConv2D");
	test.createResultsFiles(); /*create results file to log failed tests*/

	/*loop through convd parameters as defined in the vectors above and test if they
	 *
	 * (1  compile and
	 * (2) run on the hardware
	 * (3) Produce accurate results
	 * */

	int counter=0;

	for (inputChannelsIt = inputChannels.begin(); inputChannelsIt < inputChannels.end(); inputChannelsIt++) {
		for (kernelDimensionsIt = kernelDimensions.begin(); kernelDimensionsIt < kernelDimensions.end(); kernelDimensionsIt++) {
			for (outputChannelsIt = outputChannels.begin(); outputChannelsIt < outputChannels.end(); outputChannelsIt++) {
				for (strideIt = stride.begin(); strideIt < stride.end(); strideIt++) {
					for (padIt = pad.begin(); padIt < pad.end(); padIt++) {


						std::cout << "****************" <<  std::endl;
						std::cout << "**************** " <<  std::endl;
						std::cout << "CONV2D TEST NUMBER " << counter << std::endl;
						/*reset case name*/
						test.caseName="HWConv2D";
						std::cout << "**************** " <<  std::endl;
						std::cout << "****************" <<  std::endl;

						test.addParam("input_tensor_shape","ih","224");
						test.addParam("input_tensor_shape","iw","224");
						test.addParam("input_tensor_shape","ic",std::to_string(*inputChannelsIt));
						test.addParam("input_tensor_shape","ib","1");
						test.addParam("convolution_operation","kh", std::to_string(*kernelDimensionsIt));
						test.addParam("convolution_operation","kw", std::to_string(*kernelDimensionsIt));
						test.addParam("convolution_operation","ph", std::to_string(*padIt));
						test.addParam("convolution_operation","pw", std::to_string(*padIt));
						test.addParam("convolution_operation","sh", std::to_string(*strideIt));
						test.addParam("convolution_operation","sw", std::to_string(*strideIt));
						test.addParam("convolution_operation","no", std::to_string(*outputChannelsIt));
						test.addParam("convolution_operation","wf","xavier");
						test.addParam("convolution_operation","ws","0.1");
						test.addParam("convolution_operation","bt","constant");
						test.addParam("convolution_operation","bv","2");

						/*Test if compilation successful*/
						test.generatePrototxt_2dconv();  /*create prototxt*/

						/*Test if compilation successful*/
						std::string command1 = "$MDK_HOME/projects/Fathom/src2/mvNCCompile.py ./test.prototxt --new-parser --cpp";
						EXPECT_EQ (0, system(command1.c_str())) << "ERROR: non 0 return from compile";

						/*Test if run on hardware successful*/
						std::string command2 = "python3 $MCM_HOME/python/tools/mcmRunHW.py --blob ./cpp.blob --image ./test.png --result blob_result";
						auto hardware_result = test.execute(command2.c_str());
						std::cout << "Result running on hardware is " << hardware_result << std::endl;
						EXPECT_EQ (0, hardware_result) << "ERROR: non 0 return from mcmRunHW.py";


						/*Write result to file for hardware test*/
						test.hardwareResults.open(test.failedHardwareRunFileName_.c_str(),std::ofstream::out | std::ofstream::app);
						test.hs << "TEST : " << test.caseName;
						if(hardware_result) {
							test.hs << " : FAIL" <<"\n";
						}
						else{
							test.hs << " : PASS" <<"\n";

						}
						test.hs << "";
						test.hardwareResults << test.hs.str();
						test.hs.str("");
						test.hardwareResults.close();



						/*test accuracy in range*/
						std::string command3 = "python3 $MCM_HOME/python/tools/mcmCheckRef.py --reference ./Fathom_expected.npy --result blob_result.npy";
						auto accuracy_result = test.execute(command3.c_str());
						std::cout << "return from mcmCheckRef.py " << accuracy_result << std::endl;
						EXPECT_EQ (0, accuracy_result) << "ERROR: non 0 return from mcmCheck";

						/*Write result to file for accuracy test*/
						test.accuracyResults.open(test.failedAccuracyFileName_.c_str(),std::ofstream::out | std::ofstream::app);
						test.as << "TEST : " << test.caseName;
						if(accuracy_result) {
							test.as << " : FAIL" <<"\n2";
						}
						else {
							test.as << " : PASS" <<"\n";
						}
						test.as << "";
						test.accuracyResults << test.as.str();
						test.as.str("");
						test.accuracyResults.close();
						EXPECT_EQ (0, system(command1.c_str())) << "ERROR: non 0 return from compile";

						counter++;
					}
				}
			}
		}
	}
}


TEST (nce1, DISABLE_Parallel_network_concat)
{

	/*input channel dimensions*/
	std::vector<int> inputChannels{15, 16, 17}; //Squeezenet
	std::vector<int>::iterator inputChannelsIt;

	/*kernel dimensions*/
	std::vector<int> kernelDimensions{1,3,7}; //Squeezenet
	std::vector<int>::iterator kernelDimensionsIt;

	/*output channels dimensions*/
	std::vector<int> outputChannels{16, 32, 48, 64, 96, 128, 192, 256, 1000}; //Squeezenet
	std::vector<int>::iterator outputChannelsIt;

	/*Stride dimensions*/
	std::vector<int> stride{1, 2}; //Squeezenet
	std::vector<int>::iterator strideIt;

	/*Pad  dimensions*/
	std::vector<int> pad{1, 2}; //Squeezenet
	std::vector<int>::iterator padIt;

	/*Pool  dimensions*/
	std::vector<std::string> poolType{"MAX", "AVERAGE"}; //Squeezenet
	std::vector<int> poolKernelSize{3,7}; //Squeezenet
	std::vector<int> poolStride{2}; //Squeezenet
	std::vector<int>::iterator poolIt;

	MCMtest test("HWConcat");
	test.createResultsFiles(); /*create results file to log failed tests*/


	/*loop through convd parameters as defined in the vectors above and test if they
	 *
	 * (1  compile and
	 * (2) run on the hardware
	 * (3) Produce accurate results
	 * */

	int counter = 0;
	for (inputChannelsIt = inputChannels.begin(); inputChannelsIt < inputChannels.end(); inputChannelsIt++) {
		for (kernelDimensionsIt = kernelDimensions.begin(); kernelDimensionsIt < kernelDimensions.end(); kernelDimensionsIt++) {
			for (outputChannelsIt = outputChannels.begin(); outputChannelsIt < outputChannels.end(); outputChannelsIt++) {
				for (strideIt = stride.begin(); strideIt < stride.end(); strideIt++) {
					for (padIt = pad.begin(); padIt < pad.end(); padIt++) {

						std::cout << "****************" <<  std::endl;
						std::cout << "**************** " <<  std::endl;
						std::cout << "CONCAT TEST NUMBER " << counter << std::endl;
						/*reset case name*/
						test.caseName="HWConcat";
						std::cout << "**************** " <<  std::endl;
						std::cout << "****************" <<  std::endl;

						test.addParam("input_tensor_shape","ih","224");
						test.addParam("input_tensor_shape","iw","224");
						test.addParam("input_tensor_shape","ic",std::to_string(*inputChannelsIt));
						test.addParam("input_tensor_shape","ib","1");

						test.addParam("pool1","type","MAX");
						test.addParam("pool1","ks","3");
						test.addParam("pool1","s","2");

						test.addParam("conv1","kh", std::to_string(*kernelDimensionsIt));
						test.addParam("conv1","kw", std::to_string(*kernelDimensionsIt));
						test.addParam("conv1","ph", std::to_string(*padIt));
						test.addParam("conv1","pw", std::to_string(*padIt));
						test.addParam("conv1","sh", std::to_string(*strideIt));
						test.addParam("conv1","sw", std::to_string(*strideIt));
						test.addParam("conv1","no", std::to_string(*outputChannelsIt));
						test.addParam("conv1","wf","xavier");
						test.addParam("conv1","ws","0.1");
						test.addParam("conv1","bt","constant");
						test.addParam("conv1","bv","2");

						test.addParam("conv2","kh", std::to_string(*kernelDimensionsIt));
						test.addParam("conv2","kw", std::to_string(*kernelDimensionsIt));
						test.addParam("conv2","ph", std::to_string(*padIt));
						test.addParam("conv2","pw", std::to_string(*padIt));
						test.addParam("conv2","sh", std::to_string(*strideIt));
						test.addParam("conv2","sw", std::to_string(*strideIt));
						test.addParam("conv2","no", std::to_string(*outputChannelsIt));
						test.addParam("conv2","wf","xavier");
						test.addParam("conv2","ws","0.1");
						test.addParam("conv2","bt","constant");
						test.addParam("conv2","bv","2");

						test.addParam("concat1","axis","1");

						test.generatePrototxt_diamond_concat();  /*create prototxt*/

						/*Test if compilation successful*/
						std::string command1 = "$MDK_HOME/projects/Fathom/src2/mvNCCompile.py ./test.prototxt --new-parser --cpp";
						EXPECT_EQ (0, system(command1.c_str())) << "ERROR: non 0 return from compile";

						/*Test if run on hardware successful*/
						std::string command2 = "python3 $MCM_HOME/python/tools/mcmRunHW.py --blob ./cpp.blob --image ./test.png --result blob_result";
						auto hardware_result = test.execute(command2.c_str());
						std::cout << "Result running on hardware is " << hardware_result << std::endl;
						EXPECT_EQ (0, hardware_result) << "ERROR: non 0 return from mcmRunHW.py";


						/*Write result to file for hardware test*/
						test.hardwareResults.open(test.failedHardwareRunFileName_.c_str(),std::ofstream::out | std::ofstream::app);
						test.hs << "TEST : " << test.caseName;
						if(hardware_result) {
							test.hs << " : FAIL" <<"\n";
						}
						else {
							test.hs << " : PASS" <<"\n";
						}
						test.hs << "";
						test.hardwareResults << test.hs.str();
						test.hs.str("");
						test.hardwareResults.close();


						/*test accuracy in range*/
						std::string command3 = "python3 $MCM_HOME/python/tools/mcmCheckRef.py --reference ./Fathom_expected.npy --result blob_result.npy";
						auto accuracy_result = test.execute(command3.c_str());
						std::cout << "return from mcmCheckRef.py " << accuracy_result << std::endl;
						EXPECT_EQ (0, accuracy_result) << "ERROR: non 0 return from mcmCheck";

						/*Write result to file for accuracy test*/
						test.accuracyResults.open(test.failedAccuracyFileName_.c_str(),std::ofstream::out | std::ofstream::app);
						test.as << "TEST : " << test.caseName;
						if(accuracy_result) {
							test.as << " : FAIL" <<"\n";
						}
						else {
							test.as << " : PASS" <<"\n";
						}
						test.as << "";
						test.accuracyResults << test.as.str();
						test.as.str("");
						test.accuracyResults.close();

						counter++;

					}
				}
			}
		}
	}
}

TEST (nce1, DISABLE_Parallel_network_eltwise)
{

	/*input channel dimensions*/
	std::vector<int> inputChannels{15, 16, 17}; //Squeezenet
	std::vector<int>::iterator inputChannelsIt;

	/*kernel dimensions*/
	std::vector<int> kernelDimensions{1,3,7}; //Squeezenet
	std::vector<int>::iterator kernelDimensionsIt;

	/*output channels dimensions*/
	std::vector<int> outputChannels{16, 32, 48, 64, 96, 128, 192, 256, 1000}; //Squeezenet
	std::vector<int>::iterator outputChannelsIt;

	/*Stride dimensions*/
	std::vector<int> stride{1, 2}; //Squeezenet
	std::vector<int>::iterator strideIt;

	/*Pad  dimensions*/
	std::vector<int> pad{1, 2}; //Squeezenet
	std::vector<int>::iterator padIt;

	/*Pool  dimensions*/
	std::vector<std::string> poolType{"MAX", "AVERAGE"}; //Squeezenet
	std::vector<int> poolKernelSize{3,7}; //Squeezenet
	std::vector<int> poolStride{2}; //Squeezenet
	std::vector<int>::iterator poolIt;

	MCMtest test("HWEltwise");
	test.createResultsFiles(); /*create results file to log failed tests*/


	/*loop through convd parameters as defined in the vectors above and test if they
	 *
	 * (1  compile and
	 * (2) run on the hardware
	 * (3) Produce accurate results
	 * */

	int counter = 0;
	for (inputChannelsIt = inputChannels.begin(); inputChannelsIt < inputChannels.end(); inputChannelsIt++) {
		for (kernelDimensionsIt = kernelDimensions.begin(); kernelDimensionsIt < kernelDimensions.end(); kernelDimensionsIt++) {
			for (outputChannelsIt = outputChannels.begin(); outputChannelsIt < outputChannels.end(); outputChannelsIt++) {
				for (strideIt = stride.begin(); strideIt < stride.end(); strideIt++) {
					for (padIt = pad.begin(); padIt < pad.end(); padIt++) {


						std::cout << "****************" <<  std::endl;
						std::cout << "**************** " <<  std::endl;
						std::cout << "ELTWISE TEST NUMBER " << counter << std::endl;
						/*reset case name*/
						test.caseName="HWEltwise";
						std::cout << "**************** " <<  std::endl;
						std::cout << "****************" <<  std::endl;

						test.addParam("input_tensor_shape","ih","224");
						test.addParam("input_tensor_shape","iw","224");
						test.addParam("input_tensor_shape","ic",std::to_string(*inputChannelsIt));
						test.addParam("input_tensor_shape","ib","1");

						test.addParam("pool1","type","MAX");
						test.addParam("pool1","ks","3");
						test.addParam("pool1","s","2");

						test.addParam("conv1","kh", std::to_string(*kernelDimensionsIt));
						test.addParam("conv1","kw", std::to_string(*kernelDimensionsIt));
						test.addParam("conv1","ph", std::to_string(*padIt));
						test.addParam("conv1","pw", std::to_string(*padIt));
						test.addParam("conv1","sh", std::to_string(*strideIt));
						test.addParam("conv1","sw", std::to_string(*strideIt));
						test.addParam("conv1","no", std::to_string(*outputChannelsIt));
						test.addParam("conv1","wf","xavier");
						test.addParam("conv1","ws","0.1");
						test.addParam("conv1","bt","constant");
						test.addParam("conv1","bv","2");

						test.addParam("conv2","kh", std::to_string(*kernelDimensionsIt));
						test.addParam("conv2","kw", std::to_string(*kernelDimensionsIt));
						test.addParam("conv2","ph", std::to_string(*padIt));
						test.addParam("conv2","pw", std::to_string(*padIt));
						test.addParam("conv2","sh", std::to_string(*strideIt));
						test.addParam("conv2","sw", std::to_string(*strideIt));
						test.addParam("conv2","no", std::to_string(*outputChannelsIt));
						test.addParam("conv2","wf","xavier");
						test.addParam("conv2","ws","0.1");
						test.addParam("conv2","bt","constant");
						test.addParam("conv2","bv","2");

						test.addParam("eltwise","operation"," SUM");

						test.generatePrototxt_diamond_eltwise();  /*create prototxt*/

						/*Test if compilation successful*/
						std::string command1 = "$MDK_HOME/projects/Fathom/src2/mvNCCompile.py ./test.prototxt --new-parser --cpp";
						EXPECT_EQ (0, system(command1.c_str())) << "ERROR: non 0 return from compile";

						/*Test if run on hardware successful*/
						std::string command2 = "python3 $MCM_HOME/python/tools/mcmRunHW.py --blob ./cpp.blob --image ./test.png --result blob_result";
						auto hardware_result = test.execute(command2.c_str());
						std::cout << "Result running on hardware is " << hardware_result << std::endl;
						EXPECT_EQ (0, hardware_result) << "ERROR: non 0 return from mcmRunHW.py";


						/*Write result to file for hardware test*/
						test.hardwareResults.open(test.failedHardwareRunFileName_.c_str(),std::ofstream::out | std::ofstream::app);
						test.hs << "TEST : " << test.caseName;
						if(hardware_result) {
							test.hs << " : FAIL" <<"\n";
						}
						else {
							test.hs << " : PASS" <<"\n";
						}
						test.hs << "";
						test.hardwareResults << test.hs.str();
						test.hs.str("");
						test.hardwareResults.close();


						/*test accuracy in range*/
						std::string command3 = "python3 $MCM_HOME/python/tools/mcmCheckRef.py --reference ./Fathom_expected.npy --result blob_result.npy";
						auto accuracy_result = test.execute(command3.c_str());
						std::cout << "return from mcmCheckRef.py " << accuracy_result << std::endl;
						EXPECT_EQ (0, accuracy_result) << "ERROR: non 0 return from mcmCheck";

						/*Write result to file for accuracy test*/
						test.accuracyResults.open(test.failedAccuracyFileName_.c_str(),std::ofstream::out | std::ofstream::app);
						test.as << "TEST : " << test.caseName;
						if(accuracy_result) {
							test.as << " : FAIL" <<"\n";
						}
						else {
							test.as << " : PASS" <<"\n";
						}
						test.as << "";
						test.accuracyResults << test.as.str();
						test.as.str("");
						test.accuracyResults.close();

						counter++;

					}
				}
			}
		}
	}
}



