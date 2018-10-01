#include "tests/include/MCMtest.hpp"

MCMtest::MCMtest(std::string testType)
{
    caseName=testType;
    descriptor["testMetaData"]["testType"]=testType;
    descriptor["testMetaData"]["testHash"]="00000000";
}

void MCMtest::addParam(std::string opCategory, std::string opKey, std::string opValue)
{
    caseName += "_"+opKey+"-"+opValue ;
    descriptor[opCategory][opKey]=opValue;
}

void MCMtest::generatePrototxt_2dconv()
{

    std::ofstream generatedPrototextFile;
    std::string generatedPrototxtFileName_ = "test.txt";
    std::ostringstream ss;
    generatedPrototextFile.open(generatedPrototxtFileName_.c_str(),std::ofstream::out | std::ofstream::trunc);

    ss << "name: \"Test\"" << "\n";
    ss << "input: \"data\"" << "\n";
    ss << "input_shape {" << "\n";
    ss << "dim:" << descriptor["input_tensor_shape"]["ib"].get<std::string>() << "\n";
    ss << "dim:" << descriptor["input_tensor_shape"]["ic"].get<std::string>()  << "\n";
    ss << "dim:" << descriptor["input_tensor_shape"]["ih"].get<std::string>()  << "\n";
    ss << "dim:" << descriptor["input_tensor_shape"]["iw"].get<std::string>()  << "\n";
    ss << "}" << "\n";
    ss << "layer {" << "\n";
    ss << "name: \"conv1\"" << "\n";
    ss << "type: \"Convolution\"" << "\n";
    ss << "bottom: \"data\"" << "\n";
    ss << "top: \"conv1\"" << "\n";
    ss << "param { lr_mult: 1 decay_mult: 1 }"<< "\n";
    ss << "param { lr_mult: 2 decay_mult: 0 }" << "\n";
    ss << "convolution_param {" << "\n";
    ss << "num_output: " << descriptor["convolution_operation"]["no"].get<std::string>()  << "\n";;
    ss << "kernel_h:" << descriptor["convolution_operation"]["kh"].get<std::string>()  << "\n";
    ss << "kernel_w:" << descriptor["convolution_operation"]["kw"].get<std::string>()  << "\n";
    ss << "stride_h:" << descriptor["convolution_operation"]["sh"].get<std::string>()  << "\n";
    ss << "stride_w:" << descriptor["convolution_operation"]["sw"].get<std::string>()  << "\n";
    ss << "pad_h:" << descriptor["convolution_operation"]["ph"].get<std::string>()  << "\n";
    ss << "pad_w:" << descriptor["convolution_operation"]["pw"].get<std::string>()  << "\n";
    ss << "weight_filler {" << "\n";
    ss << "type: \"" << descriptor["convolution_operation"]["wf"].get<std::string>()  << "\"\n";
    ss << "std:" << descriptor["convolution_operation"]["ws"].get<std::string>()  << "\n";
    ss << "}" << "\n";
    ss << "bias_filler {" << "\n";
    ss << "type: \"" << descriptor["convolution_operation"]["bt"].get<std::string>()  << "\"\n";
    ss << "value:" << descriptor["convolution_operation"]["bv"].get<std::string>()  << "\n";
    ss << "}" << "\n";
    ss << "}" << "\n";
    ss << "}" << "\n";

    generatedPrototextFile << ss.str();
    ss.str("");
    generatedPrototextFile.close();

    return ;
}




/*This method generates prototxt for this network*/

/*
 *      Input
 *       |
 *      Pool
 *      |  |
 *    conv conv
 *      |  |
 *     eltwise
 *        |
 *      Softmax
 *
 */

void MCMtest::generatePrototxt_diamond_eltwise()
{
	std::ofstream generatedPrototextFile;
	std::string generatedPrototxtFileName_ = "test.txt";
	std::ostringstream ss;
	generatedPrototextFile.open(generatedPrototxtFileName_.c_str(),std::ofstream::out | std::ofstream::trunc);

    ss << "name: \"Test1\"" << "\n";
    ss << "input: \"data\"" << "\n";
	ss << " input_shape {" << "\n";
    ss << "dim:" << descriptor["input_tensor_shape"]["ib"].get<std::string>() << "\n";
	ss << "dim:" << descriptor["input_tensor_shape"]["ic"].get<std::string>()  << "\n";
	ss << "dim:" << descriptor["input_tensor_shape"]["ih"].get<std::string>()  << "\n";
	ss << "dim:" << descriptor["input_tensor_shape"]["iw"].get<std::string>()  << "\n";
	ss << "}" << "\n";

    ss << "layer {" << "\n";
    ss <<  "name: \"pool1\"" << "\n";
    ss <<  "type: \"Pooling\"" << "\n";
	ss <<  "bottom: \"data\"" << "\n";
	ss <<  "top: \"pool1\"" << "\n";
	ss <<  "pooling_param {" << "\n";
    ss <<  "pool:" << descriptor["pool1"]["type"].get<std::string>() << "\n";
    ss <<  "kernel_size:" << descriptor["pool1"]["ks"].get<std::string>() << "\n";
	ss <<  "stride:" << descriptor["pool1"]["s"].get<std::string>() << "\n";
	ss <<  "}" << "\n";
    ss <<  "}" << "\n";

    ss << "layer {" << "\n";
    ss << "name: \"conv1\"" << "\n";
    ss << "type: \"Convolution\"" << "\n";
    ss << "bottom: \"pool1\"" << "\n";
    ss << "top: \"conv1\"" << "\n";
    ss << "param { lr_mult: 1 decay_mult: 1 }"<< "\n";
    ss << "param { lr_mult: 2 decay_mult: 0 }" << "\n";
    ss << "convolution_param {" << "\n";
    ss << "num_output: " << descriptor["conv1"]["no"].get<std::string>()  << "\n";;
    ss << "kernel_h:" << descriptor["conv1"]["kh"].get<std::string>()  << "\n";
    ss << "kernel_w:" << descriptor["conv1"]["kw"].get<std::string>()  << "\n";
    ss << "stride_h:" << descriptor["conv1"]["sh"].get<std::string>()  << "\n";
    ss << "stride_w:" << descriptor["conv1"]["sw"].get<std::string>()  << "\n";
    ss << "pad_h:" << descriptor["conv1"]["ph"].get<std::string>()  << "\n";
    ss << "pad_w:" << descriptor["conv1"]["pw"].get<std::string>()  << "\n";
    ss << "weight_filler {" << "\n";
    ss << "type: \"" << descriptor["conv1"]["wf"].get<std::string>()  << "\"\n";
    ss << "std:" << descriptor["conv1"]["ws"].get<std::string>()  << "\n";
    ss << "}" << "\n";
    ss << "bias_filler {" << "\n";
    ss << "type: \"" << descriptor["conv1"]["bt"].get<std::string>()  << "\"\n";
    ss << "value:" << descriptor["conv1"]["bv"].get<std::string>()  << "\n";
    ss << "}" << "\n";
    ss << "}" << "\n";
    ss << "}" << "\n";

    ss << "layer {" << "\n";
    ss << "name: \"conv2\"" << "\n";
    ss << "type: \"Convolution\"" << "\n";
    ss << "bottom: \"pool1\"" << "\n";
    ss << "top: \"conv2\"" << "\n";
    ss << "param { lr_mult: 1 decay_mult: 1 }"<< "\n";
    ss << "param { lr_mult: 2 decay_mult: 0 }" << "\n";
    ss << "convolution_param {" << "\n";
    ss << "num_output: " << descriptor["conv2"]["no"].get<std::string>()  << "\n";;
    ss << "kernel_h:" << descriptor["conv2"]["kh"].get<std::string>()  << "\n";
    ss << "kernel_w:" << descriptor["conv2"]["kw"].get<std::string>()  << "\n";
    ss << "stride_h:" << descriptor["conv2"]["sh"].get<std::string>()  << "\n";
    ss << "stride_w:" << descriptor["conv2"]["sw"].get<std::string>()  << "\n";
    ss << "pad_h:" << descriptor["conv2"]["ph"].get<std::string>()  << "\n";
    ss << "pad_w:" << descriptor["conv2"]["pw"].get<std::string>()  << "\n";
    ss << "weight_filler {" << "\n";
    ss << "type: \"" << descriptor["conv2"]["wf"].get<std::string>()  << "\"\n";
    ss << "std:" << descriptor["conv2"]["ws"].get<std::string>()  << "\n";
    ss << "}" << "\n";
    ss << "bias_filler {" << "\n";
    ss << "type: \"" << descriptor["conv2"]["bt"].get<std::string>()  << "\"\n";
    ss << "value:" << descriptor["conv2"]["bv"].get<std::string>()  << "\n";
    ss << "}" << "\n";
    ss << "}" << "\n";
    ss << "}" << "\n";

    ss << "layer {" << "\n";
    ss << "name: \"eltwise-sum\"" << "\n";
	ss << "bottom: \"conv1\"" << "\n";
	ss << "bottom: \"conv2\"" << "\n";
	ss << "top: \"conv1_conv2_sum\"" << "\n";
	ss << "type: \"Eltwise\"" << "\n";
	ss << "eltwise_param {" << "\n";
	ss << "operation:" << descriptor["eltwise"]["operation"].get<std::string>()  << "\n";
	ss << "}" << "\n";
	ss << "}" << "\n";

    ss << "layer {" << "\n";
    ss << "name: \"prob\"" << "\n";
    ss << "type: \"Softmax\"" << "\n";
	ss << "bottom: \"concat1\"" << "\n";
	ss << "top: \"prob\"" << "\n";
	ss << "}" << "\n";


    generatedPrototextFile << ss.str();
    ss.str("");
    generatedPrototextFile.close();

    return ;
}

/*This method generates prototxt for this network*/

/*
 *      Input
 *       |
 *      Pool
 *      |  |
 *    conv conv
 *      |  |
 *     Concat
 *        |
 *      Softmax
 *
 */

void MCMtest::generatePrototxt_diamond_concat()
{
	std::ofstream generatedPrototextFile;
	std::string generatedPrototxtFileName_ = "test.txt";
	std::ostringstream ss;
	generatedPrototextFile.open(generatedPrototxtFileName_.c_str(),std::ofstream::out | std::ofstream::trunc);

    ss << "name: \"Test1\"" << "\n";
    ss << "input: \"data\"" << "\n";
    ss << " input_shape {" << "\n";
    ss << "dim:" << descriptor["input_tensor_shape"]["ib"].get<std::string>() << "\n";
    ss << "dim:" << descriptor["input_tensor_shape"]["ic"].get<std::string>()  << "\n";
    ss << "dim:" << descriptor["input_tensor_shape"]["ih"].get<std::string>()  << "\n";
    ss << "dim:" << descriptor["input_tensor_shape"]["iw"].get<std::string>()  << "\n";
    ss << "}" << "\n";

    ss << "layer {" << "\n";
    ss <<  "name: \"pool1\"" << "\n";
    ss <<  "type: \"Pooling\"" << "\n";
	ss <<  "bottom: \"data\"" << "\n";
	ss <<  "top: \"pool1\"" << "\n";
	ss <<  "pooling_param {" << "\n";
    ss <<  "pool:" << descriptor["pool1"]["type"].get<std::string>() << "\n";
    ss <<  "kernel_size:" << descriptor["pool1"]["ks"].get<std::string>() << "\n";
	ss <<  "stride:" << descriptor["pool1"]["s"].get<std::string>() << "\n";
	ss <<  "}" << "\n";
    ss <<  "}" << "\n";

    ss << "layer {" << "\n";
    ss << "name: \"conv1\"" << "\n";
    ss << "type: \"Convolution\"" << "\n";
    ss << "bottom: \"pool1\"" << "\n";
    ss << "top: \"conv1\"" << "\n";
    ss << "param { lr_mult: 1 decay_mult: 1 }"<< "\n";
    ss << "param { lr_mult: 2 decay_mult: 0 }" << "\n";
    ss << "convolution_param {" << "\n";
    ss << "num_output: " << descriptor["conv1"]["no"].get<std::string>()  << "\n";;
    ss << "kernel_h:" << descriptor["conv1"]["kh"].get<std::string>()  << "\n";
    ss << "kernel_w:" << descriptor["conv1"]["kw"].get<std::string>()  << "\n";
    ss << "stride_h:" << descriptor["conv1"]["sh"].get<std::string>()  << "\n";
    ss << "stride_w:" << descriptor["conv1"]["sw"].get<std::string>()  << "\n";
    ss << "pad_h:" << descriptor["conv1"]["ph"].get<std::string>()  << "\n";
    ss << "pad_w:" << descriptor["conv1"]["pw"].get<std::string>()  << "\n";
    ss << "weight_filler {" << "\n";
    ss << "type: \"" << descriptor["conv1"]["wf"].get<std::string>()  << "\"\n";
    ss << "std:" << descriptor["conv1"]["ws"].get<std::string>()  << "\n";
    ss << "}" << "\n";
    ss << "bias_filler {" << "\n";
    ss << "type: \"" << descriptor["conv1"]["bt"].get<std::string>()  << "\"\n";
    ss << "value:" << descriptor["conv1"]["bv"].get<std::string>()  << "\n";
    ss << "}" << "\n";hardware_accuracy_test.txt
    ss << "}" << "\n";
    ss << "}" << "\n";

    ss << "layer {" << "\n";
    ss << "name: \"conv2\"" << "\n";
    ss << "type: \"Convolution\"" << "\n";
    ss << "bottom: \"pool1\"" << "\n";
    ss << "top: \"conv2\"" << "\n";
    ss << "param { lr_mult: 1 decay_mult: 1 }"<< "\n";
    ss << "param { lr_mult: 2 decay_mult: 0 }" << "\n";
    ss << "convolution_param {" << "\n";
    ss << "num_output: " << descriptor["conv2"]["no"].get<std::string>()  << "\n";;
    ss << "kernel_h:" << descriptor["conv2"]["kh"].get<std::string>()  << "\n";
    ss << "kernel_w:" << descriptor["conv2"]["kw"].get<std::string>()  << "\n";
    ss << "stride_h:" << descriptor["conv2"]["sh"].get<std::string>()  << "\n";
    ss << "stride_w:" << descriptor["conv2"]["sw"].get<std::string>()  << "\n";
    ss << "pad_h:" << descriptor["conv2"]["ph"].get<std::string>()  << "\n";
    ss << "pad_w:" << descriptor["conv2"]["pw"].get<std::string>()  << "\n";
    ss << "weight_filler {" << "\n";
    ss << "type: \"" << descriptor["conv2"]["wf"].get<std::string>()  << "\"\n";
    ss << "std:" << descriptor["conv2"]["ws"].get<std::string>()  << "\n";
    ss << "}" << "\n";
    ss << "bias_filler {" << "\n";
    ss << "type: \"" << descriptor["conv2"]["bt"].get<std::string>()  << "\"\n";
    ss << "value:" << descriptor["conv2"]["bv"].get<std::string>()  << "\n";
    ss << "}" << "\n";
    ss << "}" << "\n";
    ss << "}" << "\n";

    ss << "layer {" << "\n";
    ss << "name: \"concat1\"" << "\n";
	ss << "bottom: \"conv1\"" << "\n";
	ss << "bottom: \"conv2\"" << "\n";
	ss << "top: \"concat1\"" << "\n";
	ss << "type: \"Concat\"" << "\n";
	ss << "concat_param {" << "\n";
	ss << "axis:" << descriptor["concat1"]["axis"].get<std::string>()  << "\n";
	ss << "}" << "\n";
	ss << "}" << "\n";

    ss << "layer {" << "\n";
    ss << "name: \"prob\"" << "\n";
    ss << "type: \"Softmax\"" << "\n";
	ss << "bottom: \"concat1\"" << "\n";
	ss << "top: \"prob\"" << "\n";
	ss << "}" << "\n";


    generatedPrototextFile << ss.str();
    ss.str("");
    generatedPrototextFile.close();

    return ;
}

/*This method runs command using popen to get return codes correctly*/
int MCMtest::execute(const char* cmd) {
	    char buffer[128];
	    std::string result = "";
	    FILE* pipe = popen(cmd, "r");
	    if (!pipe) throw std::runtime_error("popen() failed!");
	    try {
	        while (!feof(pipe)) {
	            if (fgets(buffer, 128, pipe) != NULL)
	                result += buffer;
	        }
	    } catch (...) {
	        pclose(pipe);
	        throw;
	    }
	    auto res = pclose(pipe);
	    return res;
	}

/*This method creates files to store the PASS/FAIL status of tests for (1) run on hardware and (2) accuracy*/
void MCMtest::createResultsFiles() {

	projectRootPath_ = mv::utils::projectRootPath();
	std::string savedTestsPath_ = mv::utils::projectRootPath() + "/tests/nce1/automated_results/";
	failedHardwareRunFileName_ = savedTestsPath_ + "hardware_run_test.txt";
	hardwareResults.open(failedHardwareRunFileName_.c_str(),std::ofstream::out | std::ofstream::trunc); /*open the source file with the truncate-option to delete previous content*/
	hardwareResults.close();

	projectRootPath_ = mv::utils::projectRootPath();
	failedAccuracyFileName_ = savedTestsPath_ + "hardware_accuracy_test.txt";
	accuracyResults.open(failedAccuracyFileName_.c_str(),std::ofstream::out | std::ofstream::trunc); /*open the source file with the truncate-option to delete previous content*/
	accuracyResults.close();


}


void MCMtest::saveResult()
{
    std::string jsonFileName = caseName + ".json";
    std::ofstream jsonFile;
    std::ostringstream jf;
    jsonFile.open(jsonFileName.c_str(),std::ofstream::out | std::ofstream::trunc);

    std::cout << "calling save result for " << caseName << std::endl;
    jf << descriptor.stringifyPretty() << std::endl;
    jsonFile << jf.str();
    jf.str("");
    jsonFile.close();

}

MCMtest::~MCMtest() {
	// TODO Auto-generated destructor stub
}

