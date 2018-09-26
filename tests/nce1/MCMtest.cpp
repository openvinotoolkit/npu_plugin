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
    std::cout << "adding params for " << caseName << std::endl;
    descriptor[opCategory][opKey]=opValue;
}

void MCMtest::generatePrototxt()
{
    std::cout << "calling prototxt generator for " << caseName << std::endl;

    std::string generatedPrototextFileName = "test.prototxt";
    std::ofstream generatedPrototextFile;
    std::ostringstream ss;
    generatedPrototextFile.open(generatedPrototextFileName.c_str(),std::ofstream::out | std::ofstream::trunc);

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

