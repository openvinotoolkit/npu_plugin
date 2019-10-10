#include "convert_tensor.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

bool ParseAndCheckCommandLine(int argc, char *argv[]) 
{
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) 
    {
        showUsage();
        return false;
    }

    if (FLAGS_t.empty()) 
        throw std::logic_error("Parameter -t is not set");
    if (FLAGS_b.empty()) 
        throw std::logic_error("Parameter -b is not set");
 
    return true;
}

std::string getFilename(std::string& path)
{
    std::string base_filename = path.substr(path.find_last_of("/\\") + 1);
    std::string::size_type const p(base_filename.find_last_of('.'));
    std::string file_without_extension = base_filename.substr(0, p);

    return file_without_extension;
}

int main(int argc, char *argv[]) 
{
    if (!ParseAndCheckCommandLine(argc, argv)) 
        return 0;

    //convert blob to json
    std::string commandline = std::string("flatc -t ") + std::getenv("GRAPHFILE") + std::string("/src/schema/graphfile.fbs --strict-json -- ") + FLAGS_b;
    //std::cout << commandline << std::endl;
    std::system(commandline.c_str());
    
    //load json and read quantization values: scale and zeropoint
    std::string json_file = getFilename(FLAGS_b) + std::string(".json");
    std::ifstream i(json_file);
    json j = json::parse(i);

    std::string dtype = j["header"]["net_output"][0]["data_dtype"].get<std::string>();
    int qZero = j["header"]["net_output"][0]["quant_zero"][0].get<int>();
    int qScale = j["header"]["net_output"][0]["quant_scale"][0].get<int>();
    std::cout << "Datatype: " << dtype << std::endl;
    std::cout << "quant_zero: " << qZero << std::endl;
    std::cout << "quant_scale: " << qScale << std::endl;

    //read size of output tensor
    int tSize = 1;
    for (int x=0; x<j["header"]["net_output"][0]["dimensions"].size(); ++x)
    {
        tSize *= j["header"]["net_output"][0]["dimensions"][x].get<int>();
    }
    std::cout << "Output size: " << tSize << std::endl;

    // Read the InferenceManagerDemo output file into a vector
    std::ifstream file(FLAGS_t, std::ios::binary);
    file.unsetf(std::ios::skipws);
    std::vector<unsigned char> outputVector;
    outputVector.reserve(tSize);
    outputVector.insert(outputVector.begin(),
        std::istream_iterator<unsigned char>(file),
        std::istream_iterator<unsigned char>());
    
    for (int i = 0; i < tSize; ++i)
        std::cout << std::hex << (int)outputVector[i] << " ";
}
