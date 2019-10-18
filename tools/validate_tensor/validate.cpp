#include "convert_tensor.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

#include <iomanip>
#include <vector>

using json = nlohmann::json;

int test()
{
    //Generate a file with some floats in it for testing
    std::ofstream fout("Dequantize.bin", std::ios::binary);
    std::ofstream qout("Quantize.bin", std::ios::binary);

    for(size_t i = 0; i < 10; ++i)
    {
        float f = 30.14159f * i;
        fout.write(reinterpret_cast<const char*>(&f), sizeof(f));

        // Quantize: quantized_value = zero_point + real_value / scale
        int q = 0 + (f / 16);
        std::cout << q << std::endl;
        qout.write(reinterpret_cast<const char*>(&q), sizeof(q));
    }
    fout.close();
    qout.close();

    std::ifstream fin("Dequantize.bin", std::ios::binary);
    if(!fin)
    {
        std::cout << " Error, Couldn't find the file" << "\n";
        return 0;
    }

    fin.seekg(0, std::ios::end);
    const size_t num_elements = fin.tellg() / sizeof(float);
    fin.seekg(0, std::ios::beg);

    std::vector<float> data(num_elements);
    fin.read(reinterpret_cast<char*>(&data[0]), num_elements*sizeof(float));
    for(size_t i = 0; i < data.size(); ++i)
        std::cout << std::fixed << data[i] << "\n";

    //
    std::ifstream fin2("Quantize.bin", std::ios::binary);
    if(!fin2)
    {
        std::cout << " Error, Couldn't find the file" << "\n";
        return 0;
    }

    fin2.seekg(0, std::ios::end);
    const size_t num_elements2 = fin2.tellg() / sizeof(int);
    fin2.seekg(0, std::ios::beg);

    std::vector<int> data2(num_elements2);
    fin.read(reinterpret_cast<char*>(&data2[0]), num_elements2*sizeof(int));
    for(size_t i = 0; i < data2.size(); ++i)
        std::cout << std::fixed << data2[i] << "\n";

    return 0;
}

bool ParseAndCheckCommandLine(int argc, char *argv[]) 
{
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) 
    {
        showUsage();
        return false;
    }

    if (FLAGS_e.empty()) 
        throw std::logic_error("Parameter -e is not set");
    if (FLAGS_a.empty()) 
        throw std::logic_error("Parameter -a is not set");
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


bool compare(std::vector<float>& actualResults, std::vector<float>& expectedResults, float tolerance)
{
    std::cout << "Comparing results ... " << std::endl;
    std::cout << "  Actual Results size: " << actualResults.size() << std::endl;
    std::cout << "  Expected Results size: " << expectedResults.size() << std::endl;
    std::cout << "  Tolerence: " << tolerance << std::endl;
    // if (actualResults.size() != expectedResults.size())
    //     return false;

    size_t actualMaxErrId = 0;
    size_t expectedMaxErrId = 0;
    std::function<void(size_t, size_t)> absoluteErrorUpdater = [&](size_t actualIdx, size_t expectedIdx) {
        auto actual = actualResults[actualIdx];
        auto expected = expectedResults[expectedIdx];
        float abs_error = fabsf(actual - expected);
        float max_abs_error = expected * (tolerance/100.0f);
        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
            actualMaxErrId = actualIdx;
            expectedMaxErrId = expectedIdx;
        }
    };
    std::cout << "Expected\tActualVal" << std::endl;
    for (size_t n = 0; n < expectedResults.size(); ++n) 
    {
        size_t expectedVal = expectedResults[n];
        size_t actualVal = actualResults[n];

        std::cout << expectedVal << "\t\t" << actualVal << std::endl;
        absoluteErrorUpdater(actualVal, expectedVal);
    }

    std::cout << "  expectedMaxErrId = " << expectedMaxErrId << std::endl 
                << "  actualMaxErrId = " << actualMaxErrId << std::endl;
    if (actualMaxErrId > 0)
        return false;
    else
        return true;   
}


int main(int argc, char *argv[]) 
{
    test();
    exit(0);

    if (!ParseAndCheckCommandLine(argc, argv)) 
        return 0;

    //convert blob to json
    std::cout << "Converting blob to json... " << std::endl;
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
    std::cout << "Querying quantization values... " << std::endl;
    std::cout << "  Datatype: " << dtype << std::endl;
    std::cout << "  quant_zero: " << qZero << std::endl;
    std::cout << "  quant_scale: " << qScale << std::endl;

    // read size of output tensor
    int tSize = 1;
    for (int x=0; x<j["header"]["net_output"][0]["dimensions"].size(); ++x)
    {
        tSize *= j["header"]["net_output"][0]["dimensions"][x].get<int>();
    }
    std::cout << "  Output size: " << tSize << std::endl;

    // Read the InferenceManagerDemo output file into a vector
    std::cout << "Reading in actual results... " << std::endl;
    std::ifstream file(FLAGS_a, std::ios::binary);
    file.unsetf(std::ios::skipws);
    std::vector<unsigned char> outputVector;
    outputVector.reserve(tSize);
    outputVector.insert(outputVector.begin(),
        std::istream_iterator<unsigned char>(file),
        std::istream_iterator<unsigned char>());
    
    // De-quantize: real_value = scale * (quantized_value - zero_point)
    // Quantize: quantized_value = zero_point + real_value / scale
    std::cout << "Dequantizing actual results... " << std::endl;
    std::vector<float> outputFP32;
    for (int i = 0; i < tSize; ++i)
    {
        unsigned int x;
        std::stringstream ss;
        ss << std::hex << (int)outputVector[i];
        ss >> x;

        float val = qScale * (static_cast<int>(x) - qZero);
        outputFP32.push_back(val);

        //std::cout << ", " << val;
    }
    //std::cout << std::endl;

    // //write to file
    // ofstream fout("data.bin", ios::out | ios::binary);
    // fout.write((char*)&outputVector[0], outputVector.size() * sizeof(float));
    // fout.close();
    // //

    // Read in expected results tensor into a vector
    std::cout << "Reading in expected results... " << std::endl;
    std::ifstream infile(FLAGS_e, std::ios::binary);
    infile.seekg(0, infile.end);
    auto totalExpected = infile.tellg()  / sizeof(float);
    std::cout << "Total entries read from Expected: " << totalExpected << std::endl;
    infile.seekg(0, infile.beg);

    std::vector<float> expectedFP32(totalExpected);
    //infile.read(reinterpret_cast<char*>(expectedFP32.data()), expectedFP32.size()*sizeof(float));
    infile.read(reinterpret_cast<char*>(&expectedFP32[0]), totalExpected*sizeof(float));
    for (int i = 0; i < 20; ++i)
        std::cout << ", " << expectedFP32[i];
    std::cout << std::endl;
    
    // compare
    bool pass = false;
    pass = compare(outputFP32, expectedFP32, FLAGS_t);
    std::cout << "Validation status: " << ((pass) ? "true" : "false"); 
    std::cout << std:: endl;
}


