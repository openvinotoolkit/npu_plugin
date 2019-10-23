#include "validate.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

#include <iomanip>
#include <vector>

using json = nlohmann::json;

int test()
{
    int Q_ZERO_POINT = 0;
    int Q_SCALE = 16;

    //Generate a file with some floats in it for testing
    std::ofstream fout("Dequantize.bin", std::ios::binary);
    std::ofstream qout("Quantize.bin", std::ios::binary);
    std::ofstream testfile("u8vals.bin", std::ios::binary);

    for(size_t i = 0; i < 10; ++i)
    {
        testfile.write(reinterpret_cast<const char*>(144u), sizeof(uint8_t));

        float f = 30.14159f * i;
        fout.write(reinterpret_cast<const char*>(&f), sizeof(f));

        // Quantize: quantized_value = zero_point + real_value / scale
        unsigned int q = Q_ZERO_POINT + f / Q_SCALE;
        std::cout << f << "\t" << q << std::endl;
        qout.write(reinterpret_cast<const char*>(&q), sizeof(q));
    }
    fout.close();
    qout.close();

    //
    std::cout << "Loading test values: " << std::endl;
    std::ifstream testIn("u8vals.bin", std::ios::in | std::ios::binary);
    testIn.seekg(0, std::ios::end);
    size_t num_vals = testIn.tellg() / sizeof(uint8_t);
    testIn.seekg(0, std::ios::beg);
    std::vector<uint8_t> dataTest(num_vals);
    testIn.read(reinterpret_cast<char*>(&dataTest[0]), num_vals*sizeof(uint8_t));
    for(size_t i = 0; i < dataTest.size(); ++i)
        std::cout << dataTest[i] << ", ";
    testIn.close();
    return 0;
    //

    std::cout << "Normal vals from file: " << std::endl;
    std::ifstream fin("Dequantize.bin", std::ios::in | std::ios::binary);
    fin.seekg(0, std::ios::end);
    const size_t num_elements = fin.tellg() / sizeof(float);
    fin.seekg(0, std::ios::beg);
    std::vector<float> data(num_elements);
    fin.read(reinterpret_cast<char*>(&data[0]), num_elements*sizeof(float));
    for(size_t i = 0; i < data.size(); ++i)
        std::cout << std::fixed << data[i] << "\n";

    std::cout << "Dequantised vals from Quantised file: " << std::endl;
    std::ifstream fin2("Quantize.bin", std::ios::in | std::ios::binary);
    fin2.seekg(0, std::ios::end);
    const size_t num_elements2 = fin2.tellg() / sizeof(unsigned int);
    fin2.seekg(0, std::ios::beg);
    std::vector<int> data2(num_elements2);
    fin2.read(reinterpret_cast<char*>(&data2[0]), num_elements2*sizeof(unsigned int));
    for(size_t i = 0; i < data2.size(); ++i)
    {
        // De-quantize: real_value = scale * (quantized_value - zero_point)
        float val = Q_SCALE * (static_cast<int>(data2[i]) - Q_ZERO_POINT);
        std::cout << data2[i] << "\t" << val << std::endl;
    }
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
    bool testResult = true;
    std::cout << "Comparing results ... " << std::endl;
    std::cout << "  Actual Results size: " << actualResults.size() << std::endl;
    std::cout << "  Expected Results size: " << expectedResults.size() << std::endl;
    std::cout << "  Tolerence: " << tolerance << "%" << std::endl;
    // if (actualResults.size() != expectedResults.size())
    //     return false;

    size_t actualMaxErrId = 0;
    size_t expectedMaxErrId = 0;
    std::function<void(size_t)> absoluteErrorUpdater = [&](size_t idx) {
        float actual = actualResults[idx];
        float expected = expectedResults[idx];
        float abs_error = fabsf(actual - expected);
        float abs_allowed_err = expected * (tolerance/100.0f);
        std::string result = "\tpass";
        if (abs_error > abs_allowed_err) {
            testResult = false;
            result = "\tfail";
        }
        if (idx < 50) // print first 50 rows
            std::cout << expected << "\t" << actual << "\t" << abs_error << "\t" << abs_allowed_err << "\t"  << result << std::endl;
    };
    std::cout << "Printing first 50 rows...\nExp\tActual\tdiff\ttolerence\tresult" << std::endl;
    for (size_t n = 0; n < expectedResults.size(); ++n) 
        absoluteErrorUpdater(n);

    return testResult;
}


int main(int argc, char *argv[]) 
{
    // test();
    // exit(0);

    if (!ParseAndCheckCommandLine(argc, argv)) 
        return 0;

    //
    // convert blob to json
    //
    std::cout << "Converting blob to json... " << std::endl;
    std::string commandline = std::string("flatc -t ") + std::getenv("GRAPHFILE") + std::string("/src/schema/graphfile.fbs --strict-json -- ") + FLAGS_b;
    std::cout << commandline << std::endl;
    std::system(commandline.c_str());
    
    //
    // load json and read quantization values: scale and zeropoint
    //
    std::string json_file = getFilename(FLAGS_b) + std::string(".json");
    std::ifstream i(json_file);
    json j = json::parse(i);

    std::string dtype = j["header"]["net_output"][0]["data_dtype"].get<std::string>();
    int qZero = j["header"]["net_output"][0]["quant_zero"][0].get<int>();
    int qScale = j["header"]["net_output"][0]["quant_scale"][0].get<int>();
    int qShift = j["header"]["net_output"][0]["quant_shift"][0].get<int>();
    std::cout << "Querying quantization values... " << std::endl;
    std::cout << "  Datatype: " << dtype << std::endl;
    std::cout << "  quant_zero: " << qZero << std::endl;
    std::cout << "  quant_scale: " << qScale << std::endl;
    std::cout << "  quant_shift: " << qShift << std::endl;

    // read size of output tensor
    int tSize = 1;
    for (int x=0; x<j["header"]["net_output"][0]["dimensions"].size(); ++x)
        tSize *= j["header"]["net_output"][0]["dimensions"][x].get<int>();
    std::cout << "  Output size: " << tSize << std::endl;

    //
    // Read the InferenceManagerDemo output file into a vector
    //
    std::cout << "Reading in actual results... ";
    std::ifstream file(FLAGS_a, std::ios::in | std::ios::binary);
    file.seekg(0, std::ios::end);
    auto totalActual = file.tellg() / sizeof(uint8_t);
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> outputVector(totalActual);
    file.read(reinterpret_cast<char*>(&outputVector[0]), totalActual*sizeof(uint8_t));
    std::cout << totalActual << std::endl;
    for(size_t i = 0; i < 20; ++i)  //print first 20 values
        std::cout << outputVector[i] << ", ";
    std::cout << std::endl;

    // de-quantize
    std::vector<float> outputFP32;
    for(size_t i = 0; i < outputVector.size(); ++i)
    {
        // De-quantize: bitshift left by qShift then multiply by scale
        float val = static_cast<uint8_t>(outputVector[i])<<qShift * qScale;
        outputFP32.push_back(val);
    }
    for(size_t i = 0; i < 20; ++i)
        std::cout << outputFP32[i] << ", ";
    std::cout << std::endl;
    
    //
    // Read in expected results tensor into a vector (CPU-plugin)
    //
    std::cout << "Reading in expected results... ";
    std::ifstream infile(FLAGS_e, std::ios::binary);
    infile.seekg(0, infile.end);
    auto totalExpected = infile.tellg()  / sizeof(float);
    std::cout << totalExpected << std::endl;
    infile.seekg(0, infile.beg);

    std::vector<float> expectedFP32(totalExpected);
    infile.read(reinterpret_cast<char*>(&expectedFP32[0]), totalExpected*sizeof(float));

    //    
    // compare
    //
    bool pass = false;
    pass = compare(outputFP32, expectedFP32, FLAGS_t);
    std::cout << "Validation status: " << ((pass) ? "true" : "false"); 
    std::cout << std:: endl;
}
