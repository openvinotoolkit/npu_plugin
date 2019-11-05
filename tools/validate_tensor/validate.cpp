#include "validate.hpp"
#include "include/mcm/utils/env_loader.hpp"
#include "include/mcm/utils/custom_math.hpp"

#include <fstream>
#include <sys/stat.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <dirent.h>
#include <iomanip>
#include <vector>

using json = nlohmann::json;

/**
 * Required environmental variables
 * VPUIP_HOME = <path to vpuip_2 repo>
 * DLDT_HOME  = <path to DLDT repo>
 * 
 * All the filenames are hardcoded. This should be updated.
 * 
 */

// Error Code Guide
const int8_t RESULT_SUCCESS  = 0;
const int8_t FAIL_GENERAL    = 1;
const int8_t FAIL_CPU_PLUGIN = 2;  // CPU plugin fails to create emulator files (output_cpu.bin, input-0.bin)
const int8_t FAIL_COMPILER   = 3;  // KMB plugin/MCM fails to create blob file  (mcm.blob)
const int8_t FAIL_RUNTIME    = 4;  // Runtime fails to generate results file    (output-0.dat)
const int8_t FAIL_VALIDATION = 5;  // Correctness Error between results files
const int8_t FAIL_ERROR      = 9;  // Error occured during run, check log


bool ParseAndCheckCommandLine(int argc, char *argv[]) 
{
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) 
    {
        showUsage();
        return false;
    }

    if (FLAGS_mode == "validate")
    {
        if (FLAGS_b.empty()) 
            throw std::logic_error("Parameter -b must be set in validation mode");
        if (FLAGS_e.empty()) 
            throw std::logic_error("Parameter -e must be set in validation mode");
        if (FLAGS_a.empty()) 
            throw std::logic_error("Parameter -a must be set in validation mode");
        if ((FLAGS_d.compare("U8") != 0 ) && (FLAGS_d.compare("FP16") != 0 ))  
            throw std::logic_error("Parameter -d is not set correctly");
    }
    else    //normal operation
    {
        if (FLAGS_m.empty()) 
            throw std::logic_error("Parameter -m is not set");
        if (FLAGS_i.empty()) 
            throw std::logic_error("Parameter -i is not set");
        if (FLAGS_k.empty()) 
            throw std::logic_error("Parameter -k is not set");
        if ((FLAGS_d.compare("U8") != 0 ) && (FLAGS_d.compare("FP16") != 0 ))  
            throw std::logic_error("Parameter -d is not set correctly");
    }
 
    return true;
}

std::string findBlob(std::string folderPath)
{
    std::string blobPath("");
    if (auto dir = opendir(folderPath.c_str())) 
    {
        while (auto f = readdir(dir)) 
        {
            if (!f->d_name || f->d_name[0] == '.')
                continue;

            if ( strstr( f->d_name, ".blob" ))
            {
                blobPath =  f->d_name;
                break;
            }
        }
        closedir(dir);
    }
    return blobPath;
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
    std::cout << "  Tolerence: " << tolerance << "%" << std::endl;
    if (actualResults.size() != expectedResults.size())
        return false;

    size_t maxErr = 0;
    std::function<void(size_t)> absoluteErrorUpdater = [&](size_t idx) {
        float actual = actualResults[idx];
        float expected = expectedResults[idx];
        float abs_error = fabsf(actual - expected);
        float abs_allowed_err = fabsf(expected * (tolerance/100.0f));
        std::string result = "\tpass";
        if (abs_error > abs_allowed_err) 
        {
            if (abs_error > maxErr) maxErr = abs_error;
            result = "\tfail";
        }
        if (idx < 250 && idx > 200) // print first 50 rows
            std::cout << expected << "\t" << actual << "\t" << abs_error << "\t" << abs_allowed_err << "\t"  << result << std::endl;
    };
    std::cout << "Printing first 50 rows...\nExp\tActual\tdiff\ttolerence\tresult" << std::endl;
    for (size_t n = 0; n < expectedResults.size(); ++n) 
        absoluteErrorUpdater(n);

    if (maxErr == 0) return true;
    else return false;
}

bool checkFilesExist(std::vector<std::string> filepaths)
{
    for(std::string fPath : filepaths)
    {
        struct stat buffer;
        if (stat (fPath.c_str(), &buffer) != 0)
        {
            std::cout << "File: " << fPath << " not produced!" << std::endl;
            return false;
        }
    }
    return true;
}

int runEmulator(std::string pathXML, std::string pathImage, std::string& blobPath)
{
    //
    // Clean any old files
    std::cout << "Deleting old emulator results files... " << std::endl;
    std::string binFolder = std::getenv("DLDT_HOME") + std::string("/bin/intel64/Debug/");
    std::vector<std::string> filesDelete = {"output_cpu.bin", "input-0.bin"};
    for (std::string fDelete : filesDelete)
        remove((binFolder + fDelete).c_str());
    
    do
    {   //delete any previous blobs (different names each time)
        blobPath = findBlob(std::getenv("DLDT_HOME") + std::string("/bin/intel64/Debug/release_kmb/release_kmb/"));
        std::string fullBlobPath = binFolder + "release_kmb/release_kmb/" + blobPath;
        std::cout << "Removing: " << fullBlobPath << std::endl;
        remove(fullBlobPath.c_str());
    } while (blobPath != "");
    
    // execute the classification sample async (CPU-plugin)
    std::cout << "Generating reference results... " << std::endl;
    std::string commandline = std::string("cd ") + std::getenv("DLDT_HOME") + "/bin/intel64/Debug  && " + 
        "./classification_sample_async -m " + pathXML + " -i " + pathImage + " -d CPU";
    std::cout << commandline << std::endl; 
    int returnVal = std::system(commandline.c_str());
    if (returnVal != 0)
    {
        std::cout << std::endl << "Error occurred running the classification_sample_async (CPU mode)!" << std::endl;
        return FAIL_ERROR;
    }
    if (!checkFilesExist( {std::getenv("DLDT_HOME") + std::string("/bin/intel64/Debug/output_cpu.bin")} ))
        return FAIL_CPU_PLUGIN;

    //
    // execute the classification sample async (KMB-plugin)
    std::cout << "Generating mcm blob through kmb-plugin... " << std::endl;
    commandline = std::string("cd ") + std::getenv("DLDT_HOME") + "/bin/intel64/Debug  && " + 
        "./classification_sample_async -m " + pathXML + " -i " + pathImage + " -d KMB";
    std::cout << commandline << std::endl;
    std::system(commandline.c_str());
    if (returnVal != 0)
    {
        std::cout << std::endl << "Error occurred running the classification_sample_async (KMB mode)!" << std::endl;
        return FAIL_ERROR;
    }
    
    blobPath = findBlob(std::getenv("DLDT_HOME") + std::string("/bin/intel64/Debug/release_kmb/release_kmb/"));
    if (blobPath == "")
    {
        std::cout << "Error! Couldn't find the generated blob in " << std::getenv("DLDT_HOME") << "/bin/intel64/Debug/release_kmb/release_kmb/" << std::endl;
        return FAIL_COMPILER;
    }
    blobPath = std::getenv("DLDT_HOME") + std::string("/bin/intel64/Debug/release_kmb/release_kmb/") + blobPath;
    return RESULT_SUCCESS;
}

bool copyFile(std::string src, std::string dest)
{
    std::string commandline = std::string("cp ") + src + std::string (" ") + dest;
    std::cout << commandline << std::endl; 
    int returnVal = std::system(commandline.c_str());
    if (returnVal != 0)
    {
        std::cout << std::endl << "Error occurred copying files!" << std::endl;
        return false;
    }
    return true;
}

int runKmbInference(std::string evmIP, std::string blobPath)
{
    //
    // Clean old results
    std::cout << "Deleting old kmb results files... " << std::endl;
    std::string outputFile = std::getenv("VPUIP_HOME") + std::string("/application/demo/InferenceManagerDemo/output-0.bin");
    remove(outputFile.c_str());

    //
    // copy the required files to InferenceManagerDemo folder
    std::string inputSrc = "./converted_image.dat";
    std::string inputDest = std::getenv("VPUIP_HOME") + std::string("/application/demo/InferenceManagerDemo/input-0.bin");
    if (!copyFile(inputSrc, inputDest))
        return FAIL_GENERAL;
    
    std::string blobDest = std::getenv("VPUIP_HOME") + std::string("/application/demo/InferenceManagerDemo/test.blob");
    if (!copyFile(blobPath, blobDest))
        return FAIL_GENERAL;

    // execute the blob
    std::string commandline = std::string("cd ") + std::getenv("VPUIP_HOME") + "/application/demo/InferenceManagerDemo  && " + 
        "make run srvIP=" + evmIP;
    std::cout << commandline << std::endl;
    int returnVal = std::system(commandline.c_str());
    if (returnVal != 0)
    {
        std::cout << std::endl << "Error occurred executing blob on runtime!" << std::endl;
        return FAIL_ERROR;
    }
    if (!checkFilesExist({outputFile}))
         return FAIL_RUNTIME;
    
    return RESULT_SUCCESS;
}

int convertBlobToJson(std::string blobPath)
{
    // convert blob to json
    std::cout << "Converting blob to json... " << std::endl;
    std::string commandline = std::string("flatc -t ") + mv::utils::projectRootPath() +
        std::string("/schema/graphfile/src/schema/graphfile.fbs --strict-json -- ") + blobPath;
    std::cout << commandline << std::endl;
    int result = std::system(commandline.c_str());
    if (result != 0)
    {
        std::cout << "Error occurred trying to convert blob to json. Please check Flatc in path and graphfiles" << std::endl;
        return FAIL_GENERAL;
    }
    else
        return RESULT_SUCCESS;
}

int validate(std::string blobPath, std::string expectedPath, std::string actualPath)
{
    uint_fast16_t typesize = (FLAGS_d.compare("U8")==0) ? 1 : 2 ;

    //
    // Read the InferenceManagerDemo output file into a vector
    std::cout << "Reading in actual results... ";
    std::ifstream file(actualPath, std::ios::in | std::ios::binary);
    file.seekg(0, std::ios::end);
    auto totalActual = file.tellg() / typesize;
    file.seekg(0, std::ios::beg);

    std::vector<float> outputFP32;
    if (FLAGS_d.compare("U8")==0)
    {
        //
        // load json and read quantization values: scale and zeropoint
        std::string json_file = getFilename(blobPath) + std::string(".json");
        std::ifstream ifile(json_file);
        json j = json::parse(ifile);

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
        for (uint32_t x = 0; x < j["header"]["net_output"][0]["dimensions"].size(); ++x)
            tSize *= j["header"]["net_output"][0]["dimensions"][x].get<int>();
        std::cout << "  Output size: " << tSize << std::endl;

        std::vector<uint8_t> outputVector(totalActual);
        file.read(reinterpret_cast<char *>(&outputVector[0]), totalActual * sizeof(uint8_t));
        std::cout << totalActual << " elements" << std::endl;

        // de-quantize
        for (size_t i = 0; i < outputVector.size(); ++i)
        {
            // De-quantize: bitshift left by qShift then multiply by scale
            float val = outputVector[i] << qShift / qScale;
            outputFP32.push_back(val);
        }
    }
    else if(FLAGS_d.compare("FP16")==0)
    {
        std::vector<u_int16_t> outputVector(totalActual);
        file.read(reinterpret_cast<char *>(&outputVector[0]), totalActual * typesize);
        std::cout << totalActual << " elements" << std::endl;
        for (size_t i = 0; i < outputVector.size(); ++i)
        {
            float val = mv::fp16_to_fp32(outputVector[i]);
            outputFP32.push_back(val);
        }
    }
    
    std::cout << "Reading in expected results... ";
    std::ifstream infile(expectedPath, std::ios::binary);
    infile.seekg(0, infile.end);
    auto totalExpected = infile.tellg() / sizeof(float);
    std::cout << totalExpected << " elements" << std::endl;
    infile.seekg(0, infile.beg);

    std::vector<float> expectedFP32(totalExpected);
    infile.read(reinterpret_cast<char*>(&expectedFP32[0]), totalExpected*sizeof(float));

    //    
    // compare
    bool pass = false;
    pass = compare(outputFP32, expectedFP32, FLAGS_t);
    std::cout << "Validation status: " << ((pass) ? "true" : "false"); 
    std::cout << std:: endl;
    if (pass)
        return RESULT_SUCCESS;
    else 
        return FAIL_VALIDATION;
}

int convertImage(std::string imagePath, std::string blobPath)
{
    // load json and read quantization values: scale and zeropoint
    std::string json_file = getFilename(blobPath) + std::string(".json");
    std::ifstream ifile(json_file);
    json j = json::parse(ifile);

    std::cout << "Querying input shape... " << std::endl;
    std::vector<std::string> inputShape;
    for (uint32_t x=0; x<j["header"]["net_input"][0]["dimensions"].size(); ++x)
        inputShape.push_back( std::to_string(j["header"]["net_input"][0]["dimensions"][x].get<int>()) );
    std::cout << "Input Shape: " << inputShape[0] << "," << inputShape[1] << "," << inputShape[2] << "," << inputShape[3] << std::endl;

    std::cout << "Querying Z/Ch Major conv... " << std::endl;
    std::vector<std::string> inputStrides;
    for (uint32_t x=0; x<j["header"]["net_input"][0]["strides"].size(); ++x)
        inputStrides.push_back( std::to_string(j["header"]["net_input"][0]["strides"][x].get<int>()) );
    std::cout << "Input Strides: " << inputStrides[0] << "," << inputStrides[1] << "," << inputStrides[2] << "," << inputStrides[3] << std::endl;

    std::string sZMajor("");
    if (! ((std::stoi(inputShape[1]) < 16) && (inputShape[2] == inputStrides[3]) ))
        sZMajor = " --zmajor";

    //
    // convert image to correct shape and order
    std::cout << "Converting image ... " << std::endl;
    std::string commandline = std::string("python3 ") + mv::utils::projectRootPath() +
        std::string("/python/tools/convert_image.py --image ") + imagePath + " --shape " + 
        inputShape[0] + "," + inputShape[1] + "," + inputShape[2] + "," + inputShape[3] + sZMajor;
    std::cout << commandline << std::endl;
    int result = std::system(commandline.c_str());
    
    if (result > 0)
    {
        std::cout << "Error occured converting image using python script";
        return FAIL_ERROR;
    }
    return RESULT_SUCCESS;    
}

int postProcessActualResults(std::string resultsPath, std::string blobPath)
{
    // load json to read output size and ch/z major transpose
    std::string json_file = getFilename(blobPath) + std::string(".json");
    std::ifstream ifile(json_file);
    json j = json::parse(ifile);

    std::cout << "Post Processing results... " << std::endl;
    std::vector<std::string> outputShape;
    for (uint32_t x=0; x<j["header"]["net_output"][0]["dimensions"].size(); ++x)
        outputShape.push_back( std::to_string(j["header"]["net_output"][0]["dimensions"][x].get<int>()) );
    std::cout << "Output Shape: " << outputShape[0] << "," << outputShape[1] << "," << outputShape[2] << "," << outputShape[3] << std::endl;

    std::cout << "Querying Z/Ch Major output... " << std::endl;
    std::vector<std::string> outputStrides;
    for (uint32_t x=0; x<j["header"]["net_output"][0]["strides"].size(); ++x)
        outputStrides.push_back( std::to_string(j["header"]["net_output"][0]["strides"][x].get<int>()) );
    std::cout << "Output Strides: " << outputStrides[0] << "," << outputStrides[1] << "," << outputStrides[2] << "," << outputStrides[3] << std::endl;

    std::string sZMajor("");
    if (! ((std::stoi(outputShape[1]) < 16) && (outputShape[2] == outputStrides[3]) ))
        sZMajor = " --zmajor";

    //
    // call python script for numpy reshape/transpose
    std::string commandline = std::string("python3 ") + mv::utils::projectRootPath() +
        std::string("/python/tools/post_process.py --file ") + resultsPath + " --shape " + 
        outputShape[0] + "," + outputShape[1] + "," + outputShape[2] + "," + outputShape[3] + sZMajor;
    std::cout << commandline << std::endl;
    int result = std::system(commandline.c_str());
    
    if (result > 0)
    {
        std::cout << "Error occured converting image using python script";
        return FAIL_ERROR;
    }
    return RESULT_SUCCESS;    
}

int main(int argc, char *argv[]) 
{
    if(std::getenv("DLDT_HOME") == NULL)
    {
        std::cout << "ERROR! Environmental variable DLDT_HOME must be set with path to DLDT repo" << std::endl << std::endl;
        return FAIL_GENERAL;
    }
    
    if(std::getenv("VPUIP_HOME") == NULL)
    {
        std::cout << "ERROR! Environmental variable VPUIP_HOME must be set with path to VPUIP_2 repo" << std::endl << std::endl;
        return FAIL_GENERAL;
    }

    if (!ParseAndCheckCommandLine(argc, argv)) 
        return FAIL_ERROR;

    if (FLAGS_mode == "validate")
    {
        //bypass all and just run the validation function
        convertBlobToJson(FLAGS_b);
        validate(FLAGS_b, FLAGS_e, FLAGS_a);
        return(0);
    }
    //
    // Normal operation
    int result = 0;

    std::string blobPath("");
    result = runEmulator(FLAGS_m, FLAGS_i, blobPath);
    if ( result > 0 ) return result;

    result = convertBlobToJson(blobPath);
    if ( result > 0 ) return result;

    result = convertImage(FLAGS_i, blobPath);
    if ( result > 0 ) return result;

    result = runKmbInference(FLAGS_k, blobPath);
    if ( result > 0 ) return result;

    std::string expectedPath = std::getenv("DLDT_HOME") + std::string("/bin/intel64/Debug/output_cpu.bin");
    std::string actualPath = std::getenv("VPUIP_HOME") + std::string("/application/demo/InferenceManagerDemo/output-0.bin");
    std::string actualPathProcessed = "./output_transposed.dat";

    result = postProcessActualResults(actualPath, blobPath);
    if ( result > 0 ) return result;

    result = validate(blobPath, expectedPath, actualPathProcessed);
    if ( result > 0 )
        return result;
    
    return(0);
}
