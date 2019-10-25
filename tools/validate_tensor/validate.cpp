#include "validate.hpp"
#include "include/mcm/utils/env_loader.hpp"

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
    }
    else    //normal operation
    {
        if (FLAGS_m.empty()) 
            throw std::logic_error("Parameter -m is not set");
        if (FLAGS_i.empty()) 
            throw std::logic_error("Parameter -i is not set");
        if (FLAGS_k.empty()) 
            throw std::logic_error("Parameter -k is not set");
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
        if (idx < 50) // print first 50 rows
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
    std::vector<std::string> filesDelete = {"output_cpu.bin", "input-0.bin", "kmb_release/kmb_release/*.blob"};
    for (std::string fDelete : filesDelete)
        remove(binFolder.append(fDelete).c_str());

    //
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
    if (!checkFilesExist({std::getenv("DLDT_HOME") + std::string("/bin/intel64/Debug/output_cpu.bin"), 
                          std::getenv("DLDT_HOME") + std::string("/bin/intel64/Debug/input.bin")} ))
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
    std::string inputSrc = std::getenv("DLDT_HOME") + std::string("/bin/intel64/Debug/input-0.bin");
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

int validate(std::string blobPath, std::string expectedPath, std::string actualPath)
{
    //
    // convert blob to json
    std::cout << "Converting blob to json... " << std::endl;
    std::string commandline = std::string("flatc -t ") + mv::utils::projectRootPath() +
        std::string("/schema/graphfile/src/schema/graphfile.fbs --strict-json -- ") + blobPath;
    std::cout << commandline << std::endl;
    std::system(commandline.c_str());
    
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
    for (uint32_t x=0; x<j["header"]["net_output"][0]["dimensions"].size(); ++x)
        tSize *= j["header"]["net_output"][0]["dimensions"][x].get<int>();
    std::cout << "  Output size: " << tSize << std::endl;

    //
    // Read the InferenceManagerDemo output file into a vector
    std::cout << "Reading in actual results... ";
    std::ifstream file(actualPath, std::ios::in | std::ios::binary);
    file.seekg(0, std::ios::end);
    auto totalActual = file.tellg() / sizeof(uint8_t);
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> outputVector(totalActual);
    file.read(reinterpret_cast<char*>(&outputVector[0]), totalActual*sizeof(uint8_t));
    std::cout << totalActual << " elements" <<  std::endl;

    // de-quantize
    std::vector<float> outputFP32;
    for(size_t i = 0; i < outputVector.size(); ++i)
    {
        // De-quantize: bitshift left by qShift then multiply by scale
        float val = outputVector[i] << qShift / qScale;
        outputFP32.push_back(val);
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
        validate(FLAGS_b, FLAGS_e, FLAGS_a);
        return(0);
    }
    //
    // Normal operation
    int result = 0;
    std::string blobPath("");
    result = runEmulator(FLAGS_m, FLAGS_i, blobPath);
    if ( result > 0 )
        return result;

    result = runKmbInference(FLAGS_k, blobPath);
    if ( result > 0 )
        return result;

    std::string expectedPath = std::getenv("DLDT_HOME") + std::string("/bin/intel64/Debug/output_cpu.bin");
    std::string actualPath = std::getenv("VPUIP_HOME") + std::string("/application/demo/InferenceManagerDemo/output-0.bin");
    result = validate(blobPath, expectedPath, actualPath);
    if ( result > 0 )
        return result;
    
    return(0);
}
