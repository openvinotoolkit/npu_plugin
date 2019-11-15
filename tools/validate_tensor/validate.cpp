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

// files used
const std::string FILE_CONVERTED_IMAGE  = "converted_image.dat";
const std::string FILE_CPU_OUTPUT       = "output_cpu.bin";
const std::string FILE_CPU_INPUT        = "input_cpu.bin";
const std::string FILE_CPU_INPUT_FP16   = "input_cpu_fp16.bin";
const std::string FILE_CPU_INPUT_NCHW   = "input_cpu_nchw.bin";
const std::string FILE_CPU_INPUT_NHWC   = "input_cpu_nhwc.bin";
const std::string DLDT_BIN_FOLDER       = "/bin/intel64/Debug/";
const std::string DLDT_BLOB_LOCATION    = "release_kmb/release_kmb/";



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
        if (FLAGS_k.empty()) 
            throw std::logic_error("Parameter -k is not set");
    }
 
    return true;
}

std::string findBlob(std::string folderPath)
{
    // OpenVino blob is written to locations based on input xml.
    // Its probably DLDT/bin/intel64/Debug/release_kmb/release_kmb
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
        std::cout << "  RESULTS SIZES DO NOT MATCH! Continuing..." << std::endl;
    //    return false;

    size_t maxErr = 0;
    float countErrs = 0;
    size_t sumDiff = 0;
    std::function<void(size_t)> absoluteErrorUpdater = [&](size_t idx) {
        float actual = actualResults[idx];
        float expected = expectedResults[idx];
        float abs_error = fabsf(actual - expected);
        float abs_allowed_err = fabsf(expected * (tolerance/100.0f));
        std::string result = "\tpass";
        if (abs_error > abs_allowed_err) 
        {
            countErrs++;
            sumDiff+=abs_error;
            if (abs_error > maxErr) maxErr = abs_error;
            result = "\tfail";
        }
        if (idx < 50) // print first 50 rows
            std::cout << expected << "\t" << actual << "\t" << abs_error << "\t" << abs_allowed_err << "\t"  << result << std::endl;
    };
    std::cout << "Printing first 50 rows...\nExp\tActual\tdiff\ttolerence\tresult" << std::endl;
    for (size_t n = 0; n < expectedResults.size(); ++n) 
        absoluteErrorUpdater(n);

    //print results report
    std::cout << "\nMetric\t\t\tActual\tThreshold\tStatus" << std::endl << "----------------------  ------  ---------\t-------" << std::endl;
    std::cout << "Incorrect Values\t" << std::setw(7) << ((countErrs/actualResults.size()) * 100) << "%" << std::setw(7) << tolerance << "%" << ((countErrs==0) ? "\t\033[1;32mPass" : "\t\033[1;31mFail") << "\033[0m" << std::endl;
    std::cout << "Highest Difference\t" << std::setw(8) << maxErr << std::setw(8) << "0" << std::setw(8) << ((maxErr==0) ? "\t\033[1;32mPass" : "\t\033[1;31mFail") << "\033[0m" << std::endl;
    std::cout << "Global Sum Difference\t" << std::setw(8) << sumDiff << std::setw(8) << "0" << std::setw(8) << ((sumDiff==0) ? "\t\033[1;32mPass" : "\t\033[1;31mFail") << "\033[0m" << std::endl << std::endl;

    if (maxErr == 0) return true;
    else return false;
}

bool checkFilesExist(std::vector<std::string> filepaths)
{
    //checks if files in the supplied vector exist
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
    std::string binFolder = std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER;
    std::vector<std::string> filesDelete = {FILE_CPU_OUTPUT, FILE_CPU_INPUT_NCHW, FILE_CPU_INPUT_NHWC};
    for (std::string fDelete : filesDelete)
        remove((binFolder + fDelete).c_str());
    
    do
    {   //delete any previous blobs (different names each time)
        blobPath = findBlob(std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + DLDT_BLOB_LOCATION);
        std::string fullBlobPath = binFolder + DLDT_BLOB_LOCATION + blobPath;
        std::cout << "Removing: " << fullBlobPath << std::endl;
        remove(fullBlobPath.c_str());
    } while (blobPath != "");
    
    // execute the classification sample async (CPU-plugin)
    std::cout << "Generating reference results... " << std::endl;
    std::string commandline = std::string("cd ") + std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + "  && " + 
        "./test_classification -m " + pathXML + " -d CPU";
    if (! FLAGS_i.empty() )
        commandline += (" -i " + pathImage);

    std::cout << commandline << std::endl; 
    int returnVal = std::system(commandline.c_str());
    if (returnVal != 0)
    {
        std::cout << std::endl << "Error occurred running the test_classification (CPU mode)!" << std::endl;
        return FAIL_ERROR;
    }
    if (!checkFilesExist( {std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + FILE_CPU_OUTPUT} ))
        return FAIL_CPU_PLUGIN;
    
    if (!checkFilesExist( {std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + FILE_CPU_INPUT_NCHW} ))
        return FAIL_CPU_PLUGIN;

    if (!checkFilesExist( {std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + FILE_CPU_INPUT_NHWC} ))
        return FAIL_CPU_PLUGIN;

    //
    // execute the classification sample async (KMB-plugin)
    std::cout << "Generating mcm blob through kmb-plugin... " << std::endl;
    commandline = std::string("cd ") + std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + " && " + 
        "./test_classification -m " + pathXML + " -d KMB";
    if (! FLAGS_i.empty() )
        commandline += (" -i " + pathImage);

    std::cout << commandline << std::endl;
    std::system(commandline.c_str());
    if (returnVal != 0)
    {
        std::cout << std::endl << "Error occurred running the test_classification (KMB mode)!" << std::endl;
        return FAIL_ERROR;
    }
    
    blobPath = findBlob(std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + DLDT_BLOB_LOCATION);
    if (blobPath == "")
    {
        std::cout << "Error! Couldn't find the generated blob in " << std::getenv("DLDT_HOME") << DLDT_BIN_FOLDER << DLDT_BLOB_LOCATION << std::endl;
        return FAIL_COMPILER;
    }
    blobPath = std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + DLDT_BLOB_LOCATION + blobPath;
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
    // Clean old results
    std::cout << "Deleting old kmb results files... " << std::endl;
    std::string outputFile = std::getenv("VPUIP_HOME") + std::string("/application/demo/InferenceManagerDemo/output-0.bin");
    remove(outputFile.c_str());

    // copy the required files to InferenceManagerDemo folder
    std::string inputCPU = std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + FILE_CPU_INPUT;
    // std::string inputCPU = FILE_CONVERTED_IMAGE;
    std::string inputDest = std::getenv("VPUIP_HOME") + std::string("/application/demo/InferenceManagerDemo/input-0.bin");
    //if (!copyFile(FILE_CONVERTED_IMAGE, inputDest)) return FAIL_GENERAL;
    if (!copyFile(inputCPU, inputDest))
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
    // Clean old file
    std::cout << "Deleting old json... " << std::endl;
    std::string outputFile = getFilename(blobPath) + std::string(".json");
    remove(outputFile.c_str());

    std::string commandline = std::string("flatc -t ") + mv::utils::projectRootPath() +
        std::string("/schema/graphfile/src/schema/graphfile.fbs --strict-json -- ") + blobPath;
    std::cout << commandline << std::endl;
    int result = std::system(commandline.c_str());
    if (result != 0)
    {
        std::cout << "Error occurred trying to convert blob to json. Please check Flatc in path and graphfiles" << std::endl;
        return FAIL_GENERAL;
    }
    if (!checkFilesExist({outputFile}))
         return FAIL_ERROR;
    
    return RESULT_SUCCESS;
}

int validate(std::string blobPath, std::string expectedPath, std::string actualPath)
{
    // load json and read quantization values: scale and zeropoint
    std::string json_file = getFilename(blobPath) + std::string(".json");
    std::ifstream ifile(json_file);
    json j = json::parse(ifile);
    std::string dtype = j["header"]["net_output"][0]["data_dtype"].get<std::string>();

    uint_fast16_t typesize = 1;
    if (dtype == "U8")
        typesize = 1;
    else if (dtype == "FP16")
        typesize = 2;
    else
    {
        std::cout << "Typesize of output layer is unsupported: " << dtype << std::endl;
        return FAIL_ERROR;
    }
    
    // Read the InferenceManagerDemo output file into a vector
    std::cout << "Reading in actual results... ";
    std::ifstream file(actualPath, std::ios::in | std::ios::binary);
    file.seekg(0, std::ios::end);
    auto totalActual = file.tellg() / typesize;
    file.seekg(0, std::ios::beg);

    std::vector<float> outputFP32;
    if (dtype.compare("U8")==0)
    {
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
            // Other De-quantize formula: real_value  = Scale * ( quantized_value - zero_point)
            //float val2 = qScale * (outputVector[i] - qZero);

            // De-quantize: bitshift left by qShift then multiply by scale
            float val = static_cast<float>(outputVector[i] << qShift) / static_cast<float>(qScale);
            outputFP32.push_back(val);
        }
    }
    else if(dtype.compare("FP16")==0)
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

    // compare
    bool pass = false;
    pass = compare(outputFP32, expectedFP32, FLAGS_t);
    std::cout << "Validation status: " << ((pass) ? "\033[1;32mPass" : "\033[1;31mFail") << "\033[0m" << std::endl; 
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

    if ((imagePath.find("bin") != std::string::npos))
    {
        std::string binFolder = std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER;
        if(sZMajor == " --zmajor")
        {
            remove((binFolder + FILE_CPU_INPUT_NCHW ).c_str());
            rename((binFolder + FILE_CPU_INPUT_NHWC).c_str(),(binFolder + FILE_CPU_INPUT).c_str() );
        }
        else
        {
            remove((binFolder + FILE_CPU_INPUT_NHWC ).c_str());
            rename((binFolder + FILE_CPU_INPUT_NCHW).c_str(),(binFolder + FILE_CPU_INPUT).c_str() );
        }
        std::string dtype = j["header"]["net_output"][0]["data_dtype"].get<std::string>();
        if (dtype == "FP16")
        {
        std::string inputDest = std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + FILE_CPU_INPUT_FP16;
        std::string inputSource = std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + FILE_CPU_INPUT;
        std::string cpubackup = std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + "input_cpu_fp32.bin";

        std::ofstream fileOut(inputDest, std::ios::out | std::ios::binary);
        std::ifstream fileIn(inputSource, std::ios::in | std::ios::binary);

        fileIn.seekg(0, std::ios::end);
        auto totalActual = fileIn.tellg() / sizeof(float);
        fileIn.seekg(0, std::ios::beg);

        std::vector<u_int16_t> inputVectorFP16;
        std::vector<float> inputVectorFP32(totalActual);
        fileIn.read(reinterpret_cast<char *>(&inputVectorFP32[0]), totalActual * sizeof(float));
        for (size_t i = 0; i < inputVectorFP32.size(); ++i)
            inputVectorFP16.push_back(mv::fp32_to_fp16(inputVectorFP32[i]));
        fileOut.write(reinterpret_cast<char *>(&inputVectorFP16[0]), totalActual * sizeof(u_int16_t) );
        fileOut.close();
        fileIn.close();
        rename(inputSource.c_str(), cpubackup.c_str());
        rename(inputDest.c_str(), inputSource.c_str());
        }
    }
    else
    {
    // Clean old file
    std::cout << "Deleting old input... " << std::endl;
    std::string outputFile = "./converted_image.dat";
    remove(outputFile.c_str());
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
    if (!checkFilesExist({outputFile}))
         return FAIL_ERROR;
    }

    return RESULT_SUCCESS;    
}

int postProcessActualResults(std::string resultsPath, std::string blobPath)
{
    // Clean old file
    std::cout << "Deleting old output... " << std::endl;
    std::string outputFile = "./output_transposed.dat";
    remove(outputFile.c_str());
    
    // load json to read output size and ch/z major transpose
    std::string json_file = getFilename(blobPath) + std::string(".json");
    std::ifstream ifile(json_file);
    json j = json::parse(ifile);

    std::cout << "Post Processing results... " << std::endl;
    std::string dtype = j["header"]["net_output"][0]["data_dtype"].get<std::string>();
    std::cout << "Datatype: " << dtype << std::endl;
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
        std::string("/python/tools/post_process.py --file ") + resultsPath + " --dtype " + dtype + " --shape " + 
        outputShape[0] + "," + outputShape[1] + "," + outputShape[2] + "," + outputShape[3] + sZMajor;
    std::cout << commandline << std::endl;
    int result = std::system(commandline.c_str());
    
    if (result > 0)
    {
        std::cout << "Error occured converting image using python script";
        return FAIL_ERROR;
    }
    if (!checkFilesExist({outputFile}))
         return FAIL_ERROR;

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

    // Normal operation
    int result = 0;

    std::string blobPath("");
    result = runEmulator(FLAGS_m, FLAGS_i, blobPath);
    if ( result > 0 ) return result;

    result = convertBlobToJson(blobPath);
    if ( result > 0 ) return result;

    if (! FLAGS_i.empty()) 
    {
        result = convertImage(FLAGS_i, blobPath);
        if ( result > 0 ) return result;
    }
    else
    {
        // both Zmajor and Cmajor inputs available. So passing any one is ok to check and delete the unwanted one
        std::string binPath = std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + FILE_CPU_INPUT_NCHW;
        result = convertImage(binPath, blobPath);
    }
    result = runKmbInference(FLAGS_k, blobPath);
    if ( result > 0 ) return result;

    std::string expectedPath = std::getenv("DLDT_HOME") + std::string("/bin/intel64/Debug/output_cpu.bin");
    std::string actualPath = std::getenv("VPUIP_HOME") + std::string("/application/demo/InferenceManagerDemo/output-0.bin");
    std::string actualPathProcessed = "./output_transposed.dat";

    //result = postProcessActualResults(actualPath, blobPath);
    //if ( result > 0 ) return result;

    //result = validate(blobPath, expectedPath, actualPathProcessed);
    result = validate(blobPath, expectedPath, actualPath);
    if ( result > 0 )
        return result;
    
    return(0);
}
