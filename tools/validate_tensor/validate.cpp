#include "validate.hpp"
#include "include/mcm/utils/env_loader.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "flatbuffers/flatbuffers.h"
#include "schema/graphfile/graphfile_generated.h"
#include <fstream>
#include <math.h>
#include <sys/stat.h>
#include <sstream>
#include <dirent.h>
#include <iomanip>
#include <vector>

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
const std::string FILE_CPU_INPUT_NCHW_RGB   = "input_cpu_nchw_rgb.bin";
const std::string FILE_CPU_INPUT_NHWC_RGB   = "input_cpu_nhwc_rgb.bin";
const std::string FILE_CPU_INPUT_NCHW_BGR   = "input_cpu_nchw_bgr.bin";
const std::string FILE_CPU_INPUT_NHWC_BGR   = "input_cpu_nhwc_bgr.bin";
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

template <typename T> void writeToFile(std::vector<T>& input, const std::string& dst) {
    std::ofstream dumper(dst, std::ios_base::binary);
    dumper.write(reinterpret_cast<char *>(&input[0]), input.size()*sizeof(T));
    dumper.close();
}

void generateGraphFile(std::string pathBlob, MVCNN::GraphFileT& graphFile)
{
    std::ifstream infile;
    infile.open(pathBlob, std::ios::binary | std::ios::in);
    infile.seekg(0,std::ios::end);
    int length = infile.tellg();
    infile.seekg(0,std::ios::beg);
    char *data = new char[length];
    infile.read(data, length);
    infile.close();

    // Autogenerated class from table Monster.
    const MVCNN::GraphFile *graphPtr = MVCNN::GetGraphFile(data);
    graphPtr->UnPackTo(&graphFile);
}

// K-L Divergence, or relative entropy, is calculated as KL(P || Q) = sum x in X P(x) * log(P(x) / Q(x))
// where P is the expected results, and Q is our observed result
double klDivergence(std::vector<float>& P, std::vector<float>& Q)
{
	double sum = 0.0;
	for(size_t i = 0; i < Q.size(); i++)
	{
		float p = P[i];
		float q = Q[i];
        if(p == 0) p = 0.0000001;
		if(q == 0) q = 0.0000001;
		float log = std::log2(p / q);
		
		sum += p * log;
	}
	
	return sum;
}

// JS Divergence, a normalized version of KL diverence for calculating differences in probability distrubution
// JS(P || Q) = 1/2 * KL(P || M) + 1/2 * KL(Q || M), where M is M = 1/2 * (P + Q)
double jsDivergence(std::vector<float>& P, std::vector<float>& Q)
{
	std::vector<float> M;
	for(size_t i = 0; i < Q.size(); i++)
	{
		M.push_back(0.5 * (P[i]+Q[i]));
	}
	float PwrtM = klDivergence(P, M);
	float QwrtM = klDivergence(Q, M);
	return (0.5 * PwrtM) + (0.5 * QwrtM);
}

// Do a softmax to turn classification results into probabilty function
// Resulting vector will be in range [0, 1] with a sum of 1
std::vector<float> softmaxResults(std::vector<float>& results)
{
    std::vector<float> normalized;
    float sum = 0;

    for(auto& i : results)
    {
        float val = exp(i);
        normalized.push_back(val);
        sum += val;
    }

    for(size_t i = 0; i < normalized.size(); i++)
    {
        normalized[i] = normalized[i] / sum;
        // If the division underflows the softmax, can't use 0 because we will take log later
        if(isnan(normalized[i]))
            normalized[i] = std::numeric_limits<float>::min();
    }

    return normalized;    
}

bool compare(std::vector<float>& actualResults, std::vector<float>& expectedResults, float tolerance, float allowedDeviation, bool fp)
{
    std::cout << "Comparing results ... " << std::endl;
    std::cout << "  Actual Results size: " << actualResults.size() << std::endl;
    std::cout << "  Expected Results size: " << expectedResults.size() << std::endl;
    std::cout << "  Tolerance: " << tolerance << "%" << std::endl;
    if(!fp)
        std::cout << "  Allowed Deviation: " << allowedDeviation << " " << std::endl;
    if (actualResults.size() != expectedResults.size())
        std::cout << "  RESULTS SIZES DO NOT MATCH! Continuing..." << std::endl;

    float maxRelErr = 0;
    float maxAbsErr = 0;
    float countErrs = 0;
    float sumDiff = 0;
    float sumSquareDiffs = 0;
    std::function<void(size_t)> absoluteErrorUpdater = [&](size_t idx) {
        float actual = actualResults[idx];
        float expected = expectedResults[idx];

        float abs_error = fabsf(actual - expected);
        float no_zero_div = 0.0;
        if(expected == 0) no_zero_div = std::numeric_limits<float>::min();
        float relative_error = fabsf(abs_error / (no_zero_div + expected));

        sumSquareDiffs += pow(abs_error, 2);
        sumDiff+=abs_error;

        if(abs_error > maxAbsErr) maxAbsErr = abs_error;
        if(relative_error > maxRelErr) maxRelErr = relative_error;

        std::string result = "\t\033[1;32mPass\033[0m";
        if (!fp and (abs_error > allowedDeviation))
        {
            countErrs++;
            result = "\t\033[1;31mfail\033[0m";
        }
        if (fp and ((relative_error*100) > (tolerance/2)))
        {
            countErrs++;
            result = "\t\033[1;31mfail\033[0m";
        }
        if (idx < 50) // print first 50 rows
            std::cout << std::setw(10) << expected << std::setw(12) << actual << std::setw(12) << abs_error << std::setw(12) << relative_error << std::setw(18)  << result  << std::endl;
    };

    std::cout << "Printing first 50 rows...\nExpected\tActual\tDifference  Relative Err\tResult" << std::endl;
    for (size_t n = 0; n < expectedResults.size(); ++n)
        absoluteErrorUpdater(n);

    // results
    float avgPixelAccuracy = (sumDiff/expectedResults.size())*100;
    float l2_err = sqrt(sumSquareDiffs) / expectedResults.size();
    float countErrsPcent = (countErrs/actualResults.size()) * 100;
    
    //print results report
    std::cout.setf( std::ios::fixed );
    std::cout << "\nMetric\t\t\t  Actual  Threshold\tStatus" << std::endl << "----------------------    ------  ---------\t-------" << std::endl;   
    std::cout << "Min Pixel Accuracy\t" << std::setw(7) << std::setprecision(3) << maxRelErr << "%" << std::setw(10)  << std::setprecision(0)  << tolerance << "%" << std::setw(8) << ((maxRelErr < tolerance) ? "\t\033[1;32mPass" : "\t\033[1;31mFail") << "\033[0m" << std::endl;
    std::cout << "Average Pixel Accuracy\t" << std::setw(7) << std::setprecision(3) << avgPixelAccuracy << "%" << std::setw(10) << std::setprecision(0) << tolerance << "%" << std::setw(8) << ((avgPixelAccuracy < tolerance) ? "\t\033[1;32mPass" : "\t\033[1;31mFail") << "\033[0m" << std::endl;
    std::cout << "% of Wrong Values\t" << std::setw(7) << std::setprecision(3) << countErrsPcent << "%" << std::setw(10) << std::setprecision(0) << tolerance << "%" << ((countErrsPcent < tolerance) ? "\t\033[1;32mPass" : "\t\033[1;31mFail") << "\033[0m" << std::endl;
    std::cout << "Pixel-wise L2 Error\t" << std::setw(7) << std::setprecision(3) << (l2_err * 100) << "%" << std::setw(10) << std::setprecision(0) << tolerance << "%" << ((l2_err < tolerance) ? "\t\033[1;32mPass" : "\t\033[1;31mFail") << "\033[0m" << std::endl;
    std::cout << "Global Sum Difference\t" << std::setw(8) << std::setprecision(3) << sumDiff << std::setw(11) << "inf" << std::setw(8) << "\t\033[1;32mPass\033[0m" << std::endl;
    std::cout << "Max Absolute Error\t" << std::setw(8) << std::setprecision(3) << maxAbsErr << std::setw(11) << "inf" << std::setw(8) << "\t\033[1;32mPass\033[0m" << std::endl << std::endl;

    std::cout << "Applying Softmax to results to create probability distribution..." << std::endl;
    std::vector<float> P = softmaxResults(expectedResults);
    std::vector<float> Q = softmaxResults(actualResults);
    double jsDiv = jsDivergence(P, Q);
    double jsDistance = sqrt(jsDiv);
	std::cout << "JS Divergence\t\t" << std::setw(10) << std::setprecision(8) << jsDiv << std::endl;
    std::cout << "JS Distance\t\t" << std::setw(8) << std::setprecision(8) << jsDistance  << std::setw(8) << std::setprecision(0) << tolerance << "%" << std::setw(8) << (jsDistance < (tolerance/100) ? "\t\033[1;32mPass" : "\t\033[1;31mFail") << "\033[0m" << std::endl << std::endl;

    // if (avgPixelAccuracy < tolerance) return true;
    if ((countErrsPcent < tolerance)) return true;
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
    std::vector<std::string> filesDelete = {FILE_CPU_OUTPUT, FILE_CPU_INPUT_NCHW_RGB, FILE_CPU_INPUT_NHWC_RGB, FILE_CPU_INPUT_NCHW_BGR, FILE_CPU_INPUT_NHWC_BGR};
    for (std::string fDelete : filesDelete)
        remove((binFolder + fDelete).c_str());
    
    do
    {   //delete any previous blobs (different names each time)
        blobPath = findBlob(std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + DLDT_BLOB_LOCATION);
        std::string fullBlobPath = binFolder + DLDT_BLOB_LOCATION + blobPath;
        std::cout << "Removing: " << fullBlobPath << std::endl;
        remove(fullBlobPath.c_str());
    } while (blobPath != "");
    
    // check if we have 2 input xml models (CPU/KMB)
    std::vector<std::string> pathXMLvector;
    if ( pathXML.find(",") != std::string::npos)
    {
        // dual xml provided
        std::stringstream sstream(pathXML);
        while( sstream.good() )
        {
            std::string subStr;
            std::getline( sstream, subStr, ',');
            pathXMLvector.push_back( subStr );
        }
    }
    else 
    {   // single xml provided
        pathXMLvector.push_back( pathXML );
    }


    // execute the classification sample async (CPU-plugin)
    std::cout << "Generating reference results... " << std::endl;
    std::string commandline = std::string("cd ") + std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + "  && " + 
        "./test_classification -m " + pathXMLvector[0] + " -d CPU";
    if (! FLAGS_i.empty() )
        commandline += (" -i " + pathImage);

    if (FLAGS_r)
        commandline += (" -r ");

    std::cout << commandline << std::endl; 
    int returnVal = std::system(commandline.c_str());
    if (returnVal != 0)
    {
        std::cout << std::endl << "Error occurred running the test_classification (CPU mode)!" << std::endl;
        return FAIL_ERROR;
    }
    if (!checkFilesExist( {std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + FILE_CPU_OUTPUT} ))
        return FAIL_CPU_PLUGIN;
    
    if (!checkFilesExist( {std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + FILE_CPU_INPUT_NCHW_BGR} ))
        return FAIL_CPU_PLUGIN;

    if (!checkFilesExist( {std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + FILE_CPU_INPUT_NHWC_BGR} ))
        return FAIL_CPU_PLUGIN;

    if (!checkFilesExist( {std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + FILE_CPU_INPUT_NCHW_RGB} ))
        return FAIL_CPU_PLUGIN;

    if (!checkFilesExist( {std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + FILE_CPU_INPUT_NHWC_RGB} ))
        return FAIL_CPU_PLUGIN;

    //
    // execute the classification sample async (KMB-plugin)
    std::cout << "Generating mcm blob through kmb-plugin... " << std::endl;
    commandline = std::string("cd ") + std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + " && " + 
        "./test_classification -m " + ((pathXMLvector.size() > 1) ? pathXMLvector[1] : pathXMLvector[0]) + " -d KMB";
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
    // if (!copyFile(FILE_CONVERTED_IMAGE, inputDest)) return FAIL_GENERAL;
    if (!copyFile(inputCPU, inputDest))
        return FAIL_GENERAL;
     
    std::string blobDest = std::getenv("VPUIP_HOME") + std::string("/application/demo/InferenceManagerDemo/test.blob");
    if (!copyFile(blobPath, blobDest))
        return FAIL_GENERAL;

    // read movisim port and runtime config from env var if exist
    std::string movisimPort = "30001";
    if(std::getenv("MOVISIM_PORT") != NULL)
        movisimPort = std::getenv("MOVISIM_PORT");

    std::string runtimeConfig = ".config";
    if(std::getenv("RUNTIME_CONFIG") != NULL)
        runtimeConfig = std::getenv("RUNTIME_CONFIG");

    // execute the blob
    std::string commandline = std::string("cd ") + std::getenv("VPUIP_HOME") + "/application/demo/InferenceManagerDemo  && " + 
        "make run CONFIG_FILE=" + runtimeConfig + " srvIP=" + evmIP + " srvPort=" + movisimPort;
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
    MVCNN::GraphFileT graphFile;
    generateGraphFile(blobPath, graphFile);

    MVCNN::DType dtype = graphFile.header->net_output[0]->data_dtype;

    uint_fast16_t typesize = 1;
    if (dtype == MVCNN::DType::DType_U8)
        typesize = 1;
    else if (dtype == MVCNN::DType::DType_FP16)
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

    float allowedDeviation = 1.0;
    bool fp = false;

    std::vector<float> outputFP32;
    if (dtype == MVCNN::DType::DType_U8)
    {
        int qZero = graphFile.header->net_output[0]->quant_zero[0];
        double qScale = graphFile.header->net_output[0]->quant_scale[0]; // was quant_real_scale[0];
        int qShift = graphFile.header->net_output[0]->quant_shift[0];
        std::cout << "Querying quantization values... " << std::endl;
        std::cout << "  Datatype: " << dtype << std::endl;
        std::cout << "  quant_zero: " << qZero << std::endl;
        std::cout << "  quant_scale: " << qScale << std::endl;
        std::cout << "  quant_shift: " << qShift << std::endl;

        allowedDeviation = ( FLAGS_t * 2.56) * qScale; // Consider the tolerance a % of int range
        allowedDeviation = allowedDeviation / 2.0; // Consider half the range above or below of a given result
        if(allowedDeviation < qScale/2.0) allowedDeviation = qScale/2.0; // No better int can be found for this value

        // read size of output tensor
        int tSize = 1;
        for (uint32_t x = 0; x < graphFile.header->net_output[0]->dimensions.size(); ++x)
            tSize *= graphFile.header->net_output[0]->dimensions[x];
        std::cout << "  Output size: " << tSize << std::endl;

        std::vector<uint8_t> outputVector(totalActual);
        file.read(reinterpret_cast<char *>(&outputVector[0]), totalActual * sizeof(uint8_t));
        std::cout << totalActual << " elements" << std::endl;

        // de-quantize
        for (size_t i = 0; i < outputVector.size(); ++i)
        {
            // De-quantize formula (1): real_value  = Scale * ( quantized_value - zero_point)
            float val = qScale * (outputVector[i] - qZero);

            // De-quantize formula (2): bitshift left by qShift then multiply by scale
            // float val = static_cast<float>(outputVector[i] << (qShift)) / static_cast<float>(qScale);
            outputFP32.push_back(val);
        }
    }
    else if (dtype == MVCNN::DType::DType_FP16)
    {
        std::vector<u_int16_t> outputVector(totalActual);
        file.read(reinterpret_cast<char *>(&outputVector[0]), totalActual * typesize);
        std::cout << totalActual << " elements" << std::endl;
        float min = mv::fp16_to_fp32(outputVector[0]);
        float max = mv::fp16_to_fp32(outputVector[0]);
        for (size_t i = 0; i < outputVector.size(); ++i)
        {
            float val = mv::fp16_to_fp32(outputVector[i]);
            // float val = static_cast<uint16_t>(outputVector[i]);
            outputFP32.push_back(val);
            if(val > max) max = val;
            if(val < min) min = val;
        }
        float range = max - min;
        allowedDeviation = 1.0 * (FLAGS_t * range/100);
        allowedDeviation = allowedDeviation / 2; // Consider half the range above or below of a given result
        fp = true;
    }
    writeToFile(outputFP32, "./output-kmb-dequantized.bin");

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
    pass = compare(outputFP32, expectedFP32, FLAGS_t, allowedDeviation, fp);
    std::cout << "Accuracy Validation status: " << ((pass) ? "\033[1;32mPass" : "\033[1;31mFail") << "\033[0m" << std::endl << std::endl; 
    if (pass)
        return RESULT_SUCCESS;
    else 
        return FAIL_VALIDATION;
}

int copyImage(std::string imagePath, std::string blobPath)
{
    // load blob and read quantization values: scale and zeropoint
    MVCNN::GraphFileT graphFile;
    generateGraphFile(blobPath, graphFile);

    std::cout << "Querying input shape... " << std::endl;
    std::vector<int> inputShape;
    for (uint32_t x=0; x<graphFile.header->net_input[0]->dimensions.size(); ++x)
        inputShape.push_back( graphFile.header->net_input[0]->dimensions[x] );
    std::cout << "Input Shape: " << inputShape[0] << "," << inputShape[1] << "," << inputShape[2] << "," << inputShape[3] << std::endl;

    std::cout << "Querying Z/Ch Major conv... " << std::endl;
    std::vector<int> inputStrides;
    for (uint32_t x=0; x<graphFile.header->net_input[0]->strides.size(); ++x)
        inputStrides.push_back( graphFile.header->net_input[0]->strides[x] );
    std::cout << "Input Strides: " << inputStrides[0] << "," << inputStrides[1] << "," << inputStrides[2] << "," << inputStrides[3] << std::endl;

    bool zMajor = false;
    if (! ((inputShape[1] < 16) && (inputShape[2] == inputStrides[3]) ))
        zMajor = true;

    if ((imagePath.find("bin") != std::string::npos) || (imagePath.find("dat") != std::string::npos))
    {
        std::string binFolder = std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER;
        if(zMajor)
        {
            if(FLAGS_r){ // Use zmajor, rgb
                remove((binFolder + FILE_CPU_INPUT_NCHW_BGR ).c_str());
                remove((binFolder + FILE_CPU_INPUT_NCHW_RGB ).c_str());
                remove((binFolder + FILE_CPU_INPUT_NHWC_BGR ).c_str());
                rename((binFolder + FILE_CPU_INPUT_NHWC_RGB).c_str(),(binFolder + FILE_CPU_INPUT).c_str() );
            }else { // Use zmajor, bgr
                remove((binFolder + FILE_CPU_INPUT_NCHW_BGR ).c_str());
                remove((binFolder + FILE_CPU_INPUT_NCHW_RGB ).c_str());
                remove((binFolder + FILE_CPU_INPUT_NHWC_RGB ).c_str());
                rename((binFolder + FILE_CPU_INPUT_NHWC_BGR).c_str(),(binFolder + FILE_CPU_INPUT).c_str() );
            }
        }
        else
        {
            if(FLAGS_r){
                remove((binFolder + FILE_CPU_INPUT_NCHW_BGR ).c_str());
                remove((binFolder + FILE_CPU_INPUT_NHWC_RGB ).c_str());
                remove((binFolder + FILE_CPU_INPUT_NHWC_BGR ).c_str());
                rename((binFolder + FILE_CPU_INPUT_NCHW_RGB).c_str(),(binFolder + FILE_CPU_INPUT).c_str() );
            }else {
                remove((binFolder + FILE_CPU_INPUT_NCHW_RGB ).c_str());
                remove((binFolder + FILE_CPU_INPUT_NHWC_RGB ).c_str());
                remove((binFolder + FILE_CPU_INPUT_NHWC_BGR ).c_str());
                rename((binFolder + FILE_CPU_INPUT_NCHW_BGR).c_str(),(binFolder + FILE_CPU_INPUT).c_str() );
            }
        }
        
        MVCNN::DType dtype = graphFile.header->net_output[0]->data_dtype;
        if (dtype == MVCNN::DType::DType_FP16)
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
        // as of compiler v2.2.2, all input must be in NHWC order
        std::string inNHWC;
        if(FLAGS_r)
            inNHWC = std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + FILE_CPU_INPUT_NHWC_RGB;
        else
            inNHWC = std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + FILE_CPU_INPUT_NHWC_BGR;
        
        std::string inNHWC_dest = std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + FILE_CPU_INPUT;
        copyFile(inNHWC, inNHWC_dest);
    }

    return RESULT_SUCCESS;    
}

int postProcessActualResults(std::string resultsPath, std::string blobPath)
{
    // Clean old file
    std::cout << "Deleting old output... " << std::endl;
    std::string outputFile = "./output_transposed.dat";
    remove(outputFile.c_str());
    
    MVCNN::GraphFileT graphFile;
    generateGraphFile(blobPath, graphFile);

    std::cout << "Post Processing results... " << std::endl;
    MVCNN::DType dtype = graphFile.header->net_output[0]->data_dtype;
    std::cout << "Datatype: " << MVCNN::EnumNameDType(dtype) << std::endl;
    std::vector<int> outputShape;
    for (uint32_t x=0; x<graphFile.header->net_output[0]->dimensions.size(); ++x)
        outputShape.push_back( graphFile.header->net_output[0]->dimensions[x] );
    std::cout << "Output Shape: " << outputShape[0] << "," << outputShape[1] << "," << outputShape[2] << "," << outputShape[3] << std::endl;

    std::cout << "Querying Z/Ch Major output... " << std::endl;
    std::vector<int> outputStrides;
    for (uint32_t x=0; x<graphFile.header->net_output[0]->strides.size(); ++x)
        outputStrides.push_back( graphFile.header->net_output[0]->strides[x] );
    std::cout << "Output Strides: " << outputStrides[0] << "," << outputStrides[1] << "," << outputStrides[2] << "," << outputStrides[3] << std::endl;

    std::string sZMajor("");
    if (! ((outputShape[1] < 16) && (outputShape[2] == outputStrides[3]) ))
        sZMajor = " --zmajor";

    // call python script for numpy reshape/transpose
    std::string dtypeStr = "U8";
    if (dtype == MVCNN::DType::DType_FP16) dtypeStr = "FP16";
    std::string commandline = std::string("python3 ") + mv::utils::projectRootPath() +
        std::string("/python/tools/post_process.py --file ") + resultsPath + " --dtype " + dtypeStr + " --shape " + 
        std::to_string(outputShape[0]) + "," + std::to_string(outputShape[1]) + "," + std::to_string(outputShape[2]) + "," + std::to_string(outputShape[3]) + sZMajor;
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

int checkInference(std::string actualResults, std::string imagePath, bool yoloNetwork=true)
{
    // convert blob to json
    std::cout << "Checking inference results ..." << std::endl;

    std::string commandline = std::string("python3 ") + mv::utils::projectRootPath() + std::string("/python/tools/output_class_reader.py ") + actualResults;
    if (yoloNetwork) 
        commandline = std::string("python3 ") + mv::utils::projectRootPath() + std::string("/python/tools/yolo_bbox.py ") + imagePath;
    std::cout << commandline << std::endl;
    int result = std::system(commandline.c_str());
    
    // read in expected inference results
    std::string expectedInferencePath = std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + std::string("/inference_results.txt");
    std::ifstream inExpected(expectedInferencePath);
    std::vector<std::string> expectedInferenceResults;
    std::string str;
    std::cout << std::endl << "Expected top 10: ";
    while (std::getline(inExpected, str))
    {
        if(str.size() > 0) {
            expectedInferenceResults.push_back(str);
            std::cout << str << " ";
        }
    }

    // read in actual inference results
    std::string actualInferencePath = std::getenv("VPUIP_HOME") + std::string("/application/demo/InferenceManagerDemo/actual_inference_results.txt");
    std::ifstream inActual(actualInferencePath);
    std::vector<std::string> actualInferenceResults;
    std::cout << std::endl << "Actual top 10:   ";
    while (std::getline(inActual, str))
    {
        if(str.size() > 0){
            actualInferenceResults.push_back(str);
            std::cout << str << " ";
        }
    }
    
    // compare
    // bool pass = (std::stoi(actualInferenceResults[0]) == std::stoi(expectedInferenceResults[0]));
    bool pass = (actualInferenceResults[0] == expectedInferenceResults[0]);
    std::cout << std::endl << "Inference Validation status (top 1): " << ((pass) ? "\033[1;32mPass" : "\033[1;31mFail") << "\033[0m" << std::endl << std::endl;
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
        convertBlobToJson(FLAGS_b);
        std::string actualPathProcessed = "./output_transposed.dat";
        postProcessActualResults(FLAGS_a, FLAGS_b);
        validate(FLAGS_b, FLAGS_e, actualPathProcessed);
        
        bool yoloNetwork=false;
        if (FLAGS_m.find("yolo") > 0)
            yoloNetwork=true;

        checkInference(FLAGS_a, FLAGS_i, yoloNetwork);
        return(0);
    }

    // Normal operation
    int result = 0;

    std::string blobPath("");
    result = runEmulator(FLAGS_m, FLAGS_i, blobPath);
    if ( result > 0 ) return result;

    result = convertBlobToJson(blobPath);
    // if ( result > 0 ) return result;

    if (! FLAGS_i.empty()) 
    {
        result = copyImage(FLAGS_i, blobPath);
        if ( result > 0 ) return result;
    }
    else
    {
        // both Zmajor and Cmajor inputs available. So passing any one is ok to check and delete the unwanted one
        std::string binPath = std::getenv("DLDT_HOME") + DLDT_BIN_FOLDER + FILE_CPU_INPUT_NCHW_BGR;
        result = copyImage(binPath, blobPath);
    }

    result = runKmbInference(FLAGS_k, blobPath);
    if ( result > 0 ) return result;

    std::string expectedPath = std::getenv("DLDT_HOME") + std::string("/bin/intel64/Debug/output_cpu.bin");
    std::string actualPath = std::getenv("VPUIP_HOME") + std::string("/application/demo/InferenceManagerDemo/output-0.bin");
    std::string actualPathProcessed = "./output_transposed.dat";

    result = postProcessActualResults(actualPath, blobPath);
    if ( result > 0 ) return result;

    validate(blobPath, expectedPath, actualPathProcessed);
    
    // master test is if the top 1's match
    bool yoloNetwork=false;
    if (FLAGS_m.find("yolo") > 0)
        yoloNetwork=true;

    return (checkInference(actualPath, FLAGS_i, yoloNetwork));
}
