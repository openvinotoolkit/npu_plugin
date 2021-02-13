#include "validate.hpp"
#include "include/mcm/utils/env_loader.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/warning_manager.hpp"
#include "flatbuffers/flatbuffers.h"
#include "schema/graphfile/graphfile_generated.h"
#include <fstream>
#include <math.h>
#include <sys/stat.h>
#include <sstream>
#include <iomanip>
#include <vector>
#include <ios>
#include <string>
#include <map>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <libssh/sftp.h>

/**
 * Required environmental variables
 * VPUIP_HOME = <path to vpuip_2 repo>
 * OPENVINO_HOME  = <path to openvino repo>
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
const std::string FILE_CPU_OUTPUT       = "output_cpu0.bin";
const std::string FILE_CPU_INPUT        = "input_cpu.bin";
const std::string FILE_CPU_INPUT_FP16   = "input_cpu_fp16.bin";
const std::string FILE_CPU_INPUT_NCHW_RGB   = "input_cpu_nchw_rgb.bin";
const std::string FILE_CPU_INPUT_NHWC_RGB   = "input_cpu_nhwc_rgb.bin";
const std::string FILE_CPU_INPUT_NCHW_BGR   = "input_cpu_nchw_bgr.bin";
const std::string FILE_CPU_INPUT_NHWC_BGR   = "input_cpu_nhwc_bgr.bin";
      std::string OPENVINO_BIN_FOLDER   = "/bin/intel64/Release/";
const std::string FILE_BLOB_NAME        = "mcm.blob";
const std::string TEST_RUNTIME          = "application/demo/InferenceManagerDemo";

std::string getEnvVarDefault(const std::string& varName, const std::string& defaultValue)
{
    const char* value = getenv(varName.c_str());
    return value ? value : defaultValue;
}

void free_channel(ssh_channel channel) {
    ssh_channel_send_eof(channel);
    ssh_channel_close(channel);
    ssh_channel_free(channel);
}

void free_session(ssh_session session) {
    ssh_disconnect(session);
    ssh_free(session);
}

void error(ssh_session session) {
    fprintf(stderr, "Error: %s\n", ssh_get_error(session));
    free_session(session);
    exit(-1);
}

int scp_receive(ssh_session session, ssh_scp scp, std::string outputFile)
{
  int rc;
  int size, mode;
  char *filename, *buffer;

  rc = ssh_scp_pull_request(scp);
  if (rc != SSH_SCP_REQUEST_NEWFILE)
  {
    fprintf(stderr, "Error receiving information about file: %s\n",
            ssh_get_error(session));
    return SSH_ERROR;
  }

  size = ssh_scp_request_get_size(scp);
  filename = strdup(ssh_scp_request_get_filename(scp));
  mode = ssh_scp_request_get_permissions(scp);
  printf("Receiving file %s, size %d, permisssions 0%o\n",
          filename, size, mode);
  free(filename);

  buffer = (char*)malloc(size);
  if (buffer == NULL)
  {
    fprintf(stderr, "Memory allocation error\n");
    return SSH_ERROR;
  }

  ssh_scp_accept_request(scp);
  int r = 0;
  while (r < size) {
      int st = ssh_scp_read(scp, buffer + r, size - r);
      r += st;
  }
  if (rc == SSH_ERROR)
  {
    fprintf(stderr, "Error receiving file data: %s\n",
            ssh_get_error(session));
    free(buffer);
    return rc;
  }

  int filedesc = open(outputFile.c_str(), O_WRONLY | O_CREAT, 0666);

  if (filedesc < 0) {
    return -1;
  }

  write(filedesc, buffer, size);
  free(buffer);

  close(filedesc);

  rc = ssh_scp_pull_request(scp);
  if (rc != SSH_SCP_REQUEST_EOF)
  {
    fprintf(stderr, "Unexpected request: %s\n",
            ssh_get_error(session));
    return SSH_ERROR;
  }

  return SSH_OK;
}

int doInferenceWithSimpleNNOnly()
{
    // Create SSH session
        ssh_session session;
        ssh_channel channel;
        int rc, port = 22;
        char buffer[1024];
        unsigned int nbytes;
        int verbosity = SSH_LOG_PROTOCOL;

        printf("Session...\n");
        session = ssh_new();
        if (session == NULL)
            exit(-1);

        ssh_options_set(session, SSH_OPTIONS_HOST, FLAGS_k.c_str());
        // ssh_options_set(session, SSH_OPTIONS_LOG_VERBOSITY, &verbosity);
        ssh_options_set(session, SSH_OPTIONS_PORT, &port);
        ssh_options_set(session, SSH_OPTIONS_USER, "root");

        printf("Connecting...\n");
        rc = ssh_connect(session);
        if (rc != SSH_OK)
            error(session);

        printf("Password Autentication...\n");
        rc = ssh_userauth_password(session, NULL, "root");
        if (rc != SSH_AUTH_SUCCESS)
            error(session);

        printf("Channel...\n");
        channel = ssh_channel_new(session);
        if (channel == NULL)
            exit(-1);

        printf("Opening...\n");
        //rc = ssh_channel_open_session(channel);
        rc = ssh_channel_open_forward(channel,"192.168.1.1",22,"localhost", 5555);
        if (rc != SSH_OK)
            error(session);

        // SFTP input-0.bin and test.blob to th EVM
        sftp_session sftp0;
        sftp_session sftp1;

        // Open two SFTP sessions
        sftp0 = sftp_new(session);
        sftp1 = sftp_new(session);

        if (sftp0 == NULL) {
            fprintf(stderr, "Error allocating SFTP session: %s\n", ssh_get_error(session));
            return SSH_ERROR;
        }
        // Initialize the SFTP session
        rc = sftp_init(sftp0);
        if (rc != SSH_OK) {
            fprintf(stderr, "Error initializing SFTP session: %s.\n", sftp_get_error(sftp0));
            sftp_free(sftp0);
            return rc;
        }

        if (sftp1 == NULL) {
            fprintf(stderr, "Error allocating SFTP session: %s\n", ssh_get_error(session));
            return SSH_ERROR;
        }
        // Initialize the SFTP session
        rc = sftp_init(sftp1);
        if (rc != SSH_OK) {
            fprintf(stderr, "Error initializing SFTP session: %s.\n", sftp_get_error(sftp1));
            sftp_free(sftp1);
            return rc;
        }

        // Open the test.blob file on the remote side
        sftp_file file0;
        sftp_file file1;

        file0 = sftp_open(sftp0, "/opt/test.blob", O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU);
        if (file0 == NULL) {
            fprintf(stderr, "Can't open test.blob for writing: %s\n", ssh_get_error(session));
            return SSH_ERROR;
        }

        file1 = sftp_open(sftp1, "/opt/input-0.bin", O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU);
        if (file1 == NULL) {
            fprintf(stderr, "Can't open input-0.bin for writing: %s\n", ssh_get_error(session));
            return SSH_ERROR;
        }

        std::string testRuntime = getEnvVarDefault("TEST_RUNTIME", TEST_RUNTIME);
        std::string blobDest = std::getenv("VPUIP_HOME") + std::string("/") + testRuntime + std::string("/test.blob");
        std::ifstream fin0(blobDest, std::ios::binary);

        while (fin0) {
            constexpr size_t max_xfer_buf_size = 260000;
            char buffer[max_xfer_buf_size];
            fin0.read(buffer, sizeof(buffer));
            if (fin0.gcount() > 0) {
                ssize_t nwritten = sftp_write(file0, buffer, fin0.gcount());
                if (nwritten != fin0.gcount()) {
                    fprintf(stderr, "Can't write data to file: %s\n", ssh_get_error(session));
                    sftp_close(file0);
                    return 1;
                }
            }
        }
        sftp_close(file0);

        std::string inputDest = std::getenv("VPUIP_HOME") + std::string("/") + testRuntime + std::string("/input-0.bin");
        std::ifstream fin1(inputDest, std::ios::binary);

        while (fin1) {
            constexpr size_t max_xfer_buf_size = 260000;
            char buffer[max_xfer_buf_size];
            fin1.read(buffer, sizeof(buffer));
            if (fin1.gcount() > 0) {
                ssize_t nwritten = sftp_write(file1, buffer, fin1.gcount());
                if (nwritten != fin1.gcount()) {
                    fprintf(stderr, "Can't write data to file: %s\n", ssh_get_error(session));
                    sftp_close(file1);
                    return 1;
                }
            }
        }
        sftp_close(file1);

        // Do inference!
        printf("Executing remote command...\n");
        rc = ssh_channel_request_exec(
                channel,
                "new_SimpleNN /opt/test.blob /opt/input-0.bin /opt/output-0.bin /opt/expected_result_sim.dat 0");
        if (rc != SSH_OK)
            error(session);

        printf("Received:\n");
        nbytes = ssh_channel_read(channel, buffer, sizeof(buffer), 0);
        while (nbytes > 0) {
            fwrite(buffer, 1, nbytes, stdout);
            nbytes = ssh_channel_read(channel, buffer, sizeof(buffer), 0);
        }

        // Scp output-0.bin
        // Start
        ssh_scp scp;
        scp = ssh_scp_new(session, SSH_SCP_READ, "/opt/output-0.bin");

        if (scp == NULL) {
            fprintf(stderr, "Error allocating scp session: %s\n", ssh_get_error(session));
            return SSH_ERROR;
        }
        rc = ssh_scp_init(scp);
        if (rc != SSH_OK) {
            fprintf(stderr, "Error initializing scp session: %s\n", ssh_get_error(session));
            ssh_scp_free(scp);
            return rc;
        }

        std::string outputFile = std::getenv("VPUIP_HOME") + std::string("/") + testRuntime + std::string("/output-0.bin");
        auto ret = scp_receive(session, scp, outputFile);
        if (ret)
            std::cout << "error reading output-0.bin" << std::endl;

        ssh_scp_close(scp);
        ssh_scp_free(scp);
        return SSH_OK;

        free_channel(channel);
        free_session(session);
}

std::string getExtension(std::string& path)
{
    std::string::size_type const p(path.find_last_of('.'));
    std::string file_extension = path.substr(p+1);

    return file_extension;
}

bool ParseAndCheckCommandLine(int argc, char *argv[])
{
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h)
    {
        showUsage();
        return false;
    }

    if (FLAGS_d)
    {
      OPENVINO_BIN_FOLDER = "/bin/intel64/Debug/";
    }

    else if (FLAGS_mode == "validate")
    {
        if (FLAGS_b.empty())
            throw std::logic_error("Parameter -b must be set in validation mode");
        if (FLAGS_e.empty())
            throw std::logic_error("Parameter -e must be set in validation mode");
        if (FLAGS_a.empty())
            throw std::logic_error("Parameter -a must be set in validation mode");
    }
    else if ((!FLAGS_app.empty()) && (FLAGS_app == "SIMPLENN")) 
    {
        if (FLAGS_m.empty())
            throw std::logic_error("Parameter -m <path to model> is not set");
        if (FLAGS_k.empty())
            throw std::logic_error("Parameter -k <evm ip address> is not set");
        if (FLAGS_p.empty())
            throw std::logic_error("Parameter -p <NUC password> is not set");
        if (! FLAGS_il.empty() )
            if (FLAGS_il != "NHWC" && FLAGS_il != "NCHW")
                throw std::logic_error("Parameter -il only supports NHWC or NCHW");
    }
    else if (FLAGS_mode != "SimpleNNInferenceOnly")//normal operation
    {
        if (FLAGS_m.empty())
            throw std::logic_error("Parameter -m <path to model> is not set");
        if (FLAGS_k.empty())
            throw std::logic_error("Parameter -k <evm ip address> is not set");
        if (! FLAGS_il.empty() )
            if (FLAGS_il != "NHWC" && FLAGS_il != "NCHW")
                throw std::logic_error("Parameter -il only supports NHWC or NCHW");
    }
    return true;
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

/*
 * Calculates intersections and unions using associative containers.
 * 1. Create intersection mapping with label as key and number of elements as value
 * 2. Create union mapping with label as key and and number of elements as value
 * 3. For each offset increase intersection and union maps
 * 4. For each union label divide intersection cardinality by union cardinality
 */
static float calculateMeanIntersectionOverUnion(const std::vector<int32_t>& vpuOut, const std::vector<int32_t>& cpuOut) 
{
    std::map<int32_t, size_t> intersectionMap;
    std::map<int32_t, size_t> unionMap;
    for (size_t pos = 0; pos < vpuOut.size() && pos < cpuOut.size(); pos++) {
        long vpuLabel = vpuOut.at(pos);
        long cpuLabel = cpuOut.at(pos);
        if (vpuLabel == cpuLabel) {
            // labels are the same -- increment intersection at label key
            // increment union at that label key only once
            // if label has not been created yet, std::map sets it to 0
            intersectionMap[vpuLabel]++;
            unionMap[vpuLabel]++;
        } else {
            // labels are different -- increment element count at both labels
            unionMap[vpuLabel]++;
            unionMap[cpuLabel]++;
        }
    }

    float totalIoU = 0.f;
    size_t nonZeroUnions = 0;
    for (const auto& unionPair : unionMap) {
        const auto& labelId = unionPair.first;
        float intersectionCardinality = intersectionMap[labelId];
        float unionCardinality = unionPair.second;
        float classIoU = intersectionCardinality / unionCardinality;
        std::cout << "Label: " << labelId << " IoU: " << classIoU << std::endl;
        nonZeroUnions++;
        totalIoU += classIoU;
    }

    if(nonZeroUnions == 0) {
        throw std::logic_error("calculateMeanIntersectionOverUnion: nonZeroUnions equal zero");
    }
    float meanIoU = totalIoU / nonZeroUnions;
    return meanIoU;
}

static std::vector<int32_t> transScoreToLabel(const std::vector<float> &scores) {
    std::vector<int32_t> labels;
    for (size_t h = 0; h < 368; h++)
        for (size_t w = 0; w < 480; w++) {
            int32_t label = 0;
            float max = 0;
            for (size_t c = 0; c < 12; c++) {
                if (scores[c*368*480 + h*480 + w] >= max) {
                    max = scores[c*368*480 + h*480 + w];
                    label = c;
                }
            }
            labels.push_back(label);
        }
    return labels;
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
    if(sum == 0) {
        throw std::logic_error("softmaxResults: sum of inputs array elements equal zero");
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
        if ((!fp) && (abs_error > allowedDeviation))
        {
            countErrs++;
            result = "\t\033[1;31mfail\033[0m";
        }
        if ((fp) && ((relative_error*100) > (tolerance/2)))
        {
            countErrs++;
            result = "\t\033[1;31mfail\033[0m";
        }
        if (idx < 50)
        {   // print first 50 rows
            std::cout << std::setw(10) << expected << std::setw(12) << actual << std::setw(12) << abs_error << std::setw(12) << relative_error << std::setw(18)  << result  << std::endl;
        }
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
    std::cout << std::endl << "====== Generate blob ======" << std::endl;
    std::cout << "Deleting old emulator results files... " << std::endl;
    std::string binFolder = std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER;
    std::vector<std::string> filesDelete = {FILE_CPU_OUTPUT, FILE_CPU_INPUT_NCHW_RGB, FILE_CPU_INPUT_NHWC_RGB, FILE_CPU_INPUT_NCHW_BGR, FILE_CPU_INPUT_NHWC_BGR};
    for (std::string fDelete : filesDelete)
        remove((binFolder + fDelete).c_str());

    //delete any previous blobs (different names each time)
    std::string fullBlobPath = std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER + FILE_BLOB_NAME;
    std::cout << "Removing: " << fullBlobPath << std::endl;
    remove(fullBlobPath.c_str());

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
    std::string commandline = std::string("cd ") + std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER + "  && " +
        "./test_classification -m " + pathXMLvector[0] + " -d CPU";
    if (! FLAGS_i.empty() )
        commandline += (" -i " + pathImage);

    if (FLAGS_r)
        commandline += (" -r ");

    // Hardcoding for ACL net, as test_classification doesn't support ip flag of FP16
    if (FLAGS_m.find("aclnet") != std::string::npos)
        commandline += (" -ip FP32");

    std::cout << commandline << std::endl;
    int returnVal = std::system(commandline.c_str());
    if (returnVal != 0)
    {
        std::cout << std::endl << "Error occurred running the test_classification (CPU mode)!" << std::endl;
        return FAIL_ERROR;
    }
    if (!checkFilesExist( {std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER + FILE_CPU_OUTPUT} ))
        return FAIL_CPU_PLUGIN;

    if (!checkFilesExist( {std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER + FILE_CPU_INPUT_NCHW_BGR} ))
        return FAIL_CPU_PLUGIN;

    if (!checkFilesExist( {std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER + FILE_CPU_INPUT_NHWC_BGR} ))
        return FAIL_CPU_PLUGIN;

    if (!checkFilesExist( {std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER + FILE_CPU_INPUT_NCHW_RGB} ))
        return FAIL_CPU_PLUGIN;

    if (!checkFilesExist( {std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER + FILE_CPU_INPUT_NHWC_RGB} ))
        return FAIL_CPU_PLUGIN;

    // execute the compile_tool (KMB-plugin)
    std::cout << "Generating mcm blob through kmb-plugin... " << std::endl;

    commandline = std::string("cd ") + std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER + " && " +
        "./compile_tool -m " + ((pathXMLvector.size() > 1) ? pathXMLvector[1] : pathXMLvector[0]) + " -d VPUX -o " + FILE_BLOB_NAME;

    // if exists, reads the input layout from commandline. Otherwise use env var
    if (! FLAGS_il.empty() )
        commandline += " -il " + FLAGS_il;
    else
    {
        bool layoutNHWC = (getEnvVarDefault("NHWC_LAYOUT", "false") == "true") ? true: false;
        if (layoutNHWC)
            commandline += " -il NHWC";
        else
            commandline += " -il NCHW";
    }

    // if exists, reads the input precision from commandline. Otherwise use env var
    if (! FLAGS_ip.empty() )
        commandline += (" -ip " + FLAGS_ip);
    else 
    {
        std::string inputPrecision = getEnvVarDefault("INPUT_PRECISION", "U8");
        commandline += " -ip " + inputPrecision;
        FLAGS_ip = inputPrecision;
    }

    commandline += " -op FP16";

    std::cout << commandline << std::endl;
    returnVal = std::system(commandline.c_str());
    if (returnVal != 0)
    {
        std::cout << std::endl << "Error occurred running the test_classification (KMB mode)!" << std::endl;
        return FAIL_ERROR;
    }

    blobPath = std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER + FILE_BLOB_NAME;
    if (!checkFilesExist( {blobPath} ))
    {
        std::cout << "Error! Couldn't find the generated blob in " << std::getenv("OPENVINO_HOME") << OPENVINO_BIN_FOLDER  << std::endl;
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

int runKmbInference(std::string evmIP, std::string blobPath) {
    // Clean old results
    std::string testRuntime = getEnvVarDefault("TEST_RUNTIME", TEST_RUNTIME);

    std::cout << "Deleting old kmb results files... " << std::endl;
    std::string outputFile = std::getenv("VPUIP_HOME") + std::string("/") + testRuntime + std::string("/output-0.bin");
    remove(outputFile.c_str());

    // copy the required files to InferenceManagerDemo folder
    std::string inputCPU = std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER + FILE_CPU_INPUT;
    std::string inputDest = std::getenv("VPUIP_HOME") + std::string("/") + testRuntime + std::string("/input-0.bin");

    // hardcoded for AclNet.
    if (FLAGS_m.find("aclnet") != std::string::npos) {  // ACLnet needs to be validated with: airplane_3_17-FP32.bin
                                                        // (for CPU) and airplane_3_17-FP32-FQU8.bin (for VPU)
        // https://github.com/movidius/migNetworkZoo/tree/master/internal/test_support
        inputCPU = FLAGS_i.replace(FLAGS_i.find(".bin"), 4, "-FQU8.bin");
        if (!checkFilesExist({inputCPU}))
            return FAIL_GENERAL;
    }

    // switch to fp16 input bin if fp16 network
    if ((!FLAGS_ip.empty()) && (FLAGS_ip == "FP16")) {
        std::cout << "FP16 mode: using image " << FILE_CPU_INPUT_FP16 << std::endl;
        inputCPU = std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER + FILE_CPU_INPUT_FP16;
    }
    if (!copyFile(inputCPU, inputDest))
        return FAIL_GENERAL;

    std::string blobDest = std::getenv("VPUIP_HOME") + std::string("/") + testRuntime + std::string("/test.blob");
    if (!copyFile(blobPath, blobDest))
        return FAIL_GENERAL;

    if ((!FLAGS_app.empty()) && (FLAGS_app == "SIMPLENN")) {
        // Create SSH session
        ssh_session session;
        ssh_channel channel;
        int rc, port = 22;
        char buffer[1024];
        unsigned int nbytes;
        int verbosity = SSH_LOG_PROTOCOL;

        printf("Session...\n");
        session = ssh_new();
        if (session == NULL)
            exit(-1);

        ssh_options_set(session, SSH_OPTIONS_HOST, FLAGS_k.c_str());
        ssh_options_set(session, SSH_OPTIONS_LOG_VERBOSITY, &verbosity);
        ssh_options_set(session, SSH_OPTIONS_PORT, &port);
        ssh_options_set(session, SSH_OPTIONS_USER, "labuser");

        printf("Connecting...\n");
        rc = ssh_connect(session);
        if (rc != SSH_OK)
            error(session);

        printf("Password Autentication...\n");
        rc = ssh_userauth_password(session, "labuser", "kmb_po_2019@irl");
        if (rc != SSH_AUTH_SUCCESS)
            error(session);

        printf("Channel...\n");
        channel = ssh_channel_new(session);
        if (channel == NULL)
            exit(-1);

        printf("Port forwarding to Host B...\n");
        //rc = ssh_channel_open_session(channel);
        rc = ssh_channel_open_forward(channel,"root@192.168.1.1",22,"localhost", 5555);
        std::cout << "rc is " << rc << std::endl;
        if (rc != SSH_OK)
        {
            error(session);
            exit(1);
        }

        printf("Port forwarding done...\n");
        // SFTP input-0.bin and test.blob to th EVM
        sftp_session sftp0;
        //sftp_session sftp1;

        // Open two SFTP sessions
        printf("Opening sftp session...\n");
        sftp0 = sftp_new(session);
        //sftp1 = sftp_new(session);

        if (sftp0 == NULL) {
            fprintf(stderr, "Error allocating SFTP session: %s\n", ssh_get_error(session));
            return SSH_ERROR;
        }
        // Initialize the SFTP session
        rc = sftp_init(sftp0);
        if (rc != SSH_OK) {
            fprintf(stderr, "Error initializing SFTP session: %s.\n", sftp_get_error(sftp0));
            sftp_free(sftp0);
            return rc;
        }
        printf("SFTP session opened...\n");

        // if (sftp1 == NULL) {
        //     fprintf(stderr, "Error allocating SFTP session: %s\n", ssh_get_error(session));
        //     return SSH_ERROR;
        // }
        // // Initialize the SFTP session
        // rc = sftp_init(sftp1);
        // if (rc != SSH_OK) {
        //     fprintf(stderr, "Error initializing SFTP session: %s.\n", sftp_get_error(sftp1));
        //     sftp_free(sftp1);
        //     return rc;
        // }

        // Open the test.blob file on the remote side
        sftp_file file0;
        //sftp_file file1;
        printf("Openign file on Host B...\n");
        file0 = sftp_open(sftp0, "/home/root/test.blob", O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU);
        if (file0 == NULL) {
            fprintf(stderr, "Can't open test.blob for writing: %s\n", ssh_get_error(session));
            return SSH_ERROR;
            exit(1);
        }

        // file1 = sftp_open(sftp1, "/home/root/input-0.bin", O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU);
        // if (file1 == NULL) {
        //     fprintf(stderr, "Can't open input-0.bin for writing: %s\n", ssh_get_error(session));
        //     return SSH_ERROR;
        // }
        std::ifstream fin0(blobDest, std::ios::binary);
        printf("file Openinn done...\n");
        while (fin0) {
            constexpr size_t max_xfer_buf_size = 260000;
            char buffer[max_xfer_buf_size];
            fin0.read(buffer, sizeof(buffer));
            if (fin0.gcount() > 0) {
                ssize_t nwritten = sftp_write(file0, buffer, fin0.gcount());
                if (nwritten != fin0.gcount()) {
                    fprintf(stderr, "Can't write data to file: %s\n", ssh_get_error(session));
                    sftp_close(file0);
                    return 1;
                }
            }
        }
        sftp_close(file0);

        // std::ifstream fin1(inputDest, std::ios::binary);

        // while (fin1) {
        //     constexpr size_t max_xfer_buf_size = 260000;
        //     char buffer[max_xfer_buf_size];
        //     fin1.read(buffer, sizeof(buffer));
        //     if (fin1.gcount() > 0) {
        //         ssize_t nwritten = sftp_write(file1, buffer, fin1.gcount());
        //         if (nwritten != fin1.gcount()) {
        //             fprintf(stderr, "Can't write data to file: %s\n", ssh_get_error(session));
        //             sftp_close(file1);
        //             return 1;
        //         }
        //     }
        // }
        // sftp_close(file1);

        // Do inference!
        printf("Executing remote command...\n");
        rc = ssh_channel_request_exec(
                channel,
                "new_SimpleNN /home/root/test.blob /home/root/input-0.bin /home/root/output-0.bin /home/root/expected_result_sim.dat 0");
        if (rc != SSH_OK)
            error(session);

        printf("Received:\n");
        nbytes = ssh_channel_read(channel, buffer, sizeof(buffer), 0);
        while (nbytes > 0) {
            fwrite(buffer, 1, nbytes, stdout);
            nbytes = ssh_channel_read(channel, buffer, sizeof(buffer), 0);
        }

        // Scp output-0.bin
        // Start
        ssh_scp scp;
        scp = ssh_scp_new(session, SSH_SCP_READ, "/home/root/output-0.bin");

        if (scp == NULL) {
            fprintf(stderr, "Error allocating scp session: %s\n", ssh_get_error(session));
            return SSH_ERROR;
        }
        rc = ssh_scp_init(scp);
        if (rc != SSH_OK) {
            fprintf(stderr, "Error initializing scp session: %s\n", ssh_get_error(session));
            ssh_scp_free(scp);
            return rc;
        }

        auto ret = scp_receive(session, scp, outputFile);
        if (ret)
            std::cout << "error reading output-0.bin" << std::endl;

        ssh_scp_close(scp);
        ssh_scp_free(scp);
        return SSH_OK;

        free_channel(channel);
        free_session(session);
    } else  // IMD
    {
        // read movisim port, runtime config and runtime options from env var if exist
        std::string movisimPort = getEnvVarDefault("MOVISIM_PORT", "30001");
        std::string runtimeConfig = getEnvVarDefault("RUNTIME_CONFIG", ".config");
        std::string runtimeOptions = getEnvVarDefault("RUNTIME_OPTIONS", "");

        // check the size of the blob file
        std::ifstream blobfile(blobPath, std::ios::in);
        blobfile.seekg(0, std::ios::end);
        long int blobfile_size = blobfile.tellg();
        std::cout << std::string("blobfile_size=") << blobfile_size << std::endl;
        long int blobfile_size_mb = blobfile_size / 1048576L;
        std::cout << std::string("blobfile_size_mb=") << blobfile_size_mb << std::endl;

        // add to runtimeOptions
        long int buffer_max_size = blobfile_size_mb + 1;
        std::string cleanMake = "";
        long int MAX_BLOB_SIZE = std::atoi(getEnvVarDefault("BLOBSIZE_FORCE_REBUILD", "80").c_str());
        if (blobfile_size_mb >= MAX_BLOB_SIZE) {
            cleanMake = " clean ";
            runtimeOptions += std::string(" CONFIG_BLOB_BUFFER_MAX_SIZE_MB=") + std::to_string(buffer_max_size);
            // TODO: Deprecated config key to remove after fully transitionning to
            // runtime versions >= NN_Runtime_v2.46.0
            runtimeOptions += std::string(" CONFIG_NN_ALIGN_WEIGHT_BUFFERS=n");
            runtimeOptions += std::string(" CONFIG_NN_WEIGHT_BUFFER_ALIGNMENT=1");
        } else {  // default buffer size to 100mb
            runtimeOptions += std::string(" CONFIG_BLOB_BUFFER_MAX_SIZE_MB=100");
        }

        // execute the blob
        std::cout << std::endl << std::string("====== Execute blob ======") << std::endl;
        std::string commandline = std::string("cd ") + std::getenv("VPUIP_HOME") + "/" + testRuntime + " && " +
                                  "make " + cleanMake + "run CONFIG_FILE=" + runtimeConfig + " srvIP=" + evmIP +
                                  " srvPort=" + movisimPort + " " + runtimeOptions;
        std::cout << commandline << std::endl;
        int returnVal = std::system(commandline.c_str());
        if (returnVal != 0) {
            std::cout << std::endl << "Error occurred executing blob on runtime!" << std::endl;
            return FAIL_ERROR;
        }
        std::cout << std::string("INFERENCE_PERFORMANCE_CHECK='") << getEnvVarDefault("INFERENCE_PERFORMANCE_CHECK", "")
                  << std::string("'") << std::endl;
        if (getEnvVarDefault("INFERENCE_PERFORMANCE_CHECK", "") != std::string("true")) {
            if (!checkFilesExist({outputFile}))
                return FAIL_RUNTIME;
        }
    }

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

    std::string commandline = std::string("flatc -t ") + std::getenv("MCM_HOME") +
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

bool validate(std::string blobPath, std::vector<std::string> expectedPaths, std::vector<std::string>& actualResultsProcessed, std::string networkType)
{
    MVCNN::GraphFileT graphFile;
    generateGraphFile(blobPath, graphFile);
    std::vector<bool> allResults;

    for (auto outIndex=0U; outIndex<graphFile.header->net_output.size(); ++outIndex)
    {
        MVCNN::DType dtype = graphFile.header->net_output[outIndex]->data_dtype;

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
        std::ifstream file(actualResultsProcessed[outIndex], std::ios::in | std::ios::binary);
        file.seekg(0, std::ios::end);
        auto totalActual = file.tellg() / typesize;
        file.seekg(0, std::ios::beg);

        float allowedDeviation = 1.0;
        bool fp = false;

        std::vector<float> outputFP32;
        if (dtype == MVCNN::DType::DType_U8)
        {
            int qZero = graphFile.header->net_output[outIndex]->quant_zero[0];
            double qScale = graphFile.header->net_output[outIndex]->quant_mult[0]; // was quant_real_scale[0];
            int qShift = graphFile.header->net_output[outIndex]->quant_shift[0];
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
            for (uint32_t x = 0; x < graphFile.header->net_output[outIndex]->dimensions.size(); ++x)
                tSize *= graphFile.header->net_output[outIndex]->dimensions[x];
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
            std::vector<uint16_t> outputVector(totalActual);
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

        bool pass = false;
        std::cout << "Reading in expected results... ";
        std::ifstream infile(expectedPaths[outIndex], std::ios::binary);
        infile.seekg(0, infile.end);
        
        if (networkType != "icnet" && networkType != "unet")
        {   // expected results are float32 for all networks (except ICNet)
            auto totalExpected = infile.tellg() / sizeof(float);
            std::cout << totalExpected << " elements" << std::endl;
            infile.seekg(0, infile.beg);

            std::vector<float> expectedFP32(totalExpected);
            infile.read(reinterpret_cast<char*>(&expectedFP32[0]), totalExpected*sizeof(float));

            // compare
            pass = compare(outputFP32, expectedFP32, FLAGS_t, allowedDeviation, fp);
        }
        else
        {   // Segmentation network comparison - ICnet CPU results are int32
            // Segmentation network comparison - Unet CPU results are float32
            float meanIoU = 0.f;
            if (networkType == "icnet")
            {
                auto totalExpected = infile.tellg() / sizeof(int32_t);
                std::cout << totalExpected << " elements" << std::endl;
                infile.seekg(0, infile.beg);

                std::vector<int32_t> expectedInt32(totalExpected);
                infile.read(reinterpret_cast<char*>(&expectedInt32[0]), totalExpected*sizeof(int32_t));

                // convert VPU results to int for comparison
                std::vector<int32_t> actualInt32(outputFP32.size());
                std::transform(outputFP32.begin(), outputFP32.end(), actualInt32.begin(),
                        [](const float &arg) { return static_cast<int>(arg); });

                // compare
                meanIoU = calculateMeanIntersectionOverUnion(actualInt32, expectedInt32);
            }
            if (networkType == "unet")
            {
                auto totalExpected = infile.tellg() / sizeof(float);
                std::cout << totalExpected << " elements" << std::endl;
                infile.seekg(0, infile.beg);
                std::vector<float> expected(totalExpected);
                infile.read(reinterpret_cast<char *>(&expected[0]), totalExpected * sizeof(float));

                // convert CPU and VPU results to int for comparison
                std::vector<int32_t> expectedUnet(totalExpected);
                expectedUnet = transScoreToLabel(expected);
                std::vector<int32_t> actualOutputUnet(totalActual);
                actualOutputUnet = transScoreToLabel(outputFP32);
                meanIoU = calculateMeanIntersectionOverUnion(actualOutputUnet, expectedUnet);
                std::cout << "The network type is unet " << std::endl;
            }

            float meanIntersectionOverUnionTolerance = 0.5f;
            std::cout << std::endl << "meanIoU: " << meanIoU << " " << "(Tolerance: >" << meanIntersectionOverUnionTolerance << ")" << std::endl;
            if (meanIntersectionOverUnionTolerance < meanIoU)
                pass = true;
        }
        std::cout << "Accuracy Validation status: " << ((pass) ? "\033[1;32mPass" : "\033[1;31mFail") << "\033[0m" << std::endl << std::endl;

        allResults.emplace_back(pass);
    }

    // gen overall result
    bool overallResult = true;
    for (bool eachResult: allResults)
        overallResult = overallResult && eachResult;

    return overallResult;
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
    std::cout << "Input Shape (NCHW): " << inputShape[0] << "," << inputShape[1] << "," << inputShape[2] << "," << inputShape[3] << std::endl;

    std::cout << "Querying Z/Ch Major conv... " << std::endl;
    std::vector<int> inputStrides;
    for (uint32_t x=0; x<graphFile.header->net_input[0]->strides.size(); ++x)
        inputStrides.push_back( graphFile.header->net_input[0]->strides[x] );
    std::cout << "Input Strides (KNCHW): " << inputStrides[0] << "," << inputStrides[1] << "," << inputStrides[2] << "," << inputStrides[3] << "," << inputStrides[4] <<std::endl;

    // for channel major to be true, (NCHW), stride along width should be '1'
    bool zMajor = false;
    if (! ((inputShape[1] < 16) && (inputStrides[4] == 1) ))
        zMajor = true;

    if (!(imagePath.find("bin") != std::string::npos) || (imagePath.find("dat") != std::string::npos))
    {
        std::string binFolder = std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER;
        if(zMajor)
        {
            std::cout << "Using Z-Major image... " << std::endl;
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
            std::cout << "Using Ch-Major image ... " << std::endl;
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

        // MVCNN::DType dtype = graphFile.header->net_output[0]->data_dtype;
        MVCNN::DType dtype = graphFile.header->net_input[0]->data_dtype;
        if (dtype == MVCNN::DType::DType_FP16)
        {
            std::cout << "Creating FP16 image from: " << FILE_CPU_INPUT << std::endl;
            std::string inputDest = std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER + FILE_CPU_INPUT_FP16;
            std::string inputSource = std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER + FILE_CPU_INPUT;

            std::ofstream fileOut(inputDest, std::ios::out | std::ios::binary);
            std::ifstream fileIn(inputSource, std::ios::in | std::ios::binary);

            fileIn.seekg(0, std::ios::end);
            auto totalActual = fileIn.tellg() / sizeof(uint8_t);
            fileIn.seekg(0, std::ios::beg);

            std::vector<uint16_t> inputVectorFP16;
            std::vector<uint8_t> inputVectorInt8(totalActual);
            fileIn.read(reinterpret_cast<char *>(&inputVectorInt8[0]), totalActual * sizeof(uint8_t));
            for (size_t i = 0; i < inputVectorInt8.size(); ++i)
            {
                float val = static_cast<float>(inputVectorInt8[i]);
                inputVectorFP16.push_back(mv::fp32_to_fp16(val));
            }
                
            fileOut.write(reinterpret_cast<char *>(&inputVectorFP16[0]), totalActual * sizeof(uint16_t) );
            fileOut.close();
            fileIn.close();
        }
    }
    else
    {
        // as of compiler v2.2.2, all input must be in NHWC order
        std::string inNHWC;
        if(FLAGS_r)
            inNHWC = std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER + FILE_CPU_INPUT_NHWC_RGB;
        else
            inNHWC = std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER + FILE_CPU_INPUT_NHWC_BGR;

        std::string inNHWC_dest = std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER + FILE_CPU_INPUT;
        copyFile(inNHWC, inNHWC_dest);
    }

    return RESULT_SUCCESS;
}

int postProcessActualResults(std::vector<std::string>& actualResults, std::string blobPath, std::vector<std::string>& actualResultsProcessed)
{
    // Clean old file
    {
        std::cout << "Deleting old output... " << std::endl;
        std::string outputFile = "./output_transposed.dat";
        remove(outputFile.c_str());
    }

    MVCNN::GraphFileT graphFile;
    generateGraphFile(blobPath, graphFile);

    std::cout << "Post Processing results... " << std::endl;
    for (auto i=0U; i<graphFile.header->net_output.size(); ++i)
    {
        MVCNN::DType dtype = graphFile.header->net_output[i]->data_dtype;
        std::cout << "Datatype: " << MVCNN::EnumNameDType(dtype) << std::endl;
        std::vector<int> outputShape;
        for (uint32_t x=0; x<graphFile.header->net_output[i]->dimensions.size(); ++x)
            outputShape.push_back( graphFile.header->net_output[i]->dimensions[x] );
        std::cout << "Output Shape: " << outputShape[0] << "," << outputShape[1] << "," << outputShape[2] << "," << outputShape[3] << std::endl;

        std::cout << "Querying Z/Ch Major output... " << std::endl;
        std::vector<int> outputStrides;
        for (uint32_t x=0; x<graphFile.header->net_output[i]->strides.size(); ++x)
            outputStrides.push_back( graphFile.header->net_output[i]->strides[x] );
        std::cout << "Output Strides: " << outputStrides[0] << "," << outputStrides[1] << "," << outputStrides[2] << "," << outputStrides[3] << std::endl;

        std::string sZMajor("");
        if (! ((outputShape[1] < 16) && (outputShape[2] == outputStrides[3]) ))
            sZMajor = " --zmajor";

        // call python script for numpy reshape/transpose
        std::string dtypeStr = "U8";
        if (dtype == MVCNN::DType::DType_FP16) dtypeStr = "FP16";
        std::string outputFile="./output_transposed" + std::to_string(i) + ".dat";
        std::string commandline = std::string("python3 ") + std::getenv("MCM_HOME") +
            std::string("/python/tools/post_process.py --file ") + actualResults[i] + " --dtype " + dtypeStr + " --shape " +
            std::to_string(outputShape[0]) + "," + std::to_string(outputShape[1]) + "," + std::to_string(outputShape[2]) + "," + std::to_string(outputShape[3]) + sZMajor + 
            " --output " + outputFile;
        std::cout << commandline << std::endl;
        int result = std::system(commandline.c_str());
        if (result > 0)
        {
            std::cout << "Error occured converting image using python script";
            return FAIL_ERROR;
        }
        if (!checkFilesExist({outputFile}))
            return FAIL_ERROR;
        
        // add post-processed file for validation later
        actualResultsProcessed.emplace_back(outputFile);
    }
    return RESULT_SUCCESS;
}

bool checkInference(std::string actualResults, std::string expectedResults, std::string imagePath, std::string networkType = "classification")
{
    if(getEnvVarDefault("INFERENCE_PERFORMANCE_CHECK", "") == std::string("true"))
    {
        // InferencePerformanceCheck has no results to report
        return true;
    }

    std::cout << "Checking inference results ..." << std::endl;

    std::string commandline = std::string("python3 ") + std::getenv("MCM_HOME")  + std::string("/python/tools/output_class_reader.py ") + actualResults + " " + expectedResults;
    if (networkType == "yolo") 
        commandline = std::string("python3 ") + std::getenv("MCM_HOME")  + std::string("/python/tools/yolo_bbox.py ") + imagePath + " " + actualResults + " " + expectedResults;
    else if (networkType == "ssd") 
        commandline = std::string("python3 ") + std::getenv("MCM_HOME")  + std::string("/python/tools/ssd_bbox.py ") + imagePath;


    std::cout << commandline << std::endl;
    int returnVal = std::system(commandline.c_str());
    if (returnVal == -1)
    {
        std::cout << std::endl << "Error occurred running the inference!" << std::endl;
        return FAIL_ERROR;
    }

    // read in expected inference results
    std::string expectedInferencePath = std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER + std::string("/inference_results.txt");
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
    bool pass = (actualInferenceResults[0] == expectedInferenceResults[0]);
    std::cout << std::endl << "Inference Validation status (top 1): " << ((pass) ? "\033[1;32mPass" : "\033[1;31mFail") << "\033[0m" << std::endl << std::endl;
    return pass;
}


int main(int argc, char *argv[])
{
    if(std::getenv("OPENVINO_HOME") == NULL)
    {
        std::cout << "ERROR! Environmental variable OPENVINO_HOME must be set with path to OPENVINO repo" << std::endl << std::endl;
        return FAIL_GENERAL;
    }

    if(std::getenv("VPUIP_HOME") == NULL)
    {
        std::cout << "ERROR! Environmental variable VPUIP_HOME must be set with path to VPUIP_2 repo" << std::endl << std::endl;
        return FAIL_GENERAL;
    }

    if (!ParseAndCheckCommandLine(argc, argv))
        return FAIL_ERROR; 

    std::string networkType="classification";
    if (FLAGS_m.find("yolo") != std::string::npos)
        networkType="yolo";
    else if (FLAGS_m.find("ssd") != std::string::npos)
        networkType="ssd";
    else if (FLAGS_m.find("icnet") != std::string::npos)
        networkType="icnet";
    else if (FLAGS_m.find("unet") != std::string::npos)
        networkType="unet";

    std::vector<std::string> actualResults;
    std::vector<std::string> actualResultsProcessed;
    std::vector<std::string> expectedPaths;

    if (FLAGS_mode == "SimpleNNInferenceOnly")
    {
        doInferenceWithSimpleNNOnly();
        return(0);
    }

    if (FLAGS_mode == "validate")
    {
        MVCNN::GraphFileT graphFile;
        generateGraphFile(FLAGS_b, graphFile);
        int32_t countOutputs = graphFile.header->net_output.size();

        //bypass all and just run the validation function
        for (auto count=0; count<countOutputs; ++count)
        {   // output-0.bin -> output-1.bin
            std::string outFile = FLAGS_a.substr(0, FLAGS_a.length()-5) + std::to_string(count) + ".bin";
            actualResults.emplace_back(outFile);

            // output_cpu
            std::string expectedFile = FLAGS_e.substr(0, FLAGS_e.length()-5) + std::to_string(count) + ".bin";
            expectedPaths.emplace_back(expectedFile);
        }
        postProcessActualResults(actualResults, FLAGS_b, actualResultsProcessed);
        validate(FLAGS_b, expectedPaths, actualResultsProcessed, networkType);

        if (networkType!="icnet")
        {
            for (auto idx=0; idx<countOutputs; ++idx)
                checkInference(actualResults[idx], expectedPaths[idx], FLAGS_i, networkType);
        }
        return(0);
    }

    // Normal operation
    int result = 0;

    std::string blobPath = std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER + FILE_BLOB_NAME;
    result = runEmulator(FLAGS_m, FLAGS_i, blobPath);
    if ( result > 0 ) return result;

    if (! FLAGS_i.empty())
    {
        result = copyImage(FLAGS_i, blobPath);
        if ( result > 0 ) return result;
    }
    else
    {
        // both Zmajor and Cmajor inputs available. So passing any one is ok to check and delete the unwanted one
        std::string binPath = std::getenv("OPENVINO_HOME") + OPENVINO_BIN_FOLDER + FILE_CPU_INPUT_NCHW_BGR;
        result = copyImage(binPath, blobPath);
    }

    result = runKmbInference(FLAGS_k, blobPath);
    if ( result > 0 ) return result;

    MVCNN::GraphFileT graphFile;
    generateGraphFile(blobPath, graphFile);
    int32_t countOutputs = graphFile.header->net_output.size();
    std::string expectedPath = std::getenv("OPENVINO_HOME") + std::string(OPENVINO_BIN_FOLDER + FILE_CPU_OUTPUT);
    std::string actualPath = std::getenv("VPUIP_HOME") + std::string("/") + getEnvVarDefault("TEST_RUNTIME", TEST_RUNTIME) + std::string("/output-0.bin");

    for (auto count=0; count<countOutputs; ++count)
    {   
        std::string outFile = actualPath.substr(0, actualPath.length()-5) + std::to_string(count) + ".bin";
        actualResults.emplace_back(outFile);

        // output_cpu
        std::string expectedFile = expectedPath.substr(0, expectedPath.length()-5) + std::to_string(count) + ".bin";
        expectedPaths.emplace_back(expectedFile);
    }

    bool testPass=false;
    if(getEnvVarDefault("INFERENCE_PERFORMANCE_CHECK", "") != std::string("true"))
    {
        result = postProcessActualResults(actualResults, blobPath, actualResultsProcessed);
        if ( result > 0 ) return result;

        testPass=validate(blobPath, expectedPaths, actualResultsProcessed, networkType);
    }

    // master test is if top 1's match - not for ICnet, it uses result of validate()
    if (networkType != "icnet" && networkType != "unet")
    {
        testPass = true;
        for (auto idx=0; idx<countOutputs; ++idx) {
            bool thisResult = checkInference(actualResults[idx], expectedPaths[idx], FLAGS_i, networkType);
            testPass = testPass && thisResult;
        }
    }

    if(testPass) return RESULT_SUCCESS;
    else return FAIL_VALIDATION;
}

// #include <libssh/sftp.h>
// int main(int argc, char* argv[]) {

//     // Create SSH session
//     ssh_session session;
//     ssh_channel channel;
//     int rc, port = 22;
//     char buffer[1024];
//     unsigned int nbytes;
//     int verbosity = SSH_LOG_PROTOCOL;

//     printf("Session...\n");
//     session = ssh_new();
//     if (session == NULL)
//         exit(-1);

//     ssh_options_set(session, SSH_OPTIONS_HOST, "hostA_IP.com");
//     ssh_options_set(session, SSH_OPTIONS_LOG_VERBOSITY, &verbosity);
//     ssh_options_set(session, SSH_OPTIONS_PORT, &port);
//     ssh_options_set(session, SSH_OPTIONS_USER, "user");

//     printf("Connecting...\n");
//     rc = ssh_connect(session);
//     if (rc != SSH_OK)
//         error(session);

//     printf("Password Autentication...\n");
//     rc = ssh_userauth_password(session, "user", "userpassword");
//     if (rc != SSH_AUTH_SUCCESS)
//         error(session);

//     printf("Channel...\n");
//     channel = ssh_channel_new(session);
//     if (channel == NULL)
//         exit(-1);

//     printf("Port forwarding to Host B...\n");
//     rc = ssh_channel_open_forward(channel, "root@hostB_IP.com", 22, "localhost", 5555);
//     if (rc != SSH_OK) {
//         error(session);
//         exit(1);
//     }
//     printf("Port forwarding done...\n");

//     // Open SFTP session
//     sftp_session sftp0;
//     printf("Opening sftp session...\n");
//     sftp0 = sftp_new(session);

//     if (sftp0 == NULL) {
//         fprintf(stderr, "Error allocating SFTP session: %s\n", ssh_get_error(session));
//         return SSH_ERROR;
//     }
//     // Initialize the SFTP session
//     rc = sftp_init(sftp0);
//     if (rc != SSH_OK) {
//         fprintf(stderr, "Error initializing SFTP session: %s.\n", sftp_get_error(sftp0));
//         sftp_free(sftp0);
//         return rc;
//     }
//     printf("SFTP session opened...\n");

//     sftp_file file0;
//     printf("Openign file on Host B...\n");
//     file0 = sftp_open(sftp0, "/home/root/test.blob", O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU);
//     if (file0 == NULL) {
//         fprintf(stderr, "Can't open test.blob for writing: %s\n", ssh_get_error(session));
//         return SSH_ERROR;
//         exit(1);
//     }
// }