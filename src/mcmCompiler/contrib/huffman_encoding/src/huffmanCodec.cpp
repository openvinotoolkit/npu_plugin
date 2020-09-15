#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>

#include "Huffman.hpp"
#include "huffmanCodec.hpp"
#include "logging.hpp"

#ifdef HDE_USE_DPI
#include <svdpi.h>
#endif

huffmanCodec::huffmanCodec(
    uint32_t pBitsPerSym,
    uint32_t pMaxEncSyms,
    uint32_t pVerbosity,
    uint32_t pBlockSizeBytes,
    bool pStatsOnly,
    bool pBypass,
    set<char> pExclusions)
{
    stringstream rptStream;
    mInputDataPlayhead = 0;
    mHuffmanCodecConfig = new huffmanCodecConfig;

    huffmanCodecConfigDefaults();

    mHuffmanCodecConfig->blockSizeBytes = pBlockSizeBytes;
    mHuffmanCodecConfig->bitsPerSym = pBitsPerSym;
    mHuffmanCodecConfig->maxEncSyms = pMaxEncSyms;
    mHuffmanCodecConfig->exclusions = pExclusions;
    mHuffmanCodecConfig->statsOnly = pStatsOnly;
    mHuffmanCodecConfig->bypass = pBypass;
    mHuffmanCodecConfig->verbosity = pVerbosity;

    mHuffmanCodec = new Huffman(mHuffmanCodecConfig->maxEncSyms);

    rptStream << "huffmanCodec: custom constructor" << endl;
    Report(3, rptStream);
}

huffmanCodec::~huffmanCodec()
{
    stringstream rptStream;
    delete mHuffmanCodec;
    delete mHuffmanCodecConfig;

    rptStream << "huffmanCodec: destructor" << endl;
    Report(3, rptStream);
}

vector<char> huffmanCodec::getInputDataWithExclusions(const uint32_t &inputDataLength,
                                                      const uint8_t *inputData,
                                                      const uint32_t stepSize,
                                                      const set<char> &exclusions = {})
{
    stringstream rptStream;
    vector<char> nd;
    uint32_t len = 0;
    char *data;

    data = new char[mHuffmanCodecConfig->blockSizeBytes];

    rptStream << "getInputDataWithExclusions: start (idl: " << inputDataLength << " step: " << stepSize << " excls: " << exclusions.size() << ")" << endl;
    Report(2, rptStream);

    do
    {
        len = ((inputDataLength - mInputDataPlayhead) > mHuffmanCodecConfig->blockSizeBytes) ? mHuffmanCodecConfig->blockSizeBytes : (inputDataLength - mInputDataPlayhead);

        rptStream << "getInputDataWithExclusions: block (len: " << len << " playhead: " << mInputDataPlayhead << ")" << endl;
        Report(2, rptStream);

        if ((len > 0) && (len <= inputDataLength))
        {
            rptStream << "getInputDataWithExclusions: memcpy " << len << " bytes into data buffer" << endl;
            Report(3, rptStream);

            memcpy(&data[0], &inputData[mInputDataPlayhead], len);
            mInputDataPlayhead += len;
        }

        for (uint32_t i = 0; i < len && nd.size() < stepSize; i++)
        {
            if (exclusions.find(data[i]) != exclusions.end())
                continue;
            nd.push_back(data[i]);
        }
    } while (!(mInputDataPlayhead >= inputDataLength) && nd.size() != stepSize);

    rptStream << "getInputDataWithExclusions: returning " << nd.size() << " bytes" << endl;
    Report(2, rptStream);

    delete[] data;
    return nd;
}

uint32_t huffmanCodec::huffmanCodecDecompressArray(uint32_t &inputDataLength,
                                                   uint8_t *inputData,
                                                   uint8_t *outputData)
{
    stringstream rptStream;
    std::string lvpDummyFilename = "";
    uint32_t outputDataLength = 0;

    rptStream << "*************************************************************" << endl;
    rptStream << "***          Running De-Compression Model                 ***" << endl;
    rptStream << "*************************************************************" << endl;
    Report(1, rptStream);

    outputDataLength = mHuffmanCodec->readEncodedData(lvpDummyFilename,
                                                      lvpDummyFilename,
                                                      inputData,
                                                      inputDataLength,
                                                      outputData,
                                                      READ_FROM_BUFFER,
                                                      WRITE_TO_BUFFER);

    rptStream << "*************************************************************" << endl;
    Report(1, rptStream);

    return outputDataLength;
}

uint32_t huffmanCodec::huffmanCodecCompressArray(uint32_t &inputDataLength,
                                                 uint8_t *inputData,
                                                 uint8_t *outputData)
{
    stringstream rptStream;
    uint32_t outputDataLength = 0;
    uint64_t lviInputDataTallyBits = 0;
    uint64_t lviOutputDataTallyBits = 0;
    uint64_t lviEntropyTallyBits = 0;
    uint64_t lviEncodedDataBits = 0;
    huffmanOutputDataRouting_t outputDataRouting = WRITE_TO_BUFFER;
    vector<char> lvvOutputDataBuffer;
    vector<char> nd;
    string lvsDummyString = "";
    float lvfCompressRatio = 0.0;

    rptStream << "*************************************************************" << endl;
    rptStream << "***            Running Compression Model                  ***" << endl;
    rptStream << "*************************************************************" << endl;
    Report(1, rptStream);

    rptStream << "huffmanCodecCompress: starting with IDL " << inputDataLength << " bytes" << endl;
    Report(2, rptStream);

    lvvOutputDataBuffer.clear();
    mInputDataPlayhead = 0;

    while ((nd = getInputDataWithExclusions(inputDataLength,
                                            inputData,
                                            mHuffmanCodecConfig->blockSizeBytes,
                                            mHuffmanCodecConfig->exclusions))
               .size() > 0)
    {
        // this reads in all the data in the nd vector, except the excluded symbols
        // up the the set STEP size

        auto freq = mHuffmanCodec->getSymFreqs(nd.data(),
                                               nd.size(),
                                               mHuffmanCodecConfig->bitsPerSym); // get the symbol frequencies

        rptStream << "huffmanCodecCompress: reset codec" << endl;
        Report(2, rptStream);
        mHuffmanCodec->reset();

        rptStream << "huffmanCodecCompress: encode" << endl;
        Report(2, rptStream);
        HuffmanResult_t lvtBlockStats = mHuffmanCodec->encode(freq,
                                                              mHuffmanCodecConfig->bitsPerSym,
                                                              mHuffmanCodecConfig->bypass);

        lviInputDataTallyBits = lviInputDataTallyBits + lvtBlockStats.totalSize;
        lviEntropyTallyBits = lviEntropyTallyBits + lvtBlockStats.entropySize;

        rptStream << "huffmanCodecCompress: write to buffer" << endl;
        Report(2, rptStream);

        lviEncodedDataBits = mHuffmanCodec->writeEncodedData(nd.data(),
                                                             nd.size(),
                                                             lvsDummyString,
                                                             &lvvOutputDataBuffer,
                                                             mHuffmanCodecConfig->bypass,
                                                             mHuffmanCodecConfig->statsOnly,
                                                             outputDataRouting);

        lviOutputDataTallyBits = lviOutputDataTallyBits + lviEncodedDataBits;
        lvfCompressRatio = (100.0 * lviEncodedDataBits / lvtBlockStats.totalSize);

        rptStream << "huffmanCodecCompress: block compress size = " << lviEncodedDataBits << " bits" << endl;
        rptStream << "huffmanCodecCompress: block total size    = " << lvtBlockStats.totalSize << " bits" << endl;
        rptStream << "huffmanCodecCompress: block Compression ratio: " << fixed << setw(5) << setprecision(2) << lvfCompressRatio << " %" << endl;
        Report(2, rptStream);
    }

    outputDataLength = lvvOutputDataBuffer.size();

    memcpy(reinterpret_cast<void *>(outputData), reinterpret_cast<void *>(lvvOutputDataBuffer.data()), outputDataLength);

    lvfCompressRatio = (100.0 * lviOutputDataTallyBits / lviInputDataTallyBits);

    rptStream << "*************************************************************" << endl;
    rptStream << "huffmanCodecCompress: OVERALL TOTALS" << endl;
    rptStream << "huffmanCodecCompress: Compressed size:   " << lviOutputDataTallyBits << " bits" << endl;
    rptStream << "huffmanCodecCompress: Original size:     " << lviInputDataTallyBits << " bits" << endl;
    rptStream << "huffmanCodecCompress: Compression ratio: " << fixed << setw(5) << setprecision(2) << lvfCompressRatio << " %" << endl;
    Report(1, rptStream);
    rptStream << "*************************************************************" << endl;
    rptStream << "huffmanCodecCompress: finishing with ODL " << outputDataLength << " bytes" << endl;
    rptStream << "*************************************************************" << endl;
    Report(1, rptStream);

    return outputDataLength;
}

void huffmanCodec::huffmanCodecConfigDefaults()
{
    mHuffmanCodecConfig->blockSizeBytes = 4096;
    mHuffmanCodecConfig->compressFlag = 1;
    mHuffmanCodecConfig->bitsPerSym = 8;
    mHuffmanCodecConfig->maxEncSyms = 16;
    mHuffmanCodecConfig->exclusions.clear();
    mHuffmanCodecConfig->statsOnly = false;
    mHuffmanCodecConfig->bypass = false;
    mHuffmanCodecConfig->verbosity = 2; // set between 0-5,
                                        // 0 shows basic info,
                                        // 3 shows Metadata and some other useful stuff,
                                        // 5 shows all available info
}

#ifdef ENABLE_HDE_C_API
extern "C" huffmanCodec *huffmanCodecInitC(
    unsigned int pBitsPerSym,
    unsigned int pMaxEncSyms,
    unsigned int pVerbosity,
    unsigned int pBlockSizeBytes,
    bool pStatsOnly,
    bool pBypass)
{
    huffmanCodec *lvpHuffmanCodec;

    lvpHuffmanCodec = new huffmanCodec(pBitsPerSym,
                                       pMaxEncSyms,
                                       pVerbosity,
                                       pBlockSizeBytes,
                                       pStatsOnly,
                                       pBypass,
                                       {});

    return lvpHuffmanCodec;
}

extern "C" unsigned int huffmanCodecCompressArrayC(char *outputData,
                                                   unsigned long *outputDataLength,
                                                   char *inputData,
                                                   unsigned long inputDataLength,
                                                   huffmanCodec *huffmanCodecPtr)
{

    *outputDataLength = (unsigned long)huffmanCodecPtr->huffmanCodecCompressArray((uint32_t &)inputDataLength,
                                                                                  (uint8_t *)(inputData),
                                                                                  (uint8_t *)(outputData));
    if (outputDataLength > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

extern "C" unsigned int huffmanCodecDecompressArrayC(char *outputData,
                                                     unsigned long *outputDataLength,
                                                     char *inputData,
                                                     unsigned long inputDataLength,
                                                     huffmanCodec *huffmanCodecPtr)
{
    *outputDataLength = (unsigned long)huffmanCodecPtr->huffmanCodecDecompressArray((uint32_t &)inputDataLength,
                                                                                    (uint8_t *)inputData,
                                                                                    (uint8_t *)outputData);
    if (outputDataLength > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

#endif
