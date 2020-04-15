#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <cmath>
#include <cstdio>
#include <iostream>

#include "Huffman.hpp"

#ifdef HDE_USE_DPI
#include <svdpi.h>
#endif

using namespace std;

extern "C"
{
    extern int ARGS_verbosity; //!< debug level; 0 off
}

struct huffmanCodecConfig
{
    uint32_t bitsPerSym;
    uint32_t maxEncSyms;
    uint32_t verbosity;
    uint32_t blockSizeBytes;

    bool compressFlag;
    bool statsOnly;
    bool bypass;

    set<char> exclusions;
};

class huffmanCodec
{
public:
    huffmanCodec(
        uint32_t pBitsPerSym = 8,
        uint32_t pMaxEncSyms = 16,
        uint32_t pVerbosity = 0,
        uint32_t pBlockSizeBytes = 4096,
        bool pStatsOnly = false,
        bool pBypass = false,
        set<char> pExclusions = {});
    ~huffmanCodec();
    void huffmanCodecConfigDefaults();

    uint32_t huffmanCodecCompressArray(unsigned char *inputData,
                                       int inputDataLength,
                                       unsigned char *outputData,
                                       int outputDataLength)
    {
        return this->huffmanCodecCompressArray((uint32_t &)inputDataLength, inputData, outputData);
    }

    uint32_t huffmanCodecDecompressArray(unsigned char *inputData,
                                         int inputDataLength,
                                         unsigned char *outputData,
                                         int outputDataLength)
    {
        return this->huffmanCodecDecompressArray((uint32_t &)inputDataLength, inputData, outputData);
    }

    uint32_t huffmanCodecCompressArray(uint32_t &inputDataLength,
                                       uint8_t *inputData,
                                       uint8_t *outputData);

    uint32_t huffmanCodecDecompressArray(uint32_t &inputDataLength,
                                         uint8_t *inputData,
                                         uint8_t *outputData);

    Huffman *mHuffmanCodec;

private:
    huffmanCodecConfig *mHuffmanCodecConfig;
    uint32_t mInputDataPlayhead;

    vector<char> getInputDataWithExclusions(const uint32_t &inputDataLength,
                                            const uint8_t *inputData,
                                            const uint32_t stepSize,
                                            const set<char> &exclusions);
};

#ifdef ENABLE_HDE_C_API

extern "C" huffmanCodec *huffmanCodecInitC(
    unsigned int pBitsPerSym,
    unsigned int pMaxEncSyms,
    unsigned int pVerbosity,
    unsigned int pBlockSizeBytes,
    bool pStatsOnly,
    bool pBypass);

extern "C" unsigned int huffmanCodecCompressArrayC(char *outputData,
                                                   unsigned long *outputDataLength,
                                                   char *inputData,
                                                   unsigned long inputDataLength,
                                                   huffmanCodec *huffmanCodecPtr);

extern "C" unsigned int huffmanCodecDecompressArrayC(char *outputData,
                                                     unsigned long *outputDataLength,
                                                     char *inputData,
                                                     unsigned long inputDataLength,
                                                     huffmanCodec *huffmanCodecPtr);

#endif
