#pragma once
#include <fstream>
#include <set>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>

#include "bitCompactor.h"

#ifdef BTC_USE_DPI
    #include <svdpi.h>
#endif

using namespace std;

class BitCompactorCodec
{
public:
    using memoryRead  = function<void(uint64_t address, uint8_t * const data, size_t size)>;
    using memoryWrite = function<void(uint64_t address, uint8_t const * const data, size_t size)>;

    struct BitCompactorCodecConfig
    {
        uint32_t bitsPerSym;
        uint32_t maxEncSyms;
        uint32_t verbosity;
        uint32_t blockSizeBytes;
        uint32_t superBlockSizeBytes;
        uint32_t minFixedBitLn;

        bool     compressFlag;
        bool     statsOnly;
        bool     bypass;
        bool     dualEncodeEn;
        bool     procBinningEn;
        bool     procBitmapEn;
        bool     mixedBlockSizeEn;
        bool     ratioEn;

        int      alignMode;
        string   inFileName;
        string   outFileName;

        set<char> exclusions;
    };

    typedef enum {
        WRITE_TO_BUFFER,
        WRITE_TO_FILE
        }  BitCompactorOutputDataRouting_t;

    typedef enum {
        READ_FROM_BUFFER,
        READ_FROM_FILE
        }  BitCompactorInputDataRouting_t;

    BitCompactorCodec(
        uint32_t     pVerbosity = 0,
        uint32_t     pBlockSizeBytes = 8,
        uint32_t     pSuperBlockSizeBytes = 4096,
        bool         pStatsOnly = false,
        bool         pBypass = false,
        set<char>    pExclusions = {}
        );
    BitCompactorCodec(
        memoryRead   pMemoryRead,
        memoryWrite  pMemoryWrite,
        uint32_t     pVerbosity = 0,
        uint32_t     pBlockSizeBytes = 8,
        uint32_t     pSuperBlockSizeBytes = 4096,
        bool         pStatsOnly = false,
        bool         pBypass = false,
        set<char>    pExclusions = {}
        );
    ~BitCompactorCodec ();

    BitCompactorCodec(const BitCompactorCodec &) = delete;
    BitCompactorCodec& operator= (const BitCompactorCodec &) = delete;

    void BitCompactorCodecConfigDefaults ();
    bool BitCompactorCodecConfigParseCLI (int argc, char** argv);
    void BitCompactorCodecConfigUpdate ();

    uint32_t BitCompactorCodecCompressArray  ( 
                                               const uint64_t readAddress,  // read address for memory access via function pointer (ignored otherwise)
                                               const uint64_t writeAddress, // write address for memory access via function pointer (ignored otherwise) 
                                               const uint32_t &inputDataLength,
                                               const uint8_t  *inputData,
                                               uint8_t        *outputData
                                               );

    uint32_t BitCompactorCodecDecompressArray( 
                                               const uint64_t  readAddress,  // read address for memory access via function pointer (ignored otherwise) 
                                               const uint64_t  writeAddress, // write address for memory access via function pointer (ignored otherwise)
                                               const uint32_t &inputDataLength,
                                               const uint8_t  *inputData,
                                               uint8_t        *outputData
                                               );

    void BitCompactorCodecDecompressStream( const uint8_t  inputData[32],
                                            std::vector<uint8_t>& outputData
                                            );

    uint32_t BitCompactorCodecGetCompressErrorCount ();
    uint32_t BitCompactorCodecGetDecompressErrorCount ();

    vector<char> BitCompactorCodecGetInputData ( 
                                              const uint32_t   &inputDataLength,
                                              const uint8_t    *inputData,
                                              const uint32_t    stepSize,
                                              const set<char>  &exclusions
                                              );

    BitCompactor *mBitCompactor;
private:
    memoryRead  mMemoryRead;
    memoryWrite mMemoryWrite;
    uint64_t    mBaseReadAddress;
    uint64_t    mCurrReadAddress;
    uint64_t    mBaseWriteAddress;
    uint64_t    mCurrWriteAddress;
    BitCompactorCodecConfig *mBitCompactorCodecConfig;
    uint32_t            mInputDataPlayhead;
};

#ifdef ENABLE_BITC_C_API

extern "C" BitCompactorCodec * BitCompactorCodecInitC(
    unsigned int   pVerbosity,
    unsigned int   pBlockSizeBytes,
    unsigned int   pSuperBlockSizeBytes,
    bool           pStatsOnly,
    bool           pBypass
    );

extern "C" unsigned int BitCompactorCodecCompressArrayC(   char*          outputData,
                                                      unsigned long* outputDataLength,
                                                      char*          inputData,
                                                      unsigned long  inputDataLength,
                                                      BitCompactorCodec  *BitCompactorCodecPtr
                                                      );

extern "C" unsigned int BitCompactorCodecDecompressArrayC( char*          outputData,
                                                      unsigned long* outputDataLength,
                                                      char*          inputData,
                                                      unsigned long  inputDataLength,
                                                      BitCompactorCodec * BitCompactorCodecPtr
                                                      );

#endif

