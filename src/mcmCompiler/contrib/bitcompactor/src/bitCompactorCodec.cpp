#include <fstream>
#include <set>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>

#include "bitCompactorCodec.h"

#ifdef BTC_USE_DPI
    #include <svdpi.h>
#endif

#ifdef _WIN32
#pragma warning(disable : 4996) // should be safe to remove after kw fixes
#endif
BitCompactorCodec::BitCompactorCodec(
    uint32_t     pVerbosity,
    uint32_t     pBlockSizeBytes,
    uint32_t     pSuperBlockSizeBytes,
    bool         pStatsOnly,
    bool         pBypass,
    set<char>    pExclusions
    ) :
    mBaseReadAddress(0),
    mCurrReadAddress(0),
    mBaseWriteAddress(0),
    mCurrWriteAddress(0)
{
    mInputDataPlayhead  = 0;

    mBitCompactor            = new BitCompactor();
    mBitCompactorCodecConfig = new BitCompactorCodecConfig;

    BitCompactorCodecConfigDefaults();

    mBitCompactorCodecConfig->exclusions        = pExclusions;
    mBitCompactorCodecConfig->blockSizeBytes    = pBlockSizeBytes;
    mBitCompactorCodecConfig->superBlockSizeBytes    = pSuperBlockSizeBytes;
    mBitCompactorCodecConfig->statsOnly         = pStatsOnly;
    mBitCompactorCodecConfig->bypass            = pBypass;
    mBitCompactorCodecConfig->verbosity         = pVerbosity;

    mMemoryRead  = nullptr;
    mMemoryWrite = nullptr;

    BTC_REPORT_INFO (mBitCompactorCodecConfig->verbosity, 3, "BitCompactorCodec: custom constructor");
}

BitCompactorCodec::BitCompactorCodec(
    memoryRead   pMemoryRead,
    memoryWrite  pMemoryWrite,
    uint32_t     pVerbosity,
    uint32_t     pBlockSizeBytes,
    uint32_t     pSuperBlockSizeBytes,
    bool         pStatsOnly,
    bool         pBypass,
    set<char>    pExclusions
    ) :
    mBaseReadAddress(0),
    mCurrReadAddress(0),
    mBaseWriteAddress(0),
    mCurrWriteAddress(0)
{
    mInputDataPlayhead  = 0;

    mBitCompactor            = new BitCompactor();
    mBitCompactorCodecConfig = new BitCompactorCodecConfig;

    BitCompactorCodecConfigDefaults();

    mBitCompactorCodecConfig->exclusions        = pExclusions;
    mBitCompactorCodecConfig->blockSizeBytes    = pBlockSizeBytes;
    mBitCompactorCodecConfig->superBlockSizeBytes    = pSuperBlockSizeBytes;
    mBitCompactorCodecConfig->statsOnly         = pStatsOnly;
    mBitCompactorCodecConfig->bypass            = pBypass;
    mBitCompactorCodecConfig->verbosity         = pVerbosity;

    mMemoryRead  = pMemoryRead;
    mMemoryWrite = pMemoryWrite;

    BTC_REPORT_INFO (mBitCompactorCodecConfig->verbosity, 3, "BitCompactorCodec: custom constructor with memory access callbacks defined");
}

BitCompactorCodec::~BitCompactorCodec()
{
    BTC_REPORT_INFO (mBitCompactorCodecConfig->verbosity, 3, "BitCompactorCodec: destructor");

    delete mBitCompactor;
    delete mBitCompactorCodecConfig;
}

vector<char> BitCompactorCodec::BitCompactorCodecGetInputData (
                                                        const uint32_t &inputDataLength,
                                                        const uint8_t  *inputData,
                                                        const uint32_t  stepSize,
                                                        const set<char> &exclusions = { }
                                                        )
{
    stringstream rptStream;
    vector<char> nd;
    uint32_t  len = 0;
    char *data;

    data = new char[mBitCompactorCodecConfig->blockSizeBytes];

    rptStream.str(""); rptStream << __FUNCTION__ << ": start (idl: " << inputDataLength << " step: " << stepSize << " excls: " << exclusions.size() << ")";
    BTC_REPORT_INFO (mBitCompactorCodecConfig->verbosity, 2, rptStream.str().c_str());

    do
    {
        len = ( ( inputDataLength - mInputDataPlayhead ) > mBitCompactorCodecConfig->blockSizeBytes ) ?
              mBitCompactorCodecConfig->blockSizeBytes : ( inputDataLength - mInputDataPlayhead );

        rptStream.str(""); rptStream << __FUNCTION__ << ": block (len: " << len << " playhead: " << mInputDataPlayhead << ")";
        BTC_REPORT_INFO (mBitCompactorCodecConfig->verbosity, 2, rptStream.str().c_str());

        if ( ( len > 0 ) && ( len <= inputDataLength ) )
        {
            if ( mMemoryRead )
            {
                rptStream.str(""); rptStream << __FUNCTION__ << ": memoryRead " << len << " bytes into data buffer (mCurrReadAddress 0x" << hex << mCurrReadAddress << ", mBaseReadAddress 0x" << hex << mBaseReadAddress << ")";
                BTC_REPORT_INFO (mBitCompactorCodecConfig->verbosity, 3, rptStream.str().c_str());

                mMemoryRead ( mCurrReadAddress, reinterpret_cast<uint8_t*>(&data[0]), static_cast<size_t>(len) );
                mInputDataPlayhead += len;
                mCurrReadAddress   += len;
            }
            else
            {
                rptStream.str(""); rptStream << __FUNCTION__ << ": memcpy " << len << " bytes into data buffer";
                BTC_REPORT_INFO (mBitCompactorCodecConfig->verbosity, 3, rptStream.str().c_str());

                memcpy ( &data[0], &inputData[mInputDataPlayhead], len );
                mInputDataPlayhead += len;
            }
        }

        for ( uint32_t i = 0; i < len && nd.size() < stepSize; i++)
        {
            if ( exclusions.find( data[i] ) != exclusions.end())
                continue;
            nd.push_back(data[i]);
        }
    } while (!(mInputDataPlayhead >= inputDataLength) && nd.size() != stepSize);

    rptStream.str(""); rptStream << __FUNCTION__ << ": returning " << nd.size() << " bytes" <<  endl;
    BTC_REPORT_INFO (mBitCompactorCodecConfig->verbosity, 2, rptStream.str().c_str());

    delete[] data;
    return nd;
}

uint32_t BitCompactorCodec::BitCompactorCodecDecompressArray( const uint64_t  readAddress,
                                                              const uint64_t  writeAddress,
                                                              const uint32_t &inputDataLength,
                                                              const uint8_t  *inputData,
                                                              uint8_t  *outputData
                                                              )
{
    stringstream rptStream;
    std::string lvpDummyFilename = "";
    uint32_t outputDataLength = 0;

    rptStream.str("");
    rptStream << "*************************************************************" << endl;
    rptStream << "***          Running De-Compression Model                 ***" << endl;
    rptStream << "*************************************************************";
    BTC_REPORT_INFO (mBitCompactorCodecConfig->verbosity, 1, rptStream.str().c_str());

    BitCompactor::btcmpctr_compress_wrap_args_t lvtBitCompactorDecompressArgs;

    lvtBitCompactorDecompressArgs.bypass_en      = mBitCompactorCodecConfig->bypass;
    lvtBitCompactorDecompressArgs.dual_encode_en = mBitCompactorCodecConfig->dualEncodeEn;
    lvtBitCompactorDecompressArgs.proc_bin_en    = mBitCompactorCodecConfig->procBinningEn;
    lvtBitCompactorDecompressArgs.proc_btmap_en  = mBitCompactorCodecConfig->procBitmapEn;
    lvtBitCompactorDecompressArgs.align          = mBitCompactorCodecConfig->alignMode;
    lvtBitCompactorDecompressArgs.verbosity      = mBitCompactorCodecConfig->verbosity;
    lvtBitCompactorDecompressArgs.SblkSize       = mBitCompactorCodecConfig->blockSizeBytes;
    lvtBitCompactorDecompressArgs.LblkSize       = mBitCompactorCodecConfig->superBlockSizeBytes;
    lvtBitCompactorDecompressArgs.mixedBlkSize   = mBitCompactorCodecConfig->mixedBlockSizeEn;

    // if the memoryRead callback is set
    // then set the mBaseReadAddress and mCurrReadAddress to this value
    //
    if ( mMemoryRead )
    {
        mBaseReadAddress = readAddress;
        mCurrReadAddress = readAddress;
    }
    else
    {
        mBaseReadAddress = 0x0;
        mCurrReadAddress = 0x0;
    }

    // if the memoryWrite callback is set
    // then set the mBaseWriteAddress and mCurrWriteAddress to this value
    //
    if ( mMemoryWrite )
    {
        mBaseWriteAddress = writeAddress;
        mCurrWriteAddress = writeAddress;
    }
    else
    {
        mBaseWriteAddress = 0x0;
        mCurrWriteAddress = 0x0;
    }    

    if ( mMemoryRead && mMemoryWrite )
    {
        // read inputDataLength bytes into a pre-sized vector of uint8_t using mMemoryRead callback
        vector<uint8_t> lvtUpstreamReadBuffer;
        vector<uint8_t> lvtUpstreamWriteBuffer;
        lvtUpstreamReadBuffer.resize(inputDataLength);
        lvtUpstreamWriteBuffer.resize(BTC_MAX_DECOMPRESS_FACTOR*inputDataLength);

        mMemoryRead(mCurrReadAddress, lvtUpstreamReadBuffer.data(), inputDataLength);

        // provide pointer to this vector's data container to DecompressWrap
        // provide pointer to output data storage vector sized for worst-case ratio to DecompressWrap
        mBitCompactor->DecompressWrap(  const_cast<unsigned char*>(reinterpret_cast<const uint8_t*>(lvtUpstreamReadBuffer.data())),
                                        const_cast<int*>(reinterpret_cast<const int*>(&inputDataLength)),
                                        reinterpret_cast<unsigned char*>(lvtUpstreamWriteBuffer.data()),
                                        reinterpret_cast<int*>(&outputDataLength),
                                        &lvtBitCompactorDecompressArgs
                                        );       

        // after DecompressWrap call, write contents of output data storage vector upstream via write callback
        mMemoryWrite(mCurrWriteAddress, lvtUpstreamWriteBuffer.data(), outputDataLength);

        mCurrReadAddress  += inputDataLength;
        mCurrWriteAddress += outputDataLength;
    }
    else
    {
        // in this use case, storage for input and output data is already in place
        // no read/write callbacks need to be done; input data is already in the input buffer,
        // output buffer is already sized to store worst case output
        mBitCompactor->DecompressWrap(  const_cast<unsigned char*>(reinterpret_cast<const uint8_t*>(inputData)),
                                        const_cast<int*>(reinterpret_cast<const int*>(&inputDataLength)),
                                        reinterpret_cast<unsigned char*>(outputData),
                                        reinterpret_cast<int*>(&outputDataLength),
                                        &lvtBitCompactorDecompressArgs
                                        );
    }

    rptStream.str(""); rptStream << "*************************************************************";
    BTC_REPORT_INFO (mBitCompactorCodecConfig->verbosity, 1, rptStream.str().c_str());

    if ( mBitCompactor->getDecodeErrorCount() == 0 )
    {
        return outputDataLength;
    }
    else
    {
        return 0;
        // BTC_DEC_INT decompress error occurred; call BitCompactorCodecGetDecompressErrorCount() method to return error count
    }
}

bool BitCompactorCodec::BitCompactorCodecConfigParseCLI(int argc, char** argv)
{
    if (argc <= 1)
    {
        printf("No Arguments Specified!!!\n");
        return false;
    }
    argc--;
    argv++;

    BitCompactorCodecConfigDefaults();

    while(argc > 0)
    {
        if(strcmp(*argv, "-c") == 0) {
            argc--;
            argv++;
            mBitCompactorCodecConfig->compressFlag = true;
            // Get outFile Name to compress
            mBitCompactorCodecConfig->outFileName = *argv;
            argc--;
            argv++;
        } else if(strcmp(*argv, "-d") == 0) {
            argc--;
            argv++;
            mBitCompactorCodecConfig->compressFlag = false;
            // Get outFile Name to compress
            mBitCompactorCodecConfig->outFileName = *argv;
            argc--;
            argv++;
        } else if(strcmp(*argv, "-f") == 0) {
            argc--;
            argv++;
            mBitCompactorCodecConfig->inFileName = *argv;
            argc--;
            argv++;
        } else if(strcmp(*argv, "-v") == 0) {
            argc--;
            argv++;

            // @ajgorman: KW fix
            try {
              size_t pos;
              string arg = *argv;
              int verbosityLevel = stoi(arg, &pos);
              if (pos < arg.size()) {
                  mBitCompactorCodecConfig->verbosity = 0;
                  BTC_REPORT_ERROR("Invalid verbosity specified on command line, defaulting to 0");
              }
              mBitCompactorCodecConfig->verbosity = verbosityLevel;
            } catch (...) {
                mBitCompactorCodecConfig->verbosity = 0;
                BTC_REPORT_ERROR("Invalid verbosity specified on command line, defaulting to 0");
            }

            argc--;
            argv++;
        } else if(strcmp(*argv, "-ratio") == 0) {
            argc--;
            argv++;
            mBitCompactorCodecConfig->ratioEn = true;
        } else if(strcmp(*argv, "-mixed_blkSize_en") == 0) {
            argc--;
            argv++;
            mBitCompactorCodecConfig->mixedBlockSizeEn = true;
        } else if(strcmp(*argv, "-proc_bin_en") == 0) {
            argc--;
            argv++;
            mBitCompactorCodecConfig->procBinningEn = true;
        } else if(strcmp(*argv, "-proc_btmap_en") == 0) {
            argc--;
            argv++;
            mBitCompactorCodecConfig->procBitmapEn = true;
        } else if(strcmp(*argv, "-dual_encode_dis") == 0) {
            argc--;
            argv++;
            mBitCompactorCodecConfig->dualEncodeEn = false;
        } else if(strcmp(*argv, "-bypass_en") == 0) {
            argc--;
            argv++;
            mBitCompactorCodecConfig->bypass = true;
        } else if(strcmp(*argv, "-align") == 0) {
            argc--;
            argv++;

            // @ajgorman: KW fix
            try {
              size_t pos;
              string arg = *argv;
              int align = stoi(arg, &pos);
              if (pos < arg.size()) {
                  BTC_REPORT_ERROR("Invalid align mode specified on command line, exiting...");
                  exit(1);
              }
              mBitCompactorCodecConfig->alignMode = align;
            } catch (...) {
                BTC_REPORT_ERROR("Invalid align mode specified on command line, exiting...");
                exit(1);
            }

            argc--;
            argv++;
        } else if(strcmp(*argv, "-minfixbitln") == 0) {
            argc--;
            argv++;

            // @ajgorman: KW fix
            try {
              size_t pos;
              string arg = *argv;
              int minFixedBitLn = stoi(arg, &pos);
              if (pos < arg.size()) {
                  BTC_REPORT_ERROR("Invalid minFixedBitLn specified on command line, exiting...");
                  exit(1);
              }
              mBitCompactorCodecConfig->minFixedBitLn = minFixedBitLn;
            } catch (...) {
                BTC_REPORT_ERROR("Invalid minFixedBitLn specified on command line, exiting...");
                exit(1);
            }

            argc--;
            argv++;
        } else {
            printf("Unknown Argument %s\n", *argv);
            argc--;
        }
    }

    BitCompactorCodecConfigUpdate();

    return true;
}        

// Set engine configuration from wrapper configuration
//
void BitCompactorCodec::BitCompactorCodecConfigUpdate()
{
    mBitCompactor->mBitCompactorConfig->blockSize      = mBitCompactorCodecConfig->blockSizeBytes;
    mBitCompactor->mBitCompactorConfig->superBlockSize = mBitCompactorCodecConfig->superBlockSizeBytes;
    mBitCompactor->mBitCompactorConfig->minFixedBitLn  = mBitCompactorCodecConfig->minFixedBitLn;
    mBitCompactor->mBitCompactorConfig->cmprs          = ( mBitCompactorCodecConfig->compressFlag )? 1 : 0;
    mBitCompactor->mBitCompactorConfig->bypass_en      = ( mBitCompactorCodecConfig->bypass )? 1 : 0;
    mBitCompactor->mBitCompactorConfig->dual_encode_en = ( mBitCompactorCodecConfig->dualEncodeEn )? 1 : 0;
    mBitCompactor->mBitCompactorConfig->proc_bin_en    = ( mBitCompactorCodecConfig->procBinningEn )? 1 : 0;
    mBitCompactor->mBitCompactorConfig->proc_btmap_en  = ( mBitCompactorCodecConfig->procBitmapEn )? 1 : 0;
    mBitCompactor->mBitCompactorConfig->mixedBlkSize   = ( mBitCompactorCodecConfig->mixedBlockSizeEn )? 1 : 0;
    mBitCompactor->mBitCompactorConfig->align          = mBitCompactorCodecConfig->alignMode;
    mBitCompactor->mBitCompactorConfig->ratio          = ( mBitCompactorCodecConfig->ratioEn )? 1 : 0;

    // @ajgorman: KW fix
    unsigned int maxInCharsToCopy = min((sizeof(mBitCompactor->mBitCompactorConfig->inFileName)/sizeof(char))-1, mBitCompactorCodecConfig->inFileName.length());
    size_t inCharsCopied = mBitCompactorCodecConfig->inFileName.copy(mBitCompactor->mBitCompactorConfig->inFileName, maxInCharsToCopy, 0);
    mBitCompactor->mBitCompactorConfig->inFileName[inCharsCopied] = '\0';

    // @ajgorman: KW fix
    unsigned int maxOutCharsToCopy = min((sizeof(mBitCompactor->mBitCompactorConfig->outFileName)/sizeof(char))-1, mBitCompactorCodecConfig->outFileName.length());
    size_t outCharsCopied = mBitCompactorCodecConfig->outFileName.copy(mBitCompactor->mBitCompactorConfig->outFileName, maxOutCharsToCopy, 0);
    mBitCompactor->mBitCompactorConfig->outFileName[outCharsCopied] = '\0';

    mBitCompactor->mBitCompactorConfig->verbosity      = mBitCompactorCodecConfig->verbosity;
    
}

void BitCompactorCodec::BitCompactorCodecDecompressStream( const uint8_t  inputData[32],
                                          std::vector<uint8_t>& outputData
                                          )
{
    //mBitCompactor->readEncodedDataStreaming(inputData, outputData);
}

uint32_t BitCompactorCodec::BitCompactorCodecGetCompressErrorCount ()
{
    return mBitCompactor->getEncodeErrorCount();
}

uint32_t BitCompactorCodec::BitCompactorCodecGetDecompressErrorCount ()
{
    return mBitCompactor->getDecodeErrorCount();
}

uint32_t BitCompactorCodec::BitCompactorCodecCompressArray(
                                                    const uint64_t  readAddress,
                                                    const uint64_t  writeAddress,
                                                    const uint32_t &inputDataLength,
                                                    const uint8_t  *inputData,
                                                    uint8_t        *outputData
                                                    )
{
    stringstream rptStream;
    uint32_t                   outputDataLength        = 0;
    uint64_t                   lviInputDataTallyBytes  = 0;
    uint64_t                   lviOutputDataTallyBytes = 0;
    uint64_t                   lviEncodedDataBytes     = 0;
    vector<char>               lvvOutputDataBuffer;
    vector<char>               nd;
    string                     lvsDummyString          = "";
    float                      lvfCompressRatio        = 0.0;

    rptStream.str("");
    rptStream << "*************************************************************" << endl;
    rptStream << "***            Running Compression Model                  ***" << endl;
    rptStream << "*************************************************************";
    BTC_REPORT_INFO (mBitCompactorCodecConfig->verbosity, 1, rptStream.str().c_str());

    rptStream.str(""); rptStream << "BitCompactorCodecCompress: starting with IDL " << inputDataLength << " bytes" <<  endl;
    BTC_REPORT_INFO (mBitCompactorCodecConfig->verbosity, 2, rptStream.str().c_str());

    lvvOutputDataBuffer.clear();
    mInputDataPlayhead = 0;

    BitCompactor::btcmpctr_compress_wrap_args_t lvtBitCompactorCompressArgs;

    lvtBitCompactorCompressArgs.bypass_en      = mBitCompactorCodecConfig->bypass;
    lvtBitCompactorCompressArgs.dual_encode_en = mBitCompactorCodecConfig->dualEncodeEn;
    lvtBitCompactorCompressArgs.proc_bin_en    = mBitCompactorCodecConfig->procBinningEn;
    lvtBitCompactorCompressArgs.proc_btmap_en  = mBitCompactorCodecConfig->procBitmapEn;
    lvtBitCompactorCompressArgs.align          = mBitCompactorCodecConfig->alignMode;
    lvtBitCompactorCompressArgs.verbosity      = mBitCompactorCodecConfig->verbosity;
    lvtBitCompactorCompressArgs.SblkSize       = mBitCompactorCodecConfig->blockSizeBytes;
    lvtBitCompactorCompressArgs.LblkSize       = mBitCompactorCodecConfig->superBlockSizeBytes;
    lvtBitCompactorCompressArgs.mixedBlkSize   = mBitCompactorCodecConfig->mixedBlockSizeEn;
    lvtBitCompactorCompressArgs.minFixedBitLn  = mBitCompactorCodecConfig->minFixedBitLn;

    // if the memoryRead callback is set
    // then set the mBaseReadAddress and mCurrReadAddress to this value
    //
    if ( mMemoryRead )
    {
        mBaseReadAddress = readAddress;
        mCurrReadAddress = readAddress;
    }
    else
    {
        mBaseReadAddress = 0x0;
        mCurrReadAddress = 0x0;
    }

    // if the memoryWrite callback is set
    // then set the mBaseWriteAddress and mCurrWriteAddress to this value
    //
    if ( mMemoryWrite )
    {
        mBaseWriteAddress = writeAddress;
        mCurrWriteAddress = writeAddress;
    }
    else
    {
        mBaseWriteAddress = 0x0;
        mCurrWriteAddress = 0x0;
    }

    while ( ( nd = BitCompactorCodecGetInputData(
                                               inputDataLength,
                                               inputData,
                                               inputDataLength, // don't chop into blocks at this point
                                               mBitCompactorCodecConfig->exclusions) ).size() > 0 )
    {
        // this reads in all the data in the nd vector, except the excluded symbols
        // up the the set STEP size

        rptStream.str(""); rptStream << "BitCompactorCodecCompress: reset codec" <<  endl;
        BTC_REPORT_INFO (mBitCompactorCodecConfig->verbosity, 2, rptStream.str().c_str());
        mBitCompactor->reset();

        rptStream.str(""); rptStream << "BitCompactorCodecCompress: encode" <<  endl;
        BTC_REPORT_INFO (mBitCompactorCodecConfig->verbosity, 2, rptStream.str().c_str());

        uint8_t* lvtCompressBuffer = new uint8_t[nd.size()*2]; // sized to allow for no compression, plus overhead

        lviEncodedDataBytes = mBitCompactor->CompressArray(reinterpret_cast<unsigned char*>(nd.data()),
                                                           nd.size(),
                                                           reinterpret_cast<unsigned char*>(lvtCompressBuffer),
                                                           0, // unused
                                                           &lvtBitCompactorCompressArgs);

        lviInputDataTallyBytes = lviInputDataTallyBytes + nd.size();

        rptStream.str(""); rptStream << "BitCompactorCodecCompress: write to buffer" <<  endl;
        BTC_REPORT_INFO (mBitCompactorCodecConfig->verbosity, 2, rptStream.str().c_str());
      
        lvvOutputDataBuffer.resize(lvvOutputDataBuffer.size()+lviEncodedDataBytes);

        if ( mMemoryWrite )
        {
            // Upstream memory write via memory access callback
            mMemoryWrite(mCurrWriteAddress, lvtCompressBuffer, lviEncodedDataBytes);

            mCurrWriteAddress += lviEncodedDataBytes;
        }
        else
        {
            // C++ STL vector container guarantees contiguous storage of underlying elements,
            // so it's safe to do a memcpy of the compressed data bytes onto the end of any 
            // existing stored data, provided we do the resize first...
            
            void* lvpOutputDataBufferDst = reinterpret_cast<void*>(&lvvOutputDataBuffer.at(lvvOutputDataBuffer.size()-lviEncodedDataBytes));

            memcpy(lvpOutputDataBufferDst, lvtCompressBuffer, lviEncodedDataBytes);
        }

        delete[] lvtCompressBuffer; // dispose of the temporary compress buffer

        lviOutputDataTallyBytes = lviOutputDataTallyBytes + lviEncodedDataBytes;
        lvfCompressRatio = (100.0 * lviEncodedDataBytes / nd.size());

        rptStream.str("");
        rptStream << "BitCompactorCodecCompress: block compress size = " << lviEncodedDataBytes << " bits" << endl;
        rptStream << "BitCompactorCodecCompress: block total size    = " << nd.size() << " bits" << endl;
        rptStream << "BitCompactorCodecCompress: block Compression ratio: " << fixed << setw(5) << setprecision(2) << lvfCompressRatio << " %";
        BTC_REPORT_INFO (mBitCompactorCodecConfig->verbosity, 2, rptStream.str().c_str());
    }

    outputDataLength = lvvOutputDataBuffer.size();

    memcpy ( reinterpret_cast<void*>(outputData), reinterpret_cast<void*>(lvvOutputDataBuffer.data()), outputDataLength );

    lvfCompressRatio = (lviInputDataTallyBytes <= 0) ? 0 : ( 100.0 * lviOutputDataTallyBytes / lviInputDataTallyBytes );

    rptStream.str("");
    rptStream << "*************************************************************" << endl;
    rptStream << "BitCompactorCodecCompress: OVERALL TOTALS" << endl;
    rptStream << "BitCompactorCodecCompress: Compressed size:   " << lviOutputDataTallyBytes << " bits" << endl;
    rptStream << "BitCompactorCodecCompress: Original size:     " << lviInputDataTallyBytes  << " bits" << endl;
    rptStream << "BitCompactorCodecCompress: Compression ratio: " << fixed << setw(5) << setprecision(2) << lvfCompressRatio << " %";
    BTC_REPORT_INFO (mBitCompactorCodecConfig->verbosity, 1, rptStream.str().c_str());

    if ( mBitCompactorCodecConfig->ratioEn )
    {
        rptStream.str("");
        rptStream << lviInputDataTallyBytes << "," << lviOutputDataTallyBytes << "," << fixed << setw(5) << setprecision(2) << lvfCompressRatio ;
        BTC_REPORT_INFO (mBitCompactorCodecConfig->verbosity, 0, rptStream.str().c_str());
    }
    
    rptStream.str("");
    rptStream << "*************************************************************" <<  endl;
    rptStream << "BitCompactorCodecCompress: finishing with ODL " << outputDataLength << " bytes" <<  endl;
    rptStream << "*************************************************************";
    BTC_REPORT_INFO (mBitCompactorCodecConfig->verbosity, 1, rptStream.str().c_str());

    if ( mBitCompactor->getEncodeErrorCount() == 0 )
    {
        return outputDataLength;
    }
    else
    {
        return 0;
        // Compress error occurred; call BitCompactorCodecGetCompressErrorCount() method to return error count
    }
}

void BitCompactorCodec::BitCompactorCodecConfigDefaults()
{
    mBitCompactorCodecConfig->bitsPerSym = 8; // redundo?
    mBitCompactorCodecConfig->maxEncSyms = 16; // redundo?
    mBitCompactorCodecConfig->blockSizeBytes = 64;
    mBitCompactorCodecConfig->superBlockSizeBytes = 4096;
    mBitCompactorCodecConfig->minFixedBitLn = 3;
    mBitCompactorCodecConfig->compressFlag = 1;
    mBitCompactorCodecConfig->exclusions.clear();
    mBitCompactorCodecConfig->statsOnly = false;
    mBitCompactorCodecConfig->bypass = false;
    mBitCompactorCodecConfig->dualEncodeEn = true;
    mBitCompactorCodecConfig->procBinningEn = false;
    mBitCompactorCodecConfig->procBitmapEn = false;
    mBitCompactorCodecConfig->mixedBlockSizeEn = false;
    mBitCompactorCodecConfig->alignMode = 1;
    mBitCompactorCodecConfig->ratioEn = false;
    mBitCompactorCodecConfig->inFileName = "";
    mBitCompactorCodecConfig->outFileName = "";

    mBitCompactorCodecConfig->verbosity = 0;     // set between 0-5,
                                            // 0 shows basic info,
                                            // 3 shows Metadata and some other useful stuff,
                                            // 5 shows all available info

    BitCompactorCodecConfigUpdate();
}



#ifdef ENABLE_BTC_C_API
extern "C"  BitCompactorCodec *BitCompactorCodecInitC(
    unsigned int   pVerbosity,
    unsigned int   pBlockSizeBytes,
    unsigned int   pSuperBlockSizeBytes,
    bool           pStatsOnly,
    bool           pBypass
    )
{
    BitCompactorCodec     *lvpBitCompactorCodec;

    lvpBitCompactorCodec = new BitCompactorCodec( 
                                        pVerbosity,
                                        pBlockSizeBytes,
                                        pSuperBlockSizeBytes,
                                        pStatsOnly,
                                        pBypass,
                                        {}
                                        );

    return lvpBitCompactorCodec;
}

extern "C" unsigned int BitCompactorCodecCompressArrayC(   char*          outputData,
                                                      unsigned long* outputDataLength,
                                                      char*          inputData,
                                                      unsigned long  inputDataLength,
                                                      BitCompactorCodec  *BitCompactorCodecPtr
                                                      )
{

    *outputDataLength =(unsigned long) BitCompactorCodecPtr->BitCompactorCodecCompressArray( (uint32_t &) inputDataLength,
                                                                                   (uint8_t *)(inputData),
                                                                                   (uint8_t *)(outputData)
                                                                                  );
    if (outputDataLength > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}


extern "C" unsigned int BitCompactorCodecDecompressArrayC( char*          outputData,
                                                      unsigned long* outputDataLength,
                                                      char*          inputData,
                                                      unsigned long  inputDataLength,
                                                      BitCompactorCodec * BitCompactorCodecPtr
                                                      )
{
    *outputDataLength = (unsigned long) BitCompactorCodecPtr->BitCompactorCodecDecompressArray( (uint32_t &) inputDataLength,
                                                                                      (uint8_t *) inputData,
                                                                                      (uint8_t *) outputData
                                                                                     );
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
