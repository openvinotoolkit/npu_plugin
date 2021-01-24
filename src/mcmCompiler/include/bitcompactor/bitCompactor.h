#pragma once
#include <cstdint>
#include <istream>
#include <iostream>
#include <set>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cassert>

/* Define constants for the algorithm here
 *
 */
#ifndef __BTCMPCTR__H__
#define __BTCMPCTR__H__
#ifdef BTC_USE_DPI
    #include <svdpi.h>
#endif

// Block Size
#define BLKSIZE 64
#define BIGBLKSIZE 4096
// Dual Length Mode
// Indicates if Bit lenght of the compressed data 
// should be included in the header
#define DL_INC_BL 
// ALGO
#define BITC_ALG_NONE 16
#define LFTSHFTPROC 5
#define BINEXPPROC 4
#define BTEXPPROC 6
#define ADDPROC 3
#define SIGNSHFTADDPROC 2
#define NOPROC 0
#define SIGNSHFTPROC 1

#define EOFR 0
#define CMPRSD 3
#define UNCMPRSD 2
#define LASTBLK 1
// Header bit positions index
#define BITLN 5
#define ALGO 2

#define INCR_STATE_UC(state,len)  (*state)++; if(*state == 8) { *state = 0; (*len)++; } 
#define ELEM_SWAP(a,b) { register elem_type t=(a);(a)=(b);(b)=t; }

//-----------------------------------------------------
//64B block size related constants
//-----------------------------------------------------
#define NUMALGO 8
#define MINPRDCT_IDX 0
#define MINSPRDCT_IDX 1
#define MUPRDCT_IDX 2
#define NOPRDCT_IDX 3
#define NOSPRDCT_IDX 4
#define MEDPRDCT_IDX 5
#define BINCMPCT_IDX 6
#define BTMAP_IDX 7

#define MAXSYMS 16
#define NUMSYMSBL 4
//-----------------------------------------------------
// 4K Block Size related constants
//-----------------------------------------------------

#define NUM4KALGO 2
#define BINCMPCT4K_IDX 0
#define BTMAP4K_IDX 1

#define MAXSYMS4K 64
#define NUMSYMSBL4K 6

//-----------------------------------------------------
#define FILENAME_SIZE 256


// Function Declarations

// Alignment
#define DEFAULT_ALIGN 1
// 1 --> 32B Alignment
// 2 --> 64B Alignment
// 0 --> No  Alignment

#endif

#ifdef VPU2p6_BUILD

#include "moviReportingManager.h"
#define BTC_REPORT_INFO(verbosity,lvl,x)  if(lvl <= verbosity) { MOVI_REPORT_INFO("bitCompactor",x,((lvl<1)?MOVI_HIGH:MOVI_LOW)); }
#define BTC_REPORT_ERROR(x) { MOVI_REPORT_ERROR("bitCompactor",x); }

#else

#define BTC_REPORT_INFO(verbosity,lvl,x)  if(lvl <= verbosity) { std::cout << x << std::endl; }
#define BTC_REPORT_ERROR(x) { std::cerr << x << std::endl; }

#endif

#define BTC_MAX_DECOMPRESS_FACTOR 5
#define BTC_MAX_COMPRESS_FACTOR 2

class BitCompactor
{
public:

    // Arguments typedef.
    typedef struct btcmpctr_args_s
    {
        int cmprs;
        int decmprs;
        int blockSize;
        int superBlockSize;
        int minFixedBitLn;
        char inFileName[FILENAME_SIZE];
        char outFileName[FILENAME_SIZE];
        int ratio;
        int verbosity;
        int mixedBlkSize;
        int proc_bin_en;   // Enable the Binning preprocessing mode.
        int proc_btmap_en; // Enable the Bitmap pre-processing mode.
        int align;    // 0 -> byte align, 1->32B align, 2 -> 64B align.
        int dual_encode_en; // Enables the dual encoding mode.
        int bypass_en; // When set to 1, the compressor will treat all blocks as bypass.
    } btcmpctr_args_t;

    // Compress Wrap Arguments
    typedef struct btcmpctr_compress_wrap_args_s
    {
        int verbosity;
        int mixedBlkSize;
        int proc_bin_en;   // Enable the Binning preprocessing mode.
        int proc_btmap_en; // Enable the Bitmap pre-processing mode.
        int LblkSize; // Typicall 4096
        int SblkSize; // Typically 64
        int align;    // 0 -> byte align, 1->32B align, 2 -> 64B align.
        int dual_encode_en; // Enables the dual encoding mode.
        int bypass_en; // When set to 1, the compressor will treat all blocks as bypass.
        int minFixedBitLn; // Set minimum fixed-length symbol size in bits (0..7, default 3)
    } btcmpctr_compress_wrap_args_t;

    typedef unsigned char elem_type ;

    // Struct defining the return type of an algorithm
    //
    typedef struct btcmpctr_algo_args_s
    {
        unsigned char* inAry;   // Input Buffer to work On.
        unsigned char* minimum; // Array of bytes to Add or bins
        unsigned char* bitln;   // Bit Length after the algo is run.
        unsigned char* residual; // Pre-Processed data to be inserted into the output stream.
                  int  blkSize;  // Current Block Size to work with.
                  int* minFixedBitLn;
                  int* numSyms;  // Number of Symbols binned.
        unsigned char* bitmap;   // bit map when replacing highest frequency symbol.
                  int* numBytes; // Number of bytes in residual that is valid, when bitmap compressed.
    } btcmpctr_algo_args_t;

    // Typedef of the Algo Function pointer.
    typedef void (BitCompactor::*Algo)(btcmpctr_algo_args_t* algoArg);

    // Struct defining the chosen Algorithm and its compressed size
    typedef struct btcmpctr_algo_choice_s
    {
       // Pointer to the chosen Algo function  
       Algo chosenAlgo;
       int  cmprsdSize;
       int  algoType; // 0 64B, 1 4K
       int  none; // None chosen, hence uncompressed.
       int  algoHeader; 
       int  workingBlkSize;
       int  dual_encode;
    } btcmpctr_algo_choice_t;


    BitCompactor();
    ~BitCompactor();

    BitCompactor(const BitCompactor &) = delete;
    BitCompactor& operator= (const BitCompactor &) = delete;

    //      #     ######   ###
    //     # #    #     #   #
    //    #   #   #     #   #
    //   #     #  ######    #
    //   #######  #         #
    //   #     #  #         #
    //   #     #  #        ###

    int  DecompressWrap(       unsigned char*  src,
                               int*   srcLen,
                               unsigned char*  dst,
                               int*  dstLen,
                               btcmpctr_compress_wrap_args_t* args
                               );
    int  CompressWrap(         unsigned char* src,
                               int*           srcLen,
                               unsigned char* dst,
                               int*           dstLen,// dstLen holds the size of the output buffer.
                               btcmpctr_compress_wrap_args_t* args
                               );

    //This is a SWIG/numpy integration friendly interface for the decompression function.
    int  DecompressArray(      unsigned char* src,
                               int   srcLen,
                               unsigned char* dst,
                               int   dstLen, // unused
                               btcmpctr_compress_wrap_args_t* args
                               );
    //This is a SWIG/numpy integration friendly interface for the compression function.
    int  CompressArray(        unsigned char* src,
                               int   srcLen,
                               unsigned char* dst,
                               int   dstLen, // unused
                               btcmpctr_compress_wrap_args_t* args
                              );

    void reset();

    unsigned int getDecodeErrorCount ();
    unsigned int getEncodeErrorCount ();

    btcmpctr_args_t* mBitCompactorConfig;

    int  btcmpctr_cmprs_bound(int bufSize);

    int mVerbosityLevel;
private:

    // CompressWrap
    Algo AlgoAry[NUMALGO];
    Algo AlgoAry4K[NUM4KALGO];

    // Declare Array of Algorithms
    int AlgoAryHeaderOverhead[NUMALGO];
    int AlgoAryHeaderOverhead4K[NUM4KALGO];


    unsigned char btcmpctr_insrt_byte( unsigned char  byte,
                                       unsigned char  bitln,
                                                int*  outBufLen,
                                       unsigned char* outBuf,
                                       unsigned char  state,
                                       unsigned int*  accum,
                                                int   flush
                                     );

    unsigned char btcmpctr_insrt_hdr(int chosenAlgo,
                                     unsigned char  bitln,
                                              int*  outBufLen,
                                     unsigned char* outBuf,
                                     unsigned char  state,
                                     unsigned int*  accum,
                                     unsigned char  eofr,
                                     int            workingBlkSize,
                                     int            mixedBlkSize,
                                     int            align
                                    );

    double btcmpctr_log2( double n );

    void btcmpctr_calc_bitln(unsigned char* residual,
                             unsigned char* bitln,
                             int            blkSize,
                             int*           minFixedBitLn
                            );
    void btcmpctr_calc_dual_bitln( unsigned char* residual,
                                   unsigned char* bitln,
                                            int   blkSize,
                                   unsigned char* bitmap,
                                            int*  compressedSize
                                 );
    void btcmpctr_calc_bitln16(uint16_t* residual,
                               unsigned char* bitln
                              );
    void btcmpctr_tounsigned(signed char* inAry,
                             unsigned char* residual,
                             int            blkSize
                            );
    void btcmpctr_minprdct(
                            btcmpctr_algo_args_t* algoArg
                          );
    void btcmpctr_minSprdct(
                             btcmpctr_algo_args_t* algoArg
                           );
    void btcmpctr_muprdct(
                            btcmpctr_algo_args_t* algoArg
                          );
    void btcmpctr_medprdct(
                            btcmpctr_algo_args_t* algoArg
                          );
    void btcmpctr_noprdct(
                            btcmpctr_algo_args_t* algoArg
                         );
    void btcmpctr_noSprdct(
                            btcmpctr_algo_args_t* algoArg
                         );
    void btcmpctr_binCmpctprdct(
                                btcmpctr_algo_args_t* algoArg
                               );
    void btcmpctr_btMapprdct(
                                btcmpctr_algo_args_t* algoArg
                            );
    void btcmpctr_dummyprdct(
                                btcmpctr_algo_args_t* algoArg
                               );
    void btcmpctr_xtrct_bits(
                             unsigned char* inBuf,
                                       int* inBufLen,
                             unsigned char* state,
                             unsigned char* outByte,
                             unsigned char  numBits
                            );
    unsigned char btcmpctr_xtrct_hdr(
                                     unsigned char* inAry,
                                               int* inAryPtr,
                                     unsigned char  state,
                                     unsigned char* cmp,
                                     unsigned char* eofr,
                                     unsigned char* algo,
                                     unsigned char* bitln,
                                              int*  blkSize,
                                     unsigned char* bytes_to_add, // 1 or 2 bytes
                                     unsigned char* numSyms,
                                              int*  numBytes,
                                     unsigned char* bitmap,
                                              int   mixedBlkSize,
                                              int   dual_encode_en,
                                     unsigned char*  dual_encode
                                    );
    unsigned char btcmpctr_xtrct_bytes_wbitmap(
                                               unsigned char* inAry,
                                                         int* inAryPtr,
                                               unsigned char  state,
                                               unsigned char  bitln,
                                               unsigned char* outBuf, // Assume OutBuf is BLKSIZE
                                                         int  blkSize,
                                               unsigned char* bitmap
                                             );
    unsigned char btcmpctr_xtrct_bytes(
                                        unsigned char* inAry,
                                                  int* inAryPtr,
                                        unsigned char  state,
                                        unsigned char  bitln,
                                        unsigned char* outBuf, // Assume OutBuf is BLKSIZE
                                                  int  blkSize,
                                        unsigned char  mode16
                                      );
    void btcmpctr_tosigned(unsigned char* inAry,
                           unsigned char* outBuf
                          );
    void btcmpctr_addByte(
                          unsigned char* data_to_add,
                          unsigned char  mode16,
                          unsigned char* outBuf
                         );


    void btcmpctr_initAlgosAry(btcmpctr_compress_wrap_args_t* args);

    unsigned char btcmpctr_getAlgofrmIdx(int idx);

    unsigned char btcmpctr_get4KAlgofrmIdx(int idx);

    btcmpctr_algo_choice_t btcmpctr_ChooseAlgo64B(btcmpctr_algo_args_t* algoArg,
                                                                   int mixedBlkSize,
                                                                   int dual_encode_en
                                                );

    btcmpctr_algo_choice_t btcmpctr_ChooseAlgo4K(btcmpctr_algo_args_t* algoArg,
                                                                  int mixedBlkSize
                                               );

    elem_type quick_select(elem_type arr[], unsigned int n) ;

};

#ifdef BTC_USE_DPI

#ifdef __cplusplus
extern "C" {
#endif

#ifdef BTC_USE_DPI_STANDALONE
extern int startBTC(int argc, char* argvArray[]);
#else
extern int startBTC(int argc, const svOpenArrayHandle argvArray);
#endif

#ifdef __cplusplus
}
#endif

void btcmpctr_parse_args(int argc, char** argv, BitCompactor::btcmpctr_args_t* args);

#endif

