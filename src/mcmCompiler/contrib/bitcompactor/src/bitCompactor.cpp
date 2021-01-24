// C++ class built around the C methods in the original BitCompactor model
//

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS // should be safe to remove after kw fixes
#endif

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include "bitCompactor.h"


char debugStr[300];

BitCompactor::BitCompactor() :
        mVerbosityLevel(0)
{
    mBitCompactorConfig = new btcmpctr_args_t;
}

BitCompactor::~BitCompactor()
{
    delete mBitCompactorConfig;
}

unsigned char BitCompactor::btcmpctr_insrt_byte( unsigned char  byte,
                                   unsigned char  bitln,
                                            int*  outBufLen,
                                   unsigned char* outBuf,
                                   unsigned char  state,
                                   unsigned int*  accum,
                                            int   flush
                                 )
{
    unsigned char mask, thisBit, rem;
    // Use 32bit operations to insert a byte into the accumulator
    // Once 32bits are accumulated, push 4bytes to the outBuf 
    // and increment the outBufLen.
    #ifdef __BTCMPCTR__EN_DBG__
    sprintf(debugStr,"State = %d, BitLen = %d, Byte = %x, outByfLen = %d",(int)state,bitln,byte,*outBufLen);    
    BTC_REPORT_INFO(mVerbosityLevel,11,debugStr)
    #endif
    if (flush) {
        // Depending on state write the correct number of bytes to
        // outBuf and increment outBufLen
        if (state != 0) {
            double numbytesf = state/8.0;
            int numBytes = ceil(numbytesf);
            #ifdef __BTCMPCTR__EN_DBG__
            sprintf(debugStr,"Flushing numBytes = %d",numBytes);    
            BTC_REPORT_INFO(mVerbosityLevel,11,debugStr)
            #endif
            #ifdef __BTCMPCTR__EN_DBG__
            sprintf(debugStr,"Accum = %x",*accum);    
            BTC_REPORT_INFO(mVerbosityLevel,11,debugStr)
            #endif
            for(int i = 0; i < numBytes; i++) {
                #ifdef __BTCMPCTR__EN_DBG__
                sprintf(debugStr,"Flushing Byte = %x",(unsigned char)(*accum));    
                BTC_REPORT_INFO(mVerbosityLevel,11,debugStr)
                #endif
                *(outBuf + *outBufLen + i) = (unsigned char)(*accum);
                (*accum) >>= 8;
            }
                (*outBufLen) += numBytes;
            state = 0;
        }
            
    } else {
        // State indicates the number of bits valid in accum.
        // First see if based on state and bitln we will cross
        // over 32bits.
        if( (state + bitln) > 32) {
            // Need to first insert 32-state bits first from byte and
            // then insert the rest.
            rem = 32 - state;
            mask = ~(0xFF << rem);
            thisBit = byte & mask;
            *accum  |= (thisBit << state);
            *((unsigned int *)(outBuf+ *outBufLen)) = *accum;
            (*outBufLen) += 4;
            state = 0;
            *accum = 0;
            // Remaining bits
            byte = byte >> rem;
            rem = bitln - rem;
            mask = ~(0xFF << rem);
            thisBit = byte & mask;
            *accum  |= (thisBit << state);
            state += rem;
        } else {
            // Can insert bitln number of bits into accum
            mask    = ~(0xFF << bitln); // This should give us correct bit values from byte.
            thisBit = byte & mask;
            // Leftshift this by state
            *accum  |= (thisBit << state);
            state += bitln;
            if(state == 32) {
                *((unsigned int*)(outBuf + *outBufLen)) = *accum;
                (*outBufLen) += 4;
                state = 0;
                *accum = 0;
            }
        }
    }

    //for(int j = 0; j < 8; j++) {
    //    if(j < bitln) {
    //        mask    = 1 << j;
    //        thisBit = byte & mask; 
    //        thisBit = thisBit >> j;
    //        mask    = thisBit << state;
    //        outBuf[*outBufLen] |= mask;
    //        INCR_STATE(state,outBufLen,outBuf)
    //        sprintf(debugStr,"State = %d, Len = %d, Bit = %d",(int)state,*outBufLen,thisBit);    
    //        BTC_REPORT_INFO(mVerbosityLevel,6,debugStr)
    //    }
    //}
    return state;
}


// Insert header into the output Buffer.
unsigned char BitCompactor::btcmpctr_insrt_hdr(int chosenAlgo,
                                 unsigned char  bitln,
                                          int*  outBufLen,
                                 unsigned char* outBuf,
                                 unsigned char  state,
                                 unsigned int*  accum,
                                 unsigned char  eofr,
                                 int            workingBlkSize,
                                 int            mixedBlkSize,
                                 int            align
                                )
{
    #ifdef __BTCMPCTR__EN_DBG__
    sprintf(debugStr,"Inserting Header, outBuf Length = %d",*outBufLen);    
    BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
    #endif
    #ifdef __BTCMPCTR__EN_DBG__
    sprintf(debugStr,"Chosen Algo = %d",chosenAlgo);    
    BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
    #endif
    // Header includes the following fields
    // If eofr, then 2 bit is inserted witha value of '10'
    // else if no compression is done, then 1 bits are inserted '0'
    // else 8 bits are inserted '<3 bits bitln><3 bits algo><01>' --> [7:0] if algo != 16bit modes
    bool is16bitmode = 0;
    bool is4K        = (workingBlkSize == BIGBLKSIZE);
    bool isLastBlk    = (workingBlkSize != BLKSIZE) & !is4K;

    if (eofr) {
        //just insert the no compression bit '0' and return.
        state = btcmpctr_insrt_byte(EOFR,2,outBufLen,outBuf,state,accum,0);
        // Once the SKIP header is inserted, check the alignment requirement.
        if ( (align == 1) || (align == 2) ) {
            // 32B alignment
            // OutBufLen and State combined together will tell the current alignment.
            #ifdef __BTCMPCTR__EN_DBG__
            sprintf(debugStr,"Aligning to = %d",align);    
            BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
            #endif
            double numbytesf = state/8.0;
            unsigned int bytesinBuf = (*outBufLen + (unsigned int)ceil(numbytesf));
            unsigned int alignB = ((align == 1) ? 32 : 64);
            unsigned int numBytesToInsert = ((bytesinBuf % alignB) == 0) ? 0 : alignB - (bytesinBuf % alignB);
            unsigned int numBitsToInsert = ((state % 8) == 0) ? 0 : 8-(state %8);
            #ifdef __BTCMPCTR__EN_DBG__
            sprintf(debugStr,"numBits = %d, numBytes = %d, outBufLen = %d, state = %d",numBitsToInsert,numBytesToInsert,*outBufLen,state);    
            BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
            #endif
            state = btcmpctr_insrt_byte(0,numBitsToInsert,outBufLen,outBuf,state,accum,0);
            for(unsigned int i = 0; i < numBytesToInsert; i++) {
                state = btcmpctr_insrt_byte(0,8,outBufLen,outBuf,state,accum,0);
            }
        }
        // Below will ensure byte alignment.
        state = btcmpctr_insrt_byte(EOFR,2,outBufLen,outBuf,state,accum,1); // Flush
        return state;
    } else  if (chosenAlgo == BITC_ALG_NONE) {
        if (isLastBlk) {
            state = btcmpctr_insrt_byte(LASTBLK,2,outBufLen,outBuf,state,accum,0);
            // Insert 6 bits of block Size in bytes.
            state = btcmpctr_insrt_byte(workingBlkSize,6,outBufLen,outBuf,state,accum,0);
        } else {
            state = btcmpctr_insrt_byte(UNCMPRSD,2,outBufLen,outBuf,state,accum,0);
            if(mixedBlkSize) {
                if (is4K) {
                    state = btcmpctr_insrt_byte(1,2,outBufLen,outBuf,state,accum,0);
                } else {
                    state = btcmpctr_insrt_byte(0,2,outBufLen,outBuf,state,accum,0);
                }
            }
        }
        return state;
    } else {
        // We are guarantteed to come in here when workingBlkSize != 64
        // First form the header byte and then use a loop to insert it into 
        // the output buffer.
        //Algo is 3 bits, hence mask off the rest of the bits from chosen Algo.
        state = btcmpctr_insrt_byte(CMPRSD,2,outBufLen,outBuf,state,accum,0);
        if(mixedBlkSize) {
            if (is4K) {
                state = btcmpctr_insrt_byte(1,2,outBufLen,outBuf,state,accum,0);
            } else {
                state = btcmpctr_insrt_byte(0,2,outBufLen,outBuf,state,accum,0);
            }
        }
        state = btcmpctr_insrt_byte(chosenAlgo,3,outBufLen,outBuf,state,accum,0);
        //Bit Lenght is in the range of 1 - 8. it will get encoded to 0 - 7, with 0 indicating 8 bits.
        if ((bitln == 8) && !is16bitmode) { bitln = 0; } // Only if in 8 bit mode, in 16 bit mode this chack should change to == 16. TODO
        if ((bitln == 16) && is16bitmode) { bitln = 0; } // Only if in 8 bit mode, in 16 bit mode this chack should change to == 16. TODO
        state = btcmpctr_insrt_byte(bitln,3,outBufLen,outBuf,state,accum,0);

        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"Inserting Header, outBuf Length = %d",*outBufLen);    
        BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
        #endif
        return state;
    }
}
// Calculates log2 of number.  
double BitCompactor::btcmpctr_log2( double n )  
{  
    // log(n)/log(2) is log2.  
    return log( n ) / log( 2 );  
}  
// Calculate minimum number of bits needed to represent all the numbers
void BitCompactor::btcmpctr_calc_bitln(unsigned char* residual,
                         unsigned char* bitln,
                         int            blkSize,
                         int*           minFixedBitLn
                        )
{
    uint16_t maximum = 0;
    for(int i = 0; i < blkSize; i++) { // TODO : Get residual Length
        if(residual[i] > maximum) 
            maximum = residual[i];
    }
    #ifdef __BTCMPCTR__EN_DBG__
    sprintf(debugStr,"In calc bitln, maximum is %d",maximum);
    BTC_REPORT_INFO(mVerbosityLevel,8,debugStr)
    #endif
    // Find number of bits needed to encode the maximu.
    // Use (maximum + 1) in the log2. This helps when maximum is power of 2.
    *bitln = (maximum == 0) ? 1 : (unsigned char)(ceil(btcmpctr_log2((double)(maximum+1))));
    #ifdef __BTCMPCTR__EN_DBG__
    sprintf(debugStr,"In calc bitln, bitln is %d, minFixedBitLn is %d",*bitln,*minFixedBitLn);
    BTC_REPORT_INFO(mVerbosityLevel,8,debugStr)
    #endif
    if ( *bitln < *minFixedBitLn )
    {
        *bitln = *minFixedBitLn;
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"In calc bitln, bitln is %d, limited by minFixedBitLn %d",*bitln,*minFixedBitLn);
        BTC_REPORT_INFO(mVerbosityLevel,8,debugStr)
        #endif
    }
}

// A function that takes a buffer (residue) and calculates the dual length encoding. 
// One of the lengths could be < 8 and is indicated in a 0 in the bitmap. A 1 in the
// bitmap indicates 8bit symbols.
void BitCompactor::btcmpctr_calc_dual_bitln( unsigned char* residual,
                               unsigned char* bitln,
                                        int   blkSize,
                               unsigned char* bitmap,
                                        int*  compressedSize
                             )
{
    // First calculate/bin the bitlengths
    unsigned char bin[9]; // need to store 1 - 8 bitln
    unsigned char symbolBitln[BLKSIZE]; // Assume this is used only for 64B blocks.
    unsigned char sBitln;
    int cSize[9];
    unsigned char cumSumL,cumSumH;

    for(int i = 1; i < 9; i++) {
        bin[i] = 0;
    }
    for(int i = 0; i < blkSize; i++) {
        // Calculte the number of bits needed to encode the symbol
        sBitln = (*(residual+i) == 0) ? 1 : (unsigned char)(ceil(btcmpctr_log2((double)(*(residual+i)+1))));
        // Increment the bin for that bitlength.
        bin[sBitln]++;
        // Store the bitlenght for the i th symbol.
        symbolBitln[i] = sBitln;    
    }
    #ifdef __BTCMPCTR__EN_DBG__
    for(int i = 1; i < 9; i++) {
        sprintf(debugStr,"Num symbols with length %d is %d",i,bin[i]);
        BTC_REPORT_INFO(mVerbosityLevel,8,debugStr)
    }
    #endif
    // For each of the bins, calculate the compressed Size.
    // And find the bitln that results in the minimum compressed Size.
    for (int i = 1; i < 9; i++) {
        cumSumL = 0;
        for(int j = 1; j <= i; j++) {
            cumSumL += bin[j];
        }
        cumSumH = 0;
        for(int j = i+1; j < 9; j++) {
            cumSumH += bin[j];
        }
        cSize[i] = cumSumL*i + cumSumH*8;
        // Find the minimum compressed Size.
        if (i == 1) {
            *compressedSize = cSize[i];
            *bitln          = i;
        } else {
            if (cSize[i] < *compressedSize) {
                *compressedSize = cSize[i];
                *bitln          = i;
            }
        }
    }
    // *bitln contains the chosen bitln < 8.
    // Calculate the bitmap
    uint32_t shortSymbolCount = 0;
    uint32_t longSymbolCount = 0;
    for(int i = 0; i < blkSize; i++) {
       if (symbolBitln[i] <= *bitln) {
           bitmap[i] = 0;
           shortSymbolCount++;
       } else {
           bitmap[i] = 1;
           longSymbolCount++;
       }
    }

    // Dual-length encoding only makes sense if both symbol lengths are actually needed
    // In the case where we find we can use the short length for all symbols, we could
    // end up with a block made up of 64 1-bit symbols. This is too compressed for the RTL.
    // See https://hsdes.intel.com/appstore/article/#/18012415189
    // 
    if ( longSymbolCount == 0 )
    {
        // RTL cannot accept DL-compressed blocks made up entirely of 'short' symbols
        // Where this occurs, shortSymbolCount == blkSize, longSymbolCount == 0
        // Remedy this by forcing at least one long (8-bit) symbol at start of bitmap
        // Make adjustment to compressedSize and to bitmap[0]
        sprintf(debugStr,"calc_dual_bitln: 8-bit Symbols: 1 (forced); %0d-bit Symbols: %0d", *bitln, shortSymbolCount-1);
        BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)

        // account for the fact that using the 8-bit length for the first symbol will require more bits
        *compressedSize = *compressedSize + (8-(*bitln));
        // set element zero in bitmap to 1 (i.e. force long/8-bit symbol length)
        bitmap[0] = 1;
    }
    else
    {
        // We have at least one 'long' symbol in this block, which is fine
        sprintf(debugStr,"calc_dual_bitln: 8-bit Symbols: %0d; %0d-bit Symbols: %0d", longSymbolCount, *bitln, shortSymbolCount);
        BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
    }
}

// Calculate minimum number of bits needed to represent all the numbers
void BitCompactor::btcmpctr_calc_bitln16(uint16_t* residual,
                           unsigned char* bitln
                          )
{
    uint16_t maximum = 0;
    for(int i = 0; i < BLKSIZE/2; i++) {  // TODO : Get residual Length
        if(residual[i] > maximum) 
            maximum = residual[i];
    }
    *bitln = (maximum == 0) ? 1 : (unsigned char)(ceil(btcmpctr_log2((double)maximum)));
}
// Convert a signed number to unsigned by storing the sign bit in LSB
//
void BitCompactor::btcmpctr_tounsigned(signed char* inAry,
                         unsigned char* residual,
                         int            blkSize
                        )
{
    //if number is <0 then lose the MSB and shift a 1 to LSB.
    for(int i = 0; i < blkSize; i++) { // TODO : inAry size
        if(inAry[i] < 0) {
            residual[i] = (unsigned char) (~inAry[i] << 1) | 0x01;
        } else {
            residual[i] = (unsigned char) (inAry[i] << 1);
        }
    }
}
// Do Min Predict algo on a buffer, return minimum number and the bitln. These are pointers given to the function.
void BitCompactor::btcmpctr_minprdct(
                        btcmpctr_algo_args_t* algoArg
                      )
{
    *(algoArg->minimum) = 255;

    for(int i = 0; i < algoArg->blkSize; i++) {// TODO : inAry size
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"In minprdct, inAry[%d] is %d",i,algoArg->inAry[i]);
        BTC_REPORT_INFO(mVerbosityLevel,12,debugStr)
        #endif
        if(algoArg->inAry[i] < *algoArg->minimum) {
            *algoArg->minimum = algoArg->inAry[i];
        }
    }
    #ifdef __BTCMPCTR__EN_DBG__
    sprintf(debugStr,"In minprdct, minimum is %d",*algoArg->minimum);
    BTC_REPORT_INFO(mVerbosityLevel,7,debugStr)
    #endif
    // Subtract the minimum from the array and make a copy of the array.
    for(int i = 0; i < algoArg->blkSize; i++) {
        algoArg->residual[i] = algoArg->inAry[i] - *algoArg->minimum;
    }
    // Find Bit Length
    btcmpctr_calc_bitln(algoArg->residual,algoArg->bitln,algoArg->blkSize,algoArg->minFixedBitLn);    
}

// Do Min Signed Predict algo on a buffer, return minimum number and the bitln. These are pointers given to the function.
void BitCompactor::btcmpctr_minSprdct(
                         btcmpctr_algo_args_t* algoArg
                       )
{
    signed char  residualS[BLKSIZE], inSAry[BLKSIZE];
    signed char minS;
    for(int i = 0; i < algoArg->blkSize; i++) {// TODO : inAry size
        inSAry[i] = (signed char )algoArg->inAry[i];
    }

    *algoArg->minimum = 255;
    minS = 127;
    
    for(int i = 0; i < algoArg->blkSize; i++) {// TODO : inAry size
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"In minprdct, inSAry[%d] is %d",i,inSAry[i]);
        BTC_REPORT_INFO(mVerbosityLevel,12,debugStr)
        #endif
        if(inSAry[i] < minS) {
            minS = inSAry[i];
        }
    }
    #ifdef __BTCMPCTR__EN_DBG__
    sprintf(debugStr,"In minSprdct, minimum is %d",minS);
    BTC_REPORT_INFO(mVerbosityLevel,7,debugStr)
    #endif
    *algoArg->minimum = (unsigned char)minS;
    #ifdef __BTCMPCTR__EN_DBG__
    sprintf(debugStr,"In minSprdct, minimum is %d",*algoArg->minimum);
    BTC_REPORT_INFO(mVerbosityLevel,7,debugStr)
    #endif

    // Subtract the minimum from the array and make a copy of the array.
    for(int i = 0; i < algoArg->blkSize; i++) {
        residualS[i] = inSAry[i] - minS;
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"In minSprdct, residual[%d] = %d",i,residualS[i]);
        BTC_REPORT_INFO(mVerbosityLevel,12,debugStr)
        #endif
    }
    //Convert to Unsigned
    btcmpctr_tounsigned(residualS, algoArg->residual,algoArg->blkSize);
    // Find Bit Length
    btcmpctr_calc_bitln(algoArg->residual,algoArg->bitln,algoArg->blkSize,algoArg->minFixedBitLn);    
}
// Do Mean Signed Predict algo on a buffer, return minimum number and the bitln. These are pointers given to the function.
void BitCompactor::btcmpctr_muprdct(
                        btcmpctr_algo_args_t* algoArg
                      )
{
    signed char* inSAry = (signed char *)algoArg->inAry;
    signed char  residualS[BLKSIZE];
    signed char muS;
    double      sum,mud;

    *algoArg->minimum = 0;
    muS = 0;
    sum = 0;    
    for(int i = 0; i < algoArg->blkSize; i++) {
            sum += inSAry[i];
    }
    mud = sum/algoArg->blkSize;
    muS = (signed char)(round(mud));

    // Subtract the minimum from the array and make a copy of the array.
    for(int i = 0; i < algoArg->blkSize; i++) {
        residualS[i] = inSAry[i] - muS;
    }
    *algoArg->minimum = (unsigned char)muS;
    //Convert to Unsigned
    btcmpctr_tounsigned(residualS, algoArg->residual,algoArg->blkSize);
    // Find Bit Length
    btcmpctr_calc_bitln(algoArg->residual,algoArg->bitln,algoArg->blkSize,algoArg->minFixedBitLn);    
}
// Do Median Signed Predict algo on a buffer, return minimum number and the bitln. These are pointers given to the function.
// http://ndevilla.free.fr/median/median/src/quickselect.c

void BitCompactor::btcmpctr_medprdct(
                        btcmpctr_algo_args_t* algoArg
                      )
{
    signed char* inSAry = (signed char *)algoArg->inAry;
    signed char  residualS[BLKSIZE];
    signed char medianS;
    for(int i = 0; i < algoArg->blkSize; i++) {
        algoArg->residual[i] = algoArg->inAry[i];
    }    
    // Sort the array, check if blkSize is an odd number, if yes, then median
    // is ary[(blkSize+1)/2] else round((ary[(blkSize/2)] + ary[(blkSize/2)+1])/2)
    // http://ndevilla.free.fr/median/median/src/quickselect.c
    medianS = (signed char) quick_select((unsigned char *)algoArg->residual,algoArg->blkSize);

    // Subtract the minimum from the array and make a copy of the array.
    for(int i = 0; i < algoArg->blkSize; i++) {
        residualS[i] = inSAry[i] - medianS;
    }
    *algoArg->minimum = (unsigned char)medianS;
    //Convert to Unsigned
    btcmpctr_tounsigned(residualS, algoArg->residual,algoArg->blkSize);
    // Find Bit Length
    btcmpctr_calc_bitln(algoArg->residual,algoArg->bitln,algoArg->blkSize,algoArg->minFixedBitLn);    
}
// No Predict, just look at the maximum in the array
void BitCompactor::btcmpctr_noprdct(
                        btcmpctr_algo_args_t* algoArg
                     )
{
    // Copy inAry to residual
    for(int i = 0; i< algoArg->blkSize; i++) {
        algoArg->residual[i] = algoArg->inAry[i];
    }
    btcmpctr_calc_bitln(algoArg->inAry,algoArg->bitln,algoArg->blkSize,algoArg->minFixedBitLn);
}
// No Sign Predict. Store
void BitCompactor::btcmpctr_noSprdct(
                        btcmpctr_algo_args_t* algoArg
                     )
{
    // First convert to unsigned representation by storing MSB in LSB.
    btcmpctr_tounsigned((signed char*) algoArg->inAry, algoArg->residual,algoArg->blkSize);
    btcmpctr_calc_bitln(algoArg->residual,algoArg->bitln,algoArg->blkSize,algoArg->minFixedBitLn);    
    #ifdef __BTCMPCTR__EN_DBG__
    sprintf(debugStr,"In NoSPrdct, bitln is %d",*algoArg->bitln);
    BTC_REPORT_INFO(mVerbosityLevel,7,debugStr)
    #endif
}

// BIN the bytes in the block and check if there are <= 16 unique symbols
// If yes, return the bitln and residual. Residual is the bin number of the 
// byte in the block.
// Also need to return the array of bytes to be inserted after the header. Can be stored in *minimum.
// Need a way to return the number of symbols valid. So that the correct number of bytes can be
// inserted.
void BitCompactor::btcmpctr_binCmpctprdct(
                            btcmpctr_algo_args_t* algoArg
                           )
{
    #ifdef __BTCMPCTR__EN_DBG__
    sprintf(debugStr,"In binCmpctprdct");
    BTC_REPORT_INFO(mVerbosityLevel,7,debugStr)
    #endif
    int inBins   = 0;
    int allinBin = 1;
    int maxsyms;
    maxsyms = (algoArg->blkSize > 64) ? MAXSYMS4K : MAXSYMS;

    *algoArg->numSyms     = 1;
    algoArg->minimum[0]   = algoArg->inAry[0];
    algoArg->residual[0]  = 0;
    for(int i = 1; i < algoArg->blkSize; i++) {
        inBins = 0;
        for(int k = 0; k < *algoArg->numSyms; k++) {
            if(algoArg->inAry[i] == algoArg->minimum[k]) {
                inBins = 1;
                algoArg->residual[i] = k;
                break;
            }
        }
        if (inBins == 0) {
            if(*algoArg->numSyms == maxsyms) {
                allinBin = 0;
                break;
            }
            algoArg->minimum[*algoArg->numSyms] = algoArg->inAry[i];
            algoArg->residual[i] = *algoArg->numSyms;
           (*algoArg->numSyms)++;
        }
    }
    // Check if all symbols are in < MAXSYMS bins
    if(allinBin == 0) {
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"More than MAXSYMS symbols, returning");
        BTC_REPORT_INFO(mVerbosityLevel,7,debugStr)
        #endif
        *algoArg->bitln = 8;
        for(int i = 0; i < algoArg->blkSize; i++) {
            algoArg->residual[i] = algoArg->inAry[i];
        }
    } else {
        // Resudaul should already have the correct bin
        // index in them.
        btcmpctr_calc_bitln(algoArg->residual,algoArg->bitln,algoArg->blkSize,algoArg->minFixedBitLn);    
    }
}
// Bitmap Proc
// Bit map preproccing, find the top bin symbol and removes that symbol 
// while preserving a bitmap showing the location of the removed symbol.
void BitCompactor::btcmpctr_btMapprdct(
                            btcmpctr_algo_args_t* algoArg
                        )
{
    #ifdef __BTCMPCTR__EN_DBG__
    sprintf(debugStr,"In btMapprdct");
    BTC_REPORT_INFO(mVerbosityLevel,7,debugStr)
    #endif
    int maxIdx     = 0;
    int maxSymFreq = 0;
    int symbFreq[256];
    for(int i = 0; i < 256; i++) {
        symbFreq[i] = 0;
    }

    // Bin the Block and find the maximum frequency symbol.
    for(int i = 0; i < algoArg->blkSize; i++) {
        symbFreq[algoArg->inAry[i]]++;
    }
    // Find Max Index.
    for(int i = 0; i < 256; i++) {
        if(symbFreq[i] > maxSymFreq) {
            maxIdx = i;
            maxSymFreq = symbFreq[i];
        }
    }
    // Max Frequency Symbol is maxIdx.
    // Find this maxIdx in the inAry and create a bitmap.
    int cnt = 0;
    for(int i =0; i < algoArg->blkSize; i++) {
        if(algoArg->inAry[i] == maxIdx) {
            algoArg->bitmap[i] = 0;
        } else {
            algoArg->bitmap[i] = 1;
            algoArg->residual[cnt++] = algoArg->inAry[i];
        }
    }
    #ifdef __BTCMPCTR__EN_DBG__
    sprintf(debugStr,"In btMapprdct maxFreqSymb = %x, numBytes = %d",maxIdx,cnt);
    BTC_REPORT_INFO(mVerbosityLevel,7,debugStr)
    #endif
    *algoArg->minimum  = maxIdx;
    *algoArg->bitln    = 1;
    *algoArg->numBytes = cnt; 

}
// Dummy Proc
void BitCompactor::btcmpctr_dummyprdct(
                            btcmpctr_algo_args_t* algoArg
                           )
{
    // Choose results which show worse performance.
    *(algoArg->bitln) = 8;
    *algoArg->numBytes = algoArg->blkSize; 
    for(int i = 0; i < algoArg->blkSize; i++) {
        algoArg->residual[i] = algoArg->inAry[i];
    }
}
// Uncompression function
// Given an inBuf, Current inBuf pointer, and a state, extract specified number of bytes in outByte.
void BitCompactor::btcmpctr_xtrct_bits(
                         unsigned char* inBuf,
                                   int* inBufLen,
                         unsigned char* state,
                         unsigned char* outByte,
                         unsigned char  numBits
                        )
{
    unsigned char mask, thisBit;

    *outByte = 0;
    for(int i = 0; i< numBits; i++) {
        mask    = 1 << *state;
        thisBit = (inBuf[*inBufLen] & mask) >> *state; // Needed bit is in LSB of thisBit now.
           *outByte |= (thisBit << i);
        INCR_STATE_UC(state,inBufLen)
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"Extracting bits, state =%d, srcPtr = %d, Bit = %d",(int)*state,*inBufLen,thisBit);
        BTC_REPORT_INFO(mVerbosityLevel,6,debugStr)
        #endif
    }
}

// Extract header and give out, cmp, algo, bitln, eof, 8 or 16 bit to add.
unsigned char BitCompactor::btcmpctr_xtrct_hdr(
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
                                )
{
    unsigned char header = 0;
    // Assign default
    *eofr = 0;
    *cmp  = 0;
    *algo = BITC_ALG_NONE;
    *bitln = 8;
    *bytes_to_add = 0;
    *blkSize = 0;
    *numSyms = 0;
    *dual_encode = 0;
    unsigned char blk;
    // First extract 2bits
    btcmpctr_xtrct_bits(inAry,inAryPtr,&state,&header,2);
    if(header == EOFR) {
        // EOFR
        *eofr = 1;
        return state;
    } else if (header == CMPRSD) {
        // Compressed block
        // More header bits to extract.
        *cmp = 1;
        if(mixedBlkSize) {
            // First extract 2bits block Size
            blk = 0;
            btcmpctr_xtrct_bits(inAry,inAryPtr,&state,&blk,2);
            if(blk == 1) {
                *blkSize = BIGBLKSIZE;
            } else {
                *blkSize = BLKSIZE;
            }
        } else {
            *blkSize = BLKSIZE;
        }
    } else if (header == UNCMPRSD) {
        // Uncompressed block
        if(mixedBlkSize) {
            // First extract 2bits block Size
            blk = 0;
            btcmpctr_xtrct_bits(inAry,inAryPtr,&state,&blk,2);
            if(blk == 1) {
                *blkSize = BIGBLKSIZE;
            } else {
                *blkSize = BLKSIZE;
            }
        } else {
            *blkSize = BLKSIZE;
        }
        return state;
    } else {
        // Last block with bitstream length specified
        // Extract the next 6 bits, which holds the block Size in bytes.
        btcmpctr_xtrct_bits(inAry,inAryPtr,&state,(unsigned char *)blkSize,6);
        return state;
    }
    // Next extract Algo.
    btcmpctr_xtrct_bits(inAry,inAryPtr,&state,algo,3); // TODO Make Algo bits scalable.
    // Extract 3 bits of bitln
    btcmpctr_xtrct_bits(inAry,inAryPtr,&state,bitln,3); 
    // If dual encode is enabled extract 2 bits to decode dual_encode.
    if(dual_encode_en) {
        btcmpctr_xtrct_bits(inAry,inAryPtr,&state,dual_encode,2);
        #ifdef DL_INC_BL
        if(*dual_encode) {
            // Extract 10bits of total compressed bits.
            // Keeping it 10 to make it even.
            unsigned char dummy_cmprsd_bits;
            btcmpctr_xtrct_bits(inAry,inAryPtr,&state,&dummy_cmprsd_bits,8);
            btcmpctr_xtrct_bits(inAry,inAryPtr,&state,&dummy_cmprsd_bits,2);
        }
        #endif
    }
    // Next extract 1 bytes of data_to_add
    if( (*algo == ADDPROC) || (*algo == SIGNSHFTADDPROC) ) {
        btcmpctr_xtrct_bits(inAry,inAryPtr,&state,bytes_to_add,8); 
    }
    if( (*algo == BINEXPPROC) ) {
        // Number of symbols is not known during decompress.
        // Needs to be in the header. 5 additional bits after the header.
        int numSymsLen = (*blkSize == BIGBLKSIZE) ? NUMSYMSBL4K : NUMSYMSBL; 
        btcmpctr_xtrct_bits(inAry,inAryPtr,&state,numSyms,numSymsLen); 
        if ( (*numSyms == 0) && (numSymsLen == 4) ) { *numSyms = 16;}
        if ( (*numSyms == 0) && (numSymsLen == 6) ) { *numSyms = 64;}
        for(int i = 0; i< *numSyms; i++) {
            *(bytes_to_add+i) = 0;
            btcmpctr_xtrct_bits(inAry,inAryPtr,&state,(bytes_to_add+i),8); 
        }
    }
    if ( (*algo == BTEXPPROC) ) {
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"Extracted Algo is Bit Map Expansion");
        BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
        #endif
        // Extract high freq symbol 8 bits.
        btcmpctr_xtrct_bits(inAry,inAryPtr,&state,bytes_to_add,8);
        // Extract numBytes, 8 or 14 bits
        *numBytes = 0;
        unsigned char lB = 0;
        btcmpctr_xtrct_bits(inAry,inAryPtr,&state,&lB,8);
        *numBytes = lB;
        if((*blkSize == BIGBLKSIZE)) {
            // Extract 6 more bits.
            lB = 0;
            btcmpctr_xtrct_bits(inAry,inAryPtr,&state,&lB,6);
            (*numBytes) |= (lB << 8);
        }
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"NumBytes is %d",*numBytes);
        BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
        #endif
        // Extract the bitmap.
        for(int i = 0 ; i < *blkSize; i++) {
            btcmpctr_xtrct_bits(inAry,inAryPtr,&state,(bitmap + i),1);
        }
    }
    if(*dual_encode) {
        // Extract Bitmap
        for(int i = 0 ; i < *blkSize; i++) {
            btcmpctr_xtrct_bits(inAry,inAryPtr,&state,(bitmap + i),1);
        }
    }
    return state;
}

// Extract bytes with a bitmap
unsigned char BitCompactor::btcmpctr_xtrct_bytes_wbitmap(
                                           unsigned char* inAry,
                                                     int* inAryPtr,
                                           unsigned char  state,
                                           unsigned char  bitln,
                                           unsigned char* outBuf, // Assume OutBuf is BLKSIZE
                                                     int  blkSize,
                                           unsigned char* bitmap
                                         )
{
    int cnt = 0;
    unsigned char lbitln = bitln;
    while (cnt < blkSize) {
        if (bitln == 0) { lbitln = 8; }
        if(bitmap[cnt]) {
            #ifdef __BTCMPCTR__EN_DBG__
            sprintf(debugStr,"Extracting Byte %d",cnt);
            BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
            #endif
            btcmpctr_xtrct_bits(inAry,inAryPtr,&state,(outBuf+(cnt++)),8);
            #ifdef __BTCMPCTR__EN_DBG__
            sprintf(debugStr,"Extracted Byte %d is %d",(cnt-1),*(outBuf+cnt-1));
            BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
            #endif
        } else {
            #ifdef __BTCMPCTR__EN_DBG__
            sprintf(debugStr,"Extracting Byte %d",cnt);
            BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
            #endif
            btcmpctr_xtrct_bits(inAry,inAryPtr,&state,(outBuf+(cnt++)),lbitln);
        }
    }
    return state;

}

// Expand bits to byte, given an input buffer pointing to the exact bit, and the number of bits per symbol. produce an output byte array.
unsigned char BitCompactor::btcmpctr_xtrct_bytes(
                                    unsigned char* inAry,
                                              int* inAryPtr,
                                    unsigned char  state,
                                    unsigned char  bitln,
                                    unsigned char* outBuf, // Assume OutBuf is BLKSIZE
                                              int  blkSize,
                                    unsigned char  mode16
                                  )
{
    // This will work for 1byte symbol extract.
    int cnt = 0;
    unsigned char lbitln = bitln;
    while (cnt < blkSize) {
        if(mode16) {
            //construct 16bits output data 
            if (bitln == 0) { lbitln = 16; }
            if(lbitln > 8) {
                btcmpctr_xtrct_bits(inAry,inAryPtr,&state,(outBuf+(cnt++)),8);
                btcmpctr_xtrct_bits(inAry,inAryPtr,&state,(outBuf+(cnt++)),(lbitln-8));
            } else {
                btcmpctr_xtrct_bits(inAry,inAryPtr,&state,(outBuf+(cnt++)),lbitln);
            }
        } else {
            if (bitln == 0) { lbitln = 8; }
            btcmpctr_xtrct_bits(inAry,inAryPtr,&state,(outBuf+(cnt++)),lbitln);
        }
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"Extracting Bytes, cnt =%d",cnt);
        BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
        #endif
    }
    return state;
        
}
                                    
// tosigned - oppposite of tounsigned.
void BitCompactor::btcmpctr_tosigned(unsigned char* inAry,
                       unsigned char* outBuf
                      )
{
    // if LSB == 1, then it was a signed number.
    for(int i = 0; i < BLKSIZE; i++) {
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"Before Sign Conversion = %d",inAry[i]);
        BTC_REPORT_INFO(mVerbosityLevel,12,debugStr)
        #endif
        if((inAry[i]&1) == 1 ) {
            outBuf[i] = (unsigned char) (~inAry[i] >> 1) | 0x80;
        } else {
            outBuf[i] = (unsigned char) (inAry[i] >> 1);
        }
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"After Sign Conversion = %d",outBuf[i]);
        BTC_REPORT_INFO(mVerbosityLevel,12,debugStr)
        #endif
    }
}
// 16 and 8 bit adder into array.
void BitCompactor::btcmpctr_addByte(
                      unsigned char* data_to_add,
                      unsigned char  mode16,
                      unsigned char* outBuf
                     )
{
    int cnt = 0;
    while(cnt < BLKSIZE) {
        if(mode16) {
        } else {
            outBuf[cnt] += *data_to_add;
            cnt++;
        }
    }
}

void BitCompactor::btcmpctr_initAlgosAry(btcmpctr_compress_wrap_args_t* args)
{
    AlgoAry[MINPRDCT_IDX]       = &BitCompactor::btcmpctr_minprdct;
    AlgoAry[MINSPRDCT_IDX]      = &BitCompactor::btcmpctr_minSprdct;
    AlgoAry[MUPRDCT_IDX]        = &BitCompactor::btcmpctr_muprdct;
    AlgoAry[MEDPRDCT_IDX]       = &BitCompactor::btcmpctr_medprdct;
    AlgoAry[NOPRDCT_IDX]        = &BitCompactor::btcmpctr_noprdct;
    AlgoAry[NOSPRDCT_IDX]       = &BitCompactor::btcmpctr_noSprdct;
    if(args->proc_bin_en) {
        AlgoAry[BINCMPCT_IDX]       = &BitCompactor::btcmpctr_binCmpctprdct;
    } else {
        AlgoAry[BINCMPCT_IDX]       = &BitCompactor::btcmpctr_dummyprdct;
    }
    if(args->proc_btmap_en) {
        AlgoAry[BTMAP_IDX]       = &BitCompactor::btcmpctr_btMapprdct;
    } else {
        AlgoAry[BTMAP_IDX]       = &BitCompactor::btcmpctr_dummyprdct;
    }
    //4K Algo
    if(args->proc_btmap_en) {
        AlgoAry4K[BTMAP4K_IDX]       = &BitCompactor::btcmpctr_btMapprdct;
    } else {
        AlgoAry4K[BTMAP4K_IDX]       = &BitCompactor::btcmpctr_dummyprdct;
    }
    if(args->proc_bin_en) {
        AlgoAry4K[BINCMPCT4K_IDX]       = &BitCompactor::btcmpctr_binCmpctprdct;
    } else {
        AlgoAry4K[BINCMPCT4K_IDX]       = &BitCompactor::btcmpctr_dummyprdct;
    }
    //Initialize the Header overhead.
    AlgoAryHeaderOverhead[MINPRDCT_IDX]       = 16 + (2*args->mixedBlkSize) + (2*args->dual_encode_en); // 8bit header + 8 bit byte_to_add (minimum)
    AlgoAryHeaderOverhead[MINSPRDCT_IDX]      = 16 + (2*args->mixedBlkSize) + (2*args->dual_encode_en); // 8bit header + 8 bit byte_to_add (minimum)
    AlgoAryHeaderOverhead[MUPRDCT_IDX]        = 16 + (2*args->mixedBlkSize) + (2*args->dual_encode_en); // 8bit header + 8 bit byte_to_add (mu)
    AlgoAryHeaderOverhead[MEDPRDCT_IDX]       = 16 + (2*args->mixedBlkSize) + (2*args->dual_encode_en); // 8bit header + 8 bit byte_to_add (mu)
    AlgoAryHeaderOverhead[NOPRDCT_IDX]        = 8 + (2*args->mixedBlkSize) + (2*args->dual_encode_en);  // 8 bit header
    AlgoAryHeaderOverhead[NOSPRDCT_IDX]       = 8 + (2*args->mixedBlkSize) + (2*args->dual_encode_en);  // 8 bit header
    AlgoAryHeaderOverhead[BINCMPCT_IDX]       = 12 + (2*args->mixedBlkSize) + (2*args->dual_encode_en);  // 12 bit header. There is additional overhead based on the number of symbols which is dynamic
    AlgoAryHeaderOverhead[BTMAP_IDX]          = 8+8+8+64 + (2*args->mixedBlkSize) + (2*args->dual_encode_en);  // 8 Header, 8 topBinByte,8 ByteLength,64 Bitmap
    AlgoAryHeaderOverhead4K[BINCMPCT4K_IDX]   = 14 + (2*args->mixedBlkSize);  // 12 bit header. There is additional overhead based on the number of symbols which is dynamic
    AlgoAryHeaderOverhead4K[BTMAP4K_IDX]      = 8+8+14+4096 + (2*args->mixedBlkSize);  // 8 Header, 8 topBinByte,14 ByteLength,4096 Bitmap

}

unsigned char BitCompactor::btcmpctr_getAlgofrmIdx(int idx)
{
    if( (idx == NOPRDCT_IDX) ) {
       return NOPROC;
    } else if ( (idx == MINPRDCT_IDX) ) {
       return ADDPROC;
    } else if ( (idx == MINSPRDCT_IDX) || (idx == MUPRDCT_IDX) || (idx == MEDPRDCT_IDX) ) {
       return SIGNSHFTADDPROC;
    } else if ( (idx == NOSPRDCT_IDX) ) {
       return SIGNSHFTPROC;
    } else if ( (idx == BINCMPCT_IDX) ) {
       return BINEXPPROC;
    } else if ( (idx == BTMAP_IDX) ) {
       return BTEXPPROC;
    } else {
       return BITC_ALG_NONE;
    }  
}

unsigned char BitCompactor::btcmpctr_get4KAlgofrmIdx(int idx)
{
    if ( (idx == BINCMPCT4K_IDX) ) {
       return BINEXPPROC;
    } else if ( (idx == BTMAP4K_IDX) ) {
       return BTEXPPROC;
    } else {
       return BITC_ALG_NONE;
    }  
}

BitCompactor::btcmpctr_algo_choice_t BitCompactor::btcmpctr_ChooseAlgo64B(btcmpctr_algo_args_t* algoArg,
                                                               int mixedBlkSize,
                                                               int dual_encode_en
                                            )
{
    btcmpctr_algo_choice_t algoChoice;
    int minSize, minSizeDual;
    int chosenAlgo, chosenAlgoDual(0);
    int workingBlkSize = (algoArg->blkSize);
    int cmprsdSize, cmprsdSizeDual(0), dualCpSize;
    unsigned char dualBitln;
    unsigned char bitmap[BLKSIZE];
   
    minSize        = (workingBlkSize*8) + (mixedBlkSize ? 4 : 2);
    minSizeDual    = (workingBlkSize*8) + (mixedBlkSize ? 4 : 2);
    chosenAlgo     = BITC_ALG_NONE;
    chosenAlgoDual = BITC_ALG_NONE;
    *(algoArg->bitln)       = 8;
    // Run Through the 64B Algo's
    for(int i = 0; i< NUMALGO; i++) {
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"Calling Algo %d",i);
        BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
        #endif
        (this->*AlgoAry[i])(algoArg);
        // If BTMAP_IDX then multiple is numBytes
        int numBytes = workingBlkSize;
        if(i == BTMAP_IDX) {
            numBytes = *(algoArg->numBytes);
        }
        cmprsdSize = AlgoAryHeaderOverhead[i] + (numBytes * (*(algoArg->bitln))) ;
        if((dual_encode_en) & (i != BTMAP4K_IDX) ) {
            btcmpctr_calc_dual_bitln(algoArg->residual,&dualBitln,workingBlkSize,(unsigned char*)bitmap,&dualCpSize);
            #ifdef DL_INC_BL
            cmprsdSizeDual = AlgoAryHeaderOverhead[i] + dualCpSize + 64 + 10;
            #else
            cmprsdSizeDual = AlgoAryHeaderOverhead[i] + dualCpSize + 64 ;
            #endif
        }
        if(i == BINCMPCT_IDX) {
            cmprsdSize += ((*(algoArg->numSyms))*8);
            if(dual_encode_en) {
                cmprsdSizeDual += ((*(algoArg->numSyms))*8);
            }
        }
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"Compressed Size in bits is  %d",cmprsdSize);
        BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
        #endif
        if(cmprsdSize < minSize) {
            minSize = cmprsdSize;
            chosenAlgo = i;
        }
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"Dual Compressed Size in bits is  %d",cmprsdSizeDual);
        BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
        #endif
        if((cmprsdSizeDual < minSizeDual) && dual_encode_en && (i != BTMAP4K_IDX)) {
            minSizeDual    = cmprsdSizeDual;
            chosenAlgoDual = i;
        }
    }
    #ifdef __BTCMPCTR__EN_DBG__
    sprintf(debugStr,"Chosen Algo is %d,",chosenAlgo);    
    BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
    #endif
    if((minSizeDual < minSize) && dual_encode_en) {
        // Choose Dual Mode encoding
        algoChoice.dual_encode = 1;
    } else {
        algoChoice.dual_encode = 0;
    }
    #ifdef __BTCMPCTR__EN_DBG__
    sprintf(debugStr,"Dual Encode Mode is  %d,",algoChoice.dual_encode);    
    BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
    #endif
    algoChoice.none       = algoChoice.dual_encode ? (chosenAlgoDual == BITC_ALG_NONE) : (chosenAlgo == BITC_ALG_NONE); 
    algoChoice.algoHeader = btcmpctr_getAlgofrmIdx(algoChoice.dual_encode ? chosenAlgoDual : chosenAlgo);
    algoChoice.chosenAlgo = algoChoice.none ? &BitCompactor::btcmpctr_dummyprdct : (AlgoAry[algoChoice.dual_encode ? chosenAlgoDual : chosenAlgo]);
    algoChoice.cmprsdSize = algoChoice.dual_encode ? minSizeDual : minSize;
    algoChoice.algoType   = 0;
    algoChoice.workingBlkSize = workingBlkSize;
    return algoChoice;

}

BitCompactor::btcmpctr_algo_choice_t BitCompactor::btcmpctr_ChooseAlgo4K(btcmpctr_algo_args_t* algoArg,
                                                              int mixedBlkSize
                                           )
{
    btcmpctr_algo_choice_t algoChoice = {};
    int minSize;
    int chosenAlgo;
    int workingBlkSize = (algoArg->blkSize);
    int cmprsdSize;

    minSize     = (workingBlkSize*8) + (mixedBlkSize ? 4 : 2);
    chosenAlgo  = BITC_ALG_NONE;
    *(algoArg->bitln)       = 8;
    // Run Through the 64B Algo's
    for(int i = 0; i< NUM4KALGO; i++) {
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"Calling Algo %d",i);
        BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
        #endif
        (this->*AlgoAry4K[i])(algoArg);
        int numBytes = workingBlkSize;
        if(i == BTMAP4K_IDX) {
            numBytes = *(algoArg->numBytes);
        }
        cmprsdSize = AlgoAryHeaderOverhead4K[i] + (numBytes * (*(algoArg->bitln)));
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"Compressed Size in bits is  %d",cmprsdSize);
        BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
        #endif
        if(i == BINCMPCT4K_IDX) {
            cmprsdSize += ((*(algoArg->numSyms))*8);
        }
        if(cmprsdSize < minSize) {
            minSize = cmprsdSize;
            chosenAlgo = i;
        }
    }
    #ifdef __BTCMPCTR__EN_DBG__
    sprintf(debugStr,"Chosen Algo is %d,",chosenAlgo);    
    BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
    #endif
    algoChoice.algoHeader = btcmpctr_get4KAlgofrmIdx(chosenAlgo);

    // @ajgorman: KW fix
    if (chosenAlgo == BITC_ALG_NONE) {
        algoChoice.none = 0x1;
        algoChoice.chosenAlgo = nullptr; // should never be used if ‘none’ is asserted
    } else {
        algoChoice.none = 0x0;
        algoChoice.chosenAlgo = AlgoAry4K[chosenAlgo]; // chosenAlgo should be < NUM4KALGO
    }

    algoChoice.cmprsdSize = minSize;
    algoChoice.algoType   = 1;
    algoChoice.workingBlkSize = workingBlkSize;

    return algoChoice;
}

//Compress Wrap
//This is a SWIG/numpy integration friendly interface for the compression function.
//
int BitCompactor::CompressArray(unsigned char* src,
                                    int   srcLen,
                           unsigned char* dst,
                                    int   dstLen, // unused
                           btcmpctr_compress_wrap_args_t* args
                          )
{
        int resultLen;
        int status;
        status = CompressWrap(src,&srcLen,dst,&resultLen,args);
        if(status) {
            return resultLen;
        } else {
            return 0;
        }
}

void BitCompactor::reset()
{
    // does nothing
}

int BitCompactor::CompressWrap(unsigned char* src,
                           int*           srcLen,
                           unsigned char* dst,
                           int*           dstLen,// dstLen holds the size of the output buffer.
                           btcmpctr_compress_wrap_args_t* args
                           )
{
    mVerbosityLevel = args->verbosity;
    btcmpctr_initAlgosAry(args);    
    // Keep a copy of the destination buffer size allocated
    // to be used in checks. The final destination size will
    // be returned in this pointer. 
    //
    unsigned char bitln;
    unsigned char minimum[MAXSYMS4K];
    int cmprsdSize, workingBlkSize,dualCpSize;
    int srcCnt = 0;
    int chosenAlgo;
    int state = 0;
    int blkCnt = 0;
    int numSyms = 0;
    unsigned char residual[BIGBLKSIZE];
    unsigned char bitmap[BIGBLKSIZE];
    int numBytes;
    unsigned int accum = 0;
    btcmpctr_algo_args_t algoArg;
    btcmpctr_algo_choice_t chosenAlgos[BIGBLKSIZE/BLKSIZE];
    btcmpctr_algo_choice_t chosenAlgos4K = {};
    int smCntr = 0;
    int numSmBlks = 0;
    int bigBlkSize;
     

    *dstLen = 0;
    //dst[*dstLen] = 0;
    // Choose the correct Algo to run.
    #ifdef __BTCMPCTR__EN_DBG__
    sprintf(debugStr,"Source Length = %d",*srcLen);
    BTC_REPORT_INFO(mVerbosityLevel,1,debugStr)
    #endif
    algoArg.minimum  = (unsigned char *)&minimum;
    algoArg.bitln    = &bitln;
    algoArg.numSyms  = &numSyms;
    algoArg.bitmap   = (unsigned char *)&bitmap;
    algoArg.numBytes = &numBytes;
    algoArg.residual = (unsigned char *)&residual;
    algoArg.minFixedBitLn = &(args->minFixedBitLn); 

    while (srcCnt < *srcLen) {   
    /* While(srcCnt < srcLen)
     *
     *  If (srcCnt + 4K < srcLen):
     *      // Can run 4K Algo
     *      (Chosen4KAlgo,CmprsdSize4K) = Run_Choose4kAlgo();
     *      (<Chosen64BAlgoAry>,TotCmprsdSize64B) = Run_Choose64BAlgo();
     *      If (CmprsdSize4K <= TotCmprsdSize64B):
     *          Chosen = 4K
     *      Else
     *          Chosen = 64B
     *  Else:
     *      // Last <4K block will always get chopped up by 64B blocks
     *      (<Chosen64BAlgoAry>,TotCmprsdSize) = Run_Choose64BAlgo();
     *      Chosen = 64B
     *  
     *  If (Chosen = 4K):
     *      Run Chosen4KAlgo();
     *      Insert Header();
     *      Insert Data();
     *  Else:
     *      Foreach Algo in <Chosen64BAlgoAry>:
     *          Run Algo(); // To get residue
     *          Insert Header();
     *          Insert Data();
     *       
     *      End
     */
        if( (srcCnt + BIGBLKSIZE) > *srcLen) {
            bigBlkSize = (*srcLen - srcCnt);
        } else {
            bigBlkSize = BIGBLKSIZE;
        }
        // Choose 4K Algorithm
        if((bigBlkSize == BIGBLKSIZE) && args->mixedBlkSize) {
           algoArg.inAry = (src + srcCnt);
           algoArg.blkSize = bigBlkSize;
           chosenAlgos4K = btcmpctr_ChooseAlgo4K(&algoArg,args->mixedBlkSize);
        }

        // Work on the 64B block size.
        cmprsdSize = 0;
        smCntr = 0;
        numSmBlks = 0;
        while(smCntr < bigBlkSize) {

            if( (smCntr + BLKSIZE) > bigBlkSize) {
                workingBlkSize = (bigBlkSize - smCntr);
            } else {
                workingBlkSize = BLKSIZE;
            }
            algoArg.inAry = (src + srcCnt + smCntr);
            algoArg.blkSize = workingBlkSize;
            // Call the Algo Choice.
            #ifdef __BTCMPCTR__EN_DBG__
            sprintf(debugStr,"Trying to find best algo for this blockCnt = %d",blkCnt);
            BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
            #endif
            chosenAlgos[numSmBlks].workingBlkSize = workingBlkSize;
            if (workingBlkSize == BLKSIZE) {
                chosenAlgos[numSmBlks] = btcmpctr_ChooseAlgo64B(&algoArg,args->mixedBlkSize,args->dual_encode_en);
                #ifdef __BTCMPCTR__EN_DBG__
                sprintf(debugStr," blkCnt = %d",blkCnt);    
                BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
                #endif
            }
            cmprsdSize += chosenAlgos[numSmBlks].cmprsdSize;
            numSmBlks++;
            smCntr += workingBlkSize;
            #ifdef __BTCMPCTR__EN_DBG__
            sprintf(debugStr,"bigBlkSize = %d, smCntr = %d, numSmBlks = %d",bigBlkSize,smCntr,numSmBlks);    
            BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
            #endif
        }
        // IF    Compressed size of 4K block < totoal comporessed size of 64B blocks, and if this is not the last <4K block. and if mixed Block size is enabled.
        if( (((chosenAlgos4K.cmprsdSize <= cmprsdSize) && (bigBlkSize == BIGBLKSIZE)) || args->bypass_en) && args->mixedBlkSize) {
            //---------------------------------------------------------------------------------
            // Chosen 4K Blocks 
            //---------------------------------------------------------------------------------
            #ifdef __BTCMPCTR__EN_DBG__
            sprintf(debugStr,"Running chosen algo");    
            BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
            #endif
            if ( (chosenAlgos4K.workingBlkSize < BIGBLKSIZE) || args->bypass_en) {
                // Force an Algo for the last block.
                chosenAlgos4K.none = 1;
            }
            if(chosenAlgos4K.none != 1) {
                algoArg.inAry = (src + srcCnt);
                algoArg.blkSize = chosenAlgos4K.workingBlkSize;
                (this->*chosenAlgos4K.chosenAlgo)(&algoArg);
                chosenAlgo = chosenAlgos4K.algoHeader;
             } else {
                bitln = 8;
                for(int i = 0; i < chosenAlgos4K.workingBlkSize; i++) {
                    residual[i] = *(src + srcCnt + i);
                }
                chosenAlgo = BITC_ALG_NONE;
            }
            // Insert Header
            #ifdef __BTCMPCTR__EN_DBG__
            sprintf(debugStr,"Inserting Header, chosen Algo in 4K is %d",chosenAlgo);    
            BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
            #endif
            state = btcmpctr_insrt_hdr(chosenAlgo, bitln, dstLen, dst, state, &accum,0,chosenAlgos4K.workingBlkSize,args->mixedBlkSize,0);
            // Insert Post Header bytes
            // Insert the symbols in case of BINEXPPROC.
            if ( (chosenAlgo == BINEXPPROC) ) {
               // Insert 6 bits of numSyms
               int numSymsToInsrt = (numSyms == 64) && (NUMSYMSBL4K == 6) ? 0 : numSyms;
               state = btcmpctr_insrt_byte(numSymsToInsrt,NUMSYMSBL4K,dstLen,dst,state,&accum,0);
               // Insert the number of symbols.
               for(int i = 0; i < numSyms; i++) {
                   #ifdef __BTCMPCTR__EN_DBG__
                   sprintf(debugStr,"Inserting Binned Header %d, %d", i, minimum[i]);    
                   BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
                   #endif
                   state = btcmpctr_insrt_byte(minimum[i],8,dstLen,dst,state,&accum,0);
               }
            }
            if ( (chosenAlgo == BTEXPPROC) ) {
                // Insert 8bits of max freq symbol.
                state = btcmpctr_insrt_byte(minimum[0],8,dstLen,dst,state,&accum,0);
                // insert 14bits of byte length (14 to keep an even number of header bits)
                state = btcmpctr_insrt_byte(numBytes,8,dstLen,dst,state,&accum,0);
                state = btcmpctr_insrt_byte((numBytes>>8),6,dstLen,dst,state,&accum,0);
                // Insert 4096 bits of bitmap
                for(int i = 0; i < chosenAlgos4K.workingBlkSize; i++) {
                   state = btcmpctr_insrt_byte(bitmap[i],1,dstLen,dst,state,&accum,0);
                }
            }
                    
            // Insert data.
            #ifdef __BTCMPCTR__EN_DBG__
            sprintf(debugStr,"Inserting Data");    
            BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
            #endif
            int dataSize = (chosenAlgo == BTEXPPROC) ? numBytes : chosenAlgos4K.workingBlkSize;
            for(int i = 0; i< dataSize; i++) {
                #ifdef __BTCMPCTR__EN_DBG__
                sprintf(debugStr,"Inserting Data cnt =%d Data = %d, Src Data = %x",i,residual[i],*(src + srcCnt + i));    
                BTC_REPORT_INFO(mVerbosityLevel,6,debugStr)
                #endif
                state = btcmpctr_insrt_byte(residual[i],bitln,dstLen,dst,state,&accum,0);
            }
        
        } else {
            //---------------------------------------------------------------------------------
            // Chosen 64B Blocks 
            //---------------------------------------------------------------------------------
            // ChosenAlgo must be re-run with to correctly generate the compressed 
            smCntr = 0;
            for(int smBlk = 0; smBlk < numSmBlks ; smBlk++) { 
                #ifdef __BTCMPCTR__EN_DBG__
                sprintf(debugStr,"Running chosen algo for small block count = %d with blockSize = %d",smBlk,chosenAlgos[smBlk].workingBlkSize);    
                BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
                #endif
                if ( (chosenAlgos[smBlk].workingBlkSize < BLKSIZE) || args->bypass_en) {
                    // Force an Algo for the last block.
                    chosenAlgos[smBlk].none = 1;
                    chosenAlgos[smBlk].dual_encode = 0;
                }
                if(chosenAlgos[smBlk].none != 1) {
                    algoArg.inAry = (src + srcCnt + smCntr);
                    algoArg.blkSize = chosenAlgos[smBlk].workingBlkSize;
                    (this->*chosenAlgos[smBlk].chosenAlgo)(&algoArg);
                    chosenAlgo = chosenAlgos[smBlk].algoHeader;
                    if(chosenAlgos[smBlk].dual_encode) {
                        // Re-run dualEncode calculate length
                        btcmpctr_calc_dual_bitln((unsigned char*)&residual,&bitln,chosenAlgos[smBlk].workingBlkSize,(unsigned char*)bitmap,&dualCpSize);
                    }
                 } else {
                    bitln = 8;
                    for(int i = 0; i < chosenAlgos[smBlk].workingBlkSize; i++) {
                        residual[i] = *(src + srcCnt + smCntr + i);
                    }
                    chosenAlgo = BITC_ALG_NONE;
                }
                // Insert Header
                #ifdef __BTCMPCTR__EN_DBG__
                sprintf(debugStr,"Inserting Header");    
                BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
                #endif
                state = btcmpctr_insrt_hdr(chosenAlgo, bitln, dstLen, dst, state, &accum,0,chosenAlgos[smBlk].workingBlkSize,args->mixedBlkSize,0);
                // Insert Post Header bytes
                if(chosenAlgo != BITC_ALG_NONE) {
                    #ifdef __BTCMPCTR__EN_DBG__
                    sprintf(debugStr,"Inserting Header");    
                    BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
                    #endif
                    if(chosenAlgos[smBlk].dual_encode) {
                        state = btcmpctr_insrt_byte(1,2,dstLen,dst,state,&accum,0);
                        #ifdef DL_INC_BL
                        // Insert the bit length
                        //calculae the bitlength
                        uint16_t cpBitLen = 0;
                        for(int i = 0; i < chosenAlgos[smBlk].workingBlkSize; i++) {
                            cpBitLen = bitmap[i] ? cpBitLen+8 : cpBitLen+bitln;
                        }
                        // Insert 10 bits.
                        state = btcmpctr_insrt_byte(cpBitLen,8,dstLen,dst,state,&accum,0);
                        cpBitLen >>= 8;
                        state = btcmpctr_insrt_byte(cpBitLen,2,dstLen,dst,state,&accum,0);
                        #endif
                    } else {
                        state = btcmpctr_insrt_byte(0,2,dstLen,dst,state,&accum,0);
                    }
                }
                if( (chosenAlgo == ADDPROC) || (chosenAlgo == SIGNSHFTADDPROC) ) {
                    // Insert one more byte.
                    #ifdef __BTCMPCTR__EN_DBG__
                    sprintf(debugStr,"Inserting Header plus 1 more byte");    
                    BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
                    #endif
                    state = btcmpctr_insrt_byte(minimum[0],8,dstLen,dst,state,&accum,0);
                }
                // Insert the symbols in case of BINEXPPROC.
                if ( (chosenAlgo == BINEXPPROC) ) {
                   // Insert 5 bits of numSyms
                   int numSymsToInsrt = (numSyms == 16) && (NUMSYMSBL == 4) ? 0 : numSyms;
                   state = btcmpctr_insrt_byte(numSymsToInsrt,NUMSYMSBL,dstLen,dst,state,&accum,0);
                   // Insert the number of symbols.
                   for(int i = 0; i < numSyms; i++) {
                       #ifdef __BTCMPCTR__EN_DBG__
                       sprintf(debugStr,"Inserting Binned Header %d, %d", i, minimum[i]);    
                       BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
                       #endif
                       state = btcmpctr_insrt_byte(minimum[i],8,dstLen,dst,state,&accum,0);
                   }
                }
                if ( (chosenAlgo == BTEXPPROC) ) {
                    // Insert 8bits of max freq symbol.
                    state = btcmpctr_insrt_byte(minimum[0],8,dstLen,dst,state,&accum,0);
                    // insert 8bits of byte length (8 to keep an even number of header bits)
                    state = btcmpctr_insrt_byte(numBytes,8,dstLen,dst,state,&accum,0);
                    // Insert 64 bits of bitmap
                    for(int i = 0; i < chosenAlgos[smBlk].workingBlkSize; i++) {
                       state = btcmpctr_insrt_byte(bitmap[i],1,dstLen,dst,state,&accum,0);
                    }
                }
                // Insert the Bitmap for dual encode
                if(args->dual_encode_en) {
                    if(chosenAlgos[smBlk].dual_encode) {
                        for(int i = 0; i < chosenAlgos[smBlk].workingBlkSize; i++) {
                           state = btcmpctr_insrt_byte(bitmap[i],1,dstLen,dst,state,&accum,0);
                        }
                    }
                }
                        
                // Insert data.
                #ifdef __BTCMPCTR__EN_DBG__
                sprintf(debugStr,"Inserting Data");    
                BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
                #endif
                int dataSize = (chosenAlgo == BTEXPPROC) ? numBytes : chosenAlgos[smBlk].workingBlkSize;
                for(int i = 0; i< dataSize; i++) {
                    #ifdef __BTCMPCTR__EN_DBG__
                    sprintf(debugStr,"Inserting Data cnt =%d Data = %d, Src Data = %x",i,residual[i],*(src + srcCnt + smCntr + i));    
                    BTC_REPORT_INFO(mVerbosityLevel,6,debugStr)
                    #endif
                    if( (args->dual_encode_en) && (chosenAlgos[smBlk].dual_encode) && bitmap[i] ) {
                        state = btcmpctr_insrt_byte(residual[i],8,dstLen,dst,state,&accum,0);
                    } else {
                        state = btcmpctr_insrt_byte(residual[i],bitln,dstLen,dst,state,&accum,0);
                    }
                }
                //
                #ifdef __BTCMPCTR__EN_DBG__
                sprintf(debugStr,"Destination length is %d",*dstLen);    
                BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
                #endif
                smCntr += chosenAlgos[smBlk].workingBlkSize;
            }
        }
        srcCnt += bigBlkSize;
        blkCnt++;
    }
    // Insert end of stream bits.
    #ifdef __BTCMPCTR__EN_DBG__
    sprintf(debugStr,"Inserting End of Stream");    
    BTC_REPORT_INFO(mVerbosityLevel,6,debugStr)
    #endif
    state = btcmpctr_insrt_hdr(0,0,dstLen,dst,state,&accum,1,0,0,args->align);    
    // Check if state is non-zero, if so,  need to increment dstLen.
    if(state != 0) {
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"ERROR: state != 0 at the end of compression");    
        BTC_REPORT_INFO(mVerbosityLevel,0,debugStr)
        #endif
    }
    // All Done!!
    return 1;
}


// DecompressWrap
//This is a SWIG/numpy integration friendly interface for the decompression function.
//
int BitCompactor::DecompressArray(unsigned char* src,
                                      int   srcLen,
                             unsigned char* dst,
                                      int   dstLen, // unused
                             btcmpctr_compress_wrap_args_t* args
                            )
{
        int resultLen;
        int status;
        status = DecompressWrap(src,&srcLen,dst,&resultLen,args);
        if(status) {
            return resultLen;
        } else {
            return 0;
        }
}

int BitCompactor::DecompressWrap(  unsigned char* src,
                                        int*  srcLen,
                               unsigned char* dst,
                                         int* dstLen,
                               btcmpctr_compress_wrap_args_t* args
                            )
{
    unsigned char state = 0;
    unsigned char cmp, eofr,algo,bitln, numSyms;
    int blkSize;
    unsigned char bytes_to_add[MAXSYMS4K];
    unsigned char bitmap[BIGBLKSIZE];
    unsigned char bitmapBytes[BIGBLKSIZE];
    int numBytes = 0;
    int blkCnt = 0;
    int srcLenTrk = 0;
    *dstLen = 0;
    mVerbosityLevel = args->verbosity;
    int mixedBlkSize = args->mixedBlkSize;
    int dual_encode_en = args->dual_encode_en;
    unsigned char dual_encode;

    while ( ( srcLenTrk < *srcLen ) ) {
        // Extract Header
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"Extracting Header for blockCnt = %d",blkCnt);
        BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
        #endif
        state = btcmpctr_xtrct_hdr(src,&srcLenTrk,state,&cmp,&eofr,&algo,&bitln,&blkSize,(unsigned char*)bytes_to_add,&numSyms,&numBytes,(unsigned char *)bitmap,mixedBlkSize,dual_encode_en,&dual_encode);
        #ifdef __BTCMPCTR__EN_DBG__
        if ( srcLenTrk > ( (*srcLen) - 1 ) && ( (*srcLen) > 0 ) )
        {
            // no more compressed source data to process; srcLenTrk has reached the end of the array
            sprintf(debugStr,"src = 0x--, srcLen = %d, srcLenTrk = %d, EOFR = %d, Compressed = %d, Algo = %d, Bit Length = %d, Block Size = %d [reached end of source data]", *srcLen,srcLenTrk,eofr,cmp,algo,bitln,blkSize);
        }
        else
        {
            // still more compressed source data left to process
            sprintf(debugStr,"src = 0x%02x, srcLen = %d, srcLenTrk = %d, EOFR = %d, Compressed = %d, Algo = %d, Bit Length = %d, Block Size = %d",src[srcLenTrk],*srcLen,srcLenTrk,eofr,cmp,algo,bitln,blkSize);
        }
        BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
        #endif
        
        // https://hsdes.intel.com/appstore/article/#/18011468189
        // if we're at EOFR, OR if srcLenTrk has reached the end of
        // the compressed source data, stop processing at this point
        // (the 'while' loop condition should then cause us to finish)
        if( ( srcLenTrk > ( (*srcLen) - 1 ) ) || eofr )
        {
            continue;
        }
        //
        if(cmp) {
            //Compressed block. Process further based on Algo
            #ifdef __BTCMPCTR__EN_DBG__
            sprintf(debugStr,"Compressed Block, Extracting data");
            BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
            #endif
            if ( algo == BTEXPPROC ) {
                // bitln set to 0, since we always extract 8bits for the non high freq calculations.
                state = btcmpctr_xtrct_bytes(src,&srcLenTrk,state,0,bitmapBytes,numBytes,0);
            } else {
                if(dual_encode) {
                    state = btcmpctr_xtrct_bytes_wbitmap(src,&srcLenTrk,state,bitln,(dst+*dstLen),blkSize,(unsigned char *)bitmap);
                } else {
                    state = btcmpctr_xtrct_bytes(src,&srcLenTrk,state,bitln,(dst+*dstLen),blkSize,0);
                }
            }
            if ( (algo == SIGNSHFTADDPROC) || (algo == SIGNSHFTPROC) ) {
                // Convert to signed
                #ifdef __BTCMPCTR__EN_DBG__
                sprintf(debugStr,"Converting to Signed form");
                BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
                #endif
                btcmpctr_tosigned((dst+*dstLen),(dst+*dstLen));
            }
            // Add any bytes.
            if ( (algo == ADDPROC) || (algo == SIGNSHFTADDPROC) ) {
                // add bytes_to_add
                #ifdef __BTCMPCTR__EN_DBG__
                sprintf(debugStr,"Adding Data");
                BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
                #endif
                btcmpctr_addByte((unsigned char*)bytes_to_add,0,(dst+*dstLen));
            }
            // Reconstruct bitmap by using lookup into bytes_to_add vector.
            if ( (algo == BINEXPPROC) ) {
                for(int i = 0; i < blkSize; i++) {
                    *(dst+*dstLen+i) = bytes_to_add[*(dst+*dstLen+i)];
                }
            }
            if ( (algo == BTEXPPROC) ) {
                // Extract numBytes from 
                int cnt = 0;
                for(int i = 0; i < blkSize; i ++) {
                    if(!bitmap[i]) {
                        *(dst+*dstLen+i) = bytes_to_add[0];
                    } else {
                        *(dst+*dstLen+i) = bitmapBytes[cnt++];
                    }
                }
                //assert cnt == numBytes;
            }
        } else {
            // Uncompressed block
            #ifdef __BTCMPCTR__EN_DBG__
            sprintf(debugStr,"Uncompressed Data block..");
            BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
            #endif
            state = btcmpctr_xtrct_bytes(src,&srcLenTrk,state,bitln,(dst+*dstLen),blkSize,0);
        }
        (*dstLen) += blkSize;
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debugStr,"Src length = %d",srcLenTrk);
        BTC_REPORT_INFO(mVerbosityLevel,5,debugStr)
        #endif
        blkCnt++;
    }
    return 1;


}

int BitCompactor::btcmpctr_cmprs_bound(int bufSize) 
{
    return ceil(((ceil(bufSize/BLKSIZE) * 4) + 2)/8) + bufSize + 1 + 64;
}

BitCompactor::elem_type BitCompactor::quick_select(elem_type arr[], unsigned int n)
{
    unsigned int low, high ;
    unsigned int median;
    unsigned int middle, ll, hh;

    low = 0 ; high = n-1 ; median = (low + high) / 2;
    for (;;) {
        if (high <= low) /* One element only */
            return arr[median] ;

        if (high == low + 1) {  /* Two elements only */
            if (arr[low] > arr[high])
                ELEM_SWAP(arr[low], arr[high]) ;
            return arr[median] ;
        }

    /* Find median of low, middle and high items; swap into position low */
    middle = (low + high) / 2;
    if (arr[middle] > arr[high])    ELEM_SWAP(arr[middle], arr[high]) ;
    if (arr[low] > arr[high])       ELEM_SWAP(arr[low], arr[high]) ;
    if (arr[middle] > arr[low])     ELEM_SWAP(arr[middle], arr[low]) ;

    /* Swap low item (now in position middle) into position (low+1) */
    ELEM_SWAP(arr[middle], arr[low+1]) ;

    /* Nibble from each end towards middle, swapping items when stuck */
    ll = low + 1;
    hh = high;
    for (;;) {
        do ll++; while (arr[low] > arr[ll]) ;
        do hh--; while (arr[hh]  > arr[low]) ;

        if (hh < ll)
        break;

        ELEM_SWAP(arr[ll], arr[hh]) ;
    }

    /* Swap middle item (in position low) back into correct position */
    ELEM_SWAP(arr[low], arr[hh]) ;

    /* Re-set active partition */
        if (hh <= median) 
            low = ll;
        if (hh >= median)
            high = hh - 1;
    }
}

unsigned int BitCompactor::getDecodeErrorCount()
{
    return 0; // TBD: implement error counting for decompress/compress
}

unsigned int BitCompactor::getEncodeErrorCount()
{
    return 0; // TBD: implement error counting for decompress/compress
}

#ifdef BTC_USE_DPI

// API call provided for UVM testbench
//
#ifdef BTC_USE_DPI_STANDALONE
// this function call avoids extern functions provided only in the SV/simulator link context
// (allows testing standalone)
extern int startBTC(int argc, char* argvArray[]) 
#else
extern int startBTC(int argc, const svOpenArrayHandle argvArray) 
#endif
{
    // Extract argv from argvArray
    char * argv[argc];
    char debug[1024];

    for (int i = 0; i < argc; i++)
    {
#ifdef BTC_USE_DPI_STANDALONE
        argv[i] = argvArray[i];
#else
        argv[i] = *(char **)svGetArrElemPtr(argvArray, i);
#endif
    }
    // Create an instance of the BitCompactor class (minus the codec wrapper)
    BitCompactor bitCompactor;
    int verbosityM;

    BitCompactor::btcmpctr_args_t args;
    btcmpctr_parse_args(argc, argv, &args);
    verbosityM = args.verbosity;
    BitCompactor::btcmpctr_compress_wrap_args_t cmprs_args;

    int fileSize;
    int outBufSize;
    FILE *inFile;
    FILE *outFile;
    int error;
    unsigned char* inBuf;
    unsigned char* outBuf;

    inFile = fopen(args.inFileName, "rb");
    if(NULL == inFile) {
        printf("unable to open file %s for reading", args.inFileName);
        return 0;
    }
    // Get File Size.
    fseek(inFile,0,SEEK_END);
    fileSize = ftell(inFile);    
    fseek(inFile,0,SEEK_SET);
    inBuf = (unsigned char*)malloc(fileSize);
    if (NULL == inBuf) {
        printf("Cannot allocate memory to fill inBuf");
        return 0;
    }
    #ifdef __BTCMPCTR__EN_DBG__
    sprintf(debug,"Reading inFile %s",args.inFileName);
    BTC_REPORT_INFO(verbosityM,1,debug)
    #endif
    fread(inBuf,fileSize,1,inFile);
    fclose(inFile);

    // Call compress wrap
    if(args.cmprs) {
        // Allocate outBuf based on Max compressed Size.
        outBufSize = bitCompactor.btcmpctr_cmprs_bound(fileSize);
        outBuf = (unsigned char*) malloc(outBufSize);
        if (NULL == outBuf) {
            printf("Cannot allocate memory to fill outBuf");
            return 0;
        }
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debug,"Compression mode, calling compress_wrap, received output buffer Len limit as %d",outBufSize);
        BTC_REPORT_INFO(verbosityM,1,debug)
        #endif
        cmprs_args.verbosity      = args.verbosity;
        cmprs_args.mixedBlkSize   = args.mixedBlkSize;
        cmprs_args.minFixedBitLn  = args.minFixedBitLn;
        cmprs_args.proc_bin_en    = args.proc_bin_en;
        cmprs_args.proc_btmap_en  = args.proc_btmap_en;
        cmprs_args.dual_encode_en = args.dual_encode_en;
        cmprs_args.bypass_en      = args.bypass_en;
        cmprs_args.align          = args.align;
        error = bitCompactor.CompressWrap(inBuf, &fileSize, outBuf, &outBufSize,&cmprs_args);
        if(error == 0) {
            BTC_REPORT_ERROR("Something went wrong during compress");
            return 0;
        }
        // Write outBuf to File.
        outFile = fopen(args.outFileName,"wb");
        if(NULL == outFile) {
            sprintf(debug,"unable to open file %s for reading", args.outFileName);
                        BTC_REPORT_ERROR(debug);
            return 0;
        }
        fwrite(outBuf,outBufSize,1,outFile);
        fclose(outFile);
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debug,"Source Size = %d",fileSize);
        BTC_REPORT_INFO(verbosityM,1,debug)
        #endif
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debug,"Destination Size = %d",outBufSize);
        BTC_REPORT_INFO(verbosityM,1,debug)
        #endif
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debug,"Compression Ratio = %f",(((double)outBufSize/(double)fileSize)*100));
        BTC_REPORT_INFO(verbosityM,1,debug)
        #endif
        if(args.ratio) {
            #ifdef __BTCMPCTR__EN_DBG__
            sprintf(debug,"%d,%d,%f",fileSize,outBufSize,(((double)outBufSize/(double)fileSize)*100));
            BTC_REPORT_INFO(verbosityM,0,debug)
            #endif
        }
    }

    // Call decompress wrap
    if (args.decmprs) {
        // Allocate outBuf based on Max compressed Size.
        outBufSize = fileSize*10; // Assume a compression efficiency of 10%, too much
        outBuf = (unsigned char*) malloc(outBufSize);
        if (NULL == outBuf) {
            BTC_REPORT_ERROR("Cannot allocate memory to fill outBuf");
            return 0;
        }
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debug,"Decompression mode, calling decompress_wrap");
        BTC_REPORT_INFO(verbosityM,1,debug)
        #endif
        cmprs_args.verbosity      = args.verbosity;
        cmprs_args.mixedBlkSize   = args.mixedBlkSize;
        cmprs_args.dual_encode_en = args.dual_encode_en;
        cmprs_args.bypass_en      = args.bypass_en;
        error = bitCompactor.DecompressWrap(inBuf,&fileSize,outBuf,&outBufSize,&cmprs_args);
        if(error == 0) {
            BTC_REPORT_ERROR("Something went wrong during compress");
            return 0;
        }
        // Write outBuf to File.
        outFile = fopen(args.outFileName,"wb");
        if(NULL == outFile) {
                        sprintf(debug,"unable to open file %s for reading", args.outFileName);
                        BTC_REPORT_ERROR(debug);
            return 0;
        }
        fwrite(outBuf,outBufSize,1,outFile);
        fclose(outFile);
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debug,"Source Size = %d",fileSize);
        BTC_REPORT_INFO(verbosityM,1,debug)
        #endif
        #ifdef __BTCMPCTR__EN_DBG__
        sprintf(debug,"Destination Size = %d",outBufSize);
        BTC_REPORT_INFO(verbosityM,1,debug)
        #endif
    }

    return 1;
}

// utility function to decode commandline arguments supplied by DPI
void btcmpctr_parse_args(int argc, char** argv, BitCompactor::btcmpctr_args_t* args)
{
    if(argc <= 1) {
        BTC_REPORT_ERROR("No Arguments Specified!!!");
        //exit;
    }
    argc--;
    argv++;
    char linebuf[5];
    args->ratio = 0;
    args->mixedBlkSize = 0;
    args->proc_bin_en = 0;
    args->proc_btmap_en = 0;
    args->dual_encode_en = 1;
    args->bypass_en = 0;
    args->verbosity = 0;
    args->align = DEFAULT_ALIGN;
    args->minFixedBitLn = 3; // vpu2p7 default min fixed bit length (deliberately limiting max achievable compression ratio)

    while(argc > 0) {
        if(strcmp(*argv, "-c") == 0) {
            argc--;
            argv++;
            args->cmprs = 1;
            args->decmprs = 0;
            // Get outFile Name to compress
            strcpy(args->outFileName, *argv);
            argc--;
            argv++;
        } else if(strcmp(*argv, "-d") == 0) {
            argc--;
            argv++;
            args->cmprs = 0;
            args->decmprs = 1;
            // Get outFile Name to compress
            strcpy(args->outFileName, *argv);
            argc--;
            argv++;
        } else if(strcmp(*argv, "-f") == 0) {
            argc--;
            argv++;
            strcpy(args->inFileName, *argv);
            argc--;
            argv++;
        } else if(strcmp(*argv, "-v") == 0) {
            argc--;
            argv++;
            strcpy(linebuf,*argv);
            args->verbosity = atoi(linebuf);
            argc--;
            argv++;
        } else if(strcmp(*argv, "-ratio") == 0) {
            argc--;
            argv++;
            args->ratio = 1;
        } else if(strcmp(*argv, "-mixed_blkSize_en") == 0) {
            argc--;
            argv++;
            args->mixedBlkSize = 1;
        } else if(strcmp(*argv, "-proc_bin_en") == 0) {
            argc--;
            argv++;
            args->proc_bin_en=1;
        } else if(strcmp(*argv, "-proc_btmap_en") == 0) {
            argc--;
            argv++;
            args->proc_btmap_en=1;
        } else if(strcmp(*argv, "-dual_encode_dis") == 0) {
            argc--;
            argv++;
            args->dual_encode_en=0;
        } else if(strcmp(*argv, "-bypass_en") == 0) {
            argc--;
            argv++;
            args->bypass_en=1;
        } else if(strcmp(*argv, "-align") == 0) {
            argc--;
            argv++;
            strcpy(linebuf,*argv);
            args->align = atoi(linebuf);
            argc--;
            argv++;
        } else if(strcmp(*argv, "-minfixbitln") == 0) {
            argc--;
            argv++;
            strcpy(linebuf,*argv);
            args->minFixedBitLn = atoi(linebuf);
            argc--;
            argv++;
        } else {
            char msgBuf[1024];
            sprintf(msgBuf,"Unknown Argument %s", *argv);
            BTC_REPORT_ERROR(msgBuf);
            argc--;
        }
    }
}        


#endif
