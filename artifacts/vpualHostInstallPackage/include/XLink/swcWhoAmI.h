///
/// INTEL CONFIDENTIAL
/// Copyright 2020. Intel Corporation.
/// This software and the related documents are Intel copyrighted materials, 
/// and your use of them is governed by the express license under which they were provided to you ("License"). 
/// Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
/// transmit this software or the related documents without Intel's prior written permission.
/// This software and the related documents are provided as is, with no express or implied warranties, 
/// other than those that are expressly stated in the License.
///
/// @file      swcWhoAmI.h
/// 

#ifndef SWC_WHOAMI_H_
#define SWC_WHOAMI_H_

#ifndef __PC__
#include <registersMyriad.h>
#endif

typedef enum {
    PROCESS_DEBUGGER = 0,
    PROCESS_LEON_OS,
    PROCESS_LEON_RT,
    PROCESS_SHAVE0,
    PROCESS_SHAVE1,
    PROCESS_SHAVE2,
    PROCESS_SHAVE3,
    PROCESS_SHAVE4,
    PROCESS_SHAVE5,
    PROCESS_SHAVE6,
    PROCESS_SHAVE7,
    PROCESS_SHAVE8,
    PROCESS_SHAVE9,
    PROCESS_SHAVE10,
    PROCESS_SHAVE11,
    #if CFG_NUM_SHAVES > 12
        PROCESS_SHAVE12,
        PROCESS_SHAVE13,
        PROCESS_SHAVE14,
        PROCESS_SHAVE15,
    #endif
    NUMBER_OF_PROCESSES,
    ERROR_ID,
} swcProcessorType;

#if defined(__sparc__)
    #define LEON_CORE
#elif  defined(__arm__)
    #define ARM_CORE
#elif  defined(__aarch64__)
    #define AARCH64_CORE
#else
    #define SHAVE_CORE
#endif

static inline swcProcessorType swcWhoAmI(void);

static inline swcProcessorType swcWhoAmI(void)
{
    #ifdef LEON_CORE
        // We are one of the leon processors
        unsigned int pcrValue;
        asm ("rd %%asr17, %[pcrValue]" :
                [pcrValue] "=r" (pcrValue) );
        int processorIndex = pcrValue >> 28;
        if (processorIndex)
            return PROCESS_LEON_RT;
        return PROCESS_LEON_OS;
    #endif
    #ifdef AARCH64_CORE
        //TODO: DO low level ARM core identification
        return PROCESS_LEON_OS;
    #endif   
    #ifdef ARM_CORE
        //TODO: DO low level ARM core identification
        return PROCESS_LEON_RT;
    #endif    
    #ifdef SHAVE_CORE
        // We are a shave
        int shaveNumber;
        shaveNumber = __builtin_shave_getcpuid();
        switch(shaveNumber){
        case 0: return PROCESS_SHAVE0;
        case 1: return PROCESS_SHAVE1;
        case 2: return PROCESS_SHAVE2;
        case 3: return PROCESS_SHAVE3;
        case 4: return PROCESS_SHAVE4;
        case 5: return PROCESS_SHAVE5;
        case 6: return PROCESS_SHAVE6;
        case 7: return PROCESS_SHAVE7;
        case 8: return PROCESS_SHAVE8;
        case 9: return PROCESS_SHAVE9;
        case 10: return PROCESS_SHAVE10;
        case 11: return PROCESS_SHAVE11;
        #if CFG_NUM_SHAVES > 12
            case 12: return PROCESS_SHAVE12;
            case 13: return PROCESS_SHAVE13;
            case 14: return PROCESS_SHAVE14;
            case 15: return PROCESS_SHAVE15;
        #endif
        default:
            break;
        }

    #endif
        return ERROR_ID;
}

static inline const char * swcGetProcessorName(int process)
{
    switch(process)
    {
        case PROCESS_DEBUGGER: return "DBG";
        case PROCESS_LEON_OS: return "LOS";
        case PROCESS_LEON_RT: return "LRT";
        case PROCESS_SHAVE0: return "S00";
        case PROCESS_SHAVE1: return "S01";
        case PROCESS_SHAVE2: return "S02";
        case PROCESS_SHAVE3: return "S03";
        case PROCESS_SHAVE4: return "S04";
        case PROCESS_SHAVE5: return "S05";
        case PROCESS_SHAVE6: return "S06";
        case PROCESS_SHAVE7: return "S07";
        case PROCESS_SHAVE8: return "S08";
        case PROCESS_SHAVE9: return "S09";
        case PROCESS_SHAVE10: return "S10";
        case PROCESS_SHAVE11: return "S11";
    #if CFG_NUM_SHAVES > 12
            case PROCESS_SHAVE12: return "S12";
            case PROCESS_SHAVE13: return "S13";
            case PROCESS_SHAVE14: return "S14";
            case PROCESS_SHAVE15: return "S15";
    #endif
        default:
            break;
    }
    return "UNK";
}

#endif
