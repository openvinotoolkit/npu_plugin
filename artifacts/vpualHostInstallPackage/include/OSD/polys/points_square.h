#include "osdDefs.h"

//square

OsdPoint vertsSquare[] ALIGNED(16) = {
    #define SQSZ 480.0f
    { 10.0f       , 10.0f        }, // [0]
    { 10.0f + SQSZ, 10.0f        }, // [1]
    { 10.0f + SQSZ, 10.0f + SQSZ }, // [2]
    { 10.0f       , 10.0f + SQSZ }, // [3]
    #undef SQSZ
};