#include "osdDefs.h"

//bell
#define MID 153.0f   //on X-axis

OsdPoint vertsBell[] ALIGNED(16) = {

    {MID +   0.0f,   0.0f},

   //[Right lobe]
    {MID +  13.0f,   3.0f},
    {MID +  30.0f,  16.0f},
    {MID +  34.0f,  25.0f},
    {MID +  44.0f,  29.0f},
    {MID +  54.0f,  35.0f},
    {MID +  64.0f,  43.0f},
    {MID +  71.0f,  57.0f},
    {MID +  77.0f,  80.0f},
    {MID +  90.0f, 124.0f},
    {MID +  99.0f, 151.0f},
    {MID + 110.0f, 173.0f},
    {MID + 122.0f, 178.0f},
    {MID + 130.0f, 185.0f},
    {MID + 137.0f, 194.0f},
    {MID + 137.0f, 194.0f},
    {MID + 146.0f, 210.0f},
    {MID + 153.0f, 227.0f},
    {MID + 153.0f, 233.0f},
    {MID + 146.0f, 239.0f},
    {MID + 129.0f, 246.0f},
    {MID + 108.0f, 251.0f},
    {MID +  86.0f, 255.0f},
    {MID +  65.0f, 258.0f},
    {MID +  35.0f, 261.0f},

    {MID +   0.0f, 262.0f},

   //[Left lobe]
    {MID -  35.0f, 261.0f},
    {MID -  65.0f, 258.0f},
    {MID -  86.0f, 255.0f},
    {MID - 108.0f, 251.0f},
    {MID - 129.0f, 246.0f},
    {MID - 146.0f, 239.0f},
    {MID - 153.0f, 233.0f},
    {MID - 153.0f, 227.0f},
    {MID - 146.0f, 210.0f},
    {MID - 137.0f, 194.0f},
    {MID - 137.0f, 194.0f},
    {MID - 130.0f, 185.0f},
    {MID - 122.0f, 178.0f},
    {MID - 110.0f, 173.0f},
    {MID -  99.0f, 151.0f},
    {MID -  90.0f, 124.0f},
    {MID -  77.0f,  80.0f},
    {MID -  71.0f,  57.0f},
    {MID -  64.0f,  43.0f},
    {MID -  54.0f,  35.0f},
    {MID -  44.0f,  29.0f},
    {MID -  34.0f,  25.0f},
    {MID -  30.0f,  16.0f},
    {MID -  13.0f,   3.0f},
};

#undef MID