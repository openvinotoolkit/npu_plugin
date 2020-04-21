#include "osdDefs.h"

#define LN_W  400 //line width

OsdPoint vertsThick1[] ALIGNED(16) = {
  #define THK 1.0f
    { 90.0f     , 54.0f },
    { 90.0f     , 54.0f + THK },
    { 90.0f+LN_W, 54.0f + THK },
    { 90.0f+LN_W, 54.0f }
  #undef THK
};

OsdPoint vertsThick2[] ALIGNED(16) = {
#define THK 2.0f
    { 90.0f     , 54.0f },
    { 90.0f     , 54.0f + THK },
    { 90.0f+LN_W, 54.0f + THK },
    { 90.0f+LN_W, 54.0f }
#undef THK
};

OsdPoint vertsThick3[] ALIGNED(16) = {
#define THK 3.0f
    { 90.0f     , 54.0f },
    { 90.0f     , 54.0f + THK },
    { 90.0f+LN_W, 54.0f + THK },
    { 90.0f+LN_W, 54.0f }
#undef THK
};
OsdPoint vertsThick5[] ALIGNED(16) = {
#define THK 5.0f
    { 90.0f     , 54.0f },
    { 90.0f     , 54.0f + THK },
    { 90.0f+LN_W, 54.0f + THK },
    { 90.0f+LN_W, 54.0f }
#undef THK
};
#undef LN_W