#include "osdDefs.h"

//Small triangle
OsdPoint vertsSmall[] ALIGNED(16) = {
  #define TSZ 4.0f
    { 90.0f      , 70.0f },
    { 90.0f+TSZ  , 70.0f + TSZ/2 },
    { 90.0f+TSZ/2, 70.0f + TSZ },
  #undef  TSZ
};