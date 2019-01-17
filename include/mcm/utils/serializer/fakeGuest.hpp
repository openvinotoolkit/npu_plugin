#include <stdio.h>
#include <iostream>
#include <fstream>

#ifndef __FAKE_GUEST__


class fGraphGuest{
    public:
        unsigned CMX_SIZE = 356*1024;
        unsigned version_major = 0;
        unsigned version_minor = 0;
        unsigned version_patch = 0;


};

#define __FAKE_GUEST__
#endif