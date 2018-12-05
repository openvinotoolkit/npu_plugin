#include <stdio.h>
#include <iostream>
#include <fstream>

#ifndef __FAKE_HOST__

class fTaskHost{
    public:
        unsigned referenceIDs = {1};
        unsigned * dependency;
        unsigned * consumers;
        unsigned taskID = 123;

        bool sparse_in = false;
        bool sparse_out = false;
        bool sparse_param = false;

        short op = 0;   // TODO: Should be enum DPULayerType.CONV
        short clusterID = 0;

        unsigned inWidth = 56;
        unsigned inHeight = 56;
        unsigned inChan = 64;
        unsigned outChan = 64;
        unsigned short kH = 1;
        unsigned short kW = 1;

        unsigned in_data = 0; // TODO: Tensor Ref
        unsigned out_data = 0; // TODO: Tensor Ref
        unsigned param_data = 0; // TODO: Tensor Ref

        unsigned actType = 0;
        unsigned oduType = 0;

        unsigned mpe = 1;
        unsigned ppe_param[16] = {0,0,0,0,0,0,0,0,0,0,0};
        unsigned dpuID[16] = {0, 1, 2, 3, 4};
        unsigned oXs[16] = {0, 0, 0, 32, 32};
        unsigned oXe[16] = {31, 31, 31, 55, 55};
        unsigned oYs[16] = {0, 20, 30, 0, 28};
        unsigned oYe[16] = {19, 39, 55, 27, 55};
        unsigned oZs[16] = {0, 0, 0, 0, 0};
        unsigned oZe[16] = {63, 63, 63, 63, 63};
};

class fGraphHost
{
    public:
        unsigned CMX_SIZE = 356*1024;
        unsigned version[3] = {3, 0, 1};
        std::string githash = "1b30ae2e04abf47f98a2ebc14d125f5cab9e7288";

        unsigned shaveMask = 1;
        unsigned nce1Mask = 2;
        unsigned dpuMask = 3;
        unsigned leonCMX = 1000;
        unsigned nnCMX = 2000;
        unsigned ddrScratch = 1234;

        unsigned dims[4] = {1, 3, 256, 256};
        unsigned strides[5] = {2, 6, 512, 1536, 393216};

        unsigned taskAmount = 321;
        unsigned layerAmount = 123;
        fTaskHost tasks[1];



};

#define __FAKE_HOST__
#endif