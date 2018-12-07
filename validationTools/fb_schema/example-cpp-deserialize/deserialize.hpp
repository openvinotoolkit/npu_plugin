#include <stdio.h>
#include <iostream>
#include <fstream>

#include "graphfile_generated.h"

#ifndef __DESERIALIZE__

using namespace MVCNN;

#define STR(s) #s

// Enum labels are useful for debugging.
// Using field names from blob to trigger compilation error if they change
// to force us to keep the code up to date.
const std::string dpuLayerTypeLabels[] =
{
    STR(DPULayerType_Conv),
    STR(DPULayerType_MaxPool),
    STR(DPULayerType_MaxPoolWithIndex)
};

const std::string specificTaskLabels[] =
{
    STR(SpecificTask_MvTensorTask),
    STR(SpecificTask_UPADMATask),
    STR(SpecificTask_NNDMATask),
    STR(SpecificTask_NCE1Task),
    STR(SpecificTask_NCE2Task),
    STR(SpecificTask_NNTensorTask),
    STR(SpecificTask_ControllerTask)
};

const std::string dtypeLabels[] =
{
    STR(DType_NOT_SET),
    STR(DType_FP32),
    STR(DType_FP16),
    STR(DType_FP8),
    STR(DType_U32),
    STR(DType_U16),
    STR(DType_U8),
    STR(DType_I32),
    STR(DType_I16),
    STR(DType_I8),
    STR(DType_I4),
    STR(DType_I2),
    STR(DType_I4X),
    STR(DType_BIN),
    STR(DType_LOG)
};

class Blob
{
private:
    char* data;

public:
    Blob(const char* s)
    {
        std::ifstream myFile;

        myFile.open(s, std::ios::binary | std::ios::in);
        myFile.seekg(0,std::ios::end);
        int length = myFile.tellg();
        myFile.seekg(0,std::ios::beg);

        std::cout << "Reading " << length << " bytes from "
                  << s << std::endl;

        data = new char[length];
        assert(data != nullptr);

        myFile.read(data, length );
        myFile.close();
    }
    Blob(const Blob&) = delete;

    ~Blob()
    {
        if (data != nullptr)
            delete data;
    }

    char* get_ptr()
    {
        return data;
    }
};

void deserialize(const GraphFile* const graph, bool print);

#define __DESERIALIZE__
#endif
