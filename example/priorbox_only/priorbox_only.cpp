//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "build/meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include "iostream"
#include "fstream"

int main()
{
    std::string path = std::getenv("MCM_HOME");
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();

    // Define Params
    unsigned imgW = 300
    unsigned imgH = 300
    unsigned width = 19
    unsigned height = 19
    unsigned crc = 0x726E32BC

    PriorBoxParams params_struct;
    params_struct.num_min_sizes = 1
    params_struct.num_max_sizes = 0
    params_struct.num_aspect_ratios = 1
    params_struct.num_variances = 4
    params_struct.flip = 1
    params_struct.clip = 0
    params_struct.step_w = 0.0f
    params_struct.step_h = 0.0f
    params_struct.offset = 0.5f
    params_struct.params = {60.0f, 2.0f, 0.1f, 0.1f, 0.2f, 0.2f}
    params = params_struct

    std::string compDescPath = path + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}


TestCase t1 = {300, 300, 19, 19, 0x726E32BC,
    { 1, 0, 1, 4, 1, 0, 0.0f, 0.0f, 0.5f, {60.0f, 2.0f, 0.1f, 0.1f, 0.2f, 0.2f}}};

typedef struct
{
   u32 num_min_sizes;
   u32 num_max_sizes;
   u32 num_aspect_ratios;
   u32 num_variances;
   u32 flip;
   u32 clip;
   float step_w;
   float step_h;
   float offset;
   float params[8]; // 1 min size, 0/1 max size, always 4 variance, 1 or 2 aspect
} PriorBoxParams;

typedef struct
{
   u32 imgW;
   u32 imgH;
   u32 width;
   u32 height;
   u32 crc;
   PriorBoxParams params;
} TestCase;