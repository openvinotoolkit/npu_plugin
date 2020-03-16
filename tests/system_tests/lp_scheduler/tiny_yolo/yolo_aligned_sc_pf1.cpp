#include <unistd.h>

//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>



struct InputParams {

  InputParams() : comp_descriptor_(NULL) {}

  bool parse_args(int argc, char **argv) {
    int opt;
    char const * const options = "d:";

    while ((opt = getopt(argc, argv, options)) != -1) {
      switch (opt) {
        case 'd':
          comp_descriptor_ = optarg;
          break;
        default:
          usage();
          return false;
      }
    }

    if (!comp_descriptor_) { 
      usage();
      return false; 
    }
    return true;
  }

  void usage() const {
    fprintf(stderr, "./three_layer_conv_model -d {comp_descriptor}\n");
  }

  const char *comp_descriptor_;
};  // struct InputParams //


int main(int argc, char **argv)
{

    InputParams params;

    if (!params.parse_args(argc, argv)) { return -1; }

    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({416,416,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#33");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3*3*3*16);
    auto weights0 = om.constantInt(weightsData0,{3,3,3,16}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{122},{0.0022838988807052374},{-0.2793084979057312},{0.30308571457862854}}, "conv1#0_weights#1");
    auto conv0 = om.conv(input0, weights0, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "conv1#34");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (16);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{16}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.791293288988527e-05},{-inf},{inf}}, "conv1#0_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    auto pool0 = om.maxPool(bias_c0, {2, 2}, {2, 2}, {0, 0, 0, 0}, true, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "pool1/max_pool#35");

    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (3*3*16*32);
    auto weights1 = om.constantInt(weightsData1,{3,3,16,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{119},{0.002742463955655694},{-0.32530343532562256},{0.374024897813797}}, "conv2#4_weights#5");
    auto conv1 = om.conv(pool0, weights1, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "conv2#36");

    std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights1 = om.constantInt(biasWeightsData1,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.075476120604435e-05},{-inf},{inf}}, "conv2#4_bias#6");
    auto bias_c1 = om.bias(conv1, biasWeights1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    auto pool1 = om.maxPool(bias_c1, {2, 2}, {2, 2}, {0, 0, 0, 0}, true, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "pool2/max_pool#37");

    std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (3*3*32*64);
    auto weights2 = om.constantInt(weightsData2,{3,3,32,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.0032946206629276276},{-0.42331647872924805},{0.4168117642402649}}, "conv3#8_weights#9");
    auto conv2 = om.conv(pool1, weights2, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "conv3#38");

    std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights2 = om.constantInt(biasWeightsData2,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2920080735057127e-05},{-inf},{inf}}, "conv3#8_bias#10");
    auto bias_c2 = om.bias(conv2, biasWeights2, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    auto pool2 = om.maxPool(bias_c2, {2, 2}, {2, 2}, {0, 0, 0, 0}, true, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "pool3/max_pool#39");

    std::vector<int64_t> weightsData3 = mv::utils::generateSequence<int64_t> (3*3*64*128);
    auto weights3 = om.constantInt(weightsData3,{3,3,64,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{132},{0.003287225728854537},{-0.43268874287605286},{0.40555381774902344}}, "conv4#12_weights#13");
    auto conv3 = om.conv(pool2, weights3, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "conv4#40");

    std::vector<int64_t> biasWeightsData3 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights3 = om.constantInt(biasWeightsData3,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.2891081496491097e-05},{-inf},{inf}}, "conv4#12_bias#14");
    auto bias_c3 = om.bias(conv3, biasWeights3, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    auto pool3 = om.maxPool(bias_c3, {2, 2}, {2, 2}, {0, 0, 0, 0}, true, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "pool4/max_pool#41");

    std::vector<int64_t> weightsData4 = mv::utils::generateSequence<int64_t> (3*3*128*256);
    auto weights4 = om.constantInt(weightsData4,{3,3,128,256}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.0035929458681493998},{-0.46311089396476746},{0.45309028029441833}}, "conv5#16_weights#17");
    auto conv4 = om.conv(pool3, weights4, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "conv5#42");

    std::vector<int64_t> biasWeightsData4 = mv::utils::generateSequence<int64_t> (256);
    auto biasWeights4 = om.constantInt(biasWeightsData4,{256}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4089983778831083e-05},{-inf},{inf}}, "conv5#16_bias#18");
    auto bias_c4 = om.bias(conv4, biasWeights4, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    auto pool4 = om.maxPool(bias_c4, {2, 2}, {2, 2}, {0, 0, 0, 0}, true, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "pool5/max_pool#43");

    std::vector<int64_t> weightsData5 = mv::utils::generateSequence<int64_t> (3*3*256*512);
    auto weights5 = om.constantInt(weightsData5,{3,3,256,512}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{133},{0.0037452178075909615},{-0.4963448643684387},{0.4586856961250305}}, "conv6#20_weights#21");
    auto conv5 = om.conv(pool4, weights5, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "conv6#44");

    std::vector<int64_t> biasWeightsData5 = mv::utils::generateSequence<int64_t> (512);
    auto biasWeights5 = om.constantInt(biasWeightsData5,{512}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.4687128896184731e-05},{-inf},{inf}}, "conv6#20_bias#22");
    auto bias_c5 = om.bias(conv5, biasWeights5, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}});

    auto pool5 = om.maxPool(bias_c5, {2, 2}, {1, 1}, {0, 0, 0, 0}, true, mv::DType("UInt8"), {{0},{0.003921568859368563},{0.0},{1.0}}, "pool6/max_pool#45");

    std::vector<int64_t> weightsData6 = mv::utils::generateSequence<int64_t> (3*3*512*1024);
    auto weights6 = om.constantInt(weightsData6,{3,3,512,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.004124246072024107},{-0.5335013270378113},{0.5181813836097717}}, "conv7#24_weights#25");
    auto conv6 = om.conv(pool5, weights6, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("UInt8"), {{0},{0.125490203499794},{0.0},{32.0}}, "conv7#46");

    std::vector<int64_t> biasWeightsData6 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights6 = om.constantInt(biasWeightsData6,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{1.6173513358808123e-05},{-inf},{inf}}, "conv7#24_bias#26");
    auto bias_c6 = om.bias(conv6, biasWeights6, mv::DType("UInt8"), {{0},{0.125490203499794},{0.0},{32.0}});

    std::vector<int64_t> weightsData7 = mv::utils::generateSequence<int64_t> (3*3*1024*1024);
    auto weights7 = om.constantInt(weightsData7,{3,3,1024,1024}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{124},{0.004363630432635546},{-0.5415508151054382},{0.5711749792098999}}, "conv8#27_weights#28");
    auto conv7 = om.conv(bias_c6, weights7, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("UInt8"), {{0},{0.125490203499794},{0.0},{32.0}}, "conv8#47");

    std::vector<int64_t> biasWeightsData7 = mv::utils::generateSequence<int64_t> (1024);
    auto biasWeights7 = om.constantInt(biasWeightsData7,{1024}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0005475928774103522},{-inf},{inf}}, "conv8#27_bias#29");
    auto bias_c7 = om.bias(conv7, biasWeights7, mv::DType("UInt8"), {{0},{0.125490203499794},{0.0},{32.0}});

    std::vector<int64_t> weightsData8 = mv::utils::generateSequence<int64_t> (3*3*1024*128);
    auto weights8 = om.constantInt(weightsData8,{3,3,1024,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{129},{0.004149848595261574},{-0.5362045764923096},{0.5220068097114563}}, "conv9/conv9#30_weights#31");
    auto conv8 = om.conv(bias_c7, weights8, {1, 1}, {1, 1, 1, 1}, 1, 1, mv::DType("UInt8"), {{4},{0.2549019753932953},{-1.0196079015731812},{63.98039245605469}}, "conv9/conv9#48");

    std::vector<int64_t> biasWeightsData8 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights8 = om.constantInt(biasWeightsData8,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0005207653157413006},{-inf},{inf}}, "conv9/conv9#30_bias#32");
    auto bias_c8 = om.bias(conv8, biasWeights8, mv::DType("UInt8"), {{4},{0.2549019753932953},{-1.0196079015731812},{63.98039245605469}});

    om.output(bias_c8);

    std::string compDescPath = params.comp_descriptor_;
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
