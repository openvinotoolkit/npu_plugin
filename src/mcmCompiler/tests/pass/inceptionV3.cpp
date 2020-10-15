//This file is the parsed network which is created through python.
#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include "iostream"
#include "fstream"

int main()
{
    std::string path = std::getenv("MDK_HOME");
    double inf = std::numeric_limits<double>::infinity();

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    auto input0 = om.input({299,299,3,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.007843137718737125},{-1.0},{1.0}}, "input#315");

    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (3*3*3*32);
    auto weights0 = om.constantInt(weightsData0,{3,3,3,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{145},{0.04352348670363426},{-6.285919666290283},{4.769046306610107}}, "InceptionV3/InceptionV3/Conv2d_1a_3x3/Relu_weights#1");
    auto conv0 = om.conv(input0, weights0, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.06630117446184158},{0.0},{16.90679931640625}}, "InceptionV3/InceptionV3/Conv2d_1a_3x3/Relu#316");

    std::vector<int64_t> biasWeightsData0 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights0 = om.constantInt(biasWeightsData0,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00034136069007217884},{-inf},{inf}}, "InceptionV3/InceptionV3/Conv2d_1a_3x3/Relu_bias#2");
    auto bias_c0 = om.bias(conv0, biasWeights0, {{0},{0.06630117446184158},{0.0},{16.90679931640625}});

    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (3*3*32*32);
    auto weights1 = om.constantInt(weightsData1,{3,3,32,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{81},{0.011024107225239277},{-0.8817759156227112},{1.9183472394943237}}, "InceptionV3/InceptionV3/Conv2d_2a_3x3/Relu_weights#4");
    auto conv1 = om.conv(bias_c0, weights1, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.07182303816080093},{0.0},{18.31487464904785}}, "InceptionV3/InceptionV3/Conv2d_2a_3x3/Relu#317");

    std::vector<int64_t> biasWeightsData1 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights1 = om.constantInt(biasWeightsData1,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0007309112115763128},{-inf},{inf}}, "InceptionV3/InceptionV3/Conv2d_2a_3x3/Relu_bias#5");
    auto bias_c1 = om.bias(conv1, biasWeights1, {{0},{0.07182303816080093},{0.0},{18.31487464904785}});

    std::vector<int64_t> weightsData2 = mv::utils::generateSequence<int64_t> (3*3*32*64);
    auto weights2 = om.constantInt(weightsData2,{3,3,32,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{107},{0.008458223193883896},{-0.8930862545967102},{1.2553025484085083}}, "InceptionV3/InceptionV3/Conv2d_2b_3x3/Relu_weights#7");
    auto conv2 = om.conv(bias_c1, weights2, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.06411805748939514},{0.0},{16.35010528564453}}, "InceptionV3/InceptionV3/Conv2d_2b_3x3/Relu#318");

    std::vector<int64_t> biasWeightsData2 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights2 = om.constantInt(biasWeightsData2,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0006074953125789762},{-inf},{inf}}, "InceptionV3/InceptionV3/Conv2d_2b_3x3/Relu_bias#8");
    auto bias_c2 = om.bias(conv2, biasWeights2, {{0},{0.06411805748939514},{0.0},{16.35010528564453}});

    auto pool0 = om.maxPool(bias_c2, {3, 3}, {2, 2}, {0, 0, 0, 0}, true, {{0},{0.06411805748939514},{0.0},{16.35010528564453}}, "InceptionV3/InceptionV3/MaxPool_3a_3x3/MaxPool#319");

    std::vector<int64_t> weightsData3 = mv::utils::generateSequence<int64_t> (1*1*64*80);
    auto weights3 = om.constantInt(weightsData3,{1,1,64,80}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{106},{0.00860925205051899},{-0.8998916745185852},{1.2868584394454956}}, "InceptionV3/InceptionV3/Conv2d_3b_1x1/Relu_weights#11");
    auto conv3 = om.conv(pool0, weights3, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.058248285204172134},{0.0},{14.853312492370605}}, "InceptionV3/InceptionV3/Conv2d_3b_1x1/Relu#320");

    std::vector<int64_t> biasWeightsData3 = mv::utils::generateSequence<int64_t> (80);
    auto biasWeights3 = om.constantInt(biasWeightsData3,{80}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0005520085687749088},{-inf},{inf}}, "InceptionV3/InceptionV3/Conv2d_3b_1x1/Relu_bias#12");
    auto bias_c3 = om.bias(conv3, biasWeights3, {{0},{0.058248285204172134},{0.0},{14.853312492370605}});

    std::vector<int64_t> weightsData4 = mv::utils::generateSequence<int64_t> (3*3*80*192);
    auto weights4 = om.constantInt(weightsData4,{3,3,80,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{131},{0.005702306982129812},{-0.7423749566078186},{0.7060109972953796}}, "InceptionV3/InceptionV3/Conv2d_4a_3x3/Relu_weights#14");
    auto conv4 = om.conv(bias_c3, weights4, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.04986247420310974},{0.0},{12.714930534362793}}, "InceptionV3/InceptionV3/Conv2d_4a_3x3/Relu#321");

    std::vector<int64_t> biasWeightsData4 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights4 = om.constantInt(biasWeightsData4,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00033214958966709673},{-inf},{inf}}, "InceptionV3/InceptionV3/Conv2d_4a_3x3/Relu_bias#15");
    auto bias_c4 = om.bias(conv4, biasWeights4, {{0},{0.04986247420310974},{0.0},{12.714930534362793}});

    auto pool1 = om.maxPool(bias_c4, {3, 3}, {2, 2}, {0, 0, 0, 0}, true, {{0},{0.04986247420310974},{0.0},{12.714930534362793}}, "InceptionV3/InceptionV3/MaxPool_5a_3x3/MaxPool#322");

    std::vector<int64_t> weightsData5 = mv::utils::generateSequence<int64_t> (1*1*192*64);
    auto weights5 = om.constantInt(weightsData5,{1,1,192,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.004454410634934902},{-0.5758478045463562},{0.555572509765625}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_0/Conv2d_0a_1x1/Relu_weights#19");
    auto conv5 = om.conv(pool1, weights5, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.055134955793619156},{0.0},{14.05941390991211}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_0/Conv2d_0a_1x1/Relu#324");

    std::vector<int64_t> biasWeightsData5 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights5 = om.constantInt(biasWeightsData5,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00022210793395061046},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_0/Conv2d_0a_1x1/Relu_bias#20");
    auto bias_c5 = om.bias(conv5, biasWeights5, {{0},{0.055134955793619156},{0.0},{14.05941390991211}});

    std::vector<int64_t> weightsData6 = mv::utils::generateSequence<int64_t> (1*1*192*48);
    auto weights6 = om.constantInt(weightsData6,{1,1,192,48}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{128},{0.003966687712818384},{-0.50564044713974},{0.5018982887268066}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_1/Conv2d_0a_1x1/Relu_weights#22");
    auto conv6 = om.conv(pool1, weights6, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.03920495882630348},{0.0},{9.997264862060547}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_1/Conv2d_0a_1x1/Relu#325");

    std::vector<int64_t> biasWeightsData6 = mv::utils::generateSequence<int64_t> (48);
    auto biasWeights6 = om.constantInt(biasWeightsData6,{48}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00019778887508437037},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_1/Conv2d_0a_1x1/Relu_bias#23");
    auto bias_c6 = om.bias(conv6, biasWeights6, {{0},{0.03920495882630348},{0.0},{9.997264862060547}});

    std::vector<int64_t> weightsData7 = mv::utils::generateSequence<int64_t> (5*5*48*64);
    auto weights7 = om.constantInt(weightsData7,{5,5,48,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{115},{0.0042505874298512936},{-0.4827670454978943},{0.5968821048736572}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_1/Conv2d_0b_5x5/Relu_weights#25");
    auto conv7 = om.conv(bias_c6, weights7, {1, 1}, {2, 2, 2, 2}, 1, 1, {{0},{0.055134955793619156},{0.0},{14.05941390991211}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_1/Conv2d_0b_5x5/Relu#326");

    std::vector<int64_t> biasWeightsData7 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights7 = om.constantInt(biasWeightsData7,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00016664410941302776},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_1/Conv2d_0b_5x5/Relu_bias#26");
    auto bias_c7 = om.bias(conv7, biasWeights7, {{0},{0.055134955793619156},{0.0},{14.05941390991211}});

    std::vector<int64_t> weightsData8 = mv::utils::generateSequence<int64_t> (1*1*192*64);
    auto weights8 = om.constantInt(weightsData8,{1,1,192,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{89},{0.005482192151248455},{-0.484860897064209},{0.9076159596443176}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_2/Conv2d_0a_1x1/Relu_weights#28");
    auto conv8 = om.conv(pool1, weights8, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.049495816230773926},{0.0},{12.62143325805664}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_2/Conv2d_0a_1x1/Relu#327");

    std::vector<int64_t> biasWeightsData8 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights8 = om.constantInt(biasWeightsData8,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0002733556611929089},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_2/Conv2d_0a_1x1/Relu_bias#29");
    auto bias_c8 = om.bias(conv8, biasWeights8, {{0},{0.049495816230773926},{0.0},{12.62143325805664}});

    std::vector<int64_t> weightsData9 = mv::utils::generateSequence<int64_t> (3*3*64*96);
    auto weights9 = om.constantInt(weightsData9,{3,3,64,96}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{121},{0.005696559324860573},{-0.6830973625183105},{0.763828694820404}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_2/Conv2d_0b_3x3/Relu_weights#31");
    auto conv9 = om.conv(bias_c8, weights9, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.04927249997854233},{0.0},{12.56448745727539}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_2/Conv2d_0b_3x3/Relu#328");

    std::vector<int64_t> biasWeightsData9 = mv::utils::generateSequence<int64_t> (96);
    auto biasWeights9 = om.constantInt(biasWeightsData9,{96}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0002819558430928737},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_2/Conv2d_0b_3x3/Relu_bias#32");
    auto bias_c9 = om.bias(conv9, biasWeights9, {{0},{0.04927249997854233},{0.0},{12.56448745727539}});

    std::vector<int64_t> weightsData10 = mv::utils::generateSequence<int64_t> (3*3*96*96);
    auto weights10 = om.constantInt(weightsData10,{3,3,96,96}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{81},{0.004391361027956009},{-0.35252583026885986},{0.7628799080848694}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_2/Conv2d_0c_3x3/Relu_weights#34");
    auto conv10 = om.conv(bias_c9, weights10, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.055134955793619156},{0.0},{14.05941390991211}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_2/Conv2d_0c_3x3/Relu#329");

    std::vector<int64_t> biasWeightsData10 = mv::utils::generateSequence<int64_t> (96);
    auto biasWeights10 = om.constantInt(biasWeightsData10,{96}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0002163733443012461},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_2/Conv2d_0c_3x3/Relu_bias#35");
    auto bias_c10 = om.bias(conv10, biasWeights10, {{0},{0.055134955793619156},{0.0},{14.05941390991211}});

    auto pool2 = om.averagePool(pool1, {3, 3}, {1, 1}, {1, 1, 1, 1}, true, {{0},{0.04986247420310974},{0.0},{12.714930534362793}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_3/AvgPool_0a_3x3/AvgPool#323");

    std::vector<int64_t> weightsData11 = mv::utils::generateSequence<int64_t> (1*1*192*32);
    auto weights11 = om.constantInt(weightsData11,{1,1,192,32}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{160},{0.008971055969595909},{-1.4265706539154053},{0.8520775437355042}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_3/Conv2d_0b_1x1/Relu_weights#37");
    auto conv11 = om.conv(pool2, weights11, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.055134955793619156},{0.0},{14.05941390991211}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_3/Conv2d_0b_1x1/Relu#330");

    std::vector<int64_t> biasWeightsData11 = mv::utils::generateSequence<int64_t> (32);
    auto biasWeights11 = om.constantInt(biasWeightsData11,{32}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0004473190347198397},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5b/Branch_3/Conv2d_0b_1x1/Relu_bias#38");
    auto bias_c11 = om.bias(conv11, biasWeights11, {{0},{0.055134955793619156},{0.0},{14.05941390991211}});

    auto concat0 = om.concat({bias_c5, bias_c7, bias_c10, bias_c11}, "C", {{0},{0.055134955793619156},{0.0},{14.05941390991211}}, "InceptionV3/InceptionV3/Mixed_5b/concat#331");

    std::vector<int64_t> weightsData12 = mv::utils::generateSequence<int64_t> (1*1*256*64);
    auto weights12 = om.constantInt(weightsData12,{1,1,256,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{116},{0.005221945233643055},{-0.5988596081733704},{0.7275145053863525}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_0/Conv2d_0a_1x1/Relu_weights#42");
    auto conv12 = om.conv(concat0, weights12, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.0661386027932167},{0.0},{16.86534309387207}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_0/Conv2d_0a_1x1/Relu#333");

    std::vector<int64_t> biasWeightsData12 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights12 = om.constantInt(biasWeightsData12,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00028791173826903105},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_0/Conv2d_0a_1x1/Relu_bias#43");
    auto bias_c12 = om.bias(conv12, biasWeights12, {{0},{0.0661386027932167},{0.0},{16.86534309387207}});

    std::vector<int64_t> weightsData13 = mv::utils::generateSequence<int64_t> (1*1*256*48);
    auto weights13 = om.constantInt(weightsData13,{1,1,256,48}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{138},{0.00434325123205781},{-0.5954484939575195},{0.5077372789382935}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_1/Conv2d_0b_1x1/Relu_weights#45");
    auto conv13 = om.conv(concat0, weights13, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.03856371343135834},{0.0},{9.833746910095215}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_1/Conv2d_0b_1x1/Relu#334");

    std::vector<int64_t> biasWeightsData13 = mv::utils::generateSequence<int64_t> (48);
    auto biasWeights13 = om.constantInt(biasWeightsData13,{48}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00023946496366988868},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_1/Conv2d_0b_1x1/Relu_bias#46");
    auto bias_c13 = om.bias(conv13, biasWeights13, {{0},{0.03856371343135834},{0.0},{9.833746910095215}});

    std::vector<int64_t> weightsData14 = mv::utils::generateSequence<int64_t> (5*5*48*64);
    auto weights14 = om.constantInt(weightsData14,{5,5,48,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{118},{0.0037733963690698147},{-0.44283658266067505},{0.5156061053276062}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_1/Conv_1_0c_5x5/Relu_weights#48");
    auto conv14 = om.conv(bias_c13, weights14, {1, 1}, {2, 2, 2, 2}, 1, 1, {{0},{0.0661386027932167},{0.0},{16.86534309387207}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_1/Conv_1_0c_5x5/Relu#335");

    std::vector<int64_t> biasWeightsData14 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights14 = om.constantInt(biasWeightsData14,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00014551618369296193},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_1/Conv_1_0c_5x5/Relu_bias#49");
    auto bias_c14 = om.bias(conv14, biasWeights14, {{0},{0.0661386027932167},{0.0},{16.86534309387207}});

    std::vector<int64_t> weightsData15 = mv::utils::generateSequence<int64_t> (1*1*256*64);
    auto weights15 = om.constantInt(weightsData15,{1,1,256,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{115},{0.005050840321928263},{-0.5756707191467285},{0.7072426676750183}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_2/Conv2d_0a_1x1/Relu_weights#51");
    auto conv15 = om.conv(concat0, weights15, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.033666182309389114},{0.0},{8.58487606048584}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_2/Conv2d_0a_1x1/Relu#336");

    std::vector<int64_t> biasWeightsData15 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights15 = om.constantInt(biasWeightsData15,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00027847784804180264},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_2/Conv2d_0a_1x1/Relu_bias#52");
    auto bias_c15 = om.bias(conv15, biasWeights15, {{0},{0.033666182309389114},{0.0},{8.58487606048584}});

    std::vector<int64_t> weightsData16 = mv::utils::generateSequence<int64_t> (3*3*64*96);
    auto weights16 = om.constantInt(weightsData16,{3,3,64,96}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{101},{0.00458966288715601},{-0.45770204067230225},{0.708072304725647}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_2/Conv2d_0b_3x3/Relu_weights#54");
    auto conv16 = om.conv(bias_c15, weights16, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.038608841598033905},{0.0},{9.845254898071289}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_2/Conv2d_0b_3x3/Relu#337");

    std::vector<int64_t> biasWeightsData16 = mv::utils::generateSequence<int64_t> (96);
    auto biasWeights16 = om.constantInt(biasWeightsData16,{96}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00015451641229446977},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_2/Conv2d_0b_3x3/Relu_bias#55");
    auto bias_c16 = om.bias(conv16, biasWeights16, {{0},{0.038608841598033905},{0.0},{9.845254898071289}});

    std::vector<int64_t> weightsData17 = mv::utils::generateSequence<int64_t> (3*3*96*96);
    auto weights17 = om.constantInt(weightsData17,{3,3,96,96}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{67},{0.006374058313667774},{-0.42051592469215393},{1.1984949111938477}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_2/Conv2d_0c_3x3/Relu_weights#57");
    auto conv17 = om.conv(bias_c16, weights17, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.0661386027932167},{0.0},{16.86534309387207}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_2/Conv2d_0c_3x3/Relu#338");

    std::vector<int64_t> biasWeightsData17 = mv::utils::generateSequence<int64_t> (96);
    auto biasWeights17 = om.constantInt(biasWeightsData17,{96}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0002460950054228306},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_2/Conv2d_0c_3x3/Relu_bias#58");
    auto bias_c17 = om.bias(conv17, biasWeights17, {{0},{0.0661386027932167},{0.0},{16.86534309387207}});

    auto pool3 = om.averagePool(concat0, {3, 3}, {1, 1}, {1, 1, 1, 1}, true, {{0},{0.055134955793619156},{0.0},{14.05941390991211}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_3/AvgPool_0a_3x3/AvgPool#332");

    std::vector<int64_t> weightsData18 = mv::utils::generateSequence<int64_t> (1*1*256*64);
    auto weights18 = om.constantInt(weightsData18,{1,1,256,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{116},{0.008546429686248302},{-0.9843059182167053},{1.1864871978759766}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_3/Conv2d_0b_1x1/Relu_weights#60");
    auto conv18 = om.conv(pool3, weights18, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.0661386027932167},{0.0},{16.86534309387207}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_3/Conv2d_0b_1x1/Relu#339");

    std::vector<int64_t> biasWeightsData18 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights18 = om.constantInt(biasWeightsData18,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00047120702220126987},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5c/Branch_3/Conv2d_0b_1x1/Relu_bias#61");
    auto bias_c18 = om.bias(conv18, biasWeights18, {{0},{0.0661386027932167},{0.0},{16.86534309387207}});

    auto concat1 = om.concat({bias_c12, bias_c14, bias_c17, bias_c18}, "C", {{0},{0.0661386027932167},{0.0},{16.86534309387207}}, "InceptionV3/InceptionV3/Mixed_5c/concat#340");

    std::vector<int64_t> weightsData19 = mv::utils::generateSequence<int64_t> (1*1*288*64);
    auto weights19 = om.constantInt(weightsData19,{1,1,288,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{167},{0.005469807423651218},{-0.9075607061386108},{0.4817703366279602}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_0/Conv2d_0a_1x1/Relu_weights#65");
    auto conv19 = om.conv(concat1, weights19, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.04834429547190666},{0.0},{12.327795028686523}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_0/Conv2d_0a_1x1/Relu#342");

    std::vector<int64_t> biasWeightsData19 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights19 = om.constantInt(biasWeightsData19,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00036176538560539484},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_0/Conv2d_0a_1x1/Relu_bias#66");
    auto bias_c19 = om.bias(conv19, biasWeights19, {{0},{0.04834429547190666},{0.0},{12.327795028686523}});

    std::vector<int64_t> weightsData20 = mv::utils::generateSequence<int64_t> (1*1*288*48);
    auto weights20 = om.constantInt(weightsData20,{1,1,288,48}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{114},{0.004662926774471998},{-0.5253708958625793},{0.6590125560760498}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_1/Conv2d_0a_1x1/Relu_weights#68");
    auto conv20 = om.conv(concat1, weights20, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.04102246090769768},{0.0},{10.46072769165039}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_1/Conv2d_0a_1x1/Relu#343");

    std::vector<int64_t> biasWeightsData20 = mv::utils::generateSequence<int64_t> (48);
    auto biasWeights20 = om.constantInt(biasWeightsData20,{48}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0003083994670305401},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_1/Conv2d_0a_1x1/Relu_bias#69");
    auto bias_c20 = om.bias(conv20, biasWeights20, {{0},{0.04102246090769768},{0.0},{10.46072769165039}});

    std::vector<int64_t> weightsData21 = mv::utils::generateSequence<int64_t> (5*5*48*64);
    auto weights21 = om.constantInt(weightsData21,{5,5,48,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{114},{0.002590085146948695},{-0.29344117641448975},{0.3644404709339142}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_1/Conv2d_0b_5x5/Relu_weights#71");
    auto conv21 = om.conv(bias_c20, weights21, {1, 1}, {2, 2, 2, 2}, 1, 1, {{0},{0.04834429547190666},{0.0},{12.327795028686523}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_1/Conv2d_0b_5x5/Relu#344");

    std::vector<int64_t> biasWeightsData21 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights21 = om.constantInt(biasWeightsData21,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00010625167487887666},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_1/Conv2d_0b_5x5/Relu_bias#72");
    auto bias_c21 = om.bias(conv21, biasWeights21, {{0},{0.04834429547190666},{0.0},{12.327795028686523}});

    std::vector<int64_t> weightsData22 = mv::utils::generateSequence<int64_t> (1*1*288*64);
    auto weights22 = om.constantInt(weightsData22,{1,1,288,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{137},{0.0066154650412499905},{-0.9013203382492065},{0.7790077924728394}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_2/Conv2d_0a_1x1/Relu_weights#74");
    auto conv22 = om.conv(concat1, weights22, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.04998011514544487},{0.0},{12.744929313659668}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_2/Conv2d_0a_1x1/Relu#345");

    std::vector<int64_t> biasWeightsData22 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights22 = om.constantInt(biasWeightsData22,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00043753761565312743},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_2/Conv2d_0a_1x1/Relu_bias#75");
    auto bias_c22 = om.bias(conv22, biasWeights22, {{0},{0.04998011514544487},{0.0},{12.744929313659668}});

    std::vector<int64_t> weightsData23 = mv::utils::generateSequence<int64_t> (3*3*64*96);
    auto weights23 = om.constantInt(weightsData23,{3,3,64,96}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{126},{0.004606219008564949},{-0.577035665512085},{0.5929439067840576}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_2/Conv2d_0b_3x3/Relu_weights#77");
    auto conv23 = om.conv(bias_c22, weights23, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.040454305708408356},{0.0},{10.315848350524902}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_2/Conv2d_0b_3x3/Relu#346");

    std::vector<int64_t> biasWeightsData23 = mv::utils::generateSequence<int64_t> (96);
    auto biasWeights23 = om.constantInt(biasWeightsData23,{96}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00023021934612188488},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_2/Conv2d_0b_3x3/Relu_bias#78");
    auto bias_c23 = om.bias(conv23, biasWeights23, {{0},{0.040454305708408356},{0.0},{10.315848350524902}});

    std::vector<int64_t> weightsData24 = mv::utils::generateSequence<int64_t> (3*3*96*96);
    auto weights24 = om.constantInt(weightsData24,{3,3,96,96}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{84},{0.0034518393222242594},{-0.2865237891674042},{0.590243399143219}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_2/Conv2d_0c_3x3/Relu_weights#80");
    auto conv24 = om.conv(bias_c23, weights24, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.04834429547190666},{0.0},{12.327795028686523}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_2/Conv2d_0c_3x3/Relu#347");

    std::vector<int64_t> biasWeightsData24 = mv::utils::generateSequence<int64_t> (96);
    auto biasWeights24 = om.constantInt(biasWeightsData24,{96}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0001396417646901682},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_2/Conv2d_0c_3x3/Relu_bias#81");
    auto bias_c24 = om.bias(conv24, biasWeights24, {{0},{0.04834429547190666},{0.0},{12.327795028686523}});

    auto pool4 = om.averagePool(concat1, {3, 3}, {1, 1}, {1, 1, 1, 1}, true, {{0},{0.0661386027932167},{0.0},{16.86534309387207}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_3/AvgPool_0a_3x3/AvgPool#341");

    std::vector<int64_t> weightsData25 = mv::utils::generateSequence<int64_t> (1*1*288*64);
    auto weights25 = om.constantInt(weightsData25,{1,1,288,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{103},{0.008292028680443764},{-0.8489067554473877},{1.2572684288024902}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_3/Conv2d_0b_1x1/Relu_weights#83");
    auto conv25 = om.conv(pool4, weights25, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.04834429547190666},{0.0},{12.327795028686523}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_3/Conv2d_0b_1x1/Relu#348");

    std::vector<int64_t> biasWeightsData25 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights25 = om.constantInt(biasWeightsData25,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0005484231514856219},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_5d/Branch_3/Conv2d_0b_1x1/Relu_bias#84");
    auto bias_c25 = om.bias(conv25, biasWeights25, {{0},{0.04834429547190666},{0.0},{12.327795028686523}});

    auto concat2 = om.concat({bias_c19, bias_c21, bias_c24, bias_c25}, "C", {{0},{0.04834429547190666},{0.0},{12.327795028686523}}, "InceptionV3/InceptionV3/Mixed_5d/concat#349");

    std::vector<int64_t> weightsData26 = mv::utils::generateSequence<int64_t> (3*3*288*384);
    auto weights26 = om.constantInt(weightsData26,{3,3,288,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{82},{0.0028063678182661533},{-0.22646643221378326},{0.48635098338127136}}, "InceptionV3/InceptionV3/Mixed_6a/Branch_0/Conv2d_1a_1x1/Relu_weights#88");
    auto conv26 = om.conv(concat2, weights26, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.04834429547190666},{0.0},{12.327795028686523}}, "InceptionV3/InceptionV3/Mixed_6a/Branch_0/Conv2d_1a_1x1/Relu#351");

    std::vector<int64_t> biasWeightsData26 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights26 = om.constantInt(biasWeightsData26,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00013567187124863267},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6a/Branch_0/Conv2d_1a_1x1/Relu_bias#89");
    auto bias_c26 = om.bias(conv26, biasWeights26, {{0},{0.04834429547190666},{0.0},{12.327795028686523}});

    std::vector<int64_t> weightsData27 = mv::utils::generateSequence<int64_t> (1*1*288*64);
    auto weights27 = om.constantInt(weightsData27,{1,1,288,64}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{126},{0.005199098028242588},{-0.6521255373954773},{0.668445348739624}}, "InceptionV3/InceptionV3/Mixed_6a/Branch_1/Conv2d_0a_1x1/Relu_weights#91");
    auto conv27 = om.conv(concat2, weights27, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.039479810744524},{0.0},{10.067351341247559}}, "InceptionV3/InceptionV3/Mixed_6a/Branch_1/Conv2d_0a_1x1/Relu#352");

    std::vector<int64_t> biasWeightsData27 = mv::utils::generateSequence<int64_t> (64);
    auto biasWeights27 = om.constantInt(biasWeightsData27,{64}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00025134673342108727},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6a/Branch_1/Conv2d_0a_1x1/Relu_bias#92");
    auto bias_c27 = om.bias(conv27, biasWeights27, {{0},{0.039479810744524},{0.0},{10.067351341247559}});

    std::vector<int64_t> weightsData28 = mv::utils::generateSequence<int64_t> (3*3*64*96);
    auto weights28 = om.constantInt(weightsData28,{3,3,64,96}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{106},{0.0038477531634271145},{-0.4053335189819336},{0.5719957947731018}}, "InceptionV3/InceptionV3/Mixed_6a/Branch_1/Conv2d_0b_3x3/Relu_weights#94");
    auto conv28 = om.conv(bias_c27, weights28, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.039638299494981766},{0.0},{10.107766151428223}}, "InceptionV3/InceptionV3/Mixed_6a/Branch_1/Conv2d_0b_3x3/Relu#353");

    std::vector<int64_t> biasWeightsData28 = mv::utils::generateSequence<int64_t> (96);
    auto biasWeights28 = om.constantInt(biasWeightsData28,{96}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00015190856356639415},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6a/Branch_1/Conv2d_0b_3x3/Relu_bias#95");
    auto bias_c28 = om.bias(conv28, biasWeights28, {{0},{0.039638299494981766},{0.0},{10.107766151428223}});

    std::vector<int64_t> weightsData29 = mv::utils::generateSequence<int64_t> (3*3*96*96);
    auto weights29 = om.constantInt(weightsData29,{3,3,96,96}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{88},{0.0032767783850431442},{-0.28487005829811096},{0.5474316477775574}}, "InceptionV3/InceptionV3/Mixed_6a/Branch_1/Conv2d_1a_1x1/Relu_weights#97");
    auto conv29 = om.conv(bias_c28, weights29, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.04834429547190666},{0.0},{12.327795028686523}}, "InceptionV3/InceptionV3/Mixed_6a/Branch_1/Conv2d_1a_1x1/Relu#354");

    std::vector<int64_t> biasWeightsData29 = mv::utils::generateSequence<int64_t> (96);
    auto biasWeights29 = om.constantInt(biasWeightsData29,{96}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00012988591333851218},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6a/Branch_1/Conv2d_1a_1x1/Relu_bias#98");
    auto bias_c29 = om.bias(conv29, biasWeights29, {{0},{0.04834429547190666},{0.0},{12.327795028686523}});

    auto pool5 = om.maxPool(concat2, {3, 3}, {2, 2}, {0, 0, 0, 0}, true, {{0},{0.04834429547190666},{0.0},{12.327795028686523}}, "InceptionV3/InceptionV3/Mixed_6a/Branch_2/MaxPool_1a_3x3/MaxPool#350");

    auto concat3 = om.concat({bias_c26, bias_c29, pool5}, "C", {{0},{0.04834429547190666},{0.0},{12.327795028686523}}, "InceptionV3/InceptionV3/Mixed_6a/concat#355");

    std::vector<int64_t> weightsData30 = mv::utils::generateSequence<int64_t> (1*1*768*192);
    auto weights30 = om.constantInt(weightsData30,{1,1,768,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{107},{0.0046876417472958565},{-0.4979884922504425},{0.6926724910736084}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_0/Conv2d_0a_1x1/Relu_weights#102");
    auto conv30 = om.conv(concat3, weights30, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.07881999760866165},{0.0},{20.09910011291504}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_0/Conv2d_0a_1x1/Relu#357");

    std::vector<int64_t> biasWeightsData30 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights30 = om.constantInt(biasWeightsData30,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0002266207302454859},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_0/Conv2d_0a_1x1/Relu_bias#103");
    auto bias_c30 = om.bias(conv30, biasWeights30, {{0},{0.07881999760866165},{0.0},{20.09910011291504}});

    std::vector<int64_t> weightsData31 = mv::utils::generateSequence<int64_t> (1*1*768*128);
    auto weights31 = om.constantInt(weightsData31,{1,1,768,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{80},{0.004490106366574764},{-0.35642749071121216},{0.7840595245361328}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_1/Conv2d_0a_1x1/Relu_weights#105");
    auto conv31 = om.conv(concat3, weights31, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.03768473118543625},{0.0},{9.609606742858887}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_1/Conv2d_0a_1x1/Relu#358");

    std::vector<int64_t> biasWeightsData31 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights31 = om.constantInt(biasWeightsData31,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00021707102132495493},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_1/Conv2d_0a_1x1/Relu_bias#106");
    auto bias_c31 = om.bias(conv31, biasWeights31, {{0},{0.03768473118543625},{0.0},{9.609606742858887}});

    std::vector<int64_t> weightsData32 = mv::utils::generateSequence<int64_t> (7*1*128*128);
    auto weights32 = om.constantInt(weightsData32,{7,1,128,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{80},{0.00547377485781908},{-0.4322487413883209},{0.9580901265144348}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_1/Conv2d_0b_1x7/Relu_weights#108");
    auto conv32 = om.conv(bias_c31, weights32, {1, 1}, {3, 3, 0, 0}, 1, 1, {{0},{0.04628434032201767},{0.0},{11.802506446838379}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_1/Conv2d_0b_1x7/Relu#359");

    std::vector<int64_t> biasWeightsData32 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights32 = om.constantInt(biasWeightsData32,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0002062777493847534},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_1/Conv2d_0b_1x7/Relu_bias#109");
    auto bias_c32 = om.bias(conv32, biasWeights32, {{0},{0.04628434032201767},{0.0},{11.802506446838379}});

    std::vector<int64_t> weightsData33 = mv::utils::generateSequence<int64_t> (1*7*128*192);
    auto weights33 = om.constantInt(weightsData33,{1,7,128,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{97},{0.0050449129194021225},{-0.48552289605140686},{0.7958850264549255}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_1/Conv2d_0c_7x1/Relu_weights#111");
    auto conv33 = om.conv(bias_c32, weights33, {1, 1}, {0, 0, 3, 3}, 1, 1, {{0},{0.07881999760866165},{0.0},{20.09910011291504}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_1/Conv2d_0c_7x1/Relu#360");

    std::vector<int64_t> biasWeightsData33 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights33 = om.constantInt(biasWeightsData33,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00023350046831183136},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_1/Conv2d_0c_7x1/Relu_bias#112");
    auto bias_c33 = om.bias(conv33, biasWeights33, {{0},{0.07881999760866165},{0.0},{20.09910011291504}});

    std::vector<int64_t> weightsData34 = mv::utils::generateSequence<int64_t> (1*1*768*128);
    auto weights34 = om.constantInt(weightsData34,{1,1,768,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{153},{0.0044212955981493},{-0.6731512546539307},{0.44985783100128174}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_2/Conv2d_0a_1x1/Relu_weights#114");
    auto conv34 = om.conv(concat3, weights34, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.033814314752817154},{0.0},{8.622650146484375}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_2/Conv2d_0a_1x1/Relu#361");

    std::vector<int64_t> biasWeightsData34 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights34 = om.constantInt(biasWeightsData34,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00021374440984800458},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_2/Conv2d_0a_1x1/Relu_bias#115");
    auto bias_c34 = om.bias(conv34, biasWeights34, {{0},{0.033814314752817154},{0.0},{8.622650146484375}});

    std::vector<int64_t> weightsData35 = mv::utils::generateSequence<int64_t> (1*7*128*128);
    auto weights35 = om.constantInt(weightsData35,{1,7,128,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{85},{0.0036225812509655952},{-0.30433785915374756},{0.615797758102417}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_2/Conv2d_0b_7x1/Relu_weights#117");
    auto conv35 = om.conv(bias_c34, weights35, {1, 1}, {0, 0, 3, 3}, 1, 1, {{0},{0.032610274851322174},{0.0},{8.315620422363281}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_2/Conv2d_0b_7x1/Relu#362");

    std::vector<int64_t> biasWeightsData35 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights35 = om.constantInt(biasWeightsData35,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00012249509745743126},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_2/Conv2d_0b_7x1/Relu_bias#118");
    auto bias_c35 = om.bias(conv35, biasWeights35, {{0},{0.032610274851322174},{0.0},{8.315620422363281}});

    std::vector<int64_t> weightsData36 = mv::utils::generateSequence<int64_t> (7*1*128*128);
    auto weights36 = om.constantInt(weightsData36,{7,1,128,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{83},{0.004558204207569361},{-0.37426066398620605},{0.7835232019424438}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_2/Conv2d_0c_1x7/Relu_weights#120");
    auto conv36 = om.conv(bias_c35, weights36, {1, 1}, {3, 3, 0, 0}, 1, 1, {{0},{0.034052688628435135},{0.0},{8.683435440063477}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_2/Conv2d_0c_1x7/Relu#363");

    std::vector<int64_t> biasWeightsData36 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights36 = om.constantInt(biasWeightsData36,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00014864429249428213},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_2/Conv2d_0c_1x7/Relu_bias#121");
    auto bias_c36 = om.bias(conv36, biasWeights36, {{0},{0.034052688628435135},{0.0},{8.683435440063477}});

    std::vector<int64_t> weightsData37 = mv::utils::generateSequence<int64_t> (1*7*128*128);
    auto weights37 = om.constantInt(weightsData37,{1,7,128,128}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{83},{0.004249798599630594},{-0.3497583866119385},{0.7296904921531677}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_2/Conv2d_0d_7x1/Relu_weights#123");
    auto conv37 = om.conv(bias_c36, weights37, {1, 1}, {0, 0, 3, 3}, 1, 1, {{0},{0.05173087865114212},{0.0},{13.191373825073242}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_2/Conv2d_0d_7x1/Relu#364");

    std::vector<int64_t> biasWeightsData37 = mv::utils::generateSequence<int64_t> (128);
    auto biasWeights37 = om.constantInt(biasWeightsData37,{128}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00014471706526819617},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_2/Conv2d_0d_7x1/Relu_bias#124");
    auto bias_c37 = om.bias(conv37, biasWeights37, {{0},{0.05173087865114212},{0.0},{13.191373825073242}});

    std::vector<int64_t> weightsData38 = mv::utils::generateSequence<int64_t> (7*1*128*192);
    auto weights38 = om.constantInt(weightsData38,{7,1,128,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{102},{0.0032412372529506683},{-0.3261047899723053},{0.49716946482658386}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_2/Conv2d_0e_1x7/Relu_weights#126");
    auto conv38 = om.conv(bias_c37, weights38, {1, 1}, {3, 3, 0, 0}, 1, 1, {{0},{0.07881999760866165},{0.0},{20.09910011291504}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_2/Conv2d_0e_1x7/Relu#365");

    std::vector<int64_t> biasWeightsData38 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights38 = om.constantInt(biasWeightsData38,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00016767204215284437},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_2/Conv2d_0e_1x7/Relu_bias#127");
    auto bias_c38 = om.bias(conv38, biasWeights38, {{0},{0.07881999760866165},{0.0},{20.09910011291504}});

    auto pool6 = om.averagePool(concat3, {3, 3}, {1, 1}, {1, 1, 1, 1}, true, {{0},{0.04834429547190666},{0.0},{12.327795028686523}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_3/AvgPool_0a_3x3/AvgPool#356");

    std::vector<int64_t> weightsData39 = mv::utils::generateSequence<int64_t> (1*1*768*192);
    auto weights39 = om.constantInt(weightsData39,{1,1,768,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{144},{0.010166983120143414},{-1.4563055038452148},{1.1261082887649536}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_3/Conv2d_0b_1x1/Relu_weights#129");
    auto conv39 = om.conv(pool6, weights39, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.07881999760866165},{0.0},{20.09910011291504}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_3/Conv2d_0b_1x1/Relu#366");

    std::vector<int64_t> biasWeightsData39 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights39 = om.constantInt(biasWeightsData39,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0004915156168863177},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6b/Branch_3/Conv2d_0b_1x1/Relu_bias#130");
    auto bias_c39 = om.bias(conv39, biasWeights39, {{0},{0.07881999760866165},{0.0},{20.09910011291504}});

    auto concat4 = om.concat({bias_c30, bias_c33, bias_c38, bias_c39}, "C", {{0},{0.07881999760866165},{0.0},{20.09910011291504}}, "InceptionV3/InceptionV3/Mixed_6b/concat#367");

    std::vector<int64_t> weightsData40 = mv::utils::generateSequence<int64_t> (1*1*768*192);
    auto weights40 = om.constantInt(weightsData40,{1,1,768,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{115},{0.004857946652919054},{-0.5524792075157166},{0.6814392805099487}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_0/Conv2d_0a_1x1/Relu_weights#134");
    auto conv40 = om.conv(concat4, weights40, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.08443223685026169},{0.0},{21.53022003173828}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_0/Conv2d_0a_1x1/Relu#369");

    std::vector<int64_t> biasWeightsData40 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights40 = om.constantInt(biasWeightsData40,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0003829033812507987},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_0/Conv2d_0a_1x1/Relu_bias#135");
    auto bias_c40 = om.bias(conv40, biasWeights40, {{0},{0.08443223685026169},{0.0},{21.53022003173828}});

    std::vector<int64_t> weightsData41 = mv::utils::generateSequence<int64_t> (1*1*768*160);
    auto weights41 = om.constantInt(weightsData41,{1,1,768,160}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{142},{0.006818308029323816},{-0.9618598222732544},{0.7699904441833496}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_1/Conv2d_0a_1x1/Relu_weights#137");
    auto conv41 = om.conv(concat4, weights41, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.05015585571527481},{0.0},{12.789743423461914}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_1/Conv2d_0a_1x1/Relu#370");

    std::vector<int64_t> biasWeightsData41 = mv::utils::generateSequence<int64_t> (160);
    auto biasWeights41 = om.constantInt(biasWeightsData41,{160}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0005374190513975918},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_1/Conv2d_0a_1x1/Relu_bias#138");
    auto bias_c41 = om.bias(conv41, biasWeights41, {{0},{0.05015585571527481},{0.0},{12.789743423461914}});

    std::vector<int64_t> weightsData42 = mv::utils::generateSequence<int64_t> (7*1*160*160);
    auto weights42 = om.constantInt(weightsData42,{7,1,160,160}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{111},{0.004268056247383356},{-0.46856772899627686},{0.6155185103416443}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_1/Conv2d_0b_1x7/Relu_weights#140");
    auto conv42 = om.conv(bias_c41, weights42, {1, 1}, {3, 3, 0, 0}, 1, 1, {{0},{0.04093127325177193},{0.0},{10.437474250793457}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_1/Conv2d_0b_1x7/Relu#371");

    std::vector<int64_t> biasWeightsData42 = mv::utils::generateSequence<int64_t> (160);
    auto biasWeights42 = om.constantInt(biasWeightsData42,{160}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00021406800078693777},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_1/Conv2d_0b_1x7/Relu_bias#141");
    auto bias_c42 = om.bias(conv42, biasWeights42, {{0},{0.04093127325177193},{0.0},{10.437474250793457}});

    std::vector<int64_t> weightsData43 = mv::utils::generateSequence<int64_t> (1*7*160*192);
    auto weights43 = om.constantInt(weightsData43,{1,7,160,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{113},{0.004993530455976725},{-0.5583935976028442},{0.7099630832672119}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_1/Conv2d_0c_7x1/Relu_weights#143");
    auto conv43 = om.conv(bias_c42, weights43, {1, 1}, {0, 0, 3, 3}, 1, 1, {{0},{0.08443223685026169},{0.0},{21.53022003173828}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_1/Conv2d_0c_7x1/Relu#372");

    std::vector<int64_t> biasWeightsData43 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights43 = om.constantInt(biasWeightsData43,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00020439154468476772},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_1/Conv2d_0c_7x1/Relu_bias#144");
    auto bias_c43 = om.bias(conv43, biasWeights43, {{0},{0.08443223685026169},{0.0},{21.53022003173828}});

    std::vector<int64_t> weightsData44 = mv::utils::generateSequence<int64_t> (1*1*768*160);
    auto weights44 = om.constantInt(weightsData44,{1,1,768,160}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{106},{0.005904658231884241},{-0.6179603338241577},{0.8818228244781494}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_2/Conv2d_0a_1x1/Relu_weights#146");
    auto conv44 = om.conv(concat4, weights44, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.044526606798172},{0.0},{11.354284286499023}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_2/Conv2d_0a_1x1/Relu#373");

    std::vector<int64_t> biasWeightsData44 = mv::utils::generateSequence<int64_t> (160);
    auto biasWeights44 = om.constantInt(biasWeightsData44,{160}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00046540514449588954},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_2/Conv2d_0a_1x1/Relu_bias#147");
    auto bias_c44 = om.bias(conv44, biasWeights44, {{0},{0.044526606798172},{0.0},{11.354284286499023}});

    std::vector<int64_t> weightsData45 = mv::utils::generateSequence<int64_t> (1*7*160*160);
    auto weights45 = om.constantInt(weightsData45,{1,7,160,160}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{104},{0.0034263664856553078},{-0.35174331068992615},{0.5185537934303284}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_2/Conv2d_0b_7x1/Relu_weights#149");
    auto conv45 = om.conv(bias_c44, weights45, {1, 1}, {0, 0, 3, 3}, 1, 1, {{0},{0.05707687884569168},{0.0},{14.554604530334473}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_2/Conv2d_0b_7x1/Relu#374");

    std::vector<int64_t> biasWeightsData45 = mv::utils::generateSequence<int64_t> (160);
    auto biasWeights45 = om.constantInt(biasWeightsData45,{160}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00015256447659339756},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_2/Conv2d_0b_7x1/Relu_bias#150");
    auto bias_c45 = om.bias(conv45, biasWeights45, {{0},{0.05707687884569168},{0.0},{14.554604530334473}});

    std::vector<int64_t> weightsData46 = mv::utils::generateSequence<int64_t> (7*1*160*160);
    auto weights46 = om.constantInt(weightsData46,{7,1,160,160}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{99},{0.004499785602092743},{-0.43955039978027344},{0.7033951282501221}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_2/Conv2d_0c_1x7/Relu_weights#152");
    auto conv46 = om.conv(bias_c45, weights46, {1, 1}, {3, 3, 0, 0}, 1, 1, {{0},{0.060697637498378754},{0.0},{15.477897644042969}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_2/Conv2d_0c_1x7/Relu#375");

    std::vector<int64_t> biasWeightsData46 = mv::utils::generateSequence<int64_t> (160);
    auto biasWeights46 = om.constantInt(biasWeightsData46,{160}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0002568337076809257},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_2/Conv2d_0c_1x7/Relu_bias#153");
    auto bias_c46 = om.bias(conv46, biasWeights46, {{0},{0.060697637498378754},{0.0},{15.477897644042969}});

    std::vector<int64_t> weightsData47 = mv::utils::generateSequence<int64_t> (1*7*160*160);
    auto weights47 = om.constantInt(weightsData47,{1,7,160,160}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{98},{0.003951098769903183},{-0.3815431594848633},{0.6220359802246094}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_2/Conv2d_0d_7x1/Relu_weights#155");
    auto conv47 = om.conv(bias_c46, weights47, {1, 1}, {0, 0, 3, 3}, 1, 1, {{0},{0.08116589486598969},{0.0},{20.697303771972656}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_2/Conv2d_0d_7x1/Relu#376");

    std::vector<int64_t> biasWeightsData47 = mv::utils::generateSequence<int64_t> (160);
    auto biasWeights47 = om.constantInt(biasWeightsData47,{160}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0002398223732598126},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_2/Conv2d_0d_7x1/Relu_bias#156");
    auto bias_c47 = om.bias(conv47, biasWeights47, {{0},{0.08116589486598969},{0.0},{20.697303771972656}});

    std::vector<int64_t> weightsData48 = mv::utils::generateSequence<int64_t> (7*1*160*192);
    auto weights48 = om.constantInt(weightsData48,{7,1,160,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{96},{0.003994532395154238},{-0.38077834248542786},{0.6338329315185547}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_2/Conv2d_0e_1x7/Relu_weights#158");
    auto conv48 = om.conv(bias_c47, weights48, {1, 1}, {3, 3, 0, 0}, 1, 1, {{0},{0.08443223685026169},{0.0},{21.53022003173828}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_2/Conv2d_0e_1x7/Relu#377");

    std::vector<int64_t> biasWeightsData48 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights48 = om.constantInt(biasWeightsData48,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0003242198145017028},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_2/Conv2d_0e_1x7/Relu_bias#159");
    auto bias_c48 = om.bias(conv48, biasWeights48, {{0},{0.08443223685026169},{0.0},{21.53022003173828}});

    auto pool7 = om.averagePool(concat4, {3, 3}, {1, 1}, {1, 1, 1, 1}, true, {{0},{0.07881999760866165},{0.0},{20.09910011291504}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_3/AvgPool_0a_3x3/AvgPool#368");

    std::vector<int64_t> weightsData49 = mv::utils::generateSequence<int64_t> (1*1*768*192);
    auto weights49 = om.constantInt(weightsData49,{1,1,768,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{123},{0.007271741051226854},{-0.8853006362915039},{0.9617215991020203}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_3/Conv2d_0b_1x1/Relu_weights#161");
    auto conv49 = om.conv(pool7, weights49, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.08443223685026169},{0.0},{21.53022003173828}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_3/Conv2d_0b_1x1/Relu#378");

    std::vector<int64_t> biasWeightsData49 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights49 = om.constantInt(biasWeightsData49,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0005731586134061217},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6c/Branch_3/Conv2d_0b_1x1/Relu_bias#162");
    auto bias_c49 = om.bias(conv49, biasWeights49, {{0},{0.08443223685026169},{0.0},{21.53022003173828}});

    auto concat5 = om.concat({bias_c40, bias_c43, bias_c48, bias_c49}, "C", {{0},{0.08443223685026169},{0.0},{21.53022003173828}}, "InceptionV3/InceptionV3/Mixed_6c/concat#379");

    std::vector<int64_t> weightsData50 = mv::utils::generateSequence<int64_t> (1*1*768*192);
    auto weights50 = om.constantInt(weightsData50,{1,1,768,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{156},{0.007884848862886429},{-1.2210692167282104},{0.7816823720932007}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_0/Conv2d_0a_1x1/Relu_weights#166");
    auto conv50 = om.conv(concat5, weights50, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.053837697952985764},{0.0},{13.728612899780273}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_0/Conv2d_0a_1x1/Relu#381");

    std::vector<int64_t> biasWeightsData50 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights50 = om.constantInt(biasWeightsData50,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0006657353951595724},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_0/Conv2d_0a_1x1/Relu_bias#167");
    auto bias_c50 = om.bias(conv50, biasWeights50, {{0},{0.053837697952985764},{0.0},{13.728612899780273}});

    std::vector<int64_t> weightsData51 = mv::utils::generateSequence<int64_t> (1*1*768*160);
    auto weights51 = om.constantInt(weightsData51,{1,1,768,160}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{135},{0.004366664215922356},{-0.586275041103363},{0.5228577256202698}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_1/Conv2d_0a_1x1/Relu_weights#169");
    auto conv51 = om.conv(concat5, weights51, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.0726492702960968},{0.0},{18.525564193725586}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_1/Conv2d_0a_1x1/Relu#382");

    std::vector<int64_t> biasWeightsData51 = mv::utils::generateSequence<int64_t> (160);
    auto biasWeights51 = om.constantInt(biasWeightsData51,{160}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0003686872369144112},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_1/Conv2d_0a_1x1/Relu_bias#170");
    auto bias_c51 = om.bias(conv51, biasWeights51, {{0},{0.0726492702960968},{0.0},{18.525564193725586}});

    std::vector<int64_t> weightsData52 = mv::utils::generateSequence<int64_t> (7*1*160*160);
    auto weights52 = om.constantInt(weightsData52,{7,1,160,160}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{77},{0.006336270831525326},{-0.4825599789619446},{1.1268528699874878}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_1/Conv2d_0b_1x7/Relu_weights#172");
    auto conv52 = om.conv(bias_c51, weights52, {1, 1}, {3, 3, 0, 0}, 1, 1, {{0},{0.06458134949207306},{0.0},{16.468244552612305}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_1/Conv2d_0b_1x7/Relu#383");

    std::vector<int64_t> biasWeightsData52 = mv::utils::generateSequence<int64_t> (160);
    auto biasWeights52 = om.constantInt(biasWeightsData52,{160}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00046032547834329307},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_1/Conv2d_0b_1x7/Relu_bias#173");
    auto bias_c52 = om.bias(conv52, biasWeights52, {{0},{0.06458134949207306},{0.0},{16.468244552612305}});

    std::vector<int64_t> weightsData53 = mv::utils::generateSequence<int64_t> (1*7*160*192);
    auto weights53 = om.constantInt(weightsData53,{1,7,160,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{102},{0.004404191859066486},{-0.4468778073787689},{0.6717869639396667}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_1/Conv2d_0c_7x1/Relu_weights#175");
    auto conv53 = om.conv(bias_c52, weights53, {1, 1}, {0, 0, 3, 3}, 1, 1, {{0},{0.053837697952985764},{0.0},{13.728612899780273}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_1/Conv2d_0c_7x1/Relu#384");

    std::vector<int64_t> biasWeightsData53 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights53 = om.constantInt(biasWeightsData53,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00028442867915146053},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_1/Conv2d_0c_7x1/Relu_bias#176");
    auto bias_c53 = om.bias(conv53, biasWeights53, {{0},{0.053837697952985764},{0.0},{13.728612899780273}});

    std::vector<int64_t> weightsData54 = mv::utils::generateSequence<int64_t> (1*1*768*160);
    auto weights54 = om.constantInt(weightsData54,{1,1,768,160}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{111},{0.005913903936743736},{-0.6528329849243164},{0.8492985963821411}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_2/Conv2d_0a_1x1/Relu_weights#178");
    auto conv54 = om.conv(concat5, weights54, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.06219832971692085},{0.0},{15.860573768615723}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_2/Conv2d_0a_1x1/Relu#385");

    std::vector<int64_t> biasWeightsData54 = mv::utils::generateSequence<int64_t> (160);
    auto biasWeights54 = om.constantInt(biasWeightsData54,{160}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0004993241163901985},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_2/Conv2d_0a_1x1/Relu_bias#179");
    auto bias_c54 = om.bias(conv54, biasWeights54, {{0},{0.06219832971692085},{0.0},{15.860573768615723}});

    std::vector<int64_t> weightsData55 = mv::utils::generateSequence<int64_t> (1*7*160*160);
    auto weights55 = om.constantInt(weightsData55,{1,7,160,160}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{94},{0.005815011914819479},{-0.543221116065979},{0.9337919354438782}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_2/Conv2d_0b_7x1/Relu_weights#181");
    auto conv55 = om.conv(bias_c54, weights55, {1, 1}, {0, 0, 3, 3}, 1, 1, {{0},{0.05706518515944481},{0.0},{14.55162239074707}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_2/Conv2d_0b_7x1/Relu#386");

    std::vector<int64_t> biasWeightsData55 = mv::utils::generateSequence<int64_t> (160);
    auto biasWeights55 = om.constantInt(biasWeightsData55,{160}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00036168404039926827},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_2/Conv2d_0b_7x1/Relu_bias#182");
    auto bias_c55 = om.bias(conv55, biasWeights55, {{0},{0.05706518515944481},{0.0},{14.55162239074707}});

    std::vector<int64_t> weightsData56 = mv::utils::generateSequence<int64_t> (7*1*160*160);
    auto weights56 = om.constantInt(weightsData56,{7,1,160,160}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{108},{0.004232329782098532},{-0.4541642963886261},{0.6208474636077881}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_2/Conv2d_0c_1x7/Relu_weights#184");
    auto conv56 = om.conv(bias_c55, weights56, {1, 1}, {3, 3, 0, 0}, 1, 1, {{0},{0.04300113767385483},{0.0},{10.965290069580078}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_2/Conv2d_0c_1x7/Relu#387");

    std::vector<int64_t> biasWeightsData56 = mv::utils::generateSequence<int64_t> (160);
    auto biasWeights56 = om.constantInt(biasWeightsData56,{160}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00024151869001798332},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_2/Conv2d_0c_1x7/Relu_bias#185");
    auto bias_c56 = om.bias(conv56, biasWeights56, {{0},{0.04300113767385483},{0.0},{10.965290069580078}});

    std::vector<int64_t> weightsData57 = mv::utils::generateSequence<int64_t> (1*7*160*160);
    auto weights57 = om.constantInt(weightsData57,{1,7,160,160}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{149},{0.0064451307989656925},{-0.950759768486023},{0.6863034963607788}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_2/Conv2d_0d_7x1/Relu_weights#187");
    auto conv57 = om.conv(bias_c56, weights57, {1, 1}, {0, 0, 3, 3}, 1, 1, {{0},{0.04716057702898979},{0.0},{12.025947570800781}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_2/Conv2d_0d_7x1/Relu#388");

    std::vector<int64_t> biasWeightsData57 = mv::utils::generateSequence<int64_t> (160);
    auto biasWeights57 = om.constantInt(biasWeightsData57,{160}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0002771479485090822},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_2/Conv2d_0d_7x1/Relu_bias#188");
    auto bias_c57 = om.bias(conv57, biasWeights57, {{0},{0.04716057702898979},{0.0},{12.025947570800781}});

    std::vector<int64_t> weightsData58 = mv::utils::generateSequence<int64_t> (7*1*160*192);
    auto weights58 = om.constantInt(weightsData58,{7,1,160,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{157},{0.006102328188717365},{-0.9521626830101013},{0.5978286862373352}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_2/Conv2d_0e_1x7/Relu_weights#190");
    auto conv58 = om.conv(bias_c57, weights58, {1, 1}, {3, 3, 0, 0}, 1, 1, {{0},{0.053837697952985764},{0.0},{13.728612899780273}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_2/Conv2d_0e_1x7/Relu#389");

    std::vector<int64_t> biasWeightsData58 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights58 = om.constantInt(biasWeightsData58,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00028778932755813},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_2/Conv2d_0e_1x7/Relu_bias#191");
    auto bias_c58 = om.bias(conv58, biasWeights58, {{0},{0.053837697952985764},{0.0},{13.728612899780273}});

    auto pool8 = om.averagePool(concat5, {3, 3}, {1, 1}, {1, 1, 1, 1}, true, {{0},{0.08443223685026169},{0.0},{21.53022003173828}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_3/AvgPool_0a_3x3/AvgPool#380");

    std::vector<int64_t> weightsData59 = mv::utils::generateSequence<int64_t> (1*1*768*192);
    auto weights59 = om.constantInt(weightsData59,{1,1,768,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{154},{0.013549362309277058},{-2.0706124305725098},{1.3709255456924438}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_3/Conv2d_0b_1x1/Relu_weights#193");
    auto conv59 = om.conv(pool8, weights59, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.053837697952985764},{0.0},{13.728612899780273}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_3/Conv2d_0b_1x1/Relu#390");

    std::vector<int64_t> biasWeightsData59 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights59 = om.constantInt(biasWeightsData59,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.001144002890214324},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6d/Branch_3/Conv2d_0b_1x1/Relu_bias#194");
    auto bias_c59 = om.bias(conv59, biasWeights59, {{0},{0.053837697952985764},{0.0},{13.728612899780273}});

    auto concat6 = om.concat({bias_c50, bias_c53, bias_c58, bias_c59}, "C", {{0},{0.053837697952985764},{0.0},{13.728612899780273}}, "InceptionV3/InceptionV3/Mixed_6d/concat#391");

    std::vector<int64_t> weightsData60 = mv::utils::generateSequence<int64_t> (1*1*768*192);
    auto weights60 = om.constantInt(weightsData60,{1,1,768,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{112},{0.006958557292819023},{-0.7740282416343689},{0.9934453368186951}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_0/Conv2d_0a_1x1/Relu_weights#198");
    auto conv60 = om.conv(concat6, weights60, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.05141126736998558},{0.0},{13.109872817993164}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_0/Conv2d_0a_1x1/Relu#393");

    std::vector<int64_t> biasWeightsData60 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights60 = om.constantInt(biasWeightsData60,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00037463271291926503},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_0/Conv2d_0a_1x1/Relu_bias#199");
    auto bias_c60 = om.bias(conv60, biasWeights60, {{0},{0.05141126736998558},{0.0},{13.109872817993164}});

    std::vector<int64_t> weightsData61 = mv::utils::generateSequence<int64_t> (1*1*768*192);
    auto weights61 = om.constantInt(weightsData61,{1,1,768,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{97},{0.0061390940099954605},{-0.5921565294265747},{0.9671733975410461}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_1/Conv2d_0a_1x1/Relu_weights#201");
    auto conv61 = om.conv(concat6, weights61, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.04406578838825226},{0.0},{11.236776351928711}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_1/Conv2d_0a_1x1/Relu#394");

    std::vector<int64_t> biasWeightsData61 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights61 = om.constantInt(biasWeightsData61,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0003305147110950202},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_1/Conv2d_0a_1x1/Relu_bias#202");
    auto bias_c61 = om.bias(conv61, biasWeights61, {{0},{0.04406578838825226},{0.0},{11.236776351928711}});

    std::vector<int64_t> weightsData62 = mv::utils::generateSequence<int64_t> (7*1*192*192);
    auto weights62 = om.constantInt(weightsData62,{7,1,192,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{72},{0.005854951683431864},{-0.4169313907623291},{1.0702263116836548}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_1/Conv2d_0b_1x7/Relu_weights#204");
    auto conv62 = om.conv(bias_c61, weights62, {1, 1}, {3, 3, 0, 0}, 1, 1, {{0},{0.05123240500688553},{0.0},{13.064263343811035}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_1/Conv2d_0b_1x7/Relu#395");

    std::vector<int64_t> biasWeightsData62 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights62 = om.constantInt(biasWeightsData62,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00025800307048484683},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_1/Conv2d_0b_1x7/Relu_bias#205");
    auto bias_c62 = om.bias(conv62, biasWeights62, {{0},{0.05123240500688553},{0.0},{13.064263343811035}});

    std::vector<int64_t> weightsData63 = mv::utils::generateSequence<int64_t> (1*7*192*192);
    auto weights63 = om.constantInt(weightsData63,{1,7,192,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{121},{0.002567456802353263},{-0.3081381618976593},{0.3439958393573761}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_1/Conv2d_0c_7x1/Relu_weights#207");
    auto conv63 = om.conv(bias_c62, weights63, {1, 1}, {0, 0, 3, 3}, 1, 1, {{0},{0.05141126736998558},{0.0},{13.109872817993164}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_1/Conv2d_0c_7x1/Relu#396");

    std::vector<int64_t> biasWeightsData63 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights63 = om.constantInt(biasWeightsData63,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00013153698819223791},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_1/Conv2d_0c_7x1/Relu_bias#208");
    auto bias_c63 = om.bias(conv63, biasWeights63, {{0},{0.05141126736998558},{0.0},{13.109872817993164}});

    std::vector<int64_t> weightsData64 = mv::utils::generateSequence<int64_t> (1*1*768*192);
    auto weights64 = om.constantInt(weightsData64,{1,1,768,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{130},{0.0046445284970104694},{-0.6008613705635071},{0.5788488984107971}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_2/Conv2d_0a_1x1/Relu_weights#210");
    auto conv64 = om.conv(concat6, weights64, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.0520210899412632},{0.0},{13.26537799835205}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_2/Conv2d_0a_1x1/Relu#397");

    std::vector<int64_t> biasWeightsData64 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights64 = om.constantInt(biasWeightsData64,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0002500507398508489},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_2/Conv2d_0a_1x1/Relu_bias#211");
    auto bias_c64 = om.bias(conv64, biasWeights64, {{0},{0.0520210899412632},{0.0},{13.26537799835205}});

    std::vector<int64_t> weightsData65 = mv::utils::generateSequence<int64_t> (1*7*192*192);
    auto weights65 = om.constantInt(weightsData65,{1,7,192,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{94},{0.003432407509535551},{-0.31881389021873474},{0.5530176162719727}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_2/Conv2d_0b_7x1/Relu_weights#213");
    auto conv65 = om.conv(bias_c64, weights65, {1, 1}, {0, 0, 3, 3}, 1, 1, {{0},{0.04245786368846893},{0.0},{10.82675552368164}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_2/Conv2d_0b_7x1/Relu#398");

    std::vector<int64_t> biasWeightsData65 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights65 = om.constantInt(biasWeightsData65,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00017855757323559374},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_2/Conv2d_0b_7x1/Relu_bias#214");
    auto bias_c65 = om.bias(conv65, biasWeights65, {{0},{0.04245786368846893},{0.0},{10.82675552368164}});

    std::vector<int64_t> weightsData66 = mv::utils::generateSequence<int64_t> (7*1*192*192);
    auto weights66 = om.constantInt(weightsData66,{7,1,192,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{102},{0.004284923430532217},{-0.4336787760257721},{0.654691755771637}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_2/Conv2d_0c_1x7/Relu_weights#216");
    auto conv66 = om.conv(bias_c65, weights66, {1, 1}, {3, 3, 0, 0}, 1, 1, {{0},{0.04603631794452667},{0.0},{11.73926067352295}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_2/Conv2d_0c_1x7/Relu#399");

    std::vector<int64_t> biasWeightsData66 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights66 = om.constantInt(biasWeightsData66,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00018192869902122766},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_2/Conv2d_0c_1x7/Relu_bias#217");
    auto bias_c66 = om.bias(conv66, biasWeights66, {{0},{0.04603631794452667},{0.0},{11.73926067352295}});

    std::vector<int64_t> weightsData67 = mv::utils::generateSequence<int64_t> (1*7*192*192);
    auto weights67 = om.constantInt(weightsData67,{1,7,192,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{112},{0.0029152212664484978},{-0.3236146867275238},{0.4168515205383301}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_2/Conv2d_0d_7x1/Relu_weights#219");
    auto conv67 = om.conv(bias_c66, weights67, {1, 1}, {0, 0, 3, 3}, 1, 1, {{0},{0.04794910177588463},{0.0},{12.227021217346191}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_2/Conv2d_0d_7x1/Relu#400");

    std::vector<int64_t> biasWeightsData67 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights67 = om.constantInt(biasWeightsData67,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00013420604227576405},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_2/Conv2d_0d_7x1/Relu_bias#220");
    auto bias_c67 = om.bias(conv67, biasWeights67, {{0},{0.04794910177588463},{0.0},{12.227021217346191}});

    std::vector<int64_t> weightsData68 = mv::utils::generateSequence<int64_t> (7*1*192*192);
    auto weights68 = om.constantInt(weightsData68,{7,1,192,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{103},{0.0022881305776536465},{-0.23450914025306702},{0.3466760218143463}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_2/Conv2d_0e_1x7/Relu_weights#222");
    auto conv68 = om.conv(bias_c67, weights68, {1, 1}, {3, 3, 0, 0}, 1, 1, {{0},{0.05141126736998558},{0.0},{13.109872817993164}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_2/Conv2d_0e_1x7/Relu#401");

    std::vector<int64_t> biasWeightsData68 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights68 = om.constantInt(biasWeightsData68,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00010971380834234878},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_2/Conv2d_0e_1x7/Relu_bias#223");
    auto bias_c68 = om.bias(conv68, biasWeights68, {{0},{0.05141126736998558},{0.0},{13.109872817993164}});

    auto pool9 = om.averagePool(concat6, {3, 3}, {1, 1}, {1, 1, 1, 1}, true, {{0},{0.053837697952985764},{0.0},{13.728612899780273}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_3/AvgPool_0a_3x3/AvgPool#392");

    std::vector<int64_t> weightsData69 = mv::utils::generateSequence<int64_t> (1*1*768*192);
    auto weights69 = om.constantInt(weightsData69,{1,1,768,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{126},{0.00730943726375699},{-0.9128758311271667},{0.9437211751937866}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_3/Conv2d_0b_1x1/Relu_weights#225");
    auto conv69 = om.conv(pool9, weights69, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.05141126736998558},{0.0},{13.109872817993164}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_3/Conv2d_0b_1x1/Relu#402");

    std::vector<int64_t> biasWeightsData69 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights69 = om.constantInt(biasWeightsData69,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.000393523252569139},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_6e/Branch_3/Conv2d_0b_1x1/Relu_bias#226");
    auto bias_c69 = om.bias(conv69, biasWeights69, {{0},{0.05141126736998558},{0.0},{13.109872817993164}});

    auto concat7 = om.concat({bias_c60, bias_c63, bias_c68, bias_c69}, "C", {{0},{0.05141126736998558},{0.0},{13.109872817993164}}, "InceptionV3/InceptionV3/Mixed_6e/concat#403");

    std::vector<int64_t> weightsData70 = mv::utils::generateSequence<int64_t> (1*1*768*192);
    auto weights70 = om.constantInt(weightsData70,{1,1,768,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{115},{0.006470506079494953},{-0.7351618409156799},{0.9083467125892639}}, "InceptionV3/InceptionV3/Mixed_7a/Branch_0/Conv2d_0a_1x1/Relu_weights#230");
    auto conv70 = om.conv(concat7, weights70, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.04326530918478966},{0.0},{11.03265380859375}}, "InceptionV3/InceptionV3/Mixed_7a/Branch_0/Conv2d_0a_1x1/Relu#405");

    std::vector<int64_t> biasWeightsData70 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights70 = om.constantInt(biasWeightsData70,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00033265689853578806},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7a/Branch_0/Conv2d_0a_1x1/Relu_bias#231");
    auto bias_c70 = om.bias(conv70, biasWeights70, {{0},{0.04326530918478966},{0.0},{11.03265380859375}});

    std::vector<int64_t> weightsData71 = mv::utils::generateSequence<int64_t> (3*3*192*320);
    auto weights71 = om.constantInt(weightsData71,{3,3,192,320}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{79},{0.003502116771414876},{-0.2725469172000885},{0.6169907450675964}}, "InceptionV3/InceptionV3/Mixed_7a/Branch_0/Conv2d_1a_3x3/Relu_weights#233");
    auto conv71 = om.conv(bias_c70, weights71, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.05141126736998558},{0.0},{13.109872817993164}}, "InceptionV3/InceptionV3/Mixed_7a/Branch_0/Conv2d_1a_3x3/Relu#406");

    std::vector<int64_t> biasWeightsData71 = mv::utils::generateSequence<int64_t> (320);
    auto biasWeights71 = om.constantInt(biasWeightsData71,{320}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0001515201583970338},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7a/Branch_0/Conv2d_1a_3x3/Relu_bias#234");
    auto bias_c71 = om.bias(conv71, biasWeights71, {{0},{0.05141126736998558},{0.0},{13.109872817993164}});

    std::vector<int64_t> weightsData72 = mv::utils::generateSequence<int64_t> (1*1*768*192);
    auto weights72 = om.constantInt(weightsData72,{1,1,768,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{116},{0.00858816783875227},{-0.9893528819084167},{1.1920417547225952}}, "InceptionV3/InceptionV3/Mixed_7a/Branch_1/Conv2d_0a_1x1/Relu_weights#236");
    auto conv72 = om.conv(concat7, weights72, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.058338236063718796},{0.0},{14.876250267028809}}, "InceptionV3/InceptionV3/Mixed_7a/Branch_1/Conv2d_0a_1x1/Relu#407");

    std::vector<int64_t> biasWeightsData72 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights72 = om.constantInt(biasWeightsData72,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00044152859481982887},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7a/Branch_1/Conv2d_0a_1x1/Relu_bias#237");
    auto bias_c72 = om.bias(conv72, biasWeights72, {{0},{0.058338236063718796},{0.0},{14.876250267028809}});

    std::vector<int64_t> weightsData73 = mv::utils::generateSequence<int64_t> (7*1*192*192);
    auto weights73 = om.constantInt(weightsData73,{7,1,192,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{116},{0.003553818678483367},{-0.4100940525531769},{0.49257591366767883}}, "InceptionV3/InceptionV3/Mixed_7a/Branch_1/Conv2d_0b_1x7/Relu_weights#239");
    auto conv73 = om.conv(bias_c72, weights73, {1, 1}, {3, 3, 0, 0}, 1, 1, {{0},{0.05407462269067764},{0.0},{13.789029121398926}}, "InceptionV3/InceptionV3/Mixed_7a/Branch_1/Conv2d_0b_1x7/Relu#408");

    std::vector<int64_t> biasWeightsData73 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights73 = om.constantInt(biasWeightsData73,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00020732352277264},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7a/Branch_1/Conv2d_0b_1x7/Relu_bias#240");
    auto bias_c73 = om.bias(conv73, biasWeights73, {{0},{0.05407462269067764},{0.0},{13.789029121398926}});

    std::vector<int64_t> weightsData74 = mv::utils::generateSequence<int64_t> (1*7*192*192);
    auto weights74 = om.constantInt(weightsData74,{1,7,192,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{120},{0.0028677876107394695},{-0.3426341116428375},{0.38578397035598755}}, "InceptionV3/InceptionV3/Mixed_7a/Branch_1/Conv2d_0c_7x1/Relu_weights#242");
    auto conv74 = om.conv(bias_c73, weights74, {1, 1}, {0, 0, 3, 3}, 1, 1, {{0},{0.035801906138658524},{0.0},{9.129486083984375}}, "InceptionV3/InceptionV3/Mixed_7a/Branch_1/Conv2d_0c_7x1/Relu#409");

    std::vector<int64_t> biasWeightsData74 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights74 = om.constantInt(biasWeightsData74,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00015507453645113856},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7a/Branch_1/Conv2d_0c_7x1/Relu_bias#243");
    auto bias_c74 = om.bias(conv74, biasWeights74, {{0},{0.035801906138658524},{0.0},{9.129486083984375}});

    std::vector<int64_t> weightsData75 = mv::utils::generateSequence<int64_t> (3*3*192*192);
    auto weights75 = om.constantInt(weightsData75,{3,3,192,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{93},{0.005856209900230169},{-0.5412960648536682},{0.9461812973022461}}, "InceptionV3/InceptionV3/Mixed_7a/Branch_1/Conv2d_1a_3x3/Relu_weights#245");
    auto conv75 = om.conv(bias_c74, weights75, {2, 2}, {0, 0, 0, 0}, 1, 1, {{0},{0.05141126736998558},{0.0},{13.109872817993164}}, "InceptionV3/InceptionV3/Mixed_7a/Branch_1/Conv2d_1a_3x3/Relu#410");

    std::vector<int64_t> biasWeightsData75 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights75 = om.constantInt(biasWeightsData75,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0002096634852932766},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7a/Branch_1/Conv2d_1a_3x3/Relu_bias#246");
    auto bias_c75 = om.bias(conv75, biasWeights75, {{0},{0.05141126736998558},{0.0},{13.109872817993164}});

    auto pool10 = om.maxPool(concat7, {3, 3}, {2, 2}, {0, 0, 0, 0}, true, {{0},{0.05141126736998558},{0.0},{13.109872817993164}}, "InceptionV3/InceptionV3/Mixed_7a/Branch_2/MaxPool_1a_3x3/MaxPool#404");

    auto concat8 = om.concat({bias_c71, bias_c75, pool10}, "C", {{0},{0.05141126736998558},{0.0},{13.109872817993164}}, "InceptionV3/InceptionV3/Mixed_7a/concat#411");

    std::vector<int64_t> weightsData76 = mv::utils::generateSequence<int64_t> (1*1*1280*320);
    auto weights76 = om.constantInt(weightsData76,{1,1,1280,320}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{107},{0.0062094880267977715},{-0.6554938554763794},{0.9217161536216736}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_0/Conv2d_0a_1x1/Relu_weights#250");
    auto conv76 = om.conv(concat8, weights76, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.03830382600426674},{0.0},{9.767476081848145}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_0/Conv2d_0a_1x1/Relu#413");

    std::vector<int64_t> biasWeightsData76 = mv::utils::generateSequence<int64_t> (320);
    auto biasWeights76 = om.constantInt(biasWeightsData76,{320}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00031923764618113637},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_0/Conv2d_0a_1x1/Relu_bias#251");
    auto bias_c76 = om.bias(conv76, biasWeights76, {{0},{0.03830382600426674},{0.0},{9.767476081848145}});

    std::vector<int64_t> weightsData77 = mv::utils::generateSequence<int64_t> (1*1*1280*384);
    auto weights77 = om.constantInt(weightsData77,{1,1,1280,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{79},{0.004471372347325087},{-0.34916406869888306},{0.7865645289421082}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_1/Conv2d_0a_1x1/Relu_weights#253");
    auto conv77 = om.conv(concat8, weights77, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.03215353563427925},{0.0},{8.199151992797852}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_1/Conv2d_0a_1x1/Relu#414");

    std::vector<int64_t> biasWeightsData77 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights77 = om.constantInt(biasWeightsData77,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00022987891861703247},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_1/Conv2d_0a_1x1/Relu_bias#254");
    auto bias_c77 = om.bias(conv77, biasWeights77, {{0},{0.03215353563427925},{0.0},{8.199151992797852}});

    std::vector<int64_t> weightsData78 = mv::utils::generateSequence<int64_t> (3*1*384*384);
    auto weights78 = om.constantInt(weightsData78,{3,1,384,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{88},{0.003912406042218208},{-0.3412354588508606},{0.652515709400177}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_1/Conv2d_0b_1x3/Relu_weights#256");
    auto conv78 = om.conv(bias_c77, weights78, {1, 1}, {1, 1, 0, 0}, 1, 1, {{0},{0.03830382600426674},{0.0},{9.767476081848145}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_1/Conv2d_0b_1x3/Relu#415");

    std::vector<int64_t> biasWeightsData78 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights78 = om.constantInt(biasWeightsData78,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0001257976982742548},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_1/Conv2d_0b_1x3/Relu_bias#257");
    auto bias_c78 = om.bias(conv78, biasWeights78, {{0},{0.03830382600426674},{0.0},{9.767476081848145}});

    std::vector<int64_t> weightsData79 = mv::utils::generateSequence<int64_t> (1*3*384*384);
    auto weights79 = om.constantInt(weightsData79,{1,3,384,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{91},{0.006459605880081654},{-0.5832034945487976},{1.0575363636016846}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_1/Conv2d_0b_3x1/Relu_weights#259");
    auto conv79 = om.conv(bias_c77, weights79, {1, 1}, {0, 0, 1, 1}, 1, 1, {{0},{0.03830382600426674},{0.0},{9.767476081848145}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_1/Conv2d_0b_3x1/Relu#416");

    std::vector<int64_t> biasWeightsData79 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights79 = om.constantInt(biasWeightsData79,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00020769918046426028},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_1/Conv2d_0b_3x1/Relu_bias#260");
    auto bias_c79 = om.bias(conv79, biasWeights79, {{0},{0.03830382600426674},{0.0},{9.767476081848145}});

    auto concat9 = om.concat({bias_c78, bias_c79}, "C", {{0},{0.03830382600426674},{0.0},{9.767476081848145}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_1/concat#417");

    std::vector<int64_t> weightsData80 = mv::utils::generateSequence<int64_t> (1*1*1280*448);
    auto weights80 = om.constantInt(weightsData80,{1,1,1280,448}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{98},{0.004376340191811323},{-0.42393532395362854},{0.6876550316810608}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_2/Conv2d_0a_1x1/Relu_weights#263");
    auto conv80 = om.conv(concat8, weights80, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.033504221588373184},{0.0},{8.54357624053955}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_2/Conv2d_0a_1x1/Relu#418");

    std::vector<int64_t> biasWeightsData80 = mv::utils::generateSequence<int64_t> (448);
    auto biasWeights80 = om.constantInt(biasWeightsData80,{448}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00022499318583868444},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_2/Conv2d_0a_1x1/Relu_bias#264");
    auto bias_c80 = om.bias(conv80, biasWeights80, {{0},{0.033504221588373184},{0.0},{8.54357624053955}});

    std::vector<int64_t> weightsData81 = mv::utils::generateSequence<int64_t> (3*3*448*384);
    auto weights81 = om.constantInt(weightsData81,{3,3,448,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{101},{0.0021207702811807394},{-0.2112676352262497},{0.3274080157279968}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_2/Conv2d_0b_3x3/Relu_weights#266");
    auto conv81 = om.conv(bias_c80, weights81, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.029130704700946808},{0.0},{7.4283294677734375}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_2/Conv2d_0b_3x3/Relu#419");

    std::vector<int64_t> biasWeightsData81 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights81 = om.constantInt(biasWeightsData81,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{7.105475378921255e-05},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_2/Conv2d_0b_3x3/Relu_bias#267");
    auto bias_c81 = om.bias(conv81, biasWeights81, {{0},{0.029130704700946808},{0.0},{7.4283294677734375}});

    std::vector<int64_t> weightsData82 = mv::utils::generateSequence<int64_t> (3*1*384*384);
    auto weights82 = om.constantInt(weightsData82,{3,1,384,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{116},{0.004879996180534363},{-0.563081681728363},{0.6764373779296875}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_2/Conv2d_0c_1x3/Relu_weights#269");
    auto conv82 = om.conv(bias_c81, weights82, {1, 1}, {1, 1, 0, 0}, 1, 1, {{0},{0.03830382600426674},{0.0},{9.767476081848145}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_2/Conv2d_0c_1x3/Relu#420");

    std::vector<int64_t> biasWeightsData82 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights82 = om.constantInt(biasWeightsData82,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00014215773262549192},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_2/Conv2d_0c_1x3/Relu_bias#270");
    auto bias_c82 = om.bias(conv82, biasWeights82, {{0},{0.03830382600426674},{0.0},{9.767476081848145}});

    std::vector<int64_t> weightsData83 = mv::utils::generateSequence<int64_t> (1*3*384*384);
    auto weights83 = om.constantInt(weightsData83,{1,3,384,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{103},{0.00310041313059628},{-0.31690430641174316},{0.47060060501098633}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_2/Conv2d_0d_3x1/Relu_weights#272");
    auto conv83 = om.conv(bias_c81, weights83, {1, 1}, {0, 0, 1, 1}, 1, 1, {{0},{0.03830382600426674},{0.0},{9.767476081848145}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_2/Conv2d_0d_3x1/Relu#421");

    std::vector<int64_t> biasWeightsData83 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights83 = om.constantInt(biasWeightsData83,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{9.031721128849313e-05},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_2/Conv2d_0d_3x1/Relu_bias#273");
    auto bias_c83 = om.bias(conv83, biasWeights83, {{0},{0.03830382600426674},{0.0},{9.767476081848145}});

    auto concat10 = om.concat({bias_c82, bias_c83}, "C", {{0},{0.03830382600426674},{0.0},{9.767476081848145}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_2/concat#422");

    auto pool11 = om.averagePool(concat8, {3, 3}, {1, 1}, {1, 1, 1, 1}, true, {{0},{0.05141126736998558},{0.0},{13.109872817993164}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_3/AvgPool_0a_3x3/AvgPool#412");

    std::vector<int64_t> weightsData84 = mv::utils::generateSequence<int64_t> (1*1*1280*192);
    auto weights84 = om.constantInt(weightsData84,{1,1,1280,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{100},{0.008386661298573017},{-0.8303586840629578},{1.2998533248901367}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_3/Conv2d_0b_1x1/Relu_weights#276");
    auto conv84 = om.conv(pool11, weights84, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.03830382600426674},{0.0},{9.767476081848145}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_3/Conv2d_0b_1x1/Relu#423");

    std::vector<int64_t> biasWeightsData84 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights84 = om.constantInt(biasWeightsData84,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0004311688826419413},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7b/Branch_3/Conv2d_0b_1x1/Relu_bias#277");
    auto bias_c84 = om.bias(conv84, biasWeights84, {{0},{0.03830382600426674},{0.0},{9.767476081848145}});

    auto concat11 = om.concat({bias_c76, concat9, concat10, bias_c84}, "C", {{0},{0.03830382600426674},{0.0},{9.767476081848145}}, "InceptionV3/InceptionV3/Mixed_7b/concat#424");

    std::vector<int64_t> weightsData85 = mv::utils::generateSequence<int64_t> (1*1*2048*320);
    auto weights85 = om.constantInt(weightsData85,{1,1,2048,320}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{85},{0.014261765405535698},{-1.203641414642334},{2.41884708404541}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_0/Conv2d_0a_1x1/Relu_weights#281");
    auto conv85 = om.conv(concat11, weights85, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.06592372804880142},{0.0},{16.810550689697266}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_0/Conv2d_0a_1x1/Relu#426");

    std::vector<int64_t> biasWeightsData85 = mv::utils::generateSequence<int64_t> (320);
    auto biasWeights85 = om.constantInt(biasWeightsData85,{320}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0005462802364490926},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_0/Conv2d_0a_1x1/Relu_bias#282");
    auto bias_c85 = om.bias(conv85, biasWeights85, {{0},{0.06592372804880142},{0.0},{16.810550689697266}});

    std::vector<int64_t> weightsData86 = mv::utils::generateSequence<int64_t> (1*1*2048*384);
    auto weights86 = om.constantInt(weightsData86,{1,1,2048,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{112},{0.007268023211508989},{-0.8102016448974609},{1.0358762741088867}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_1/Conv2d_0a_1x1/Relu_weights#284");
    auto conv86 = om.conv(concat11, weights86, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.0314251109957695},{0.0},{8.013402938842773}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_1/Conv2d_0a_1x1/Relu#427");

    std::vector<int64_t> biasWeightsData86 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights86 = om.constantInt(biasWeightsData86,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0002783931267913431},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_1/Conv2d_0a_1x1/Relu_bias#285");
    auto bias_c86 = om.bias(conv86, biasWeights86, {{0},{0.0314251109957695},{0.0},{8.013402938842773}});

    std::vector<int64_t> weightsData87 = mv::utils::generateSequence<int64_t> (3*1*384*384);
    auto weights87 = om.constantInt(weightsData87,{3,1,384,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{44},{0.009618469513952732},{-0.40944981575012207},{2.0336413383483887}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_1/Conv2d_0b_1x3/Relu_weights#287");
    auto conv87 = om.conv(bias_c86, weights87, {1, 1}, {1, 1, 0, 0}, 1, 1, {{0},{0.06592372804880142},{0.0},{16.810550689697266}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_1/Conv2d_0b_1x3/Relu#428");

    std::vector<int64_t> biasWeightsData87 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights87 = om.constantInt(biasWeightsData87,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0003022614400833845},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_1/Conv2d_0b_1x3/Relu_bias#288");
    auto bias_c87 = om.bias(conv87, biasWeights87, {{0},{0.06592372804880142},{0.0},{16.810550689697266}});

    std::vector<int64_t> weightsData88 = mv::utils::generateSequence<int64_t> (1*3*384*384);
    auto weights88 = om.constantInt(weightsData88,{1,3,384,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{47},{0.010438359342515469},{-0.483572393655777},{2.1677708625793457}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_1/Conv2d_0c_3x1/Relu_weights#290");
    auto conv88 = om.conv(bias_c86, weights88, {1, 1}, {0, 0, 1, 1}, 1, 1, {{0},{0.06592372804880142},{0.0},{16.810550689697266}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_1/Conv2d_0c_3x1/Relu#429");

    std::vector<int64_t> biasWeightsData88 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights88 = om.constantInt(biasWeightsData88,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00032802659552544355},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_1/Conv2d_0c_3x1/Relu_bias#291");
    auto bias_c88 = om.bias(conv88, biasWeights88, {{0},{0.06592372804880142},{0.0},{16.810550689697266}});

    auto concat12 = om.concat({bias_c87, bias_c88}, "C", {{0},{0.06592372804880142},{0.0},{16.810550689697266}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_1/concat#430");

    std::vector<int64_t> weightsData89 = mv::utils::generateSequence<int64_t> (1*1*2048*448);
    auto weights89 = om.constantInt(weightsData89,{1,1,2048,448}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{132},{0.005983275827020407},{-0.7840480804443359},{0.7357040047645569}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_2/Conv2d_0a_1x1/Relu_weights#294");
    auto conv89 = om.conv(concat11, weights89, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.03366456925868988},{0.0},{8.584465026855469}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_2/Conv2d_0a_1x1/Relu#431");

    std::vector<int64_t> biasWeightsData89 = mv::utils::generateSequence<int64_t> (448);
    auto biasWeights89 = om.constantInt(biasWeightsData89,{448}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00022918237664271146},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_2/Conv2d_0a_1x1/Relu_bias#295");
    auto bias_c89 = om.bias(conv89, biasWeights89, {{0},{0.03366456925868988},{0.0},{8.584465026855469}});

    std::vector<int64_t> weightsData90 = mv::utils::generateSequence<int64_t> (3*3*448*384);
    auto weights90 = om.constantInt(weightsData90,{3,3,448,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{76},{0.003713848302140832},{-0.27693870663642883},{0.6663787364959717}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_2/Conv2d_0b_3x3/Relu_weights#297");
    auto conv90 = om.conv(bias_c89, weights90, {1, 1}, {1, 1, 1, 1}, 1, 1, {{0},{0.035905711352825165},{0.0},{9.155956268310547}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_2/Conv2d_0b_3x3/Relu#432");

    std::vector<int64_t> biasWeightsData90 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights90 = om.constantInt(biasWeightsData90,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00012502509343903512},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_2/Conv2d_0b_3x3/Relu_bias#298");
    auto bias_c90 = om.bias(conv90, biasWeights90, {{0},{0.035905711352825165},{0.0},{9.155956268310547}});

    std::vector<int64_t> weightsData91 = mv::utils::generateSequence<int64_t> (3*1*384*384);
    auto weights91 = om.constantInt(weightsData91,{3,1,384,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{100},{0.0056028105318546295},{-0.5568640828132629},{0.8662497401237488}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_2/Conv2d_0c_1x3/Relu_weights#300");
    auto conv91 = om.conv(bias_c90, weights91, {1, 1}, {1, 1, 0, 0}, 1, 1, {{0},{0.06592372804880142},{0.0},{16.810550689697266}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_2/Conv2d_0c_1x3/Relu#433");

    std::vector<int64_t> biasWeightsData91 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights91 = om.constantInt(biasWeightsData91,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00020117289386689663},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_2/Conv2d_0c_1x3/Relu_bias#301");
    auto bias_c91 = om.bias(conv91, biasWeights91, {{0},{0.06592372804880142},{0.0},{16.810550689697266}});

    std::vector<int64_t> weightsData92 = mv::utils::generateSequence<int64_t> (1*3*384*384);
    auto weights92 = om.constantInt(weightsData92,{1,3,384,384}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{81},{0.0049523161724209785},{-0.39416611194610596},{0.8637221455574036}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_2/Conv2d_0d_3x1/Relu_weights#303");
    auto conv92 = om.conv(bias_c90, weights92, {1, 1}, {0, 0, 1, 1}, 1, 1, {{0},{0.06592372804880142},{0.0},{16.810550689697266}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_2/Conv2d_0d_3x1/Relu#434");

    std::vector<int64_t> biasWeightsData92 = mv::utils::generateSequence<int64_t> (384);
    auto biasWeights92 = om.constantInt(biasWeightsData92,{384}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.00017781642964109778},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_2/Conv2d_0d_3x1/Relu_bias#304");
    auto bias_c92 = om.bias(conv92, biasWeights92, {{0},{0.06592372804880142},{0.0},{16.810550689697266}});

    auto concat13 = om.concat({bias_c91, bias_c92}, "C", {{0},{0.06592372804880142},{0.0},{16.810550689697266}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_2/concat#435");

    auto pool12 = om.averagePool(concat11, {3, 3}, {1, 1}, {1, 1, 1, 1}, true, {{0},{0.03830382600426674},{0.0},{9.767476081848145}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_3/AvgPool_0a_3x3/AvgPool#425");

    std::vector<int64_t> weightsData93 = mv::utils::generateSequence<int64_t> (1*1*2048*192);
    auto weights93 = om.constantInt(weightsData93,{1,1,2048,192}, mv::DType("UInt8"), mv::Order::getZMajorID(4), {{131},{0.02265091799199581},{-2.944530487060547},{2.808802843093872}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_3/Conv2d_0b_1x1/Relu_weights#307");
    auto conv93 = om.conv(pool12, weights93, {1, 1}, {0, 0, 0, 0}, 1, 1, {{0},{0.06592372804880142},{0.0},{16.810550689697266}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_3/Conv2d_0b_1x1/Relu#436");

    std::vector<int64_t> biasWeightsData93 = mv::utils::generateSequence<int64_t> (192);
    auto biasWeights93 = om.constantInt(biasWeightsData93,{192}, mv::DType("UInt8"), mv::Order::getColMajorID(1), {{0},{0.0008676169090904295},{-inf},{inf}}, "InceptionV3/InceptionV3/Mixed_7c/Branch_3/Conv2d_0b_1x1/Relu_bias#308");
    auto bias_c93 = om.bias(conv93, biasWeights93, {{0},{0.06592372804880142},{0.0},{16.810550689697266}});

    auto concat14 = om.concat({bias_c85, concat12, concat13, bias_c93}, "C", {{0},{0.06592372804880142},{0.0},{16.810550689697266}}, "InceptionV3/InceptionV3/Mixed_7c/concat#437");

    auto pool13 = om.averagePool(concat14, {8, 8}, {2, 2}, {0, 0, 0, 0}, true, {{0},{0.06592372804880142},{0.0},{16.810550689697266}}, "InceptionV3/Logits/AvgPool_1a_8x8/AvgPool#438");

    om.output(pool13);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/debug_ma2490_resnet50-auto-strategy.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();
}
