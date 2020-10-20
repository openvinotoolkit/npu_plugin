// Network source: migNetworkZoo/internal/full_networks/IRv10/WW22/vd_kmb_models_public_ww22/tiny_yolo_v1/tf/FP16-INT8

#include "include/mcm/compiler/compilation_unit.hpp"
#include "include/mcm/utils/data_generator.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/hardware_tests.hpp"

#include <iostream>
#include <fstream>
#include <unistd.h>

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

void build_tiny_yolo_v1(mv::OpModel& model)
{
    using namespace mv;

    const auto input_1_0 = model.input("input_1", {416, 416, 3, 1}, mv::DType("UInt8"), mv::Order("NHWC"), true);
    const auto conv2d_9_Conv2D_fq_weights_1_Copy_out_low777_const_0 = model.constant("conv2d_9/Conv2D/fq_weights_1/Copy/out_low777_const", {0.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto conv2d_9_Conv2D_fq_weights_1_Copy_out_high778_const_0 = model.constant("conv2d_9/Conv2D/fq_weights_1/Copy/out_high778_const", {254.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto conv2d_8_Conv2D_fq_weights_1_Copy_out_low769_const_0 = model.constant("conv2d_8/Conv2D/fq_weights_1/Copy/out_low769_const", {0.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto conv2d_8_Conv2D_fq_weights_1_Copy_out_high770_const_0 = model.constant("conv2d_8/Conv2D/fq_weights_1/Copy/out_high770_const", {254.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto conv2d_7_Conv2D_fq_weights_1_Copy_out_low737_const_0 = model.constant("conv2d_7/Conv2D/fq_weights_1/Copy/out_low737_const", {0.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto conv2d_7_Conv2D_fq_weights_1_Copy_out_high738_const_0 = model.constant("conv2d_7/Conv2D/fq_weights_1/Copy/out_high738_const", {254.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto conv2d_6_Conv2D_fq_weights_1_Copy_out_low745_const_0 = model.constant("conv2d_6/Conv2D/fq_weights_1/Copy/out_low745_const", {0.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto conv2d_6_Conv2D_fq_weights_1_Copy_out_high746_const_0 = model.constant("conv2d_6/Conv2D/fq_weights_1/Copy/out_high746_const", {254.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto conv2d_5_Conv2D_fq_weights_1_Copy_out_low785_const_0 = model.constant("conv2d_5/Conv2D/fq_weights_1/Copy/out_low785_const", {0.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto conv2d_5_Conv2D_fq_weights_1_Copy_out_high786_const_0 = model.constant("conv2d_5/Conv2D/fq_weights_1/Copy/out_high786_const", {254.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto conv2d_4_Conv2D_fq_weights_1_Copy_out_low801_const_0 = model.constant("conv2d_4/Conv2D/fq_weights_1/Copy/out_low801_const", {0.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto conv2d_4_Conv2D_fq_weights_1_Copy_out_high802_const_0 = model.constant("conv2d_4/Conv2D/fq_weights_1/Copy/out_high802_const", {254.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto conv2d_3_Conv2D_fq_weights_1_Copy_out_low753_const_0 = model.constant("conv2d_3/Conv2D/fq_weights_1/Copy/out_low753_const", {0.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto conv2d_3_Conv2D_fq_weights_1_Copy_out_high754_const_0 = model.constant("conv2d_3/Conv2D/fq_weights_1/Copy/out_high754_const", {254.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto conv2d_2_Conv2D_fq_weights_1_Copy_out_low761_const_0 = model.constant("conv2d_2/Conv2D/fq_weights_1/Copy/out_low761_const", {0.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto conv2d_2_Conv2D_fq_weights_1_Copy_out_high762_const_0 = model.constant("conv2d_2/Conv2D/fq_weights_1/Copy/out_high762_const", {254.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto conv2d_1_Conv2D_fq_weights_1_Copy_out_low793_const_0 = model.constant("conv2d_1/Conv2D/fq_weights_1/Copy/out_low793_const", {0.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto conv2d_1_Conv2D_fq_weights_1_Copy_out_high794_const_0 = model.constant("conv2d_1/Conv2D/fq_weights_1/Copy/out_high794_const", {254.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto Constant_5868_0 = model.constant("Constant_5868", mv::utils::generateSequence<double>(125), {125}, mv::DType("Float32"), mv::Order("W"));
    const auto Constant_5856_0 = model.constant("Constant_5856", mv::utils::generateSequence<double>(1024), {1024}, mv::DType("Float32"), mv::Order("W"));
    const auto Constant_5844_0 = model.constant("Constant_5844", mv::utils::generateSequence<double>(1024), {1024}, mv::DType("Float32"), mv::Order("W"));
    const auto Constant_5832_0 = model.constant("Constant_5832", mv::utils::generateSequence<double>(512), {512}, mv::DType("Float32"), mv::Order("W"));
    const auto Constant_5820_0 = model.constant("Constant_5820", mv::utils::generateSequence<double>(256), {256}, mv::DType("Float32"), mv::Order("W"));
    const auto Constant_5808_0 = model.constant("Constant_5808", mv::utils::generateSequence<double>(128), {128}, mv::DType("Float32"), mv::Order("W"));
    const auto Constant_5796_0 = model.constant("Constant_5796", mv::utils::generateSequence<double>(64), {64}, mv::DType("Float32"), mv::Order("W"));
    const auto Constant_5784_0 = model.constant("Constant_5784", mv::utils::generateSequence<double>(32), {32}, mv::DType("Float32"), mv::Order("W"));
    const auto Constant_5772_0 = model.constant("Constant_5772", mv::utils::generateSequence<double>(16), {16}, mv::DType("Float32"), mv::Order("W"));
    const auto Constant_3711_0 = model.constant("Constant_3711", mv::utils::generateSequence<double>(1*1*1024*125), {1, 1, 1024, 125}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto Constant_3699_0 = model.constant("Constant_3699", mv::utils::generateSequence<double>(3*3*1024*1024), {3, 3, 1024, 1024}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto Constant_3687_0 = model.constant("Constant_3687", mv::utils::generateSequence<double>(3*3*512*1024), {3, 3, 512, 1024}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto Constant_3675_0 = model.constant("Constant_3675", mv::utils::generateSequence<double>(3*3*256*512), {3, 3, 256, 512}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto Constant_3663_0 = model.constant("Constant_3663", mv::utils::generateSequence<double>(3*3*128*256), {3, 3, 128, 256}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto Constant_3651_0 = model.constant("Constant_3651", mv::utils::generateSequence<double>(3*3*64*128), {3, 3, 64, 128}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto Constant_3639_0 = model.constant("Constant_3639", mv::utils::generateSequence<double>(3*3*32*64), {3, 3, 32, 64}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto Constant_3627_0 = model.constant("Constant_3627", mv::utils::generateSequence<double>(3*3*16*32), {3, 3, 16, 32}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto Constant_3615_0 = model.constant("Constant_3615", mv::utils::generateSequence<double>(3*3*3*16), {3, 3, 3, 16}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto _532536_const_0 = model.constant("532536_const", {8.882812}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _531535_const_0 = model.constant("531535_const", {-8.953125}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _530534_const_0 = model.constant("530534_const", {8.882812}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _529533_const_0 = model.constant("529533_const", {-8.953125}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _522526_const_0 = model.constant("522526_const", {11.726562}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _521525_const_0 = model.constant("521525_const", {-11.820312}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _520524_const_0 = model.constant("520524_const", {11.726562}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _519523_const_0 = model.constant("519523_const", {-11.820312}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _512516_const_0 = model.constant("512516_const", {15.640625}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _511515_const_0 = model.constant("511515_const", {-15.765625}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _510514_const_0 = model.constant("510514_const", {15.640625}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _509513_const_0 = model.constant("509513_const", {-15.765625}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _502506_const_0 = model.constant("502506_const", {22.875000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _501505_const_0 = model.constant("501505_const", {-23.062500}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _500504_const_0 = model.constant("500504_const", {22.875000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _499503_const_0 = model.constant("499503_const", {-23.062500}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _492496_const_0 = model.constant("492496_const", {24.609375}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _491495_const_0 = model.constant("491495_const", {-24.812500}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _490494_const_0 = model.constant("490494_const", {24.609375}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _489493_const_0 = model.constant("489493_const", {-24.812500}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _482486_const_0 = model.constant("482486_const", {59.062500}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _481485_const_0 = model.constant("481485_const", {-59.531250}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _480484_const_0 = model.constant("480484_const", {59.062500}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _479483_const_0 = model.constant("479483_const", {-59.531250}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _374378_const_0 = model.constant("374378_const", mv::utils::generateSequence<double>(1*1*1*1024), {1, 1, 1, 1024}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto _373377_const_0 = model.constant("373377_const", mv::utils::generateSequence<double>(1*1*1*1024), {1, 1, 1, 1024}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto conv2d_8_Conv2D_fq_weights_1_0 = model.fakeQuantize("conv2d_8/Conv2D/fq_weights_1", Constant_3699_0, conv2d_8_Conv2D_fq_weights_1_Copy_out_low769_const_0, conv2d_8_Conv2D_fq_weights_1_Copy_out_high770_const_0, _373377_const_0, _374378_const_0, 255);
    const auto _364368_const_0 = model.constant("364368_const", {32.281250}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _363367_const_0 = model.constant("363367_const", {-32.531250}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _362366_const_0 = model.constant("362366_const", {32.281250}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _361365_const_0 = model.constant("361365_const", {-32.531250}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _354358_const_0 = model.constant("354358_const", mv::utils::generateSequence<double>(1*1*1*128), {1, 1, 1, 128}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto _353357_const_0 = model.constant("353357_const", mv::utils::generateSequence<double>(1*1*1*128), {1, 1, 1, 128}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto conv2d_4_Conv2D_fq_weights_1_0 = model.fakeQuantize("conv2d_4/Conv2D/fq_weights_1", Constant_3651_0, conv2d_4_Conv2D_fq_weights_1_Copy_out_low801_const_0, conv2d_4_Conv2D_fq_weights_1_Copy_out_high802_const_0, _353357_const_0, _354358_const_0, 255);
    const auto _334338_const_0 = model.constant("334338_const", mv::utils::generateSequence<double>(1*1*1*256), {1, 1, 1, 256}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto _333337_const_0 = model.constant("333337_const", mv::utils::generateSequence<double>(1*1*1*256), {1, 1, 1, 256}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto conv2d_5_Conv2D_fq_weights_1_0 = model.fakeQuantize("conv2d_5/Conv2D/fq_weights_1", Constant_3663_0, conv2d_5_Conv2D_fq_weights_1_Copy_out_low785_const_0, conv2d_5_Conv2D_fq_weights_1_Copy_out_high786_const_0, _333337_const_0, _334338_const_0, 255);
    const auto _314318_const_0 = model.constant("314318_const", mv::utils::generateSequence<double>(1*1*1*125), {1, 1, 1, 125}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto _313317_const_0 = model.constant("313317_const", mv::utils::generateSequence<double>(1*1*1*125), {1, 1, 1, 125}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto conv2d_9_Conv2D_fq_weights_1_0 = model.fakeQuantize("conv2d_9/Conv2D/fq_weights_1", Constant_3711_0, conv2d_9_Conv2D_fq_weights_1_Copy_out_low777_const_0, conv2d_9_Conv2D_fq_weights_1_Copy_out_high778_const_0, _313317_const_0, _314318_const_0, 255);
    const auto _304308_const_0 = model.constant("304308_const", {6.507812}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _303307_const_0 = model.constant("303307_const", {-6.562500}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _302306_const_0 = model.constant("302306_const", {6.507812}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _301305_const_0 = model.constant("301305_const", {-6.562500}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _294298_const_0 = model.constant("294298_const", mv::utils::generateSequence<double>(1*1*1*1024), {1, 1, 1, 1024}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto _293297_const_0 = model.constant("293297_const", mv::utils::generateSequence<double>(1*1*1*1024), {1, 1, 1, 1024}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto conv2d_7_Conv2D_fq_weights_1_0 = model.fakeQuantize("conv2d_7/Conv2D/fq_weights_1", Constant_3687_0, conv2d_7_Conv2D_fq_weights_1_Copy_out_low737_const_0, conv2d_7_Conv2D_fq_weights_1_Copy_out_high738_const_0, _293297_const_0, _294298_const_0, 255);
    const auto _274278_const_0 = model.constant("274278_const", mv::utils::generateSequence<double>(1*1*1*64), {1, 1, 1, 64}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto _273277_const_0 = model.constant("273277_const", mv::utils::generateSequence<double>(1*1*1*64), {1, 1, 1, 64}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto conv2d_3_Conv2D_fq_weights_1_0 = model.fakeQuantize("conv2d_3/Conv2D/fq_weights_1", Constant_3639_0, conv2d_3_Conv2D_fq_weights_1_Copy_out_low753_const_0, conv2d_3_Conv2D_fq_weights_1_Copy_out_high754_const_0, _273277_const_0, _274278_const_0, 255);
    const auto _254258_const_0 = model.constant("254258_const", mv::utils::generateSequence<double>(1*1*1*16), {1, 1, 1, 16}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto _253257_const_0 = model.constant("253257_const", mv::utils::generateSequence<double>(1*1*1*16), {1, 1, 1, 16}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto conv2d_1_Conv2D_fq_weights_1_0 = model.fakeQuantize("conv2d_1/Conv2D/fq_weights_1", Constant_3615_0, conv2d_1_Conv2D_fq_weights_1_Copy_out_low793_const_0, conv2d_1_Conv2D_fq_weights_1_Copy_out_high794_const_0, _253257_const_0, _254258_const_0, 255);
    const auto _244248_const_0 = model.constant("244248_const", {253.500000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _243247_const_0 = model.constant("243247_const", {0.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _242246_const_0 = model.constant("242246_const", {253.500000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto _241245_const_0 = model.constant("241245_const", {0.000000}, {1}, mv::DType("Float32"), mv::Order("W"));
    const auto conv2d_1_Conv2D_fq_input_0_0 = model.fakeQuantize("conv2d_1/Conv2D/fq_input_0", input_1_0, _241245_const_0, _242246_const_0, _243247_const_0, _244248_const_0, 256);
    const auto batch_normalization_1_cond_FusedBatchNormV3_1_variance_Fused_Add__0 = model.conv("batch_normalization_1/cond/FusedBatchNormV3_1/variance/Fused_Add_", conv2d_1_Conv2D_fq_input_0_0, conv2d_1_Conv2D_fq_weights_1_0, {1, 1}, {1, 1, 1, 1}, 1, 1);
    const auto batch_normalization_1_cond_FusedBatchNormV3_1_variance_Fused_Add__bias_0 = model.bias("batch_normalization_1/cond/FusedBatchNormV3_1/variance/Fused_Add_:bias", batch_normalization_1_cond_FusedBatchNormV3_1_variance_Fused_Add__0, Constant_5772_0);
    const auto leaky_re_lu_1_LeakyRelu2447_0 = model.leakyRelu("leaky_re_lu_1/LeakyRelu2447", batch_normalization_1_cond_FusedBatchNormV3_1_variance_Fused_Add__bias_0, 0.100000001490116);
    const auto max_pooling2d_1_MaxPool_fq_input_0_0 = model.fakeQuantize("max_pooling2d_1/MaxPool/fq_input_0", leaky_re_lu_1_LeakyRelu2447_0, _479483_const_0, _480484_const_0, _481485_const_0, _482486_const_0, 256);
    const auto max_pooling2d_1_MaxPool_0 = model.maxPool("max_pooling2d_1/MaxPool", max_pooling2d_1_MaxPool_fq_input_0_0, {2, 2}, {2, 2}, {0, 0, 0, 0}, false);
    const auto _234238_const_0 = model.constant("234238_const", mv::utils::generateSequence<double>(1*1*1*32), {1, 1, 1, 32}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto _233237_const_0 = model.constant("233237_const", mv::utils::generateSequence<double>(1*1*1*32), {1, 1, 1, 32}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto conv2d_2_Conv2D_fq_weights_1_0 = model.fakeQuantize("conv2d_2/Conv2D/fq_weights_1", Constant_3627_0, conv2d_2_Conv2D_fq_weights_1_Copy_out_low761_const_0, conv2d_2_Conv2D_fq_weights_1_Copy_out_high762_const_0, _233237_const_0, _234238_const_0, 255);
    const auto batch_normalization_2_cond_FusedBatchNormV3_1_variance_Fused_Add__0 = model.conv("batch_normalization_2/cond/FusedBatchNormV3_1/variance/Fused_Add_", max_pooling2d_1_MaxPool_0, conv2d_2_Conv2D_fq_weights_1_0, {1, 1}, {1, 1, 1, 1}, 1, 1);
    const auto batch_normalization_2_cond_FusedBatchNormV3_1_variance_Fused_Add__bias_0 = model.bias("batch_normalization_2/cond/FusedBatchNormV3_1/variance/Fused_Add_:bias", batch_normalization_2_cond_FusedBatchNormV3_1_variance_Fused_Add__0, Constant_5784_0);
    const auto leaky_re_lu_2_LeakyRelu2467_0 = model.leakyRelu("leaky_re_lu_2/LeakyRelu2467", batch_normalization_2_cond_FusedBatchNormV3_1_variance_Fused_Add__bias_0, 0.100000001490116);
    const auto max_pooling2d_2_MaxPool_fq_input_0_0 = model.fakeQuantize("max_pooling2d_2/MaxPool/fq_input_0", leaky_re_lu_2_LeakyRelu2467_0, _489493_const_0, _490494_const_0, _491495_const_0, _492496_const_0, 256);
    const auto max_pooling2d_2_MaxPool_0 = model.maxPool("max_pooling2d_2/MaxPool", max_pooling2d_2_MaxPool_fq_input_0_0, {2, 2}, {2, 2}, {0, 0, 0, 0}, false);
    const auto batch_normalization_3_cond_FusedBatchNormV3_1_variance_Fused_Add__0 = model.conv("batch_normalization_3/cond/FusedBatchNormV3_1/variance/Fused_Add_", max_pooling2d_2_MaxPool_0, conv2d_3_Conv2D_fq_weights_1_0, {1, 1}, {1, 1, 1, 1}, 1, 1);
    const auto batch_normalization_3_cond_FusedBatchNormV3_1_variance_Fused_Add__bias_0 = model.bias("batch_normalization_3/cond/FusedBatchNormV3_1/variance/Fused_Add_:bias", batch_normalization_3_cond_FusedBatchNormV3_1_variance_Fused_Add__0, Constant_5796_0);
    const auto leaky_re_lu_3_LeakyRelu2443_0 = model.leakyRelu("leaky_re_lu_3/LeakyRelu2443", batch_normalization_3_cond_FusedBatchNormV3_1_variance_Fused_Add__bias_0, 0.100000001490116);
    const auto max_pooling2d_3_MaxPool_fq_input_0_0 = model.fakeQuantize("max_pooling2d_3/MaxPool/fq_input_0", leaky_re_lu_3_LeakyRelu2443_0, _499503_const_0, _500504_const_0, _501505_const_0, _502506_const_0, 256);
    const auto max_pooling2d_3_MaxPool_0 = model.maxPool("max_pooling2d_3/MaxPool", max_pooling2d_3_MaxPool_fq_input_0_0, {2, 2}, {2, 2}, {0, 0, 0, 0}, false);
    const auto batch_normalization_4_cond_FusedBatchNormV3_1_variance_Fused_Add__0 = model.conv("batch_normalization_4/cond/FusedBatchNormV3_1/variance/Fused_Add_", max_pooling2d_3_MaxPool_0, conv2d_4_Conv2D_fq_weights_1_0, {1, 1}, {1, 1, 1, 1}, 1, 1);
    const auto batch_normalization_4_cond_FusedBatchNormV3_1_variance_Fused_Add__bias_0 = model.bias("batch_normalization_4/cond/FusedBatchNormV3_1/variance/Fused_Add_:bias", batch_normalization_4_cond_FusedBatchNormV3_1_variance_Fused_Add__0, Constant_5808_0);
    const auto leaky_re_lu_4_LeakyRelu2459_0 = model.leakyRelu("leaky_re_lu_4/LeakyRelu2459", batch_normalization_4_cond_FusedBatchNormV3_1_variance_Fused_Add__bias_0, 0.100000001490116);
    const auto max_pooling2d_4_MaxPool_fq_input_0_0 = model.fakeQuantize("max_pooling2d_4/MaxPool/fq_input_0", leaky_re_lu_4_LeakyRelu2459_0, _509513_const_0, _510514_const_0, _511515_const_0, _512516_const_0, 256);
    const auto max_pooling2d_4_MaxPool_0 = model.maxPool("max_pooling2d_4/MaxPool", max_pooling2d_4_MaxPool_fq_input_0_0, {2, 2}, {2, 2}, {0, 0, 0, 0}, false);
    const auto batch_normalization_5_cond_FusedBatchNormV3_1_variance_Fused_Add__0 = model.conv("batch_normalization_5/cond/FusedBatchNormV3_1/variance/Fused_Add_", max_pooling2d_4_MaxPool_0, conv2d_5_Conv2D_fq_weights_1_0, {1, 1}, {1, 1, 1, 1}, 1, 1);
    const auto batch_normalization_5_cond_FusedBatchNormV3_1_variance_Fused_Add__bias_0 = model.bias("batch_normalization_5/cond/FusedBatchNormV3_1/variance/Fused_Add_:bias", batch_normalization_5_cond_FusedBatchNormV3_1_variance_Fused_Add__0, Constant_5820_0);
    const auto leaky_re_lu_5_LeakyRelu2439_0 = model.leakyRelu("leaky_re_lu_5/LeakyRelu2439", batch_normalization_5_cond_FusedBatchNormV3_1_variance_Fused_Add__bias_0, 0.100000001490116);
    const auto max_pooling2d_5_MaxPool_fq_input_0_0 = model.fakeQuantize("max_pooling2d_5/MaxPool/fq_input_0", leaky_re_lu_5_LeakyRelu2439_0, _519523_const_0, _520524_const_0, _521525_const_0, _522526_const_0, 256);
    const auto max_pooling2d_5_MaxPool_0 = model.maxPool("max_pooling2d_5/MaxPool", max_pooling2d_5_MaxPool_fq_input_0_0, {2, 2}, {2, 2}, {0, 0, 0, 0}, false);
    const auto _214218_const_0 = model.constant("214218_const", mv::utils::generateSequence<double>(1*1*1*512), {1, 1, 1, 512}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto _213217_const_0 = model.constant("213217_const", mv::utils::generateSequence<double>(1*1*1*512), {1, 1, 1, 512}, mv::DType("Float32"), mv::Order("NCHW"));
    const auto conv2d_6_Conv2D_fq_weights_1_0 = model.fakeQuantize("conv2d_6/Conv2D/fq_weights_1", Constant_3675_0, conv2d_6_Conv2D_fq_weights_1_Copy_out_low745_const_0, conv2d_6_Conv2D_fq_weights_1_Copy_out_high746_const_0, _213217_const_0, _214218_const_0, 255);
    const auto batch_normalization_6_cond_FusedBatchNormV3_1_variance_Fused_Add__0 = model.conv("batch_normalization_6/cond/FusedBatchNormV3_1/variance/Fused_Add_", max_pooling2d_5_MaxPool_0, conv2d_6_Conv2D_fq_weights_1_0, {1, 1}, {1, 1, 1, 1}, 1, 1);
    const auto batch_normalization_6_cond_FusedBatchNormV3_1_variance_Fused_Add__bias_0 = model.bias("batch_normalization_6/cond/FusedBatchNormV3_1/variance/Fused_Add_:bias", batch_normalization_6_cond_FusedBatchNormV3_1_variance_Fused_Add__0, Constant_5832_0);
    const auto leaky_re_lu_6_LeakyRelu2455_0 = model.leakyRelu("leaky_re_lu_6/LeakyRelu2455", batch_normalization_6_cond_FusedBatchNormV3_1_variance_Fused_Add__bias_0, 0.100000001490116);
    const auto max_pooling2d_6_MaxPool_fq_input_0_0 = model.fakeQuantize("max_pooling2d_6/MaxPool/fq_input_0", leaky_re_lu_6_LeakyRelu2455_0, _529533_const_0, _530534_const_0, _531535_const_0, _532536_const_0, 256);
    const auto max_pooling2d_6_MaxPool_0 = model.maxPool("max_pooling2d_6/MaxPool", max_pooling2d_6_MaxPool_fq_input_0_0, {2, 2}, {1, 1}, {0, 1, 0, 1}, false);
    const auto batch_normalization_7_cond_FusedBatchNormV3_1_variance_Fused_Add__0 = model.conv("batch_normalization_7/cond/FusedBatchNormV3_1/variance/Fused_Add_", max_pooling2d_6_MaxPool_0, conv2d_7_Conv2D_fq_weights_1_0, {1, 1}, {1, 1, 1, 1}, 1, 1);
    const auto batch_normalization_7_cond_FusedBatchNormV3_1_variance_Fused_Add__bias_0 = model.bias("batch_normalization_7/cond/FusedBatchNormV3_1/variance/Fused_Add_:bias", batch_normalization_7_cond_FusedBatchNormV3_1_variance_Fused_Add__0, Constant_5844_0);
    const auto leaky_re_lu_7_LeakyRelu2451_0 = model.leakyRelu("leaky_re_lu_7/LeakyRelu2451", batch_normalization_7_cond_FusedBatchNormV3_1_variance_Fused_Add__bias_0, 0.100000001490116);
    const auto conv2d_8_Conv2D_fq_input_0_0 = model.fakeQuantize("conv2d_8/Conv2D/fq_input_0", leaky_re_lu_7_LeakyRelu2451_0, _361365_const_0, _362366_const_0, _363367_const_0, _364368_const_0, 256);
    const auto batch_normalization_8_cond_FusedBatchNormV3_1_variance_Fused_Add__0 = model.conv("batch_normalization_8/cond/FusedBatchNormV3_1/variance/Fused_Add_", conv2d_8_Conv2D_fq_input_0_0, conv2d_8_Conv2D_fq_weights_1_0, {1, 1}, {1, 1, 1, 1}, 1, 1);
    const auto batch_normalization_8_cond_FusedBatchNormV3_1_variance_Fused_Add__bias_0 = model.bias("batch_normalization_8/cond/FusedBatchNormV3_1/variance/Fused_Add_:bias", batch_normalization_8_cond_FusedBatchNormV3_1_variance_Fused_Add__0, Constant_5856_0);
    const auto leaky_re_lu_8_LeakyRelu2463_0 = model.leakyRelu("leaky_re_lu_8/LeakyRelu2463", batch_normalization_8_cond_FusedBatchNormV3_1_variance_Fused_Add__bias_0, 0.100000001490116);
    const auto conv2d_9_Conv2D_fq_input_0_0 = model.fakeQuantize("conv2d_9/Conv2D/fq_input_0", leaky_re_lu_8_LeakyRelu2463_0, _301305_const_0, _302306_const_0, _303307_const_0, _304308_const_0, 256);
    const auto conv2d_9_BiasAdd_Add_0 = model.conv("conv2d_9/BiasAdd/Add", conv2d_9_Conv2D_fq_input_0_0, conv2d_9_Conv2D_fq_weights_1_0, {1, 1}, {0, 0, 0, 0}, 1, 1);
    const auto conv2d_9_BiasAdd_Add_bias_0 = model.bias("conv2d_9/BiasAdd/Add:bias", conv2d_9_BiasAdd_Add_0, Constant_5868_0);
    const auto conv2d_9_BiasAdd_YoloRegion_0 = model.regionYolo("conv2d_9/BiasAdd/YoloRegion", conv2d_9_BiasAdd_Add_bias_0, 4, 20, true, 5, {});
    model.output("conv2d_9/BiasAdd/YoloRegion:0", conv2d_9_BiasAdd_YoloRegion_0, mv::DType("Float16"), true);
}

int main(int argc, char **argv)
{
    InputParams params;

    if (!params.parse_args(argc, argv)) { return -1; }

    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    build_tiny_yolo_v1(om);

    std::string compDescPath = params.comp_descriptor_;
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
