/*
* {% copyright %}
*/
#ifndef NN_RUNTIME_TYPES_H_2400__
#define NN_RUNTIME_TYPES_H_2400__

#include <stdint.h>
#include <assert.h>
#include <nn_relocation.h>
#include <nn_log.h>

namespace nn
{
	namespace inference_runtime
	{
        //NCE_DPU_TENSOR_MODE wmode&amode - IDU types for activation&weights
        enum class InputTensorDType : uint8_t
        {
            FP16  = 0x0,
            U8    = 0x1,
            I8    = 0x2,
            I4    = 0x3,
            I2    = 0x4,
            I4X   = 0x5, //not used
            I2X   = 0x6, //not used
            BIN   = 0x7,
            INPUT_DTYPE_UNKNOWN
        };

        //NCE_DPU_MPE_CFG mpe_mode
        enum class MpeActivationWeightDtype : uint8_t
        {
             FP16 = 0x0,
             U8   = 0x1,
             I8   = 0x2,
             I4   = 0x3,
             I2   = 0x4,
             I4X  = 0x5, //not used
             I2X  = 0x6, //not used
             BIN  = 0x7,
             MPE_DTYPE_UNKNOWN
        };

        //NCE_DPU_ODU_CFG  dtype 4 bit field
        enum class OutputTensorDType : uint8_t
        {
             FP16  = 0x00,
             U8F   = 0x01,
             G8    = 0x02,
             I8    = 0x03,
             I32   = 0x04,
             I4    = 0x05,
             I2    = 0x06,
             LOG   = 0x07,
             BIN   = 0x08,
             OUTPUT_DTYPE_UNKNOWN
        };

        inline uint8_t getIDUDTypeSizeBits (InputTensorDType dtype)
        {
            switch (dtype)
            {
                case InputTensorDType::FP16:
                    return 16;
                case InputTensorDType::U8:
                    return 8;
                case InputTensorDType::I8:
                    return 8;
                case InputTensorDType::I4:
                    return 4;
                case InputTensorDType::I2:
                    return 2;
                case InputTensorDType::I4X:
                    return 4;
                case InputTensorDType::I2X:
                    return 2;
                case InputTensorDType::BIN:
                    return 1;
                default:
                    nnLog(MVLOG_ERROR, "Unknown IDU data type");
                    return 1;
            }
        }
	}

    namespace dpu_runtime
    {
        enum
        {
            PPE_ILIST_ENTRIES = 16,
        };

        typedef struct
        {
            /* IDU */
            struct
            {
                unsigned int se_addr;
                unsigned int sparsity_addr;
            } se_sp_addr[4];

            union
            {
                unsigned int se_sp_size;
                struct
                {
                    unsigned int sp_seg_size : 14;
                    unsigned int se_seg_size : 18;
                } se_sp_size_bf;
            } se_sp_size[3];

            union
            {
                unsigned int z_config;
                struct
                {
                    unsigned int se_z_split : 4;
                    unsigned int num_ses_in_z_dir : 9;
                    unsigned int reserved : 19;
                } z_config_bf;
            } z_config;

            union
            {
                unsigned int kernel_pad_cfg;
                struct
                {
                    unsigned int mpe_assign : 1;
                    unsigned int pad_right_en : 1;
                    unsigned int pad_left_en : 1;
                    unsigned int pad_bottom_en : 1;
                    unsigned int pad_top_en : 1;
                    unsigned int kernel_y : 4;
                    unsigned int kernel_x : 4;
                    unsigned int wt_plt_cfg : 2;
                    unsigned int act_dense : 1;
                    unsigned int wt_dense : 1;
                    unsigned int unused1 : 12;
                    unsigned int sp_se_tbl_segment : 1;
                    unsigned int rst_ctxt : 1;
                    unsigned int unused2 : 1;
                } kernel_pad_cfg_bf;
            } kernel_pad_cfg;

            unsigned int weight_size;

            //Struct padding for register memcpy
            unsigned int weight_num_placeholder;

            unsigned int weight_start;

            union
            {
                unsigned int tensor_size0;
                struct
                {
                    unsigned int tensor_size_x : 14;
                    unsigned int tensor_size_y : 14;
                    unsigned int unused : 4;
                } tensor_size0_bf;
            } tensor_size0;

            union
            {
                unsigned int tensor_size1;
                struct
                {
                    unsigned int tensor_size_z : 14;
                    unsigned int unused : 18;
                } tensor_size1_bf;
            } tensor_size1;

            unsigned int tensor_start;

            union
            {
                unsigned int tensor_mode;
                struct
                {
                    unsigned int wmode : 3;
                    unsigned int amode : 3;
                    unsigned int stride : 3;
                    unsigned int zm_input : 1;
                    unsigned int dw_input : 1;
                    unsigned int cm_input : 1;
                    unsigned int workload_operation : 2;
                    unsigned int pad_value : 16;
                    unsigned int addr_format_sel : 1;
                    unsigned int unused : 1;
                } tensor_mode_bf;
            } tensor_mode;

            unsigned int elop_sparsity_addr;
            unsigned int elop_se_addr;

            union
            {
                unsigned int elops_wload;
                struct
                {
                    unsigned int elop_wload : 1;
                    unsigned int seed_wload : 1;
                    unsigned int fifo_wr_wload : 1;
                    unsigned int unused : 29;
                } elops_wload_bf;
            } elops_wload;

            unsigned int act_offset[4];
            unsigned int base_offset_a;
            unsigned int base_offset_b;
            unsigned int wt_offset;

            /* ODU*/
            union
            {
                unsigned int odu_cfg;
                struct
                {
                    unsigned int dtype : 4;
                    unsigned int order : 1;
                    unsigned int sp_value : 8;
                    unsigned int sp_out_en : 1;
                    unsigned int sp_in_en : 1;
                    unsigned int write_sp : 1;
                    unsigned int write_pt : 1;
                    unsigned int write_ac : 1;
                    unsigned int ac_dense_mode : 1;
                    unsigned int transpose : 1;
                    unsigned int grid : 2;
                    unsigned int adr_mode : 1;
                    unsigned int packing : 1; //odu_stat_en?
                    unsigned int super_dense_mode : 1;
                    unsigned int reserved : 7;
                } odu_cfg_bf;
            } odu_cfg;

            unsigned int odu_ctx_size;
            unsigned int odu_ctx_threshold;
            unsigned int se_size;

            union
            {
                unsigned int te_dim0;
                struct
                {
                    unsigned int te_dim_y : 13;
                    unsigned int te_dim_z : 13;
                    unsigned int unused : 6;
                } te_dim0_bf;
            } te_dim0;
            union
            {
                unsigned int te_dim1;
                struct
                {
                    unsigned int te_dim_x : 13;
                    unsigned int unused : 19;
                } te_dim1_bf;
            } te_dim1;

            unsigned int pt_base;
            unsigned int sp_base;
            unsigned int base_ptr_a;
            unsigned int base_ptr_b;
            unsigned int base_adr[4];

            union
            {
                unsigned int odu_cast;
                struct
                {
                    unsigned int cast_enable : 1;
                    unsigned int reserved : 3;
                    unsigned int cast_offset : 28;
                } odu_cast_bf;
            } odu_cast[3];

            /* MPE */
            union
            {
                unsigned int mpe_cfg;
                struct
                {
                    unsigned int mpe_wtbias : 8;
                    unsigned int mpe_actbias : 8;
                    unsigned int mpe_mode : 3;
                    unsigned int mpe_dense : 1;
                    unsigned int mrm_weight_dense : 1;
                    unsigned int mrm_act_dense : 1;
                    unsigned int unused : 10;
                } mpe_cfg_bf;
            } mpe_cfg;

            unsigned int mpe_bus_data_sel;
            unsigned int mpe_bus_data;

            /* PPE */
            union
            {
                unsigned int ppe_cfg;
                struct
                {
                    unsigned int ppe_kernal_fw : 4;
                    unsigned int ppe_kernal_fh : 4;
                    unsigned int ppe_g8_bias_c : 8;
                    unsigned int ppe_g8_bias_b : 8;
                    unsigned int ppe_g8_bias_a : 8;
                } ppe_cfg_bf;
            } ppe_cfg;

            unsigned int ppe_bias;
            union
            {
                unsigned int ppe_scale;
                struct
                {
                    unsigned int ppe_scale_round_shift : 2;
                    unsigned int ppe_scale_shift : 6;
                    unsigned int unused1 : 2;
                    unsigned int ppe_scale_round : 2;
                    unsigned int unused2 : 4;
                    unsigned int ppe_scale_mult : 16;
                } ppe_scale_bf;
            } ppe_scale;

            union
            {
                unsigned int ppe_scale_ctrl;
                struct
                {
                    unsigned int ppe_scale_override : 1;
                    unsigned int unused : 31;
                } ppe_scale_ctrl_bf;
            } ppe_scale_ctrl;

            union
            {
                unsigned int ppe_prelu;
                struct {
                    unsigned int unused :8;
                    unsigned int ppe_prelu_shift :8;
                    unsigned int ppe_prelu_mult :16;
                } ppe_prelu_bf;
            } ppe_prelu;

            // NEVER used in hw logic
            union
            {
                unsigned int ppe_prelu_ctrl;
                struct {
                    unsigned int ppe_prelu_override :1;
                    unsigned int unused :31;
                } ppe_prelu_ctrl_bf;
            } ppe_prelu_ctrl;

            union
            {
                unsigned int ppe_iram_fixed_instr;
                struct
                {
                    unsigned int opcode : 6;
                    unsigned int mrm_mode : 1;
                    unsigned int rd_type : 4;
                    unsigned int rd : 5;
                    unsigned int rs1_type : 3;
                    unsigned int rs1 : 5;
                    unsigned int rs0_type : 3;
                    unsigned int rs0 : 5;
                } ppe_iram_fixed_instr_bf;
            } ppe_iram_fixed_instr;

            unsigned int ppe_scale_hclamp;
            unsigned int ppe_scale_lclamp;
            union
            {
                unsigned int ppe_remap_conv;
                struct
                {
                    unsigned int ppe_remap_shift : 6;
                    unsigned int ppe_remap_shift_lr : 1;
                    unsigned int unused : 25;
                } ppe_remap_conv_bf;
            } ppe_remap_conv;

            union
            {
                unsigned int ppe_misc;
                struct
                {
                    unsigned int ppe_scale_valid : 1;
                    unsigned int ppe_conv_valid : 1;
                    unsigned int ppe_loop_enable : 1;
                    unsigned int ppe_loop_cnt : 5;
                    unsigned int unused : 24;
                } ppe_misc_bf;
            } ppe_misc;

            unsigned int odu_mem_block_size;
        } DPU_workload_base_register_config;

        typedef struct
        {
            union
            {
                unsigned int workload_size0;
                struct
                {
                    unsigned int workload_size_x : 14;
                    unsigned int workload_size_y : 14;
                    unsigned int unused : 4;
                } workload_size0_bf;
            } workload_size0;

            union
            {
                unsigned int workload_size1;
                struct
                {
                    unsigned int workload_size_z : 14;
                    unsigned int pad_count_up : 3;
                    unsigned int pad_count_left : 3;
                    unsigned int pad_count_down : 3;
                    unsigned int pad_count_right : 3;
                    unsigned int unused : 6;
                } workload_size1_bf;
            } workload_size1;

            union
            {
                unsigned int workload_start0;
                struct
                {
                    unsigned int workload_start_x : 14;
                    unsigned int workload_start_y : 14;
                    unsigned int unused : 4;
                } workload_start0_bf;
            } workload_start0;

            union
            {
                unsigned int workload_start1;
                struct
                {
                    unsigned int workload_start_z : 14;
                    unsigned int unused : 18;
                } workload_start1_bf;
            } workload_start1;

            unsigned int weight_num;

            union
            {
                unsigned int te_end0;
                struct
                {
                    unsigned int te_end_y : 13;
                    unsigned int te_end_z : 13;
                    unsigned int unused : 6;
                } te_end0_bf;
            } te_end0;

            union
            {
                unsigned int te_end1;
                struct
                {
                    unsigned int te_end_x : 13;
                    unsigned int unused : 19;
                } te_end1_bf;
            } te_end1;

            union
            {
                unsigned int te_beg0;
                struct
                {
                    unsigned int te_beg_y : 13;
                    unsigned int te_beg_z : 13;
                    unsigned int unused : 6;
                } te_beg0_bf;
            } te_beg0;

            union
            {
                unsigned int te_beg1;
                struct
                {
                    unsigned int te_beg_x : 13;
                    unsigned int unused : 19;
                } te_beg1_bf;
            } te_beg1;
        } DPU_workitem_register_config;

        typedef DPU_workload_base_register_config DPUInvariantRegisters;
        typedef DPU_workitem_register_config DPUVariantRegisters;

        enum DPULayerTypes
        {
            NO_OP,
            DPU_CONV,
            DPU_DWCONV,
            DPU_MAXPOOL,
            DPU_AVEPOOL,
            DPU_ELTWISE,
            DPU_CMCONV,
        };

        enum ODUGrid
        {
            ODU_GRID_4x4,
            ODU_GRID_16x1,
            ODU_GRID_4x1
        };

        enum MPEGrid
        {
            MPE_GRID_4x4,
            MPE_GRID_16x1
        };

        namespace helper
        {
            template <unsigned int Bits>
            struct unsigned_type;

            template <>
            struct unsigned_type<8>
            {
                typedef unsigned char type;
            };

            template <>
            struct unsigned_type<16>
            {
                typedef unsigned short type;
            };

            template <>
            struct unsigned_type<32>
            {
                typedef unsigned int type;
            };

            template <>
            struct unsigned_type<64>
            {
                typedef unsigned long long type;
            };

        }

        enum
        {
            MAX_BARRIERS_PER_INFERENCE = 32,
        };

        typedef typename helper::unsigned_type<MAX_BARRIERS_PER_INFERENCE>::type BarrierMask;
        typedef unsigned long long PhysicalBarrierMask;

        struct BarrierUserConfig
        {
            BarrierMask wait_mask_;
            BarrierMask post_mask_;
            unsigned short start_after_;
            unsigned short clean_after_;
            unsigned int virtual_dep_;
        };

        struct BarrierGpioConfig
        {
            unsigned char group_;
            unsigned char mask_;
        };

        struct DPUInvariant
        {
            DPUInvariantRegisters registers_;
            BarrierUserConfig barriers_;
            BarrierGpioConfig barriers_gpio_;
            const unsigned int *ppe_list_;
            unsigned char ppe_list_batch_size_;
            struct
            {
                unsigned char allow_shadowing_ : 1;
                unsigned char cluster_ : 2;
            };
            unsigned short task_id_;
            unsigned int channel_major_stride_;
            unsigned int output_sparsity_offset_;
        };

        struct DPUAddresses
        {
            inference_runtime::RelativeAddress input_;
            inference_runtime::RelativeAddress output_;
            inference_runtime::RelativeAddress weightsTable_;//map of used weights
            inference_runtime::RelativeAddress weights_;//Weights set - WT_OFFSET+DATA_PTR
            inference_runtime::RelativeAddress ppe_list_;
        };

        struct DPUVariant
        {
            const DPUInvariant *invariant_;
            DPUVariantRegisters registers_;
            unsigned int output_sparsity_offset_;
            unsigned int weight_table_offset_;
            unsigned char cluster_;
            bool skip_dpu_computation_;
            unsigned short task_id_;
        };

        union se_table_entry {
          uint32_t data;
          struct {
            uint32_t base_idx :9;
            uint32_t offset :20;
            uint32_t c2 :1;
            uint32_t empty :1;
            uint32_t c1 :1;
          };
        };

    }

    namespace util
    {
        class TaskContext;
    }
}

#endif //NN_RUNTIME_TYPES_H_2400__
