#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/tensor/quantization_params.hpp"


static void loadPWLQuantParams(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(LoadPWLQuantParams)
        .setFunc(loadPWLQuantParams)
        .setDescription(
            "Loads the fixed quantization required for correct PWL operation"
        );
    }
}

// A1) Normal U8 quant output
// Compute_stage -> I32 -> Requant_stage * ((in_sc * wt_sc)/out_sc)
//      -> I8 -> + ZP -> U8
//
// HW does not support scale dequantization at operation input,
// because that would require to store dequant input data in float format
// while the HW compute pipeline is integer. So we apply input and weights
// scale alongside output scale in a requantization stage which will take
// integer values in I32 range and output int values in the requested output
// range, I8 , I13 ...
// Lets disect the requant stage: between the *(in_sc*wt_sc) and / out_sc
// operations, out datatype would be float, and / out_sc will quantize it
// to whatever range we wish.
// Requant_stage: I32 -> * (in_sc * wt_sc) -> FP -> / out_sc -> I8
// We cant store float values the pipeline so we always apply
// in_wt_sc and out_sc at the same time.
//
// B1) Normal F16 output
// Compute_stage -> S16.16 -> Requant_stage * ((in_sc * wt_sc)/out_sc)
//      -> S16.16 -> Pack_stage -> FP16
//
// We support FP16 multiplications but all the results are still stored
// in an integer fixed point accumualator of form S16.16
// HW is flexible about using setting in_sc, wt_sc and out_sc in
// such a way to support also mixed precision cases U8->FP16 or FP16->U8
//
// A2) U8 quant output + PWL table (I13 input and output)
// Compute_stage -> I32 -> Requant_stage * ((in_sc * wt_sc)/out_sc) -> CLAMP
//      -> I13 -> -> PWL -> I13 -> PostShift_stage << >> -> I8 -> + ZP -> U8
//
// The PWL table requires a fixed input and output quantization
// Same applies for both fixed and custom HW tables.
// Custom tables can be trained to support a optimal quantization range.
// So let's take the HW fixed Sigmoid table for example, we know it works
// best with [-4.0, 4.0] input range and [0.0, 1.0] output range.
// Also because PWL input is placed after the requantization stage we must ensure correct
// quantization range at requantization. For this we drop completely the fused
// (conv|pool|eltwise + sigmoid) quantization from the model and enforce the
// fixed one that PWL needs.
// For the U8 Sigmoid case we need to map [-4.0, 4.0] to the input of PWL which is
// [-4096, 4095] I13. To achieve this we set output scale to 1/1024 and clamp to
// [-4096, 4095].
// At the SIGMOID table output we have an I13 result of [0; 4095] which maps to [0.0, 1.0]
// float. Our only way to translate I13 to either I8, U8 or S16.16 is via the final post_shift
// stage which is bidirectional << >>. So we'll do [0; 4095] >> 4 -> [0, 255]
// The same forced quantization idea needs to be applied for the input tensor of the
// next comsuming operations. In this case we'll need to force a quantization of
// input float range [0.0, 1.0] with zp 0 and scale 1 / 255
//
// B2) FP16 quant output + PWL table (I13 input and output)
// Compute_stage -> S16.16 -> Requant_stage * ((in_sc * wt_sc)/out_sc) -> CLAMP
//      -> I13 (S3.10) -> PWL -> I13 (S1.12) -> PostShift_stage << >> -> S16.16 -> + ZP -> U8
//
// We can also support PWL table for FP16 output cases by taking into account the required
// fixed quantizations for table input and output.
// For example Sigmoid table input needs [-4.0, 4.0] which is achieved by clamping S16.16
// to (S3.16) and shifting >> 6 to match the I13 bits requirements for a final S3.10
// fixed point number.
// Additionally we take the I13 output of range [0.0, 1.0] and consider it a S1.12 fixed
// point result. Lastly we need to shift back to S16.16 which is done via post shift << 4.

template <typename K, typename V, typename H = std::hash<K>>
using map = std::unordered_map<K, V, H>;
using pwlFixedQuantEntry = std::vector<mv::QuantizationParams>;

void loadPWLQuantParams(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&) {

    // The preprogrammed hw PWL functions have fixed quantization requirements
    // on input and output of the activation function
    const map<std::string, map<std::string, pwlFixedQuantEntry>> pwlFixedQuantization =
    {
    {
        "Sigmoid",
        {
        {
            "UInt8",
            {
            mv::QuantizationParams({0}, {1.0 / 1015.6875}, {-4.0}, {4.0}, {0}, {1}, {4}),
            // fine tuning showed that 1/1015.6875 had the best sigmoid avg precision
            // in quantization case, better than the theoretical 1 / 1024
            mv::QuantizationParams({3}, {1.0 / 249}, {0}, {1.0})
            // to account better for sigmoid 0.0 and 1.0 results, we tweak zp and scale
            // to 3 and 1.0/249
            }
        },
        {
            "Float16",
            {
            mv::QuantizationParams({0}, {1.0 / 64}, {-4.0}, {4.0}, {0}, {1}, {-4}),
            mv::QuantizationParams({0}, {1.0}, {0}, {1.0})
            }
        }
        }
    },
    {
        "Tanh",
        {
        {
            "UInt8",
            {
            mv::QuantizationParams({0}, {1.0 / 1024}, {-4.0}, {4.0}, {0}, {1}, {5}),
            mv::QuantizationParams({128}, {1.0 / 127}, {-1.0}, {1.0})
            }
        },
        {
            "Float16",
            {
            mv::QuantizationParams({0}, {1.0 / 64}, {-4.0}, {4.0}, {0}, {1}, {-4}),
            mv::QuantizationParams({0}, {1.0}, {-1.0}, {1.0})
            }
        }
        }
    }
    };

    std::vector<std::string> fusableBaseOpTypes = {"Conv", "DepthwiseConv", "MaxPool", "Eltwise"};
    auto fusableBaseOps = model.getOpsOfTypes(fusableBaseOpTypes);

    for (auto opType : fusableBaseOpTypes)
    {
        for (auto op : fusableBaseOps[opType])
        {

            if (!op->hasAttr("postOpTypes"))
                continue;

            auto actDType = op->getInputTensor(0)->getDType().toString();
            //Note: There are PPE Types SIGMOID, TANH, EXP, SQRT, RSQRT, FLEXARB that need their output
            //quantized to 13-bits for LUT entry, then runtime converts the LUT output to the expected dtype
            std::string resolvedPWL;
            for (auto postOp : op->get<std::vector<std::string>>("postOpTypes"))
            {
                auto fixedQuantIt = pwlFixedQuantization.find(postOp);
                if (fixedQuantIt == pwlFixedQuantization.cend())
                    continue;

                if (!resolvedPWL.empty())
                {
                    pass.log(mv::Logger::MessageType::Warning,
                        "Multiple PWL functions associated per DPU task: " +
                        fixedQuantIt->first + " is ignored since " + resolvedPWL + " is already set");
                    continue;
                }

                auto typedFixedQuantIt = fixedQuantIt->second.find(actDType);

                if (typedFixedQuantIt == fixedQuantIt->second.cend()) {
                    pass.log(mv::Logger::MessageType::Error,
                        "No fixed quantization scheme associated for " +
                        fixedQuantIt->first + " with datatype " + actDType);
                    continue;
                }

                auto pwlInputQuant = typedFixedQuantIt->second[0];
                auto pwlOutputQuant = typedFixedQuantIt->second[1];

                // Use the pwl input scale to futher compute mult and shift
                op->set<mv::QuantizationParams>("pwlQuantParams", pwlInputQuant);

                for(auto output : op->getOutputTensor()){
                    if (!output->isQuantized())
                        continue;
                    output->set<mv::QuantizationParams>("quantParams", pwlOutputQuant);
                }

                resolvedPWL = postOp;
            }
        }
    }
}
