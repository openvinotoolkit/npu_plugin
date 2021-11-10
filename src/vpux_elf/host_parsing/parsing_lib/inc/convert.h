#include <host_parsed_inference.h>
#include <data_types.h>
#include <vector>

namespace parsing_lib {
    using host_parsing::DPUInvariant;
    using host_parsing::DPUVariant;
    using host_parsing::DPUInvariantRegisters;
    using host_parsing::DPUVariantRegisters;

void convertDmaTask(const DMATask &t, host_parsing::DmaDescriptor &desc);
std::vector<Relocation> getDmaRelocations();

constexpr int STRIDES(int x) { return x+1; }

enum Dimension {
    B,
    Z,
    Y,
    X,
};

// Maximum number to be stored in flatbuffers, used for sparsity map table address
//-if this value is present in the field sparsity_index it means DENSE, otherwise we have SPARSE tensor
constexpr unsigned long long DEFAULT_INDEX = 999999999999999999ULL; // 60 bits, 18 decimals

template <typename T>
constexpr auto to_underlying(T an_enum) -> std::enable_if_t<std::is_enum<T>::value, std::underlying_type_t<T>>
{
    return static_cast<std::underlying_type_t<T>>(an_enum);
}

class DPUConfigurator {
    public:
    DPUConfigurator(const Invariant &inv, const Variant &var, unsigned int cluster_count);
    DPUConfigurator(const Invariant &&, const Variant &&, unsigned int) = delete;
    bool Setup_Invariant(DPUInvariant &inv);
    bool Setup_Variant(DPUVariant &var, DPUInvariant &inv);
    // uint8_t getTile();
    std::vector<Relocation> getRelocations();

    private:
    bool Setup_Input(DPUInvariantRegisters &registers);
    bool Setup_Weights(DPUInvariantRegisters &registers);
    bool Setup_Kernel(DPUInvariantRegisters &registers);
    bool Setup_Output(DPUInvariantRegisters &registers);

    bool SetupInvariant_CMConv(DPUInvariantRegisters &registers);
    void SetupInvariant_Convolution(DPUInvariantRegisters &registers);
    void SetupInvariant_DwConvolution(DPUInvariantRegisters &registers);
    bool SetupInvariant_Eltwise(DPUInvariantRegisters &registers);
    void SetupInvariant_MaxPool(DPUInvariantRegisters &registers);

    bool Setup_PPE(DPUInvariant &invariant);

    unsigned int ConfigWorkloadSize(unsigned int size) const;
    unsigned int ConfigWorkloadStart(unsigned int start) const;

    void SetupVariant_NTHW_NTK(DPUVariantRegisters &variant);
    void SetupInvariant_Grid(DPUInvariantRegisters &invariant);

    void SetupInvariant_SOH(DPUInvariantRegisters &invariantRegisters);
    void SetupInvariant_SOH_Input(DPUInvariantRegisters &invariantRegisters);
    unsigned int SetupVariant_SOH(DPUInvariant &invariant, DPUVariant &variant);
    void Setup_Output_SOH(DPUInvariantRegisters &invariant, bool is_out_dense);
    void SetupInvariant_Input_SE_size(DPUInvariantRegisters &invariantRegisterst);

    const Invariant &srcInvariant;
    const Variant &srcVariant;
    unsigned int cluster_count;
    DPULayerType opType;
};
}
