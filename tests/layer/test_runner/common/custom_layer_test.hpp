#pragma once
#include "include/mcm/compiler/compilation_unit.hpp"
#include "tests/layer/test_runner/common/print_info_pass.hpp"
#include <vector>
#include <array>

using half = ushort;

template <uint32_t work_dims = 3>
class CustomLayerTest
{
    struct TensorInfo
    {
        mv::Shape shape;
        mv::DType dtype;
        mv::Order order;
    };
    struct ConstantInfo
    {
        std::vector<int64_t> data;
        TensorInfo tensor;
    };

    std::vector<TensorInfo> inputs;
    std::vector<TensorInfo> outputs;
    std::vector<ConstantInfo> constants;
    mv::CompilationUnit unit;
    const mv::QuantizationParams default_quant_params = mv::QuantizationParams{{0}, {1.0}, {}, {}};

    std::vector<uint8_t> pack_param_data(const std::vector<uint32_t>& layer_params) const
    {
        auto kernel_params = std::vector<uint32_t>{};
        kernel_params.reserve(work_dims * 3 + 2 + layer_params.size());

        std::copy(local_size.begin(), local_size.end(), std::back_inserter(kernel_params));
        std::copy(num_groups.begin(), num_groups.end(), std::back_inserter(kernel_params));
        std::copy(global_offset.begin(), global_offset.end(), std::back_inserter(kernel_params));
        kernel_params.push_back(work_dims);
        kernel_params.push_back(kernel_id);
        std::copy(layer_params.begin(), layer_params.end(), std::back_inserter(kernel_params));

        auto packed = std::vector<uint8_t>(kernel_params.size() * sizeof(uint32_t));
        std::copy(kernel_params.begin(), kernel_params.end(), reinterpret_cast<uint32_t *>(packed.data()));

        return packed;
    }

    static std::vector<uint8_t> read_file(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary);
        if (file.fail()) {
            throw mv::RuntimeError("Custom Layer Test", "Failed to open file:\n" + path);
        }
        return std::vector<uint8_t>(std::istreambuf_iterator<char>(file),
                                    std::istreambuf_iterator<char>());
    }

    static std::vector<uint8_t> read_kernel_file(const std::string& file_name)
    {
        const auto kernelFolder = mv::utils::projectRootPath() + "/tests/layer/test_runner/elf/";
        return read_file(kernelFolder + file_name);
    }

    static std::vector<uint8_t> read_input_file(const std::string& file_name)
    {
		const auto inputs_folder = mv::utils::projectRootPath() + "/tests/layer/test_runner/tests/";
        return read_file(inputs_folder + file_name);
    }

public:
    std::array<uint32_t, work_dims> local_size{0};
    std::array<uint32_t, work_dims> num_groups{0};
    std::array<uint32_t, work_dims> global_offset{0};
    uint32_t kernel_id{0};

    explicit CustomLayerTest(const std::string& model_name = "CustomLayerModel",
                             const std::string& compilation_descriptor_file = "release_kmb_MC-Prefetch1.json")
            : unit{model_name}
    {
        const auto compDescPath = mv::utils::projectRootPath() + "/config/compilation/" + compilation_descriptor_file;
        unit.loadCompilationDescriptor(compDescPath);
        unit.loadTargetDescriptor(mv::Target::ma2490);
        // Make logger silent
        unit.compilationDescriptor().setPassArg("GlobalConfigParams", "verbose",
        	mv::Attribute(std::string("Silent")));
        // Add printing pass
        unit.compilationDescriptor().addToGroup("serialize", "PrintInfo", "Singular", false);
    }

    mv::OpModel& get_model() {
        return unit.model();
    }

    void add_input(mv::Shape shape, mv::DType dtype, mv::Order order)
    {
        inputs.push_back({std::move(shape), std::move(dtype), std::move(order)});
    }

    void add_output(mv::Shape shape, mv::DType dtype, mv::Order order)
    {
        outputs.push_back({std::move(shape), std::move(dtype), std::move(order)});
    }

    void add_constant(const std::string& file_name, mv::Shape shape, mv::DType dtype, mv::Order order)
    {
        const auto data_from_file = read_input_file(file_name);
        assert(data_from_file.size() == shape.totalSize() * dtype.getSizeInBytes());

        // reinterpret data from file as `half` array despite of tensors dtype
        assert(dtype.getSizeInBytes() == 2 && "Only `half` input data is supported");

        auto actual_data = std::vector<half>(data_from_file.size() / 2);
        memcpy(actual_data.data(), data_from_file.data(), data_from_file.size());

        auto data_converted = [&] {
            auto result = std::vector<int64_t>{};
            result.reserve(actual_data.size());
            std::copy(actual_data.begin(), actual_data.end(), std::back_inserter(result));
            return result;
        }();

        constants.push_back({std::move(data_converted), {std::move(shape), std::move(dtype), std::move(order)}});
    }

    void run(const std::string& kernel_file, const std::vector<uint32_t>& layer_params)
    {
        std::cout << kernel_file.substr(0, kernel_file.size() - 4) << ";";

        assert(inputs.size() == 1 && outputs.size() == 1);

        const auto kernel_data = read_kernel_file(kernel_file);
        const auto param_data = pack_param_data(layer_params);

        auto& om = unit.model();

        auto layer_inputs = std::vector<mv::Data::TensorIterator>();
        layer_inputs.reserve(inputs.size() + constants.size());

        const auto input = om.input(inputs[0].shape, inputs[0].dtype, inputs[0].order,
                default_quant_params, "input0");
        layer_inputs.push_back(input);

        for (size_t i = 0; i < constants.size(); i++) {
            const auto& data = constants[i].data;
            const auto& shape = constants[i].tensor.shape;
            const auto& dtype = constants[i].tensor.dtype;
            const auto& order = constants[i].tensor.order;

            auto constant = om.constantInt(data, shape, dtype, order, default_quant_params,
                    "input_as_const" + std::to_string(i));

            layer_inputs.push_back(constant);
        }

        auto custom = om.custom(layer_inputs, kernel_data, param_data,
                outputs[0].order, outputs[0].shape, outputs[0].dtype);
        assert(custom->getShape() == outputs[0].shape);
        auto output = om.output(custom);

        unit.initialize();
        unit.run();
    }

    static uint32_t float_as_int(float f) {
        uint32_t i;
        memcpy(&i, &f, 4);
        return i;
    }

    void remove_pass(const std::string& group, const std::string& pass) {
        mv::CompilationDescriptor& compDesc = unit.compilationDescriptor();
        compDesc.remove(group, pass);
    }
};
