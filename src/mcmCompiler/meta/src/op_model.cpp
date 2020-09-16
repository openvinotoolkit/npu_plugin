/*
    DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()
*/

#include "include/mcm/op_model.hpp"
#include "include/mcm/utils/env_loader.hpp"

mv::OpModel::OpModel(const std::string& name) :
BaseOpModel(name)
{
}

mv::OpModel::OpModel(ComputationModel& other) :
BaseOpModel(other)
{
}


namespace
{
	bool recordWeightsAsText_ = false;

    template <typename T1, typename T2>
    void write(const std::vector<T1>& data, const std::string& filepath)
    {
        mv::utils::validatePath(filepath);
        std::ofstream file(filepath, std::ofstream::binary);
        T2 aux;
        for (const auto& value: data)
        {
            aux = value;
            file.write(&reinterpret_cast<char&>(aux), sizeof(aux));
        };
    }

    std::string varName(std::string name)
    {
        std::replace_if(name.begin(), name.end(), [](char c) { return !std::isalnum(c) && c != '_'; }, '_');
        if (!name.empty() && !std::isalpha(name[0]))
        {
            name = '_' + name;
        }
        return name;
    }

    std::string ltrim(std::string str)
    {
        str.erase(str.begin(), std::find_if(str.begin(), str.end(), [](char c) { return !std::isspace(c); }));
        return str;
    }
    std::string rtrim(std::string str)
    {
        str.erase(std::find_if(str.rbegin(), str.rend(), [](char c) { return !std::isspace(c); }).base(), str.end());
        return str;
    }
    std::string trim(std::string str)
    {
        return ltrim(rtrim(str));
    }
    std::vector<std::string> splitStringList(const std::string& str, char delim)
    {
        std::vector<std::string> out;

        std::istringstream istr(str);
        std::string elem;

        while (std::getline(istr, elem, delim))
        {
            elem = trim(elem);

            if (elem.empty())
            {
                continue;
            }

            out.push_back(elem);
        }

        return out;
    }

    void printParam(std::ostream* codeOut, std::ostream* dataOut, const std::string& paramName, const mv::Data::TensorIterator& tensor)
    {
        *codeOut << varName(tensor->getName());
    }
    void printParam(std::ostream* codeOut, std::ostream* dataOut, const std::string& paramName, const std::vector<mv::Data::TensorIterator>& tensors)
    {
        *codeOut << "{";
        if (!tensors.empty())
        {
            *codeOut << varName(tensors[0]->getName());
        }
        for (size_t i = 1; i < tensors.size(); ++i)
        {
            *codeOut << ", " << varName(tensors[i]->getName());
        }
        *codeOut << "}";
    }
    template <typename T>
    void printParam(std::ostream* codeOut, std::ostream* dataOut, const std::string& paramName, const std::vector<T>& attr)
    {
        if (attr.size() < 8)
        {
            *codeOut << mv::Attribute(attr).toLongString();
        }
        else
        {
            if (recordWeightsAsText_)
            {
                *codeOut << paramName;
                *dataOut << "const std::vector<" << mv::Attribute(attr[0]).getTypeName() << "> " << paramName << mv::Attribute(attr).toLongString() << ";" << std::endl;
                *dataOut << std::endl;
            }
            else
            {
                std::string T_in = typeid(T).name();
                std::string T_str = "int64_t";
                if (T_in == "l")
                    T_str = "int64_t";
                else if (T_in == "d")
                    T_str = "double";
                else 
                    T_str = T_in;
                
                std::string weightsFilename = std::string("./data/") + paramName + std::string(".bin");
                *codeOut << "read<" << T_str << "," << T_str << ">(WEIGHTS_FOLDER + \"" << weightsFilename << "\")";
                write<T,T>(attr, weightsFilename);
            }
        }
    }
    template <typename T>
    void printParam(std::ostream* codeOut, std::ostream* dataOut, const std::string& paramName, const T& attr)
    {
        *codeOut << mv::Attribute(attr).toLongString();
    }

    template <std::size_t I = 0, typename ParamTuple>
    typename std::enable_if<I == std::tuple_size<typename std::decay<ParamTuple>::type>::value, void>::type
    printParams(std::ostream* codeOut, std::ostream* dataOut, const std::string& outVarName, const std::vector<std::string>& paramNames, const ParamTuple& paramValues)
    {
    }
    template <std::size_t I = 0, typename ParamTuple>
    typename std::enable_if<I < std::tuple_size<typename std::decay<ParamTuple>::type>::value, void>::type
    printParams(std::ostream* codeOut, std::ostream* dataOut, const std::string& outVarName, const std::vector<std::string>& paramNames, const ParamTuple& paramValues)
    {
        if (I > 0)
        {
            *codeOut << ", ";
        }

        printParam(codeOut, dataOut, outVarName + "_" + paramNames.at(I), std::get<I>(paramValues));

        printParams<I + 1>(codeOut, dataOut, outVarName, paramNames, paramValues);
    }

    template <typename... Args>
    void printOp(
            std::ostream* codeOut, std::ostream* dataOut, bool recordWeightsAsText,
            const std::string& outVarName,
            const std::string& opName,
            const std::string& name,
            const std::string& paramStr,
            Args&&... args)
    {
		recordWeightsAsText_ = recordWeightsAsText;
        if (codeOut)
        {
            *codeOut << "    const auto " << outVarName << " = model." << opName << "(";
            const auto paramNames = splitStringList(paramStr, ',');
            const auto paramValues = std::forward_as_tuple(std::forward<Args>(args)...);
            printParams(codeOut, dataOut, outVarName, paramNames, paramValues);
            *codeOut << ", \"" << name << "\");" << std::endl;
        }
    }

}


mv::Data::TensorIterator mv::OpModel::align(Data::TensorIterator data, const std::size_t& dimension, const std::size_t& pad, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Align",
        {
            data
        },
        {
            { "dimension", dimension },
            { "pad", pad },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "align");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "align", name, "data,dimension, pad, quantParams", data,dimension, pad, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::argmax(Data::TensorIterator data, const int64_t& out_max_val, const int64_t& top_k, const int64_t& axis, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Argmax",
        {
            data
        },
        {
            { "out_max_val", out_max_val },
            { "top_k", top_k }, 
            { "axis", axis },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "argmax");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "argmax", name, "data,out_max_val, top_k, axis, dType, quantParams", data,out_max_val, top_k, axis, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::averagePool(Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "AveragePool",
        {
            data
        },
        {
            { "kSize", kSize },
            { "stride", stride },
            { "padding", padding }, 
            { "exclude_pad", exclude_pad },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "averagePool");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "averagePool", name, "data,kSize, stride, padding, exclude_pad, dType, quantParams", data,kSize, stride, padding, exclude_pad, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::barrierTask(const Barrier& Barrier, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "BarrierTask",
        {
        },
        {
            { "Barrier", Barrier }
        },
        name
    
    
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::batchNormalization(Data::TensorIterator data, Data::TensorIterator mean, Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, const double& eps, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "BatchNormalization",
        {
            data,
            mean,
            variance,
            offset,
            scale
        },
        {
            { "eps", eps }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "batchNormalization");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "batchNormalization", name, "data, mean, variance, offset, scale,eps, dType, quantParams", data, mean, variance, offset, scale,eps, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::bias(Data::TensorIterator data, Data::TensorIterator weights, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Bias",
        {
            data,
            weights
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "bias");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "bias", name, "data, weights,dType, quantParams", data, weights,dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::cTCDecoder(Data::TensorIterator data, Data::TensorIterator seq, const bool& ctc_merge_repeated, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "CTCDecoder",
        {
            data,
            seq
        },
        {
            { "ctc_merge_repeated", ctc_merge_repeated }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "cTCDecoder");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "cTCDecoder", name, "data, seq,ctc_merge_repeated, dType, quantParams", data, seq,ctc_merge_repeated, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::concat(const std::vector< Data::TensorIterator >& inputs, const std::string& axis, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Concat",
        inputs,
        {
            { "axis", axis },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "concat");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "concat", name, "inputs, axis, dType, quantParams", inputs, axis, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::constant(const std::vector<double>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Constant",
        {
        },
        {
            { "data", data },
            { "shape", shape },
            { "dType", dType },
            { "order", order }, 
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "constant");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "constant", name, "data, shape, dType, order, quantParams", data, shape, dType, order, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::constantDataElement(const std::vector<mv::DataElement>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "ConstantDataElement",
        {
        },
        {
            { "data", data },
            { "shape", shape },
            { "dType", dType },
            { "order", order }, 
            { "quantParams", quantParams }
        },
        name
    
    
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::constantInt(const std::vector<int64_t>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "ConstantInt",
        {
        },
        {
            { "data", data },
            { "shape", shape },
            { "dType", dType },
            { "order", order }, 
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "constantInt");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "constantInt", name, "data, shape, dType, order, quantParams", data, shape, dType, order, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::conv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor, const unsigned& group, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Conv",
        {
            data,
            weights
        },
        {
            { "stride", stride },
            { "padding", padding }, 
            { "dilationFactor", dilationFactor },
            { "group", group },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "conv");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "conv", name, "data, weights,stride, padding, dilationFactor, group, dType, quantParams", data, weights,stride, padding, dilationFactor, group, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::conversion(Data::TensorIterator data, const Order& order, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Conversion",
        {
            data
        },
        {
            { "order", order }
        },
        name
    
    
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::copy(Data::TensorIterator data, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Copy",
        {
            data
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "copy");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "copy", name, "data,dType, quantParams", data,dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::crop(Data::TensorIterator data, const std::size_t& cropVal, const std::size_t& dimension, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Crop",
        {
            data
        },
        {
            { "cropVal", cropVal }, 
            { "dimension", dimension },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "crop");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "crop", name, "data,cropVal, dimension, quantParams", data,cropVal, dimension, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::custom(const std::vector< Data::TensorIterator >& inputs, const std::vector<uint8_t>& kernelData, const std::vector<uint8_t>& paramData, const Order& outOrder, const Shape& outShape, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Custom",
        inputs,
        {
            { "kernelData", kernelData },
            { "paramData", paramData },
            { "outOrder", outOrder },
            { "outShape", outShape }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "custom");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "custom", name, "inputs, kernelData, paramData, outOrder, outShape, dType, quantParams", inputs, kernelData, paramData, outOrder, outShape, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::dMATask(Data::TensorIterator data, const DmaDirection& direction, const uint8_t& port, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "DMATask",
        {
            data
        },
        {
            { "direction", direction }, 
            { "port", port }
        },
        name
    
    
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::dPUTaskConv(const std::vector< Data::TensorIterator >& inputs, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor, const unsigned& group, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "DPUTask",
        inputs,
        {
            { "taskOp", std::string("Conv") },
            { "stride", stride },
            { "padding", padding }, 
            { "dilationFactor", dilationFactor },
            { "group", group },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::dPUTaskMaxPool(const std::vector< Data::TensorIterator >& inputs, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "DPUTask",
        inputs,
        {
            { "taskOp", std::string("MaxPool") },
            { "kSize", kSize },
            { "stride", stride },
            { "padding", padding }, 
            { "exclude_pad", exclude_pad },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::dPUTaskDepthwiseConv(const std::vector< Data::TensorIterator >& inputs, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "DPUTask",
        inputs,
        {
            { "taskOp", std::string("DepthwiseConv") },
            { "stride", stride },
            { "padding", padding }, 
            { "dilationFactor", dilationFactor },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::dPUTaskEltwise(const std::vector< Data::TensorIterator >& inputs, const std::string& eltwiseType, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "DPUTask",
        inputs,
        {
            { "taskOp", std::string("Eltwise") },
            { "eltwiseType", eltwiseType }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::deallocate(Data::TensorIterator inputs, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Deallocate",
        {
            inputs
        },
        {
        },
        name
    
    
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::deconv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor, const unsigned& group, const bool& is_depthwise, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Deconv",
        {
            data,
            weights
        },
        {
            { "stride", stride },
            { "padding", padding }, 
            { "dilationFactor", dilationFactor },
            { "group", group },
            { "is_depthwise", is_depthwise },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "deconv");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "deconv", name, "data, weights,stride, padding, dilationFactor, group, is_depthwise, dType, quantParams", data, weights,stride, padding, dilationFactor, group, is_depthwise, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::depthwiseConv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "DepthwiseConv",
        {
            data,
            weights
        },
        {
            { "stride", stride },
            { "padding", padding }, 
            { "dilationFactor", dilationFactor },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "depthwiseConv");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "depthwiseConv", name, "data, weights,stride, padding, dilationFactor, dType, quantParams", data, weights,stride, padding, dilationFactor, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::detectionOutput(const std::vector< Data::TensorIterator >& inputs, const int64_t& num_classes, const int64_t& keep_top_k, const double& nms_threshold, const int64_t& background_label_id, const int64_t& top_k, const bool& variance_encoded_in_target, const std::string& code_type, const bool& share_location, const double& confidence_threshold, const bool& clip_before_nms, const bool& clip_after_nms, const int64_t& decrease_label_id, const bool& normalized, const int64_t& input_height, const int64_t& input_width, const double& objectness_score, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "DetectionOutput",
        inputs,
        {
            { "num_classes", num_classes },
            { "keep_top_k", keep_top_k },
            { "nms_threshold", nms_threshold },
            { "background_label_id", background_label_id },
            { "top_k", top_k },
            { "variance_encoded_in_target", variance_encoded_in_target },
            { "code_type", code_type },
            { "share_location", share_location },
            { "confidence_threshold", confidence_threshold },
            { "clip_before_nms", clip_before_nms },
            { "clip_after_nms", clip_after_nms },
            { "decrease_label_id", decrease_label_id },
            { "normalized", normalized },
            { "input_height", input_height },
            { "input_width", input_width },
            { "objectness_score", objectness_score }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "detectionOutput");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "detectionOutput", name, "inputs, num_classes, keep_top_k, nms_threshold, background_label_id, top_k, variance_encoded_in_target, code_type, share_location, confidence_threshold, clip_before_nms, clip_after_nms, decrease_label_id, normalized, input_height, input_width, objectness_score, dType, quantParams", inputs, num_classes, keep_top_k, nms_threshold, background_label_id, top_k, variance_encoded_in_target, code_type, share_location, confidence_threshold, clip_before_nms, clip_after_nms, decrease_label_id, normalized, input_height, input_width, objectness_score, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::dropout(Data::TensorIterator input, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Dropout",
        {
            input
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "dropout");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "dropout", name, "input,dType, quantParams", input,dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::dummy(Data::TensorIterator data, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Dummy",
        {
            data
        },
        {
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "dummy");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "dummy", name, "data", data);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::eltwise(const std::vector< Data::TensorIterator >& inputs, const std::string& eltwiseType, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Eltwise",
        inputs,
        {
            { "eltwiseType", eltwiseType }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "eltwise");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "eltwise", name, "inputs, eltwiseType, dType, quantParams", inputs, eltwiseType, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::elu(Data::TensorIterator data, const unsigned& alpha, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Elu",
        {
            data
        },
        {
            { "alpha", alpha }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "elu");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "elu", name, "data,alpha", data,alpha);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::fakeQuantize(Data::TensorIterator data, Data::TensorIterator input_min, Data::TensorIterator input_max, Data::TensorIterator output_min, Data::TensorIterator output_max, const unsigned& levels, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "FakeQuantize",
        {
            data,
            input_min,
            input_max,
            output_min,
            output_max
        },
        {
            { "levels", levels }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "fakeQuantize");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "fakeQuantize", name, "data, input_min, input_max, output_min, output_max,levels", data, input_min, input_max, output_min, output_max,levels);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::flatten(Data::TensorIterator input, const int64_t& axis, const int64_t& end_axis, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Flatten",
        {
            input
        },
        {
            { "axis", axis },
            { "end_axis", end_axis },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "flatten");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "flatten", name, "input,axis, end_axis, dType, quantParams", input,axis, end_axis, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::fullyConnected(Data::TensorIterator data, Data::TensorIterator weights, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "FullyConnected",
        {
            data,
            weights
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "fullyConnected");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "fullyConnected", name, "data, weights,dType, quantParams", data, weights,dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::gather(Data::TensorIterator data, Data::TensorIterator indices, const unsigned& axis, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Gather",
        {
            data,
            indices
        },
        {
            { "axis", axis }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "gather");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "gather", name, "data, indices,axis, dType, quantParams", data, indices,axis, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::identity(Data::TensorIterator data, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Identity",
        {
            data
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "identity");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "identity", name, "data,dType, quantParams", data,dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::implicitConcat(const std::vector< Data::TensorIterator >& inputs, const std::string& axis, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "ImplicitConcat",
        inputs,
        {
            { "axis", axis },
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::implicitInput(Data::TensorIterator data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "ImplicitInput",
        {
            data
        },
        {
            { "shape", shape },
            { "dType", dType },
            { "order", order }, 
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "implicitInput");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "implicitInput", name, "data,shape, dType, order, quantParams", data,shape, dType, order, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::implicitInputSlice(Data::TensorIterator inputs, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "ImplicitInputSlice",
        {
            inputs
        },
        {
            { "quantParams", quantParams }
        },
        name
    
    
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::implicitJoin(const std::vector< Data::TensorIterator >& inputs, const std::string& axis, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "ImplicitJoin",
        inputs,
        {
            { "axis", axis },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::implicitOutput(Data::TensorIterator data, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "ImplicitOutput",
        {
            data
        },
        {
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "implicitOutput");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "implicitOutput", name, "data,quantParams", data,quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::implicitPermute(Data::TensorIterator inputs, const Shape& shape, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "ImplicitPermute",
        {
            inputs
        },
        {
            { "shape", shape }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::implicitReshape(Data::TensorIterator inputs, const Shape& shape, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "ImplicitReshape",
        {
            inputs
        },
        {
            { "shape", shape }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::implicitUnion(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "ImplicitUnion",
        inputs,
        {
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::input(const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const bool& networkInput, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Input",
        {
        },
        {
            { "shape", shape },
            { "dType", dType },
            { "order", order }, 
            { "quantParams", quantParams },
            { "networkInput", networkInput }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "input");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "input", name, "shape, dType, order, quantParams, networkInput", shape, dType, order, quantParams, networkInput);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::interp(Data::TensorIterator data, const double& factor, const unsigned& pad_beg, const unsigned& pad_end, const unsigned& height, const unsigned& width, const bool& align_corners, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Interp",
        {
            data
        },
        {
            { "factor", factor },
            { "pad_beg", pad_beg },
            { "pad_end", pad_end }, 
            { "height", height },
            { "width", width },
            { "align_corners", align_corners },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "interp");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "interp", name, "data,factor, pad_beg, pad_end, height, width, align_corners, dType, quantParams", data,factor, pad_beg, pad_end, height, width, align_corners, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::leakyRelu(Data::TensorIterator data, const double& alpha, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "LeakyRelu",
        {
            data
        },
        {
            { "alpha", alpha },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "leakyRelu");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "leakyRelu", name, "data,alpha, dType, quantParams", data,alpha, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::localResponseNormalization(Data::TensorIterator data, const unsigned& size, const unsigned& bias, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "LocalResponseNormalization",
        {
            data
        },
        {
            { "size", size },
            { "bias", bias }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "localResponseNormalization");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "localResponseNormalization", name, "data,size, bias", data,size, bias);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::matMul(Data::TensorIterator data0, Data::TensorIterator data1, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "MatMul",
        {
            data0,
            data1
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "matMul");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "matMul", name, "data0, data1,dType, quantParams", data0, data1,dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::maxPool(Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "MaxPool",
        {
            data
        },
        {
            { "kSize", kSize },
            { "stride", stride },
            { "padding", padding }, 
            { "exclude_pad", exclude_pad },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "maxPool");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "maxPool", name, "data,kSize, stride, padding, exclude_pad, dType, quantParams", data,kSize, stride, padding, exclude_pad, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::maximum(Data::TensorIterator inputs, const double& maximum, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Maximum",
        {
            inputs
        },
        {
            { "maximum", maximum }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "maximum");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "maximum", name, "inputs,maximum, dType, quantParams", inputs,maximum, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::minimum(Data::TensorIterator inputs, const double& minimum, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Minimum",
        {
            inputs
        },
        {
            { "minimum", minimum }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "minimum");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "minimum", name, "inputs,minimum, dType, quantParams", inputs,minimum, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::norm(Data::TensorIterator data, const double& alpha, const double& beta, const std::string& region, const unsigned& local_size, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Norm",
        {
            data
        },
        {
            { "alpha", alpha },
            { "beta", beta },
            { "region", region },
            { "local_size", local_size }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "norm");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "norm", name, "data,alpha, beta, region, local_size, dType, quantParams", data,alpha, beta, region, local_size, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::normalize(Data::TensorIterator data, Data::TensorIterator weights, const double& eps, const unsigned& across_spatial, const unsigned& channel_shared, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Normalize",
        {
            data,
            weights
        },
        {
            { "eps", eps }, 
            { "across_spatial", across_spatial },
            { "channel_shared", channel_shared },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "normalize");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "normalize", name, "data, weights,eps, across_spatial, channel_shared, dType, quantParams", data, weights,eps, across_spatial, channel_shared, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::output(Data::TensorIterator data, const DType& precision, const mv::QuantizationParams& quantParams, const bool& networkOutput, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Output",
        {
            data
        },
        {
            { "precision", precision },
            { "quantParams", quantParams },
            { "networkOutput", networkOutput }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "output");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "output", name, "data,precision, quantParams, networkOutput", data,precision, quantParams, networkOutput);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::pSROIPooling(const std::vector< Data::TensorIterator >& inputs, const std::size_t& output_dim, const std::size_t& group_size, const double& spatial_scale, const std::size_t& pooled_w, const std::size_t& pooled_h, const std::size_t& spatial_bin_x, const std::size_t& spatial_bin_y, const std::string& mode, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "PSROIPooling",
        inputs,
        {
            { "output_dim", output_dim },
            { "group_size", group_size },
            { "spatial_scale", spatial_scale },
            { "pooled_w", pooled_w },
            { "pooled_h", pooled_h },
            { "spatial_bin_x", spatial_bin_x },
            { "spatial_bin_y", spatial_bin_y },
            { "mode", mode }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "pSROIPooling");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "pSROIPooling", name, "inputs, output_dim, group_size, spatial_scale, pooled_w, pooled_h, spatial_bin_x, spatial_bin_y, mode, dType, quantParams", inputs, output_dim, group_size, spatial_scale, pooled_w, pooled_h, spatial_bin_x, spatial_bin_y, mode, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::permute(Data::TensorIterator data, const Order& order, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Permute",
        {
            data
        },
        {
            { "order", order }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "permute");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "permute", name, "data,order, dType, quantParams", data,order, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::placeholderTask(const Shape& shape, const DType& dType, const Order& order, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "PlaceholderTask",
        {
        },
        {
            { "shape", shape },
            { "dType", dType },
            { "order", order }
        },
        name
    
    
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::prelu(Data::TensorIterator data, Data::TensorIterator slope, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Prelu",
        {
            data,
            slope
        },
        {
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "prelu");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "prelu", name, "data, slope", data, slope);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::priorbox(const std::vector< Data::TensorIterator >& inputs, const unsigned& flip, const unsigned& clip, const double& step_w, const double& step_h, const double& offset, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Priorbox",
        inputs,
        {
            { "flip", flip },
            { "clip", clip },
            { "step_w", step_w },
            { "step_h", step_h },
            { "offset", offset }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "priorbox");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "priorbox", name, "inputs, flip, clip, step_w, step_h, offset, dType, quantParams", inputs, flip, clip, step_w, step_h, offset, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::proposal(const std::vector< Data::TensorIterator >& inputs, const std::vector<float>& scale, const std::vector<float>& ratio, const unsigned& base_size, const unsigned& pre_nms_topn, const unsigned& post_nms_topn, const double& nms_thresh, const unsigned& feat_stride, const unsigned& min_size, const double& pre_nms_thresh, const bool& clip_before_nms, const bool& clip_after_nms, const bool& normalize, const double& box_size_scale, const double& box_coordinate_scale, const std::string& framework, const bool& for_deformable, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Proposal",
        inputs,
        {
            { "scale", scale },
            { "ratio", ratio },
            { "base_size", base_size },
            { "pre_nms_topn", pre_nms_topn },
            { "post_nms_topn", post_nms_topn },
            { "nms_thresh", nms_thresh },
            { "feat_stride", feat_stride },
            { "min_size", min_size }, 
            { "pre_nms_thresh", pre_nms_thresh },
            { "clip_before_nms", clip_before_nms },
            { "clip_after_nms", clip_after_nms },
            { "normalize", normalize },
            { "box_size_scale", box_size_scale },
            { "box_coordinate_scale", box_coordinate_scale },
            { "framework", framework },
            { "for_deformable", for_deformable },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "proposal");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "proposal", name, "inputs, scale, ratio, base_size, pre_nms_topn, post_nms_topn, nms_thresh, feat_stride, min_size, pre_nms_thresh, clip_before_nms, clip_after_nms, normalize, box_size_scale, box_coordinate_scale, framework, for_deformable, dType, quantParams", inputs, scale, ratio, base_size, pre_nms_topn, post_nms_topn, nms_thresh, feat_stride, min_size, pre_nms_thresh, clip_before_nms, clip_after_nms, normalize, box_size_scale, box_coordinate_scale, framework, for_deformable, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::quantize(Data::TensorIterator data, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Quantize",
        {
            data
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "quantize");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "quantize", name, "data,dType, quantParams", data,dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::rOIPooling(const std::vector< Data::TensorIterator >& inputs, const unsigned& pooled_w, const unsigned& pooled_h, const double& spatial_scale, const unsigned& roi_pooling_method, const unsigned& num_rois, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "ROIPooling",
        inputs,
        {
            { "pooled_w", pooled_w },
            { "pooled_h", pooled_h },
            { "spatial_scale", spatial_scale },
            { "roi_pooling_method", roi_pooling_method },
            { "num_rois", num_rois }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "rOIPooling");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "rOIPooling", name, "inputs, pooled_w, pooled_h, spatial_scale, roi_pooling_method, num_rois, dType, quantParams", inputs, pooled_w, pooled_h, spatial_scale, roi_pooling_method, num_rois, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::refConv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor, const unsigned& group, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "RefConv",
        {
            data,
            weights
        },
        {
            { "stride", stride },
            { "padding", padding }, 
            { "dilationFactor", dilationFactor },
            { "group", group },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::regionYolo(Data::TensorIterator data, const unsigned& coords, const unsigned& classes, const bool& do_softmax, const unsigned& num, const std::vector<unsigned>& mask, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "RegionYolo",
        {
            data
        },
        {
            { "coords", coords },
            { "classes", classes },
            { "do_softmax", do_softmax }, 
            { "num", num },
            { "mask", mask },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "regionYolo");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "regionYolo", name, "data,coords, classes, do_softmax, num, mask, dType, quantParams", data,coords, classes, do_softmax, num, mask, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::relu(Data::TensorIterator data, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Relu",
        {
            data
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "relu");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "relu", name, "data,dType, quantParams", data,dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::reorder(Data::TensorIterator data, const Order& order, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Reorder",
        {
            data
        },
        {
            { "order", order }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "reorder");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "reorder", name, "data,order", data,order);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::reorgYolo(Data::TensorIterator data, const unsigned& stride, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "ReorgYolo",
        {
            data
        },
        {
            { "stride", stride }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "reorgYolo");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "reorgYolo", name, "data,stride, dType, quantParams", data,stride, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::resample(Data::TensorIterator input, const std::string& interpolation, const bool& antialias, const Shape& output_shape, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Resample",
        {
            input
        },
        {
            { "interpolation", interpolation },
            { "antialias", antialias },
            { "output_shape", output_shape }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "resample");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "resample", name, "input,interpolation, antialias, output_shape, dType, quantParams", input,interpolation, antialias, output_shape, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::reshape(Data::TensorIterator data, const Shape& shape, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Reshape",
        {
            data
        },
        {
            { "shape", shape }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "reshape");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "reshape", name, "data,shape, dType, quantParams", data,shape, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::scale(Data::TensorIterator data, Data::TensorIterator weights, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Scale",
        {
            data,
            weights
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "scale");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "scale", name, "data, weights,dType, quantParams", data, weights,dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::sigmoid(Data::TensorIterator data, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Sigmoid",
        {
            data
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "sigmoid");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "sigmoid", name, "data,dType, quantParams", data,dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::slice(Data::TensorIterator data, const Shape& begin, const Shape& size, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Slice",
        {
            data
        },
        {
            { "begin", begin },
            { "size", size }, 
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "slice");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "slice", name, "data,begin, size, quantParams", data,begin, size, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::softmax(Data::TensorIterator data, const std::string& axis, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Softmax",
        {
            data
        },
        {
            { "axis", axis },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "softmax");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "softmax", name, "data,axis, dType, quantParams", data,axis, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::sparsityMap(const std::vector<int64_t>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "SparsityMap",
        {
        },
        {
            { "data", data },
            { "shape", shape },
            { "dType", dType },
            { "order", order }, 
            { "quantParams", quantParams }
        },
        name
    
    
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::tanh(Data::TensorIterator data, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Tanh",
        {
            data
        },
        {
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "tanh");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "tanh", name, "data", data);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::tile(Data::TensorIterator data, const unsigned& axis, const unsigned& tiles, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "Tile",
        {
            data
        },
        {
            { "axis", axis },
            { "tiles", tiles }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "tile");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "tile", name, "data,axis, tiles, dType, quantParams", data,axis, tiles, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::topK(Data::TensorIterator data, const std::string& sort, const std::string& mode, const int64_t& top_k, const int64_t& axis, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "TopK",
        {
            data
        },
        {
            { "sort", sort },
            { "mode", mode },
            { "top_k", top_k }, 
            { "axis", axis },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
    if (recordModel_) { 

        const auto outputName = output != tensorEnd() ? varName(output->getName()) : (!name.empty() ? name : "topK");
        printOp(codeOut_, dataOut_, recordWeightsAsText_, outputName, "topK", name, "data,sort, mode, top_k, axis, dType, quantParams", data,sort, mode, top_k, axis, dType, quantParams);
    }
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskDummy(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Dummy") },
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskIdentity(const std::vector< Data::TensorIterator >& inputs, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Identity") },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskSoftmax(const std::vector< Data::TensorIterator >& inputs, const std::string& axis, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Softmax") },
            { "axis", axis },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskProposal(const std::vector< Data::TensorIterator >& inputs, const std::vector<float>& scale, const std::vector<float>& ratio, const unsigned& base_size, const unsigned& pre_nms_topn, const unsigned& post_nms_topn, const double& nms_thresh, const unsigned& feat_stride, const unsigned& min_size, const double& pre_nms_thresh, const bool& clip_before_nms, const bool& clip_after_nms, const bool& normalize, const double& box_size_scale, const double& box_coordinate_scale, const std::string& framework, const bool& for_deformable, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Proposal") },
            { "scale", scale },
            { "ratio", ratio },
            { "base_size", base_size },
            { "pre_nms_topn", pre_nms_topn },
            { "post_nms_topn", post_nms_topn },
            { "nms_thresh", nms_thresh },
            { "feat_stride", feat_stride },
            { "min_size", min_size }, 
            { "pre_nms_thresh", pre_nms_thresh },
            { "clip_before_nms", clip_before_nms },
            { "clip_after_nms", clip_after_nms },
            { "normalize", normalize },
            { "box_size_scale", box_size_scale },
            { "box_coordinate_scale", box_coordinate_scale },
            { "framework", framework },
            { "for_deformable", for_deformable },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskROIPooling(const std::vector< Data::TensorIterator >& inputs, const unsigned& pooled_w, const unsigned& pooled_h, const double& spatial_scale, const unsigned& roi_pooling_method, const unsigned& num_rois, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("ROIPooling") },
            { "pooled_w", pooled_w },
            { "pooled_h", pooled_h },
            { "spatial_scale", spatial_scale },
            { "roi_pooling_method", roi_pooling_method },
            { "num_rois", num_rois }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskPSROIPooling(const std::vector< Data::TensorIterator >& inputs, const std::size_t& output_dim, const std::size_t& group_size, const double& spatial_scale, const std::size_t& pooled_w, const std::size_t& pooled_h, const std::size_t& spatial_bin_x, const std::size_t& spatial_bin_y, const std::string& mode, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("PSROIPooling") },
            { "output_dim", output_dim },
            { "group_size", group_size },
            { "spatial_scale", spatial_scale },
            { "pooled_w", pooled_w },
            { "pooled_h", pooled_h },
            { "spatial_bin_x", spatial_bin_x },
            { "spatial_bin_y", spatial_bin_y },
            { "mode", mode }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskQuantize(const std::vector< Data::TensorIterator >& inputs, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Quantize") },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskReshape(const std::vector< Data::TensorIterator >& inputs, const Shape& shape, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Reshape") },
            { "shape", shape }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskRegionYolo(const std::vector< Data::TensorIterator >& inputs, const unsigned& coords, const unsigned& classes, const bool& do_softmax, const unsigned& num, const std::vector<unsigned>& mask, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("RegionYolo") },
            { "coords", coords },
            { "classes", classes },
            { "do_softmax", do_softmax }, 
            { "num", num },
            { "mask", mask },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskReorgYolo(const std::vector< Data::TensorIterator >& inputs, const unsigned& stride, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("ReorgYolo") },
            { "stride", stride }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskNormalize(const std::vector< Data::TensorIterator >& inputs, const double& eps, const unsigned& across_spatial, const unsigned& channel_shared, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Normalize") },
            { "eps", eps }, 
            { "across_spatial", across_spatial },
            { "channel_shared", channel_shared },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskPermute(const std::vector< Data::TensorIterator >& inputs, const Order& order, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Permute") },
            { "order", order }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskEltwise(const std::vector< Data::TensorIterator >& inputs, const std::string& eltwiseType, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Eltwise") },
            { "eltwiseType", eltwiseType }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskInterp(const std::vector< Data::TensorIterator >& inputs, const double& factor, const unsigned& pad_beg, const unsigned& pad_end, const unsigned& height, const unsigned& width, const bool& align_corners, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Interp") },
            { "factor", factor },
            { "pad_beg", pad_beg },
            { "pad_end", pad_end }, 
            { "height", height },
            { "width", width },
            { "align_corners", align_corners },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskDetectionOutput(const std::vector< Data::TensorIterator >& inputs, const int64_t& num_classes, const int64_t& keep_top_k, const double& nms_threshold, const int64_t& background_label_id, const int64_t& top_k, const bool& variance_encoded_in_target, const std::string& code_type, const bool& share_location, const double& confidence_threshold, const bool& clip_before_nms, const bool& clip_after_nms, const int64_t& decrease_label_id, const bool& normalized, const int64_t& input_height, const int64_t& input_width, const double& objectness_score, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("DetectionOutput") },
            { "num_classes", num_classes },
            { "keep_top_k", keep_top_k },
            { "nms_threshold", nms_threshold },
            { "background_label_id", background_label_id },
            { "top_k", top_k },
            { "variance_encoded_in_target", variance_encoded_in_target },
            { "code_type", code_type },
            { "share_location", share_location },
            { "confidence_threshold", confidence_threshold },
            { "clip_before_nms", clip_before_nms },
            { "clip_after_nms", clip_after_nms },
            { "decrease_label_id", decrease_label_id },
            { "normalized", normalized },
            { "input_height", input_height },
            { "input_width", input_width },
            { "objectness_score", objectness_score }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskPriorbox(const std::vector< Data::TensorIterator >& inputs, const unsigned& flip, const unsigned& clip, const double& step_w, const double& step_h, const double& offset, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Priorbox") },
            { "flip", flip },
            { "clip", clip },
            { "step_w", step_w },
            { "step_h", step_h },
            { "offset", offset }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskArgmax(const std::vector< Data::TensorIterator >& inputs, const int64_t& out_max_val, const int64_t& top_k, const int64_t& axis, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Argmax") },
            { "out_max_val", out_max_val },
            { "top_k", top_k }, 
            { "axis", axis },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskTopK(const std::vector< Data::TensorIterator >& inputs, const std::string& sort, const std::string& mode, const int64_t& top_k, const int64_t& axis, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("TopK") },
            { "sort", sort },
            { "mode", mode },
            { "top_k", top_k }, 
            { "axis", axis },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskNorm(const std::vector< Data::TensorIterator >& inputs, const double& alpha, const double& beta, const std::string& region, const unsigned& local_size, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Norm") },
            { "alpha", alpha },
            { "beta", beta },
            { "region", region },
            { "local_size", local_size }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskResample(const std::vector< Data::TensorIterator >& inputs, const std::string& interpolation, const bool& antialias, const Shape& output_shape, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Resample") },
            { "interpolation", interpolation },
            { "antialias", antialias },
            { "output_shape", output_shape }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskFakeQuantize(const std::vector< Data::TensorIterator >& inputs, const unsigned& levels, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("FakeQuantize") },
            { "levels", levels }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskCustom(const std::vector< Data::TensorIterator >& inputs, const std::vector<uint8_t>& kernelData, const std::vector<uint8_t>& paramData, const Order& outOrder, const Shape& outShape, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Custom") },
            { "kernelData", kernelData },
            { "paramData", paramData },
            { "outOrder", outOrder },
            { "outShape", outShape }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskSigmoid(const std::vector< Data::TensorIterator >& inputs, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Sigmoid") },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskDeconv(const std::vector< Data::TensorIterator >& inputs, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor, const unsigned& group, const bool& is_depthwise, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Deconv") },
            { "stride", stride },
            { "padding", padding }, 
            { "dilationFactor", dilationFactor },
            { "group", group },
            { "is_depthwise", is_depthwise },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskTile(const std::vector< Data::TensorIterator >& inputs, const unsigned& axis, const unsigned& tiles, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Tile") },
            { "axis", axis },
            { "tiles", tiles }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskCTCDecoder(const std::vector< Data::TensorIterator >& inputs, const bool& ctc_merge_repeated, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("CTCDecoder") },
            { "ctc_merge_repeated", ctc_merge_repeated }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskRefConv(const std::vector< Data::TensorIterator >& inputs, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor, const unsigned& group, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("RefConv") },
            { "stride", stride },
            { "padding", padding }, 
            { "dilationFactor", dilationFactor },
            { "group", group },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::uPATaskGather(const std::vector< Data::TensorIterator >& inputs, const unsigned& axis, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Gather") },
            { "axis", axis }, 
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
    return output;
}

mv::Data::TensorIterator mv::OpModel::weightsTable(const std::vector<int64_t>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    auto output = defineOp(
        "WeightsTable",
        {
        },
        {
            { "data", data },
            { "shape", shape },
            { "dType", dType },
            { "order", order }, 
            { "quantParams", quantParams }
        },
        name
    
    
    );
    return output;
}

mv::Data::OpListIterator mv::OpModel::getSourceOp(Data::TensorIterator tensor)
{
    return BaseOpModel::getSourceOp(tensor);
}
void mv::OpModel::addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr)
{
    return BaseOpModel::addAttr(op, name, attr);
}
bool mv::OpModel::isValid() const
{
    return BaseOpModel::isValid();
}
bool mv::OpModel::isValid(Data::TensorIterator tensor) const
{
    return BaseOpModel::isValid(tensor);
}
bool mv::OpModel::isValid(Data::OpListIterator op) const
{
    return BaseOpModel::isValid(op);
}
std::string mv::OpModel::getName() const
{
    return BaseOpModel::getName();
}

mv::OpModel::~OpModel()
{
}

