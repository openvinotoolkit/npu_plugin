
/// SWIG Interface File.

// Include the external numpy swig bindings.
%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
import_array();
%}

%module composition_api
%include "std_string.i"
%include exception.i
%{
    #include "include/mcm/compiler/compilation_unit.hpp"
    #include "include/mcm/tensor/quantization_params.hpp"
    #include "include/mcm/utils/env_loader.hpp"
    #include <math.h>
    #include <iostream>

    int testSWIG(){
        /// A simple test to ensure the connection between Python and C++ is working
        int test = 1;
        return test;
    }

    std::string getProjectPath()
    {
        return mv::utils::projectRootPath();
    }

    mv::CompilationUnit* getCompilationUnit()
    {
        mv::Logger::instance().log(mv::Logger::MessageType::Info, "Python SWIG bridge", "Starting MCM Composition Interface for Target Descriptor: ma2490...");
        auto unit = new mv::CompilationUnit("pySwigCU");
        unit->loadTargetDescriptor(mv::Target::ma2490);
        unit->loadCompilationDescriptor(mv::Target::ma2490);

        return unit;
    }

    mv::CompilationUnit* getCompilationUnit(const std::string& target)
    {
        mv::Logger::instance().log(mv::Logger::MessageType::Info, "Python SWIG bridge", "Starting MCM Composition Interface for Target Descriptor: " + target + " ...");
        auto unit = new mv::CompilationUnit("pySwigCU");
        if(target.compare("ma2480") == 0)
        {            unit->loadTargetDescriptor(mv::Target::ma2480);
            unit->loadCompilationDescriptor(mv::Target::ma2480);
        }
        else if(target.compare("ma2490") == 0)
        {
            unit->loadTargetDescriptor(mv::Target::ma2490);
            unit->loadCompilationDescriptor(mv::Target::ma2490);
        }
        else
        {
            //Throw an error as unsupported target descriptor type supplied
            PyErr_SetString(PyExc_Exception, "Target descriptor type not supported. Only ma2480 and ma2490 supported.");
        }

        return unit;
    }

    mv::CompilationUnit* loadCompilationDescriptor(mv::CompilationUnit *unit, const std::string& filepath)
    {
        mv::Logger::instance().log(mv::Logger::MessageType::Info, "Python SWIG bridge", "Loading custom Compilation Descriptor: " + filepath + " ...");
        //remove default descriptor and load a user defined descriptor
        unit->compilationDescriptor().clear();
        unit->loadCompilationDescriptor(filepath);

        return unit;
    }

    mv::CompositionalModel* getModel(mv::CompilationUnit *unit)
    {
        return &unit->model();
    }

    void deleteCompilationUnitObject(mv::CompilationUnit *unit)
    {
        delete unit;
    }

    void setLogLevel(const std::string& logLevel)
    {
        if(logLevel.compare("debug") == 0)
        {
            mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);
        }
        else if(logLevel.compare("info") == 0)
        {
            mv::Logger::setVerboseLevel(mv::VerboseLevel::Info);
        }
        else if(logLevel.compare("warning") == 0)
        {
            mv::Logger::setVerboseLevel(mv::VerboseLevel::Warning);
        }
        else if(logLevel.compare("silent") == 0)
        {
            mv::Logger::setVerboseLevel(mv::VerboseLevel::Silent);
        }
        else //default "error"
        {
            mv::Logger::setVerboseLevel(mv::VerboseLevel::Error);
        }
    }

    int compile(mv::CompilationUnit *unit)
    {
        unit->initialize();
        auto compOutput = unit->run();

        return 0;
    }

    // TODO: Create Generic Vector Calls
    std::array<unsigned short, 2> * get2DVector(int x, int y){
        std::array<unsigned short, 2> *arr = new std::array<unsigned short, 2>();
        arr->at(0) = x;
        arr->at(1) = y;
        return arr;
    }

    std::array<unsigned short, 4> * get4DVector(int w, int x, int y, int z){
        std::array<unsigned short, 4> *arr = new std::array<unsigned short, 4>();
        arr->at(0) = w;
        arr->at(1) = x;
        arr->at(2) = y;
        arr->at(3) = z;
        return arr;
    }

    mv::Shape * getShape(std::size_t x){
        /// Create a c++ shape object from a passed in set of dimension sizes
        mv::Shape* a = new mv::Shape({x});
        return a;
    }

    mv::Shape * getShape(std::size_t x, std::size_t y){
        /// Create a c++ shape object from a passed in set of dimension sizes
        mv::Shape* a = new mv::Shape({x, y});
        return a;
    }

    mv::Shape * getShape(std::size_t x, std::size_t y, std::size_t z){
        /// Create a c++ shape object from a passed in set of dimension sizes
        mv::Shape* a = new mv::Shape({x, y, z});
        return a;
    }

    mv::Shape * getShape(std::size_t b, std::size_t x, std::size_t y, std::size_t z){
        /// Create a c++ shape object from a passed in set of dimension sizes
        mv::Shape* a = new mv::Shape({b, x, y, z});
        return a;
    }


    mv::QuantizationParams * getQuantParams(const std::vector<int64_t>& zero_data, const std::vector<double>& scale_data,  const std::vector<double>& min,  const std::vector<double>& max){
        /// Create a c++ shape object from a passed in set of dimension sizes
        mv::QuantizationParams * quant = new mv::QuantizationParams(zero_data, scale_data, min, max);
        return quant;
    }

    mv::Order * getOrder(const std::string& framework_layout){
        /// Create a c++ shape object from a passed in set of dimension sizes
        mv::Order * order = new mv::Order(framework_layout);
        return order;
    }

    mv::DType * getDtypeFP16(){
        mv::DType * dtype = new mv::DType("Float16");
        return dtype;
    }

    std::vector<double> * getData(double * d, std::size_t len){
        /// Populate a Vector with a numpy array.
        std::vector<double> * weightsData = new std::vector<double>(d, d + len);
        return weightsData;
    }

    std::vector<int64_t> * getData(int64_t * d, std::size_t len){
        /// Populate a Vector with a numpy array.
        std::vector<int64_t> * weightsData = new std::vector<int64_t>(d, d + len);
        return weightsData;
    }

    int testConv(
        mv::Data::OpListIterator &target,
        unsigned exp_strideX,
        unsigned exp_strideY,
        unsigned exp_padX,
        unsigned exp_padY
        ){
        /// A couple of simple checks to ensure we have loaded the items correctly.

        int ret_val = 0;    // Success
        std::array<unsigned short, 2> stride = target->get<std::array<unsigned short, 2>>("stride");
        std::array<unsigned short, 4> pad = target->get<std::array<unsigned short, 4>>("padding");
        if(stride.at(0) != exp_strideX)
            ret_val = 1;
        if(stride.at(1) != exp_strideY)
            ret_val = 2;
        if(pad.at(1) != exp_padX)
            ret_val = 3;
        if(pad.at(3) != exp_padY)
            ret_val = 4;
        // TODO Consider assymetric padding

        return ret_val;
    }

    mv::Data::TensorIterator identity(mv::CompositionalModel& o,mv::Data::TensorIterator input0, const std::string& type, const mv::QuantizationParams  &quantParams, const std::string &name){
        return o.identity(input0, type, quantParams, name);
    }

    mv::Data::TensorIterator constant(mv::CompositionalModel& o, const std::vector<double>& data, const mv::Shape &shape, const mv::Order& order, const mv::QuantizationParams &quantParams, const std::string &name){
        /// Add a Constant Layer to the CompositionalModel and return the relevant iterator
        return o.constant(data, shape, mv::DType("Float64"), order, quantParams, name);
    }

    mv::Data::TensorIterator constant(mv::CompositionalModel& o, const std::vector<int64_t> &data, std::string dataType, const mv::Shape &shape, const mv::Order& order, const mv::QuantizationParams &quantParams, const std::string &name){
        return o.constantInt(data, shape, mv::DType(dataType), order, quantParams, name);
    }

    mv::Data::TensorIterator input(mv::CompositionalModel& o, const mv::Shape &shape, double type, const mv::Order& order, const mv::QuantizationParams &quantParams, const std::string& name){
        /// Add an Input Layer to the OpModel and return the relevant iterator
          return o.input(shape, mv::DType("Float64"), order, quantParams, name);
    }

    mv::Data::TensorIterator input(mv::CompositionalModel& o, const mv::Shape &shape, uint64_t type, const mv::Order& order, const mv::QuantizationParams &quantParams, const std::string& name){
        /// Add an Input Layer to the OpModel and return the relevant iterator
        return o.input(shape, mv::DType("UInt8"), order, quantParams, name);
    }

    mv::Data::TensorIterator output(mv::CompositionalModel& o, mv::Data::TensorIterator input){
        /// Add an Output Layer to the OpModel and return the relevant iterator
        return o.output(input);
    }

    mv::Data::TensorIterator maxpool2D(mv::CompositionalModel& o, mv::Data::TensorIterator input, short unsigned kernelSizeX,
        short unsigned kernelSizeY, short unsigned strideX, short unsigned strideY, short unsigned padXl, short unsigned padXr,  short unsigned padYu,short unsigned padYd, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name){
        /// Add a Max Pooling Layer to the OpModel and return the relevant iterator
        return o.maxPool(input, {kernelSizeX, kernelSizeY}, {strideX, strideY},
            {padXl, padXr, padYu, padYd}, false,"","floor", type, quantParams, name);
    }

    mv::Data::TensorIterator avgpool2D(mv::CompositionalModel& o, mv::Data::TensorIterator input, short unsigned kernelSizeX,
        short unsigned kernelSizeY, short unsigned strideX, short unsigned strideY, short unsigned padXl, short unsigned padXr,  short unsigned padYu,short unsigned padYd, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name)
    {
        return o.averagePool(input, {kernelSizeX, kernelSizeY}, {strideX, strideY}, {padXl, padXr, padYu, padYd}, false,"","floor", type, quantParams, name);
    }

    mv::Data::TensorIterator relu(mv::CompositionalModel& o, mv::Data::TensorIterator input, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name){
        return o.relu(input, type, quantParams, name);
    }

    mv::Data::TensorIterator leaky_relu(mv::CompositionalModel& o, mv::Data::TensorIterator input, double alpha, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name){
        return o.leakyRelu(input, alpha, type, quantParams, name);
    }

    mv::Data::TensorIterator sigmoid(mv::CompositionalModel& o, mv::Data::TensorIterator input, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name){
        return o.sigmoid(input, type, quantParams, name);
    }

    mv::Data::TensorIterator dropOut(mv::CompositionalModel& o, mv::Data::TensorIterator input, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name){
        return o.dropout(input, type, quantParams, name);
    }

    mv::Data::TensorIterator minimum(mv::CompositionalModel& o, mv::Data::TensorIterator input, double minimum, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name){
        return o.minimum(input, minimum, type, quantParams, name);
    }

    mv::Data::TensorIterator maximum(mv::CompositionalModel& o, mv::Data::TensorIterator input, double maximum, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name){
        return o.maximum(input, maximum, type, quantParams, name);
    }

    mv::Data::TensorIterator power(mv::CompositionalModel& o, mv::Data::TensorIterator input0, mv::Data::TensorIterator input1, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name){
        return o.eltwise({input0, input1}, "Power", type, quantParams, name);
    }

    mv::Data::TensorIterator eltwiseMinimum(mv::CompositionalModel& o, mv::Data::TensorIterator input, mv::Data::TensorIterator input1, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name){
        return o.eltwise({input, input1}, "Minimum", type, quantParams, name);
    }

    mv::Data::TensorIterator eltwiseMaximum(mv::CompositionalModel& o, mv::Data::TensorIterator input, mv::Data::TensorIterator input1, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name){
        return o.eltwise({input, input1}, "Maximum", type, quantParams, name);
    }

    mv::Data::TensorIterator eltwise(mv::CompositionalModel& o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1, const std::string& eltWiseType, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name){
        return o.eltwise({input0, input1}, eltWiseType, type, quantParams, name);
    }

    mv::Data::TensorIterator divide(mv::CompositionalModel& o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name){
        return o.eltwise({input0, input1}, "Divide", type, quantParams, name);
    }

    mv::Data::TensorIterator softmax(mv::CompositionalModel& o,mv::Data::TensorIterator input, const std::string& type, const mv::QuantizationParams  &quantParams, const std::string &name){
        return o.softmax(input, "C", type, quantParams, name);
    }

    mv::Data::TensorIterator fullyConnected(mv::CompositionalModel& o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1, const std::string& type, const mv::QuantizationParams  &quantParams, const std::string &name){
        return o.fullyConnected(input0, input1, type, quantParams, name);
    }

    mv::Data::TensorIterator conv2D(mv::CompositionalModel& o, mv::Data::TensorIterator input, mv::Data::TensorIterator filters,
        short unsigned strideX, short unsigned strideY, short unsigned padXl, short unsigned padXr,  short unsigned padYu, short unsigned padYd, short unsigned dilationFactor, short unsigned group, const std::string &type, const mv::QuantizationParams  &quantParams, const std::string& name){
        /// Add a Convolutional Layer to the OpModel and return the relevant iterator
        return o.conv(input, filters, {strideX, strideY}, {padXl, padXr, padYu, padYd}, dilationFactor, group, type, quantParams, name);
    }

    mv::Data::TensorIterator bias(mv::CompositionalModel& o, mv::Data::TensorIterator input, mv::Data::TensorIterator bias_values, const std::string& type, const mv::QuantizationParams  &quantParams, const std::string &name){
        return o.bias(input, bias_values, type, quantParams, name);
    }

    mv::Data::TensorIterator depthwiseConv2D(mv::CompositionalModel& o, mv::Data::TensorIterator input, mv::Data::TensorIterator filters,
        short unsigned strideX, short unsigned strideY, short unsigned padXl, short unsigned padXr,  short unsigned padYu,short unsigned padYd, short unsigned dilationFactor, const std::string& type, const mv::QuantizationParams  &quantParams, const std::string &name){
        /// Add a Convolutional Layer to the OpModel and return the relevant iterator
        return o.depthwiseConv(input, filters, {strideX, strideY}, {padXl, padXr, padYu, padYd}, dilationFactor, type, quantParams, name);
    }

    mv::Data::TensorIterator concat(mv::CompositionalModel& o, std::vector<mv::Data::TensorIterator> * inputs, const std::string& type, const mv::QuantizationParams  &quantParams, const std::string &name){
        /// Add a Concat Layer to the OpModel and return the relevant iterator.
        return o.concat(*inputs, "C", type, quantParams, name);
    }

    mv::Data::TensorIterator scale(mv::CompositionalModel& o,mv::Data::TensorIterator input, mv::Data::TensorIterator scale, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name){
        return o.scale(input, scale, type, quantParams, name);
    }

    std::vector<mv::Data::TensorIterator> * pushVector(std::vector<mv::Data::TensorIterator> * base, mv::Data::TensorIterator data){
        if(base == nullptr)
            base = new std::vector<mv::Data::TensorIterator>();
        base->push_back(data);
        return base;
    }


    mv::Data::OpListIterator getSourceOp(mv::CompositionalModel& o, mv::Data::TensorIterator tensor){
        // Get source operation of a tensor
        return o.getSourceOp(tensor);
    }

    mv::Data::TensorIterator matMul(mv::CompositionalModel& o, mv::Data::TensorIterator input, mv::Data::TensorIterator weights){
        return o.matMul(input, weights);
    }

    mv::Data::TensorIterator batchNorm(mv::CompositionalModel& o,mv::Data::TensorIterator input, mv::Data::TensorIterator mean, mv::Data::TensorIterator variance, mv::Data::TensorIterator offset, mv::Data::TensorIterator scale, double varianceEps){
        return o.batchNormalization(input, mean, variance, offset, scale, varianceEps);
    }


    mv::Data::TensorIterator prelu(mv::CompositionalModel& o, mv::Data::TensorIterator input, mv::Data::TensorIterator negative_slope){
        return o.prelu(input, negative_slope);
    }


    mv::Data::TensorIterator reshape(mv::CompositionalModel& o,mv::Data::TensorIterator input, const mv::Shape& shape, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name){
        return o.reshape(input, shape, type, quantParams, name);
    }

    mv::Data::TensorIterator reorgYolo(mv::CompositionalModel& o,mv::Data::TensorIterator input, const unsigned& stride, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name){
        return o.reorgYolo(input, stride, type, quantParams, name);
    }

    bool isValid(mv::CompositionalModel& o){
        return o.isValid();
    }

 %}

#include <include/mcm/computation/model/control_model.hpp>
#include <include/mcm/deployer/serializer.hpp>
#include <include/mcm/compiler/compilation_unit.hpp>

// The section below is exposing the functions within the included files,
// or the ones defined above in the module.

namespace mv
{

    namespace Data
    {
        class TensorIterator
        {
        public:
            ~TensorIterator();
        };

        class OpListIterator
        {
        public:
            ~OpListIterator();
        };
    }
}
#define SWIGWORDSIZE64

int testSWIG();
std::string getProjectPath();
mv::CompilationUnit* getCompilationUnit();
mv::CompilationUnit* getCompilationUnit(const std::string& target);
mv::CompilationUnit* loadCompilationDescriptor(mv::CompilationUnit *unit, const std::string& filepath);
mv::CompositionalModel* getModel(mv::CompilationUnit *unit);
int compile(mv::CompilationUnit *unit);
void deleteCompilationUnitObject(mv::CompilationUnit *unit);
mv::Shape * getShape(int x);
mv::Shape * getShape(int x, int y);
mv::Shape * getShape(int x, int y, int z);
mv::Shape * getShape(int b, int x, int y, int z);
std::array<unsigned short, 2> * get2DVector(int x, int y);
std::array<unsigned short, 4> * get4DVector(int w, int x, int y, int z);

// Expand a numpy array to a data pointer and a length
%include "stdint.i"
%apply (double* INPLACE_ARRAY1, std::size_t DIM1) {(double* d, std::size_t len)}
std::vector<double> * getData(double * d, std::size_t len);
%apply (int64_t* INPLACE_ARRAY1, std::size_t DIM1) {(int64_t* d, std::size_t len)}
std::vector<int64_t> * getData(int64_t * d, std::size_t len);

mv::QuantizationParams * getQuantParams(const std::vector<int64_t> &zero_data, const std::vector<double>& scale_data, const std::vector<double>& min,  const std::vector<double>& max);
mv::Order * getOrder(const std::string& framework_layout);
mv::DType * getDtypeFP16();
//Keep the order of the Wrapper
mv::Data::TensorIterator identity(mv::CompositionalModel&  o,mv::Data::TensorIterator input0, const std::string& type, const mv::QuantizationParams  &quantParams, const std::string &name);
mv::Data::TensorIterator constant(mv::CompositionalModel&  o, const std::vector<int64_t>& data, const std::string dataType, const mv::Shape &shape, const mv::Order& order, const mv::QuantizationParams  &quantParams, const std::string &name);
mv::Data::TensorIterator constant(mv::CompositionalModel&  o, const std::vector<double>& data, const mv::Shape &shape, const mv::Order& order,  const mv::QuantizationParams  &quantParams,  const std::string &name);
mv::Data::TensorIterator input(mv::CompositionalModel& o, const mv::Shape &shape, double type, const mv::Order& order, const mv::QuantizationParams &quantParams, const std::string& name);
mv::Data::TensorIterator input(mv::CompositionalModel& o, const mv::Shape &shape, uint64_t type, const mv::Order& order, const mv::QuantizationParams &quantParams, const std::string& name);
mv::Data::TensorIterator output(mv::CompositionalModel& o, mv::Data::TensorIterator input);
mv::Data::TensorIterator maxpool2D(mv::CompositionalModel& o, mv::Data::TensorIterator input, short unsigned kernelSizeX,
    short unsigned kernelSizeY, short unsigned strideX, short unsigned strideY, short unsigned padXl, short unsigned padXr,  short unsigned padYu,short unsigned padYd, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name);
mv::Data::TensorIterator avgpool2D(mv::CompositionalModel& o, mv::Data::TensorIterator input, short unsigned kernelSizeX, short unsigned kernelSizeY, short unsigned strideX, short unsigned strideY, short unsigned padXl, short unsigned padXr,  short unsigned padYu,short unsigned padYd, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name);
mv::Data::TensorIterator relu(mv::CompositionalModel& o,mv::Data::TensorIterator input, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name);
mv::Data::TensorIterator leaky_relu(mv::CompositionalModel& o,mv::Data::TensorIterator input, double alpha, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name);
mv::Data::TensorIterator sigmoid(mv::CompositionalModel& o,mv::Data::TensorIterator input, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name);
mv::Data::TensorIterator dropOut(mv::CompositionalModel& o, mv::Data::TensorIterator input, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name);
mv::Data::TensorIterator minimum(mv::CompositionalModel& o, mv::Data::TensorIterator input, double minimum, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name);
mv::Data::TensorIterator maximum(mv::CompositionalModel& o, mv::Data::TensorIterator input, double maximum, const std::string &type, const mv::QuantizationParams &quantParams, const std::string& name);
mv::Data::TensorIterator power(mv::CompositionalModel& o, mv::Data::TensorIterator input0, mv::Data::TensorIterator input1, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name);
mv::Data::TensorIterator eltwiseMinimum(mv::CompositionalModel& o, mv::Data::TensorIterator input, mv::Data::TensorIterator input1, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name);
mv::Data::TensorIterator eltwiseMaximum(mv::CompositionalModel& o, mv::Data::TensorIterator input, mv::Data::TensorIterator input1, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name);
mv::Data::TensorIterator eltwise(mv::CompositionalModel& o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1, const std::string& eltWiseType, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name);
mv::Data::TensorIterator divide(mv::CompositionalModel& o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name);
mv::Data::TensorIterator softmax(mv::CompositionalModel& o,mv::Data::TensorIterator input, const std::string& type, const mv::QuantizationParams  &quantParams, const std::string &name);
mv::Data::TensorIterator fullyConnected(mv::CompositionalModel& o,mv::Data::TensorIterator input0, mv::Data::TensorIterator input1, const std::string& type, const mv::QuantizationParams  &quantParams, const std::string &name);
mv::Data::TensorIterator conv2D(mv::CompositionalModel& o, mv::Data::TensorIterator input, mv::Data::TensorIterator filters,
    short unsigned strideX, short unsigned strideY, short unsigned padXl, short unsigned padXr,  short unsigned padYu, short unsigned padYd, short unsigned dilationFactor, short unsigned group,  const std::string& type, const mv::QuantizationParams  &quantParams, const std::string& name);
mv::Data::TensorIterator depthwiseConv2D(mv::CompositionalModel& o, mv::Data::TensorIterator input, mv::Data::TensorIterator filters,
    short unsigned strideX, short unsigned strideY, short unsigned padXl, short unsigned padXr,  short unsigned padYu,short unsigned padYd, short unsigned dilationFactor, const std::string& type, const mv::QuantizationParams  &quantParams, const std::string& name);
mv::Data::TensorIterator concat(mv::CompositionalModel& o, std::vector<mv::Data::TensorIterator> * inputs, const std::string& type, const mv::QuantizationParams  &quantParams, const std::string &name);
mv::Data::TensorIterator scale(mv::CompositionalModel& o,mv::Data::TensorIterator input, mv::Data::TensorIterator scale, const std::string& type, const mv::QuantizationParams  &quantParams, const std::string &name);
mv::Data::TensorIterator bias(mv::CompositionalModel& o, mv::Data::TensorIterator input, mv::Data::TensorIterator bias_values, const std::string& type, const mv::QuantizationParams  &quantParams, const std::string &name);

std::vector<mv::Data::TensorIterator> * pushVector(std::vector<mv::Data::TensorIterator> * base, mv::Data::TensorIterator data);
mv::Data::OpListIterator getSourceOp(mv::CompositionalModel& o, mv::Data::TensorIterator tensor);

mv::Data::TensorIterator matMul(mv::CompositionalModel& o, mv::Data::TensorIterator input, mv::Data::TensorIterator weights);
mv::Data::TensorIterator batchNorm(mv::CompositionalModel& o,mv::Data::TensorIterator input, mv::Data::TensorIterator mean, mv::Data::TensorIterator variance, mv::Data::TensorIterator offset, mv::Data::TensorIterator scale, double varianceEps);

mv::Data::TensorIterator reshape(mv::CompositionalModel& o,mv::Data::TensorIterator input, const mv::Shape& shape, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name);
mv::Data::TensorIterator reorgYolo(mv::CompositionalModel& o,mv::Data::TensorIterator input, const unsigned& stride, const std::string& type, const mv::QuantizationParams &quantParams, const std::string& name);

mv::Data::TensorIterator prelu(mv::CompositionalModel& o, mv::Data::TensorIterator input, mv::Data::TensorIterator negative_slope);
bool isValid(mv::CompositionalModel& o);
/** Sets Verbose Logging Level. Values are silent, error, warning, info, debug*/
void setLogLevel(const std::string& logLevel);

int testConv(
    mv::Data::OpListIterator &target,
    int exp_strideX,
    int exp_strideY,
    int exp_padX,
    int exp_padY
);
