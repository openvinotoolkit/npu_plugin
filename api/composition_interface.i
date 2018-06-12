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
%{
    #include <include/fathom/computation/model/op_model.hpp>
    #include <include/fathom/computation/model/model.hpp>
    #include <include/fathom/computation/model/iterator/model_iterator.hpp>
    #include <include/fathom/computation/tensor/shape.hpp>
    #include <include/fathom/computation/tensor/constant.hpp>
    #include <include/fathom/computation/model/attribute.hpp>
    #include <include/fathom/computation/model/control_model.hpp>
    #include <include/fathom/deployer/serializer.hpp>
    #include <string>

    int serialize(mv::OpModel * test_cm){
        mv::ControlModel *cm = new mv::ControlModel(*test_cm);
        mv::Serializer *gs = new mv::Serializer(mv::mvblob_mode);
        uint64_t filesize = gs->serialize(*cm, "cpp.blob");

        return filesize;   // Success, return filesize
    }

    int testSWIG(){
        /// A simple test to ensure the connection between Python and C++ is working
        int test = 1;
        return test;
    }

    mv::OpModel * getOM(){
        /// Get a blank OpModel
        mv::OpModel *om = new mv::OpModel();
        return om;
    }

    mv::Shape * getShape(int w, int x, int y, int z){
        /// Create a c++ shape object from a passed in set of dimension sizes
        mv::Shape* a = new mv::Shape(w, x, y, z);
        return a;
    }

    mv::vector<mv::float_type> * getData(float * d, size_t len){
        /// Populate a Vector with a numpy array.
        mv::vector<mv::float_type> * weightsData = new mv::vector<mv::float_type>(d, len);
        return weightsData;
    }

    mv::ConstantTensor * getConstantTensor(mv::Shape * s, mv::vector<mv::float_type> data ){
        /// Create the internal ConstantTensor which is needed for future function calls
        mv::ConstantTensor * a = new mv::ConstantTensor(*s, mv::DType::Float, mv::Order::NWHC, data);
        return a;
    }

    int testConv(
        mv::DataContext::OpListIterator &target,
        int exp_strideX,
        int exp_strideY,
        int exp_padX,
        int exp_padY
        ){
        /// A couple of simple checks to ensure we have loaded the items correctly.

        int ret_val = 0;    // Success
        if(target->getAttr("strideX").getContent<mv::byte_type>() != exp_strideX)
            ret_val = 1;
        if(target->getAttr("strideY").getContent<mv::byte_type>() != exp_strideY)
            ret_val = 2;
        if(target->getAttr("padX").getContent<mv::byte_type>() != exp_padX)
            ret_val = 3;
        if(target->getAttr("padY").getContent<mv::byte_type>() != exp_padY)
            ret_val = 4;

        return ret_val;
    }

    mv::DataContext::OpListIterator input(mv::OpModel * o, const mv::Shape &shape){
        /// Add an Input Layer to the OpModel and return the relevant iterator
        return o->input(shape, mv::DType::Float, mv::Order::NWHC);
    }

    mv::DataContext::OpListIterator output(mv::OpModel * o, mv::DataContext::OpListIterator &predecessor){
        /// Add an Output Layer to the OpModel and return the relevant iterator
        return o->output(predecessor);
    }

    mv::DataContext::OpListIterator maxpool(mv::OpModel * o, mv::DataContext::OpListIterator &predecessor, const mv::Shape &shape,
        uint8_t strideX, uint8_t strideY, uint8_t padX, uint8_t padY){
        /// Add a Max Pooling Layer to the OpModel and return the relevant iterator
        return o->maxpool(predecessor, shape, strideX, strideY, padX, padY);
    }

    mv::DataContext::OpListIterator concat(mv::OpModel * o, mv::DataContext::OpListIterator &in0, mv::DataContext::OpListIterator &in1){
        /// Add a Concat Layer to the OpModel and return the relevant iterator.
        /// Allows only two inputs at a time. More must cascade
        return o->concat(in0, in1);
    }

    mv::DataContext::OpListIterator conv(mv::OpModel * o, mv::DataContext::OpListIterator &predecessor, const mv::ConstantTensor &weights,
        uint8_t strideX, uint8_t strideY, uint8_t padX, uint8_t padY){
        /// Add a Convolutional Layer to the OpModel and return the relevant iterator
        return o->conv(predecessor, weights, strideX, strideY, padX, padY);
    }
 %}

#include <include/fathom/computation/model/op_model.hpp>
#include <include/fathom/computation/model/model.hpp>
#include <include/fathom/computation/model/iterator/model_iterator.hpp>
#include <include/fathom/computation/model/iterator/data_context.hpp>
#include <include/fathom/computation/model/control_model.hpp>
#include <include/fathom/computation/tensor/shape.hpp>
#include <include/fathom/computation/tensor/constant.hpp>
#include <include/fathom/computation/model/attribute.hpp>
#include <include/fathom/computation/api/compositional_model.hpp>
#include <include/fathom/deployer/serializer.hpp>
#include <include/fathom/computation/model/model.hpp>
#include <include/fathom/computation/op/input.hpp>
#include <include/fathom/computation/op/output.hpp>
#include <include/fathom/computation/op/conv.hpp>
#include <include/fathom/computation/op/maxpool.hpp>
#include <include/fathom/computation/op/concat.hpp>
#include <string>

// The section below is exposing the functions within the included files,
// or the ones defined above in the module.

namespace mv
{
    class OpModel
    {
    public:
        bool isValid() const;
    };
}

int testSWIG();
mv::OpModel * getOM();
mv::Shape * getShape(int w, int x, int y, int z);
mv::ConstantTensor * getConstantTensor(mv::Shape * s, mv::vector<mv::float_type> data );


// Expand a numpy array to a data pointer and a length
%include "stdint.i"
%apply (float* INPLACE_ARRAY1, int DIM1) {(float* d, int len)}
mv::vector<mv::float_type> * getData(float * d, int len);


mv::DataContext::OpListIterator input(mv::OpModel * o, const mv::Shape &shape);
mv::DataContext::OpListIterator output(mv::OpModel * o, mv::DataContext::OpListIterator &predecessor);
mv::DataContext::OpListIterator conv(mv::OpModel * o, mv::DataContext::OpListIterator &predecessor, const mv::ConstantTensor &weights,
    uint8_t strideX, uint8_t strideY, uint8_t padX, uint8_t padY);
mv::DataContext::OpListIterator maxpool(mv::OpModel * o, mv::DataContext::OpListIterator &predecessor, const mv::Shape &shape,
    uint8_t strideX, uint8_t strideY, uint8_t padX, uint8_t padY);
mv::DataContext::OpListIterator concat(mv::OpModel * o, mv::DataContext::OpListIterator &in0, mv::DataContext::OpListIterator &in1);

int testConv(
    mv::DataContext::OpListIterator &target,
    int exp_strideX,
    int exp_strideY,
    int exp_padX,
    int exp_padY
);

int serialize(mv::OpModel * test_cm);