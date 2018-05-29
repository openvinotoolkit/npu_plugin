 %module composition_api
 %{
    #include "include/fathom/computation/model/op_model.hpp"

    mv::OpModel om;

    int getOM(){
        int test = 1;
        return test;
    }

    mv::IteratorType getShape(int w, int x, int y, int z){
        return mv:Shape(w, x, y, z);
    }

    mv::DType getDtype(){
        return mv::DType::Float;
    }

    mv::Order getOrder(int enum){
        return mv::Order::NWHC;
    }

    mv::vector<mv::float_type> getData(float* d, int len){
        return mv::vector<mv::float_type> weightsData = {
           1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
           10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
           19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f
        };
    }

    mv::ConstantTensor getConstantTensor(mv::Shape s, mv::DType d, mv::Order o, mv::vector<mv::float_type> data ){
        return 0;
    }

    int getAttrByte(Attribute a){
        return a.getContent<mv::byte_type>();
    }
 %}

int getOM();