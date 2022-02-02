// RUN: vpux-translate --export-IE -o %t.xml %s && FileCheck %s --input-file %t.xml

module @Test {
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "data" : tensor<1x3x224x224xf16>
  } outputsInfo :  {
    DataInfo "prob" : tensor<1x3x224x224xf16>
  }
  func @main(%arg0: tensor<1x3x224x224xf16>) -> tensor<1x3x224x224xf16> {
    %cst = const.Declare tensor<1x1x1x1xf16> = #const.Content<dense<1.699830e-02> : tensor<1x1x1x1xf16>>
    %0 = IE.Multiply(%arg0, %cst) {auto_broadcast = "NUMPY"} : tensor<1x3x224x224xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x224x224xf16>
    %cst_0 = const.Declare tensor<1x3x1x1xf16> = #const.Content<dense<[[[[-1.766600e+00]], [[-1.985350e+00]], [[-2.103520e+00]]]]> : tensor<1x3x1x1xf16>>
    %1 = IE.Add(%0, %cst_0) {auto_broadcast = "NUMPY"} : tensor<1x3x224x224xf16>, tensor<1x3x1x1xf16> -> tensor<1x3x224x224xf16>
    return %1 : tensor<1x3x224x224xf16>
  }
}


// CHECK:       <?xml version="1.0"?>
// CHECK-NEXT:  <net
// CHECK-DAG:   name="{{[_A-Za-z0-9]+}}"
// CHECK-DAG:   version="10"
// CHECK_SAME:  >
// CHECK-NEXT:    <layers>
// CHECK-NEXT:      <layer
// CHECK-DAG:       id="0"
// CHECK-DAG:       name="{{[_A-Za-z0-9]+}}"
// CHECK-DAG:       type="Parameter"
// CHECK-DAG:       version="opset1"
// CHECK-SAME:      >
// CHECK-NEXT:        <data
// CHECK-DAG:         element_type="f16"
// CHECK-DAG:         shape="1, 3, 224, 224"
// CHECK-SAME:        />
// CHECK-NEXT:        <output>
// CHECK-NEXT:          <port
// CHECK-DAG:           id="0"
// CHECK-DAG:           precision="FP16"
// CHACK-SAME:          >
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:            <dim>3</dim>
// CHECK-NEXT:            <dim>224</dim>
// CHECK-NEXT:            <dim>224</dim>
// CHECK-NEXT:          </port>
// CHECK-NEXT:        </output>
// CHECK-NEXT:      </layer>
// CHECK-NEXT:      <layer
// CHECK-DAG:       id="1"
// CHECK-DAG:       name="{{[_A-Za-z0-9]+}}"
// CHECK-DAG:       type="Const"
// CHECK-DAG:       version="opset1"
// CHECK-SAME:      >
// CHECK-NEXT:        <data
// CHECK-DAG:         element_type="f16"
// CHECK-DAG:         offset="0"
// CHECK-DAG:         shape="1, 1, 1, 1"
// CHECK-DAG:         size="2"
// CHECK-SAME:        />
// CHECK-NEXT:        <output>
// CHECK-NEXT:          <port
// CHECK-DAG:           id="0"
// CHECK-DAG:           precision="FP16"
// CHECK-SAME:          >
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:          </port>
// CHECK-NEXT:        </output>
// CHECK-NEXT:      </layer>
// CHECK-NEXT:      <layer
// CHECK-DAG:       id="2"
// CHECK-DAG:       type="Multiply"
// CHECK-DAG:       version="opset1"
// CHECK-SAME:      >
// CHECK-NEXT:        <data auto_broadcast="numpy" />
// CHECK-NEXT:        <input>
// CHECK-NEXT:          <port
// CHECK-DAG:           id="0"
// CHECK-DAG:           precision="FP16"
// CHECK-SAME:          >
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:            <dim>3</dim>
// CHECK-NEXT:            <dim>224</dim>
// CHECK-NEXT:            <dim>224</dim>
// CHECK-NEXT:          </port>
// CHECK-NEXT:          <port
// CHECK-DAG:           id="1"
// CHECK-DAG:           precision="FP16"
// CHECK-SAME:          >
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:          </port>
// CHECK-NEXT:        </input>
// CHECK-NEXT:        <output>
// CHECK-NEXT:          <port
// CHECK_DAG:           id="2"
// CHECK-DAG:           precision="FP16"
// CHECK-SAME:          >
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:            <dim>3</dim>
// CHECK-NEXT:            <dim>224</dim>
// CHECK-NEXT:            <dim>224</dim>
// CHECK-NEXT:         </port>
// CHECK-NEXT:        </output>
// CHECK-NEXT:      </layer>
// CHECK-NEXT:      <layer
// CHACK-DAG:       id="3"
// CHACK-DAG:       name="%.*"
// CHACK-DAG:       type="Const"
// CHACK-DAG:       version="opset1"
// CHACK-SAME:      >
// CHECK-NEXT:        <data
// CHACK-DAG:         element_type="f16"
// CHACK-DAG:         offset="2"
// CHACK-DAG:         shape="1, 3, 1, 1"
// CHACK-DAG:         size="6"
// CHACK-SAME:        />
// CHECK-NEXT:        <output>
// CHECK-NEXT:          <port
// CHACK-DAG:           id="0"
// CHACK-DAG:           precision="FP16"
// CHACK-SAME:          >
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:            <dim>3</dim>
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:          </port>
// CHECK-NEXT:        </output>
// CHECK-NEXT:      </layer>
// CHECK-NEXT:      <layer
// CHACK-DAG:       id="4"
// CHACK-DAG:       name="{{[_A-Za-z0-9]+}}"
// CHACK-DAG:       type="Add"
// CHACK-DAG:       version="opset1"
// CHACK-SAME:      >
// CHECK-NEXT:        <data auto_broadcast="numpy" />
// CHECK-NEXT:        <input>
// CHECK-NEXT:          <port
// CHACK-DAG:           id="0"
// CHACK-DAG:           precision="FP16"
// CHACK-SAME:          >
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:            <dim>3</dim>
// CHECK-NEXT:            <dim>224</dim>
// CHECK-NEXT:            <dim>224</dim>
// CHECK-NEXT:          </port>
// CHECK-NEXT:          <port
// CHACK-DAG:           id="1"
// CHACK-DAG:           precision="FP16"
// CHACK-SAME:          >
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:            <dim>3</dim>
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:          </port>
// CHECK-NEXT:        </input>
// CHECK-NEXT:        <output>
// CHECK-NEXT:          <port
// CHECK_DAG:           id="2"
// CHECK-DAG:           precision="FP16"
// CHECK-SAME:          >
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:            <dim>3</dim>
// CHECK-NEXT:            <dim>224</dim>
// CHECK-NEXT:            <dim>224</dim>
// CHECK-NEXT:          </port>
// CHECK-NEXT:        </output>
// CHECK-NEXT:      </layer>
// CHECK-NEXT:      <layer
// CHACK-DAG:       id="5"
// CHACK-DAG:       name="{{[_A-Za-z0-9]+}}"
// CHACK-DAG:       type="Result"
// CHACK-DAG:       version="opset1"
// CHACK-SAME:      >
// CHECK-NEXT:        <input>
// CHECK-NEXT:          <port
// CHACK-DAG:           id="0"
// CHACK-DAG:           precision="FP16"
// CHACK-SAME:          >
// CHECK-NEXT:            <dim>1</dim>
// CHECK-NEXT:            <dim>3</dim>
// CHECK-NEXT:            <dim>224</dim>
// CHECK-NEXT:            <dim>224</dim>
// CHECK-NEXT:          </port>
// CHECK-NEXT:        </input>
// CHECK-NEXT:      </layer>
// CHECK-NEXT:    </layers>
// CHECK-NEXT:  <edges>
// CHECK-NEXT:    <edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
// CHECK-NEXT:    <edge from-layer="1" from-port="0" to-layer="2" to-port="1" />
// CHECK-NEXT:    <edge from-layer="2" from-port="2" to-layer="4" to-port="0" />
// CHECK-NEXT:    <edge from-layer="3" from-port="0" to-layer="4" to-port="1" />
// CHECK-NEXT:    <edge from-layer="4" from-port="2" to-layer="5" to-port="0" />
// CHECK-NEXT:  </edges>
// CHECK-NEXT:</net>
