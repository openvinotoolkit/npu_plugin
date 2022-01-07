//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#ifndef VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_TYPES
#define VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_TYPES

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/VPUIPRegMapped/dialect.td"


class VPUIPRegMapped_Type<string name> : TypeDef<VPUIPRegMapped_Dialect, name> {
    let mnemonic = name;
}


//
// IndexType
//


def VPUIPRegMapped_IndexType : VPUIPRegMapped_Type<"Index"> {

  let summary = "VPUIPRegMapped Index type";

  let description = [{
    An index type containing the value as a parameter
  }];

  let parameters = (ins "uint32_t":$value);

  // We define the printer inline.
  let printer = [{
    $_printer << "Index<" << getImpl()->value << ">";
  }];

  // The parser is defined here also.
  let parser = [{
    if ($_parser.parseLess())
      return Type();
    int value;
    if ($_parser.parseInteger(value))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, value);
  }];
}

#endif
