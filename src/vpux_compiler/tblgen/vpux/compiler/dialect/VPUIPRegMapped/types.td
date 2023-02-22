//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#ifndef VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_TYPES
#define VPUX_COMPILER_DIALECT_VPUIPREGMAPPED_TYPES

include "vpux/compiler/core/attributes.td"
include "vpux/compiler/dialect/VPUIPRegMapped/dialect.td"
include "vpux/compiler/dialect/VPUIPRegMapped/attributes.td"

class VPUIPRegMapped_Type<string name, list<Trait> traits = [],
                          string baseCppClass = "::mlir::Type">
        : TypeDef<VPUIPRegMapped_Dialect, name, traits, baseCppClass> {
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
    $_printer << "<" << getImpl()->value << ">";
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

//
// RegField Type
//

def VPUIPRegMapped_RegField : VPUIPRegMapped_Type<"RegField"> {
    let summary = "VPUIPRegMapped RegField Type";

    let description = "This object represents closely a RegField Type";

    let genAccessors = 1;

    let parameters = (ins
        "uint8_t":$width,
        "uint8_t":$pos,
        "uint64_t":$value,
        "std::string":$name
    );

    let assemblyFormat = [{
        $name`(`$pos `:` $width`)``=` $value
    }];
    let extraClassDeclaration = [{
        uint64_t getMap() const {
            if(getWidth() == (sizeof(uint64_t) * CHAR_BIT)) {
                return std::numeric_limits<uint64_t>::max();
            }
            uint64_t currentFieldMap = ((1ULL << getWidth()) - 1) << getPos();
            return currentFieldMap;
        }
    }];
    let genVerifyDecl = 1;
}

//
// Register Type
//

def VPUIPRegMapped_Register : VPUIPRegMapped_Type<"Register"> {
    let summary = "VPUIPRegMapped Register Type";

    let description = "This object represents closely a Register Type";

    let genAccessors = 1;

    let parameters = (ins
        "uint32_t":$size,
        "std::string":$name,
        "uint32_t":$address,
        VPUIPRegMapped_RegisterFieldArrayAttr:$regFields
    );

    let assemblyFormat = [{
        $name `(` $address `:` $size `)` `=` $regFields
    }];
    let extraClassDeclaration = [{
        Byte getSizeInBytes() const;
        std::vector<uint8_t> serialize() const;
    }];

    let genVerifyDecl = 1;
}

//
// RegMapped Type
//

def VPUIPRegMapped_RegMapped : VPUIPRegMapped_Type<"RegMapped"> {
    let summary = "VPUIPRegMapped RegMapped Type";

    let description = "This object represents closely a RegMapped Type";

    let genAccessors = 1;

    let parameters = (ins
        "std::string":$name,
        VPUIPRegMapped_RegisterArrayAttr:$regs
    );

    let assemblyFormat = [{
        $name `=` $regs
    }];
    let extraClassDeclaration = [{
        Byte getWidth() const;
        std::vector<uint8_t> serialize() const;
    }];

    let genVerifyDecl = 1;
}

//
// RegField Specialization Types
//

class VPUIPRegMapped_RegFieldWrapper<Dialect dialect, string name = "", int width = 0, int pos = 0> :
        TypeDef<dialect, "RegField_" # name, [], "vpux::VPUIPRegMapped::RegFieldType"> {
    let extraClassDeclaration = [{
    static vpux::VPUIPRegMapped::RegFieldType get(mlir::MLIRContext *context, uint64_t value) {
        return vpux::VPUIPRegMapped::RegFieldType::get(context, }] # width # [{, }] # pos # [{, value, "}] # name # [{");
    };
    static uint8_t getRegFieldWidth() {
        return }] # width # [{;
    }
  }];
}

//
// Register Type
//

class VPUIPRegMapped_RegisterWrapper<Dialect dialect, string regDialectStr, string name = "", int size = 0, int address = 0x00, list<string> regFieldTypes = []> :
         TypeDef<dialect, "Register_" # name, [], "vpux::VPUIPRegMapped::RegisterType"> {
    list<string> regFieldList = !foreach(name, regFieldTypes, regDialectStr # "RegField_" # name # "Type::get(builder.getContext(), values[\"" # name # "\"])," );
    string regFieldStr = !interleave(regFieldList, "\n");

    list<string> regFieldParams = !foreach(name, regFieldTypes, "const uint64_t " # name);
    list<string> regFieldParamsList = !foreach(name, regFieldTypes, regDialectStr # "RegField_" # name # "Type::get(builder.getContext(), " # name # ")," );
    string regFieldParamsStr = !interleave(regFieldParams, " = 0, ");
    string regFieldParamsListStr = !interleave(regFieldParamsList, "\n");

    let extraClassDeclaration = [{
    static vpux::VPUIPRegMapped::RegisterType get(mlir::OpBuilder builder, std::map<std::string, uint64_t> values) {
        auto regFields = getVPUIPRegMapped_RegisterFieldArrayAttr(builder, { }] # "\n" # regFieldStr # "\n" # [{ });
        return vpux::VPUIPRegMapped::RegisterType::get(builder.getContext(), }] # size # [{, "}] # name # [{", }] # address # [{, regFields);
    };
    static vpux::VPUIPRegMapped::RegisterType get(mlir::OpBuilder builder, }] # regFieldParamsStr # [{ = 0) {
        auto regFields = getVPUIPRegMapped_RegisterFieldArrayAttr(builder, { }] # "\n" # regFieldParamsListStr # "\n" # [{ });
        return vpux::VPUIPRegMapped::RegisterType::get(builder.getContext(), }] # size # [{, "}] # name # [{", }] # address # [{, regFields);
    };
    static unsigned getRegSize() {
        return }] # size # [{;
    }
  }];
}

//
// RegMapped Specialization Types
//

class VPUIPRegMapped_RegMappedWrapper<Dialect dialect, string regDialectStr, string name = "", list<string> registerTypes = []> :
         TypeDef<dialect, "RegMapped_" # name, [], "vpux::VPUIPRegMapped::RegisterType"> {
    list<string> regsList = !foreach(name, registerTypes, regDialectStr # "Register_" # name # "Type::get(builder, values[\"" # name # "\"])," );
    string regsStr = !interleave(regsList, "\n");

    list<string> regsDefaultList = !foreach(name, registerTypes, regDialectStr # "Register_" # name # "Type::get(builder)," );
    string regsDefaultStr = !interleave(regsDefaultList, "\n");

    let extraClassDeclaration = [{
    static vpux::VPUIPRegMapped::RegMappedType get(mlir::OpBuilder builder, std::map<std::string, std::map<std::string, uint64_t>> values) {
        auto regs = getVPUIPRegMapped_RegisterArrayAttr(builder, { }] # "\n" # regsStr # "\n" # [{ });
        return vpux::VPUIPRegMapped::RegMappedType::get(builder.getContext(), "}] # name # [{", regs);
    };
    static vpux::VPUIPRegMapped::RegMappedType get(mlir::OpBuilder builder) {
        auto regs = getVPUIPRegMapped_RegisterArrayAttr(builder, { }] # "\n" # regsDefaultStr # "\n" # [{ });
        return vpux::VPUIPRegMapped::RegMappedType::get(builder.getContext(), "}] # name # [{", regs);
    };
  }];
}

#endif
