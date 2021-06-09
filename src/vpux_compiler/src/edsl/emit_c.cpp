//
// Copyright 2021 Intel Corporation.
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

#include "vpux/compiler/edsl/emit_c.hpp"

#include <memory>
#include <string>

#include <llvm/ADT/StringSwitch.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/FormatVariadic.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Vector/VectorOps.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Support/DebugStringHelper.h>
#include <mlir/Translation.h>

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#ifdef ENABLE_PLAIDML
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/util/logging.h"
#endif

using namespace mlir;  // NOLINT

#ifdef ENABLE_PLAIDML
using namespace mlir::math;           // NOLINT
using namespace pmlc::dialect::stdx;  // NOLINT
#endif

namespace vpux {
namespace edsl {

#ifdef ENABLE_PLAIDML
class CEmitter {
public:
    explicit CEmitter(ModuleOp module, llvm::raw_ostream& os,
                      StringRef entryPoint)
            : module(module),  //
              src(os),
              entryPoint(entryPoint),
              depth(0),
              state(module) {
    }

    void emitRegions(Operation* op) {
        for (auto& region : op->getRegions()) {
            emitBlocks(region);
        }
    }

    void emitBlocks(Region& region) {
        ++depth;
        for (auto& block : region) {
            for (auto& op : block) {
                emitOp(&op);
            }
        }
        --depth;
    }

    void emitOp(Operation* op) {
        TypeSwitch<Operation*>(op)
                .Case<ModuleOp>([this](auto op) {
                    emitModuleOp(op);
                })
                .Case<FuncOp>([this](auto op) {
                    emitFuncOp(op);
                })
                // constant ops
                .Case<ConstantFloatOp>([this](auto op) {
                    emitConstantFloatOp(op);
                })
                .Case<ConstantIntOp>([this](auto op) {
                    emitConstantIntOp(op, op.getValue());
                })
                .Case<ConstantIndexOp>([this](auto op) {
                    emitConstantIntOp(op, op.getValue());
                })
                .Case<ConstantOp>([this](auto op) {
                    emitConstantOp(op);
                })
                // SCF ops
                .Case<scf::ForOp>([this](auto op) {
                    emitForOp(op);
                })
                .Case<scf::IfOp>([this](auto op) {
                    emitIfOp(op);
                })
                .Case<scf::YieldOp>([](auto /*op*/) {})
                // std ops
                .Case<CallOp>([this](auto op) {
                    emitCallOp(op);
                })
                .Case<SelectOp>([this](auto op) {
                    emitSelectOp(op);
                })
                .Case<ReturnOp>([](auto /*op*/) {})
                // memref ops
                .Case<memref::AllocOp>([this](auto op) {
                    emitAllocOp(op);
                })
                .Case<memref::LoadOp>([this](auto op) {
                    emitLoadOp(op);
                })
                .Case<memref::StoreOp>([this](auto op) {
                    emitStoreOp(op);
                })
                .Case<memref::CastOp>([this](auto op) {
                    emitCastOp(op.getOperation());
                })
                // .Case<ModuleTerminatorOp>([](auto op) {})
                // binary ops
                .Case<AddFOp>([this](auto op) {
                    emitBinaryOp(op.getOperation(), "+");
                })
                .Case<AddIOp>([this](auto op) {
                    emitBinaryOp(op.getOperation(), "+");
                })
                .Case<SubFOp>([this](auto op) {
                    emitBinaryOp(op.getOperation(), "-");
                })
                .Case<SubIOp>([this](auto op) {
                    emitBinaryOp(op.getOperation(), "-");
                })
                .Case<MulFOp>([this](auto op) {
                    emitBinaryOp(op.getOperation(), "*");
                })
                .Case<MulIOp>([this](auto op) {
                    emitBinaryOp(op.getOperation(), "*");
                })
                .Case<DivFOp>([this](auto op) {
                    emitBinaryOp(op.getOperation(), "/");
                })
                .Case<AndOp>([this](auto op) {
                    emitBinaryOp(op.getOperation(), "&");
                })
                .Case<OrOp>([this](auto op) {
                    emitBinaryOp(op.getOperation(), "|");
                })
                .Case<SignedDivIOp>([this](auto op) {
                    emitBinaryOp(op.getOperation(), "/");
                })
                .Case<UnsignedDivIOp>([this](auto op) {
                    emitBinaryOp(op.getOperation(), "/");
                })
                // math ops
                .Case<ExpOp>([this](auto op) {
                    emitIntrinsic(op.getOperation(), "exp");
                })
                .Case<ErfOp>([this](auto op) {
                    emitIntrinsic(op.getOperation(), "erf");
                })
                .Case<FloorOp>([this](auto op) {
                    emitIntrinsic(op.getOperation(), "floor");
                })
                .Case<PowOp>([this](auto op) {
                    emitIntrinsic(op.getOperation(), "power");
                })
                .Case<RoundOp>([this](auto op) {
                    emitIntrinsic(op.getOperation(), "round");
                })
                .Case<TanhOp>([this](auto op) {
                    emitIntrinsic(op.getOperation(), "tanh");
                })
                .Case<NegFOp>([this](auto op) {
                    emitIntrinsic(op.getOperation(), "-");
                })
                .Case<SqrtOp>([this](auto op) {
                    emitIntrinsic(op.getOperation(), "__builtin_shave_sau_sqt_f16_l_r");
                })
                .Case<RsqrtOp>([this](auto op) {
                    emitIntrinsic(op.getOperation(), "__builtin_shave_sau_rqt_f16_l_r");
                })
                // comparison ops
                .Case<CmpIOp>([this](auto op) {
                    emitCmpIOp(op);
                })
                .Case<CmpFOp>([this](auto op) {
                    emitCmpFOp(op);
                })
                // cast ops
                .Case<FPToSIOp>([this](auto op) {
                    emitCastOp(op.getOperation());
                })
                .Case<FPToUIOp>([this](auto op) {
                    emitCastOp(op.getOperation());
                })
                .Case<SIToFPOp>([this](SIToFPOp op) {
                    auto inType = op.in().getType();
                    if (auto inVecType = inType.dyn_cast<VectorType>()) {
                        auto operandElemType = inVecType.getElementType().cast<IntegerType>();
                        auto intermediateElemType =
                                IntegerType::get(op.getContext(), operandElemType.getWidth(), IntegerType::Signed);
                        auto intermediateType = VectorType::get(inVecType.getShape(), intermediateElemType);
                        emitCastOp(op.getOperation(), intermediateType);
                    } else {
                        auto operandType = inType.cast<IntegerType>();
                        auto intermediateType =
                                IntegerType::get(op.getContext(), operandType.getWidth(), IntegerType::Signed);
                        emitCastOp(op.getOperation(), intermediateType);
                    }
                })
                .Case<UIToFPOp>([this](UIToFPOp op) {
                    auto inType = op.getOperand().getType();
                    if (auto inVecType = inType.dyn_cast<VectorType>()) {
                        auto operandElemType = inVecType.getElementType().cast<IntegerType>();
                        auto intermediateElemType =
                                IntegerType::get(op.getContext(), operandElemType.getWidth(), IntegerType::Unsigned);
                        auto intermediateType = VectorType::get(inVecType.getShape(), intermediateElemType);
                        emitCastOp(op.getOperation(), intermediateType);
                    } else {
                        auto operandType = inType.cast<IntegerType>();
                        auto intermediateType =
                                IntegerType::get(op.getContext(), operandType.getWidth(), IntegerType::Unsigned);
                        emitCastOp(op.getOperation(), intermediateType);
                    }
                })
                .Case<IndexCastOp>([this](auto op) {
                    emitCastOp(op.getOperation());
                })
                .Case<TruncateIOp>([this](auto op) {
                    emitCastOp(op.getOperation());
                })
                .Case<FPExtOp>([this](auto op) {
                    emitCastOp(op.getOperation());
                })
                .Case<FPTruncOp>([this](auto op) {
                    emitCastOp(op.getOperation());
                })
                .Case<SignExtendIOp>([this](auto op) {
                    emitCastOp(op.getOperation());
                })
                .Case<ZeroExtendIOp>([this](auto op) {
                    emitCastOp(op.getOperation());
                })
                // vector ops
                .Case<vector::BroadcastOp>([this](auto op) {
                    emitVectorBroadcastOp(op);
                })
                .Case<vector::TransferReadOp>([this](auto op) {
                    emitVectorTransferReadOp(op);
                })
                .Case<vector::TransferWriteOp>([this](auto op) {
                    emitVectorTransferWriteOp(op);
                })
                // default case
                .Default([](Operation* op) {
                    VPUX_THROW("CEmitter: unsupported operation: {0}", debugString(*op));
                });
    }

    void emitModuleOp(ModuleOp op) {
        op.walk([&](FuncOp op) {
            emitFuncOp(op);
        });
    }

    void emitFuncOp(FuncOp op) {
        if (op.getBody().empty()) {
            return;
        }
        emitIndent();
        unsigned numArgs = op.getNumArguments();
        StringRef name = entryPoint;
        if (name.empty()) {
            name = op.getName();
        }
        src << "void " << name << '(';
        src << "void** arguments, void* settings) {\n";
        for (unsigned i = 0; i < numArgs; ++i) {
            BlockArgument arg = op.getArgument(i);
            emitIndent(depth + 1);
            emitArgument(arg);
            src << " = (" << getTypeName(arg.getType()) << ")arguments[" << i << "];\n";
        }
        emitRegions(op.getOperation());
        emitIndent();
        src << "}\n";
    }

    void emitConstantOp(ConstantOp op) {
        Value result = op.getResult();
        emitIndent();
        std::string resultName = getName(result);
        src << "const " << getTypeName(result.getType()) << ' ' << resultName << " = ";
        Attribute rhs = op.getValue();
        if (auto attr = rhs.dyn_cast<DenseElementsAttr>()) {
            ShapedType shapedType = op.getType().cast<ShapedType>();
            src << '{';
            for (unsigned i = 0; i < shapedType.getNumElements(); ++i) {
                if (i) {
                    src << ", ";
                }
                if (auto intAttr = attr.getValue(i).dyn_cast<IntegerAttr>()) {
                    src << intAttr.getInt();
                } else if (auto floatAttr = attr.getValue(i).dyn_cast<FloatAttr>()) {
                    src << floatAttr.getValueAsDouble();
                } else {
                    src << attr.getValue(i);
                }
            }
            src << '}';
        } else {
            src << rhs;
        }
        src << ";\n";
    }

    void emitConstantFloatOp(ConstantFloatOp op) {
        Value result = op.getResult();
        APFloat value = op.getValue();
        SmallVector<char, 16> str;
        if (value.isInfinity()) {
            // FIXME: smallest seems wrong for infinity
            APFloat smallest = value.getSmallest(value.getSemantics());
            smallest.toString(str);
        } else {
            value.toString(str);
        }
        emitIndent();
        src << llvm::formatv("const {0} {1} = {2};\n", getTypeName(result.getType()), getName(result), str);
    }

    void emitConstantIntOp(ConstantOp op, int64_t value) {
        Value result = op.getResult();
        emitIndent();
        src << llvm::formatv("const {0} {1} = {2};\n", getTypeName(result.getType()), getName(result), value);
    }

    void emitForOp(scf::ForOp op) {
        emitIndent();
        Value var = op.getInductionVar();
        std::string varType = getTypeName(var.getType());
        Value initValue = op.lowerBound();
        Value upperBound = op.upperBound();
        Value step = op.step();
        src << llvm::formatv("for ({0} {1} = {2}; {1} < {3}; {1} += {4}) {{\n", varType, getName(var),
                             getName(initValue), getName(upperBound), getName(step));
        emitRegions(op.getOperation());
        emitIndent();
        src << "}\n";
    }

    void emitIfOp(scf::IfOp op) {
        Value condition = op.getOperand();

        emitIndent();
        src << "if (" << getName(condition) << ") {\n";

        // then region
        emitBlocks(op.thenRegion());
        emitIndent();
        src << "}\n";

        if (!op.elseRegion().empty()) {
            // else region
            emitIndent();
            src << "else {\n";
            emitBlocks(op.elseRegion());
            emitIndent();
            src << "}\n";
        }
    }

    void emitVectorBroadcastOp(vector::BroadcastOp op) {
        auto lhs = getName(op.vector());
        auto rhs = getName(op.source());
        auto typeName = getTypeName(op.getVectorType());
        emitIndent();
        src << llvm::formatv("{0} {1} = ({0})({2});\n", typeName, lhs, rhs);
    }

    void emitVectorTransferReadOp(vector::TransferReadOp op) {
        auto lhs = getName(op.vector());
        auto rhs = getName(op.source());
        auto vectorType = op.getVectorType();
        auto typeName = getTypeName(vectorType);
        auto access = flatAccess(op.indices(), op.getShapedType().cast<MemRefType>());
        emitIndent();
        src << llvm::formatv("{0} {1} = *({0}*)&{2}[{3}];\n", typeName, lhs, rhs, access);
    }

    void emitVectorTransferWriteOp(vector::TransferWriteOp op) {
        auto lhs = getName(op.source());
        auto rhs = getName(op.vector());
        auto vectorType = op.getVectorType();
        auto typeName = getTypeName(vectorType);
        auto access = flatAccess(op.indices(), op.getShapedType().cast<MemRefType>());
        emitIndent();
        src << llvm::formatv("*({0}*)&{1}[{2}] = {3};\n", typeName, lhs, access, rhs);
    }

    void emitCallOp(CallOp op) {
        StringRef rawCallee = op.callee();
        StringRef callee = rawCallee.substr(0, rawCallee.find_last_of('$'));
        emitIndent();
        auto result = op.getResults();
        if (result.size()) {
            src << getTypeName(result[0].getType()) << ' ' << getName(result[0]) << " = ";
        }
        src << callee << '(';
        for (auto pair : llvm::enumerate(op.operands())) {
            if (pair.index()) {
                src << ", ";
            }
            src << getName(pair.value());
        }
        src << ");\n";
    }

    void emitLoadOp(memref::LoadOp op) {
        Value lhs = op.getResult();
        Value rhs = op.getMemRef();
        auto access = flatAccess(op.getIndices(), op.getMemRefType());
        emitIndent();
        src << llvm::formatv("{0} {1} = {2}[{3}];\n", getTypeName(op.getType()), getName(lhs), getName(rhs), access);
    }

    void emitStoreOp(memref::StoreOp op) {
        Value lhs = op.getMemRef();
        Value rhs = op.getValueToStore();
        auto access = flatAccess(op.getIndices(), op.getMemRefType());
        emitIndent();
        src << llvm::formatv("{0}[{1}] = {2};\n", getName(lhs), access, getName(rhs));
    }

    void emitAllocOp(memref::AllocOp op) {
        Value result = op.getResult();
        Type type = result.getType();
        ShapedType shape = type.cast<ShapedType>();
        std::string typeName = getTypeName(type);
        if (type.isa<MemRefType>() && typeName.back() == '*') {
            typeName.pop_back();
        }
        emitIndent();
        src << llvm::formatv("{0} {1}[{2}];\n", typeName, getName(result), shape.getNumElements());
    }

    std::string castExpr(Type type, StringRef expr) {
        auto typeName = getTypeName(type);
        if (type.isa<VectorType>()) {
            return llvm::formatv("mvuConvert_{0}({1})", typeName, expr);
        }
        return llvm::formatv("({0}) {1}", typeName, expr);
    }

    void emitCastOp(Operation* op, Type intermediateType = nullptr) {
        Value operand = op->getOperand(0);
        Value result = op->getResult(0);
        std::string operandName = getName(operand);
        std::string resultName = getName(result);
        auto resultType = result.getType();
        std::string resultTypeName = getTypeName(resultType);
        emitIndent();
        if (intermediateType) {
            auto tmpTypeName = getTypeName(intermediateType);
            auto subExpr = castExpr(intermediateType, operandName);
            src << llvm::formatv("{0} {1} = {2};\n", resultTypeName, resultName, castExpr(resultType, subExpr));
        } else {
            src << llvm::formatv("{0} {1} = {2};\n", resultTypeName, resultName, castExpr(resultType, operandName));
        }
    }

    void emitBinaryOp(Operation* op, StringRef opSymbol) {
        assert(op->getNumOperands() == 2 && "Expected 2 operands for binary op");
        Value result = op->getResult(0);
        Type resultType = result.getType();
        std::string typeName =
                vectorCmpType.count(result) ? getTypeName(vectorCmpType[result]) : getTypeName(resultType);
        Value lhs = op->getOperand(0);
        Value rhs = op->getOperand(1);
        emitIndent();
        src << llvm::formatv("{0} {1} = {2} {3} {4};\n", typeName, getName(result), getName(lhs), opSymbol,
                             getName(rhs));
    }

    const char* getCmpIPred(CmpIOp op) {
        switch (op.getPredicate()) {
        case CmpIPredicate::eq:
            return "==";
        case CmpIPredicate::ne:
            return "!=";
        case CmpIPredicate::slt:
            return "<";
        case CmpIPredicate::sle:
            return "<=";
        case CmpIPredicate::sgt:
            return ">";
        case CmpIPredicate::sge:
            return ">=";
        case CmpIPredicate::ult:
            return "<";
        case CmpIPredicate::ule:
            return "<=";
        case CmpIPredicate::ugt:
            return ">";
        case CmpIPredicate::uge:
            return ">=";
        }
        VPUX_THROW("CEmitter: unsupported CmpI predicate");
    }

    void emitCmpIOp(CmpIOp op) {
        emitBinaryOp(op, getCmpIPred(op));
    }

    const char* getCmpFPred(CmpFOp op) {
        switch (op.getPredicate()) {
        case CmpFPredicate::AlwaysFalse:
            return "FALSE";
        case CmpFPredicate::OEQ:
            return "==";
        case CmpFPredicate::OGT:
            return ">";
        case CmpFPredicate::OGE:
            return ">=";
        case CmpFPredicate::OLT:
            return "<";
        case CmpFPredicate::OLE:
            return "<=";
        case CmpFPredicate::ONE:
            return "!=";
        case CmpFPredicate::ORD:
            VPUX_THROW("CEmitter: CmpF.ORD is not supported");
        case CmpFPredicate::UEQ:
            return "==";
        case CmpFPredicate::UGT:
            return ">";
        case CmpFPredicate::UGE:
            return ">=";
        case CmpFPredicate::ULT:
            return "<";
        case CmpFPredicate::ULE:
            return "<=";
        case CmpFPredicate::UNE:
            return "!=";
        case CmpFPredicate::UNO:
            VPUX_THROW("CEmitter: CmpF.UNO is not supported");
        case CmpFPredicate::AlwaysTrue:
            return "TRUE";
        }
        VPUX_THROW("CEmitter: unsupported CmpF predicate");
    }

    void emitCmpFOp(CmpFOp op) {
        Value result = op.getResult();
        Value lhs = op.getOperand(0);
        Value rhs = op.getOperand(1);
        std::string typeName = getTypeName(result.getType());
        if (op.getPredicate() == CmpFPredicate::AlwaysFalse) {
            emitIndent();
            src << llvm::formatv("{0} {1} = {2};\n", typeName, getName(result), getName(rhs));
        } else if (op.getPredicate() == CmpFPredicate::AlwaysTrue) {
            emitIndent();
            src << llvm::formatv("{0} {1} = {2};\n", typeName, getName(result), getName(lhs));
        } else {
            Type lhsType = lhs.getType();
            if (auto lhsVectorType = lhsType.dyn_cast<VectorType>()) {
                unsigned bits = lhsVectorType.getElementType().getIntOrFloatBitWidth();
                auto newType = VectorType::get(lhsVectorType.getShape(), IntegerType::get(op.getContext(), bits));
                vectorCmpType[result] = newType;
            }
            emitBinaryOp(op, getCmpFPred(op));
        }
    }

    void emitSelectOp(SelectOp op) {
        Value result = op.getResult();
        Value cond = op.getCondition();
        Value trueValue = op.getTrueValue();
        Value falseValue = op.getFalseValue();
        std::string resultName = getName(result);
        std::string typeName = getTypeName(result.getType());
        std::string condName = getName(cond);
        std::string trueName = getName(trueValue);
        std::string falseName = getName(falseValue);
        emitIndent();
        if (op.getCondition().getType().isa<VectorType>()) {
            // Workaround: moviCompile doesn't support vector select
            Type condType = vectorCmpType.count(cond) ? vectorCmpType[cond] : cond.getType();
            std::string castTrue = llvm::formatv("mvuConvert_{0}({1})", getTypeName(condType), trueName);
            std::string castFalse = llvm::formatv("mvuConvert_{0}({1})", getTypeName(condType), falseName);
            src << llvm::formatv("{0} {1} = mvuConvert_{0}(({2} & {3}) + ((!{2}) & {4}));\n", typeName, resultName,
                                 condName, castTrue, castFalse);
        } else {
            src << llvm::formatv("{0} {1} = {2} ? {3} : {4};\n", typeName, resultName, condName, trueName, falseName);
        }
    }

    void emitIntrinsic(Operation* op, StringRef callee) {
        Value result = op->getResult(0);
        emitIndent();
        src << llvm::formatv("{0} {1} = {2}(", getTypeName(result.getType()), getName(result), callee);
        llvm::interleaveComma(op->getOperands(), src, [&](Value value) {
            src << getName(value);
        });
        src << ");\n";
    }

    void emitArgument(BlockArgument arg) {
        std::string type = getTypeName(arg.getType());
        if (type.find("*") == std::string::npos) {
            src << type << " " << getName(arg);
        } else {
            src << type << " restrict " << getName(arg);
        }
    }

    void emitIndent() {
        src.indent(depth * 2);
    }

    void emitIndent(int depth) {
        src.indent(depth * 2);
    }

    std::string getName(Value value) {
        auto it = valueNames.find(value);
        if (it != valueNames.end()) {
            return it->second;
        }
        std::string name;
        llvm::raw_string_ostream stream(name);
        value.printAsOperand(stream, state);
        stream.flush();
        std::replace(name.begin(), name.end(), '-', '_');
        std::replace(name.begin(), name.end(), '%', 'x');
        valueNames[value] = name;
        return name;
    }

    std::string getTypeName(Type type, bool isVector = false) {
        if (type.isIndex()) {
            return "intptr_t";
        }

        if (auto refType = type.dyn_cast<MemRefType>()) {
            return getTypeName(refType.getElementType()) + "*";
        }

        if (auto vecType = type.dyn_cast<VectorType>()) {
            return getTypeName(vecType.getElementType(), /*isVector=*/true) + std::to_string(vecType.getNumElements());
        }

        if (auto floatType = type.dyn_cast<FloatType>()) {
            switch (floatType.getWidth()) {
            case 16:
                return "half";
            case 32:
                return "float";
            case 64:
                return "double";
            }
        }

        if (type.isSignlessInteger() || type.isSignedInteger()) {
            switch (type.getIntOrFloatBitWidth()) {
            case 1:
                return isVector ? "char" : "bool";
            case 8:
                return "schar";
            case 16:
                return "short";
            case 32:
                return "int";
            case 64:
                return "longlong";
            }
        }

        if (type.isUnsignedInteger()) {
            switch (type.getIntOrFloatBitWidth()) {
            case 1:
                return isVector ? "char" : "bool";
            case 8:
                return "uchar";
            case 16:
                return "ushort";
            case 32:
                return "uint";
            case 64:
                return "ulonglong";
            }
        }

        VPUX_THROW("Unsupported type: {0}", debugString(type));
    }

    // Convert vector index exprList[i, j, k...] to flat access a*i + b*j + c*k...
    // where a, b, c are the strides in shape
    std::string flatAccess(ValueRange idxs, MemRefType type) {
        int64_t offset;
        SmallVector<int64_t, 4> strides;
        auto successStrides = getStridesAndOffset(type, strides, offset);
        assert(succeeded(successStrides) && "unexpected non-strided memref");
        (void)successStrides;

        std::string result;
        llvm::raw_string_ostream stream(result);
        if (offset) {
            stream << offset;
        }
        for (auto item : llvm::zip(idxs, strides)) {
            Value idx;
            int64_t stride;
            std::tie(idx, stride) = item;
            if (auto defOp = idx.getDefiningOp()) {
                if (m_Zero().match(defOp)) {
                    continue;
                }
            }
            if (stream.tell()) {
                stream << " + ";
            }
            stream << '(' << getName(idx) << ')';
            if (stride != 1) {
                stream << " * " << stride;
            }
        }
        if (!stream.tell()) {
            stream << '0';
        }
        return stream.str();
    }

private:
    ModuleOp module;
    llvm::raw_ostream& src;
    StringRef entryPoint;
    unsigned depth;
    AsmState state;
    DenseMap<Value, std::string> valueNames;
    DenseMap<Value, Type> vectorCmpType;
};

LogicalResult translateSCFToC(ModuleOp module, Operation* root, llvm::raw_ostream& os, StringRef entryPoint,
                              bool includeHeader) {
    try {
        // TODO: figure out a better way to extend C emitter
        if (includeHeader) {
            os << "#include <math.h>\n";
            os << "#include <moviVectorUtils.h>\n";
            os << "#include <stdbool.h>\n";
        }
        CEmitter emitter(module, os, entryPoint);
        emitter.emitOp(root);
    } catch (...) {
        return failure();
    }
    return success();
}

#else

LogicalResult translateSCFToC(ModuleOp, Operation*, llvm::raw_ostream&, StringRef, bool) {
    VPUX_THROW("translateSCFToC is only supported when ENABLE_PLAIDML=ON");
}

#endif

}  // namespace edsl
}  // namespace vpux
