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

#include "vpux/compiler/dialect/ELF/export.hpp"
#include "vpux/compiler/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"

using namespace vpux;

std::vector<uint8_t> vpux::ELF::exportToELF(mlir::ModuleOp module, Logger log) {
    log.setName("ELF BackEnd");

    log.trace("Extract '{0}' from Module (ELF File)", IE::CNNNetworkOp::getOperationName());
    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    elf::Writer elfWriter;
    // Associate the respective mlir::Operation* of
    //   CreateSectionOp/CreateLogicalSectionOp/CreateSymbolSectionOp/CreateRelocationSectionOp
    //   with the respective created elf::writer::Section* for it.
    SectionMapType sectionMap;
    // Associate the respective mlir::Operation* of a SymbolOp with the newly created
    //   elf::writer::Symbol* for it.
    SymbolMapType symbolMap;

    // Normally the ELF serialization process requires raw data to be serialized first,
    // then symbols, then relocations, simply because of data dependency.
    // However ticket #29166 plans to introduce ordering constraints on IR validity,
    // which would allow us to serialize all data in one function and just use
    // a SectionInterface rather than ops themselves.

    log.trace("Serializing '{0}' ops", ELF::CreateSectionOp::getOperationName());
    auto createSectionOps = netFunc.getOps<ELF::CreateSectionOp>();
    for (auto createSectionOp : createSectionOps) {
        createSectionOp.serialize(elfWriter, sectionMap, symbolMap);
    }

    log.trace("Serializing '{0}' ops", ELF::CreateLogicalSectionOp::getOperationName());
    auto createLogicalSectionOps = netFunc.getOps<ELF::CreateLogicalSectionOp>();
    for (auto createLogicalSectionOp : createLogicalSectionOps) {
        createLogicalSectionOp.serialize(elfWriter, sectionMap, symbolMap);
    }

    log.trace("Serializing '{0}' ops", ELF::CreateSymbolTableSectionOp::getOperationName());
    auto createSymTabOps = netFunc.getOps<ELF::CreateSymbolTableSectionOp>();
    for (auto createSymTabOp : createSymTabOps) {
        createSymTabOp.serialize(elfWriter, sectionMap, symbolMap);
    }

    log.trace("Serializing '{0}' ops", ELF::CreateRelocationSectionOp::getOperationName());
    auto createRelocSectionOps = netFunc.getOps<ELF::CreateRelocationSectionOp>();
    for (auto createRelocSection : createRelocSectionOps) {
        createRelocSection.serialize(elfWriter, sectionMap, symbolMap);
    }

    return elfWriter.generateELF();
}
