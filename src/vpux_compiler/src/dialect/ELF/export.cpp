//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/ELF/export.hpp"
#include "vpux/compiler/dialect/ELF/metadata.hpp"

using namespace vpux;

std::vector<uint8_t> vpux::ELF::exportToELF(mlir::ModuleOp module,
                                            const std::vector<vpux::PreProcessInfo>& preprocessInfo,
                                            const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                            const std::vector<std::shared_ptr<const ov::Node>>& results, Logger log) {
    log.setName("ELF BackEnd");

    log.trace("Extract '{0}' from Module (ELF File)", IE::CNNNetworkOp::getOperationName());

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

    IE::CNNNetworkOp netOp;
    mlir::FuncOp netFunc;
    IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);

    log.trace("Serializing '{0}' ops", ELF::CreateMetadataSectionOp::getOperationName());
    auto createMetadataSectionOps = netFunc.getOps<ELF::CreateMetadataSectionOp>();
    for (auto createMetadataSectionOp : createMetadataSectionOps) {
        auto metadata = vpux::ELF::constructMetadata(module, netOp, netFunc, preprocessInfo, parameters, results);
        createMetadataSectionOp.serialize(elfWriter, sectionMap, symbolMap, metadata);
    }

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
