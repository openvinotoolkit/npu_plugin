# VPUX ELF Tests

These **VPUMI37XX** MLIR files have the sole purpose of being used for **ELF generation** testing purposes.

## Test MLIR files
- Simple 1-DMA ELF generation
- NNCMX/DDR 2-DMA + Barriers ELF generation
- Activation Kernel ELF generation

## Usage

> ./vpux-translate --export-ELF <input_mlir_file> > <output_elf_file>
