/*  ---------------------------------------------------------------------------
 *  Copyright (C) 2014-2019 Movidius Ltd. All rights reserved
 *  ---------------------------------------------------------------------------
 *  File       :   shave_first_phase.ld
 *  Description:   SHAVE 1st Phase Linker Script
 *
 *  This script coalesces all the program contributions for a single SHAVE, and
 *  is intended to be used only with the '-Ur' option.
 *
 *  The naming convention allows for separate sections to be described for CMX,
 *  DDR or don't care.  It is up to the final linker script to define whether
 *  these "don't care" sections are all to be grouped together in the CMX, DDR
 *  or separated as best suits the application.
 *
 *  The compiler never emits the '.cmx' or '.ddr' prefixed sections unless the
 *  programmer has used the '__attribute__((section("...")))' extension to
 *  explicitly place their code or data in these sections.  This means that in
 *  general, the final linking phase can decide whether to place the compiler
 *  emitted code and data in CMX or DDR.
 */

TARGET("elf32-littlesparc")
OUTPUT_FORMAT("elf32-littlesparc")
OUTPUT_ARCH(sparc)

SECTIONS
{
  /* Read-only data sections */
  S.cmx.rodata     0 :
    {
      *(.cmx.rodata16 .cmx.rodata16.* .gnu.linkonce.cmx.rodata16.*)
      *(.cmx.rodata8  .cmx.rodata8.*  .gnu.linkonce.cmx.rodata8.*)
      *(.cmx.rodata4  .cmx.rodata4.*  .gnu.linkonce.cmx.rodata4.*)
      *(.cmx.rodata2  .cmx.rodata2.*  .gnu.linkonce.cmx.rodata2.*)
      *(.cmx.rodata1  .cmx.rodata1.*  .gnu.linkonce.cmx.rodata1.*)
      *(.cmx.rodata   .cmx.rodata.*   .gnu.linkonce.cmx.rodata.*)
    }
  S.ddr.rodata     0 :
    {
      *(.ddr.rodata16 .ddr.rodata16.* .gnu.linkonce.ddr.rodata16.*)
      *(.ddr.rodata8  .ddr.rodata8.*  .gnu.linkonce.ddr.rodata8.*)
      *(.ddr.rodata4  .ddr.rodata4.*  .gnu.linkonce.ddr.rodata4.*)
      *(.ddr.rodata2  .ddr.rodata2.*  .gnu.linkonce.ddr.rodata2.*)
      *(.ddr.rodata1  .ddr.rodata1.*  .gnu.linkonce.ddr.rodata1.*)
      *(.ddr.rodata   .ddr.rodata.*   .gnu.linkonce.ddr.rodata.*)
    }
  S.rodata         0 :
    { /* Collate the 16-byte aligned objects followed by the 8-byte and 4-byte aligned objects */
      *(.rodata16 .rodata16.* .gnu.linkonce.rodata16.*)
      *(.rodata8  .rodata8.*  .gnu.linkonce.rodata8.*)
      *(.rodata4  .rodata4.*  .gnu.linkonce.rodata4.*)

      /* Define the symbols that will be used by the pre-initialisation and C++ static initialisation helper */
      /* These are 4-byte aligned, but ensure that the current offset actually is 4-byte aligned */
      . = ALIGN(4);

      PROVIDE_HIDDEN(__preinit_array_start = .);
      KEEP(*(SORT_BY_INIT_PRIORITY(.preinit.*)))
      KEEP(*(.preinit))
      PROVIDE_HIDDEN(__preinit_array_end = .);
      PROVIDE(__init_array_start = .);
      KEEP(*(SORT_BY_INIT_PRIORITY(.ctors.*)))
      KEEP(*(.ctors))
      PROVIDE(__init_array_end = .);

      /* Define the symbols that will be used by the finalisation and C++ static object destruction helpers */
      PROVIDE_HIDDEN(__libc_fini = .);
      PROVIDE(__fini_array_start = .);
      KEEP(*(SORT_BY_INIT_PRIORITY(.dtors.*)))
      KEEP(*(.dtors))
      PROVIDE(__fini_array_end = .);

      /* Collate the 2-byte aligned objects followed by the 1-byte aligned objects */
      *(.rodata2 .rodata2.* .gnu.linkonce.rodata2.*)
      *(.rodata1 .rodata1.* .gnu.linkonce.rodata1.*)

      /* Finally, bring in the objects with unspecified alignment */
      *(.rodata  .rodata.*  .gnu.linkonce.rodata.*)
    }

  /* Normal data sections */
  S.cmx.data       0 :
    {
      *(.cmx.data16 .cmx.data16.* .gnu.linkonce.cmx.data16.*)
      *(.cmx.data8  .cmx.data8.*  .gnu.linkonce.cmx.data8.*)
      *(.cmx.data4  .cmx.data4.*  .gnu.linkonce.cmx.data4.*)
      *(.cmx.data2  .cmx.data2.*  .gnu.linkonce.cmx.data2.*)
      *(.cmx.data1  .cmx.data1.*  .gnu.linkonce.cmx.data1.*)
      *(.cmx.data   .cmx.data.*   .gnu.linkonce.cmx.data.*)
    }
  S.ddr.data       0 :
    {
      *(.ddr.data16 .ddr.data16.* .gnu.linkonce.ddr.data16.*)
      *(.ddr.data8  .ddr.data8.*  .gnu.linkonce.ddr.data8.*)
      *(.ddr.data4  .ddr.data4.*  .gnu.linkonce.ddr.data4.*)
      *(.ddr.data2  .ddr.data2.*  .gnu.linkonce.ddr.data2.*)
      *(.ddr.data1  .ddr.data1.*  .gnu.linkonce.ddr.data1.*)
      *(.ddr.data   .ddr.data.*   .gnu.linkonce.ddr.data.*)
    }
  S.data           0 :
    {
      *(.data16 .data16.* .gnu.linkonce.data16.*)
      *(.data8  .data8.*  .gnu.linkonce.data8.*)
      *(.data4  .data4.*  .gnu.linkonce.data4.*)
      *(.data2  .data2.*  .gnu.linkonce.data2.*)
      *(.data1  .data1.*  .gnu.linkonce.data1.*)
      *(.data   .data.*   .gnu.linkonce.data.*)

      /* Append the LLVM profiling sections */
      . = ALIGN(4);
      PROVIDE(__start___llvm_prf_cnts = .);
      *(__llvm_prf_cnts)
      PROVIDE(__stop___llvm_prf_cnts = .);

      . = ALIGN(4);
      PROVIDE(__start___llvm_prf_data = .);
      *(__llvm_prf_data)
      PROVIDE(__stop___llvm_prf_data = .);

      . = ALIGN(4);
      PROVIDE(__start___llvm_prf_names = .);
      *(__llvm_prf_names)
      PROVIDE(__stop___llvm_prf_names = .);

      . = ALIGN(4);
      PROVIDE(__start___llvm_prf_vnds = .);
      *(__llvm_prf_vnds)
      PROVIDE(__stop___llvm_prf_vnds = .);
    }

  /* Zero initialised section */
  S.bss            0 : { *(.bss .bss.* .gnu.linkonce.bss.*) }

  /* Code sections */
  S.cmx.text       0 : { *(.cmx.text .cmx.text.* .gnu.linkonce.cmx.text.*) }
  S.ddr.text       0 : { *(.ddr.text .ddr.text.* .gnu.linkonce.ddr.text.*) }
  S.text           0 : { *(.text     .text.*     .gnu.linkonce.text.*)     }

  /* Indirect Address Table (IAT) sections */
  S..iat           0 :
    {
      . = ALIGN(4);
      KEEP(*(SORT(..iat.*)))
    }
  S..iatnames      0 :
    {
      . = ALIGN(4);
      KEEP(*(SORT(..iatnames.*)))
      KEEP(*(.gnu.linkonce..iatnamesend))
    }
  S..iatstrings    0 : { *(..str..iatstrings) }

  /*
   * DWARF sections
   * DWARF symbols are relative to the start of their section, so use a VMA of zero
   */
  /* DWARF 1 */
  .debug           0 : { *(.debug) }
  .line            0 : { *(.line) }
  /* DWARF 1 extensions */
  .debug_srcinfo   0 : { *(.debug_srcinfo) }
  .debug_sfnames   0 : { *(.debug_sfnames) }
  /* DWARF 1.1 and DWARF 2 */
  .debug_aranges   0 : { *(.debug_aranges) }
  .debug_pubnames  0 : { *(.debug_pubnames) }
  /* DWARF 2 */
  .debug_info      0 : { *(.debug_info .gnu.linkonce.wi.*) }
  .debug_abbrev    0 : { *(.debug_abbrev) }
  .debug_line      0 : { *(.debug_line .debug_line.* .debug_line_end ) }
  .debug_frame     0 : { *(.debug_frame) }
  .debug_str       0 : { *(.debug_str) }
  .debug_loc       0 : { *(.debug_loc) }
  .debug_macinfo   0 : { *(.debug_macinfo) }
  /* DWARF 2 extensions */
  .debug_weaknames 0 : { *(.debug_weaknames) }
  .debug_funcnames 0 : { *(.debug_funcnames) }
  .debug_typenames 0 : { *(.debug_typenames) }
  .debug_varnames  0 : { *(.debug_varnames) }
  /* DWARF 3 */
  .debug_pubtypes  0 : { *(.debug_pubtypes) }
  .debug_ranges    0 : { *(.debug_ranges) }
  /* Additional DWARF Extensions  */
  debug_macro      0 : { *(.debug_macro) }
  /DISCARD/          : { *(.gnu.attributes) }     /* SHAVE Debugger cannot handle these yet */
  /DISCARD/          : { *(.note.GNU-stack) *(.gnu_debuglink) *(.gnu.lto_*) }
}
