OUTPUT_FORMAT("elf32-sparc")
OUTPUT_ARCH(sparc)

SECTIONS
{  
    . = 0x1c000000;      
    .text : {               
        *(.text*)
        . = ALIGN(16);
/* The below is required to handle inline functions, which are put by compiler in dedicated sections */
		*(.gnu.linkonce.text.*) 
		. = ALIGN(16);
        *(.rodata*)
        . = ALIGN(16);
        KEEP(*(.uuid.rodata*))           
        *(.uuid.rodata*)
    }

    .versiondata : {
        KEEP(*(.versiondata))
    }
}
