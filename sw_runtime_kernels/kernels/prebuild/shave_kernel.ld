OUTPUT_FORMAT("elf32-sparc")
OUTPUT_ARCH(sparc)

SECTIONS
{  
    . = 0x1d000000;      
    .text : {               
        *(.text*)
        . = ALIGN(16);
        *(.rodata*)
        . = ALIGN(16);
        KEEP(*(.uuid.rodata*))           
        *(.uuid.rodata*)
    }

    . = 0x1e000000;
    
    .arg.data : { 
        KEEP(*(.arg.data))
        *(.arg.data) 
        . = ALIGN(16);
        *(.data*) 
    }
}