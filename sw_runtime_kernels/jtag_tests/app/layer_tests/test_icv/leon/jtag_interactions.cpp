// {% copyright %}

#include "jtag_interactions.h"

volatile int var_from_deb_global __attribute__((section(".nncmx0.shared.data"))) = 0;
volatile char str_from_deb_global[ICV_TEST_TEST_FILTER_STR_MAXSIZE] __attribute__((section(".nncmx0.shared.data")));

int __attribute__((noinline)) get_from_debug_int() {
    int tmp = var_from_deb_global;
    return tmp;
}

void __attribute__((noinline)) get_from_debug_str(int length, char str[]) {
    length = std::min<int>(length, ICV_TEST_TEST_FILTER_STR_MAXSIZE - 1);
    if (str_from_deb_global[0] != 0)
        strncpy((char*)str, (char*)str_from_deb_global, length);
    str[length] = 0;
}
