// {% copyright %}

#ifndef _JTAG_INTERACTIONS_H_
#define _JTAG_INTERACTIONS_H_

#include "icv_test_suite.h"

extern volatile int var_from_deb_global;
extern volatile char str_from_deb_global[ICV_TEST_TEST_FILTER_STR_MAXSIZE];

int get_from_debug_int();
void get_from_debug_str(int length, char str[]);

inline bool get_from_debug(bool val) {
    var_from_deb_global = val ? 1 : 0;
    return bool(get_from_debug_int() != 0);
}

inline int get_from_debug(int val) {
    var_from_deb_global = val;
    return get_from_debug_int();
}

inline void get_from_debug(int length, char str[]) {
    strncpy((char*)str_from_deb_global, (char*)str, length);
    str_from_deb_global[ICV_TEST_TEST_FILTER_STR_MAXSIZE - 1] = 0;
    get_from_debug_str(length, str);
}

template<class T>
inline T get_from_debug(T& val) {
    var_from_deb_global = static_cast<int>(val);
    return static_cast<T>( get_from_debug_int());
}

#endif /* _JTAG_INTERACTIONS_H_ */
