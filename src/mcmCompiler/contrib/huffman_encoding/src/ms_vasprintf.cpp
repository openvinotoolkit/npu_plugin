#ifdef WIN32

#include "logging.hpp"
#include <stdio.h>
#include <stdarg.h>

int ms_vasprintf(char** ptr, const char* format, va_list ap)
{
	int len = _vscprintf(format, ap) + 1;
	*ptr = (char*) malloc(len);
	if (*ptr) return vsprintf_s(*ptr, len, format, ap);
	else return -1;
}

#endif //WIN32
