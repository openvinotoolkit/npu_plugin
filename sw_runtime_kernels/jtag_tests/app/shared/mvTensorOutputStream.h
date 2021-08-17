// {% copyright %}
#ifndef MV_TENSOR_OUTPUT_STREAM_H_
#define MV_TENSOR_OUTPUT_STREAM_H_

#include <stdio.h>

namespace mv
{
    namespace tensor
    {
        class OutputStream
        {
        public:
            enum Mode
            {
                Overwrite,
                Append,
            };

            OutputStream(char *buffer, unsigned int size, Mode mode = Append);

            OutputStream &operator <<(const char *text);
            OutputStream &operator <<(const void *p) { return append("0x%p", p); }

            OutputStream &operator <<(bool b);

            OutputStream &operator <<(char c);
            inline OutputStream &operator <<(unsigned char x) { return operator <<(static_cast<unsigned int>(x)); }

            inline OutputStream &operator <<(short x) { return operator <<(static_cast<int>(x)); }
            inline OutputStream &operator <<(unsigned short x) { return operator <<(static_cast<unsigned int>(x)); }

            OutputStream &operator <<(int x) { return append("%d", x); }
            OutputStream &operator <<(unsigned int x) { return append("%u", x); }

            inline OutputStream &operator <<(long x) { return operator <<(static_cast<int>(x)); }
            inline OutputStream &operator <<(unsigned long x) { return operator <<(static_cast<unsigned int>(x)); }

            OutputStream &operator <<(long long x) { return append("%lld", x); }
            OutputStream &operator <<(unsigned long long x) { return append("%llu", x); }

            inline OutputStream &operator <<(float x) { return operator <<(static_cast<double>(x)); }
            OutputStream &operator <<(double d) { return append("%f", d); }
            OutputStream &operator <<(long double d) { return append("%Lf", d); }

        private:
            char *buffer_;
            unsigned int available_;

            OutputStream(const OutputStream &);
            OutputStream &operator =(const OutputStream &);

            template <typename T, unsigned N>
            OutputStream &append(const char *format, T t)
            {
                char buf[N];
                snprintf(buf, sizeof(buf), format, t);
                append(buf);
                return *this;
            }

            template <typename T>
            inline OutputStream &append(const char *format, T t)
            {
                // This is to avoid warning about default function template arguments being C++11
                return append<T, sizeof(T) * 4>(format, t);
            }

            void append(const char *input);
        };
    }
}

#endif // MV_TENSOR_OUTPUT_STREAM_H_
