// {% copyright %}
#include "mvTensorOutputStream.h"
#include <string.h>
#include <algorithm>

namespace mv
{
    namespace tensor
    {
        OutputStream::OutputStream(char *buffer, unsigned int size, Mode mode) :
            buffer_(buffer),
            available_(buffer ? std::max(1u, size) - 1 : 0) // keep 1 byte for trailing 0
        {
            if (mode == Overwrite && buffer != 0 && size > 0)
                buffer[0] = 0;

            available_ -= strnlen(buffer_, available_);
        }

        OutputStream &OutputStream::operator <<(const char *text)
        {
            if (text != 0)
                append(text);
            return *this;
        }

        OutputStream &OutputStream::operator <<(bool b)
        {
            return operator <<(b ? "true" : "false");
        }

        OutputStream &OutputStream::operator <<(char c)
        {
            char buf[] = { c, 0 };
            append(buf);
            return *this;
        }

        void OutputStream::append(const char *input)
        {
            const unsigned int length = strnlen(input, available_);

            // not sure how strncat behaves when length is 0 and buffer_ is non-null-terminated,
            // so just skip this case explicitly
            if (length > 0)
            {
                // length > 0 implies available_ > 0 implies strlen(buffer_) < size or Overwrite
                // implies buffer_ is null-terminated in the first size characters

                strncat(buffer_, input, length);
                available_ -= length;
            }
        }
    }
}
