//
// Copyright 2021 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once
#if defined(__arm__) || defined(__aarch64__)
#include "vpux.hpp"

namespace vpux {

/**
 * @brief This class provides functionality for working with encrypted models. Only for KMB ARM platform!
 *
 */
class Encryption {
public:
    Encryption();

    /**
     * The function decrypts the model if it is encrypted
     */
    std::istream& getDecryptedStream(std::ifstream& blobStream, std::stringstream& sstream);

    /**
     * Returns true if the library is found otherwise false
     */
    bool isLibraryFound();

    ~Encryption();

private:
    typedef int (*checkFile_t)(uint8_t const* cipher_input, size_t cipher_input_size, bool* is_encrypted);
    typedef int (*decryptFile_t)(uint8_t const* cipher_input, size_t cipher_input_size, uint8_t const* plain_output,
                                 size_t* plain_output_size);

    void* _sharedLibHandle = nullptr;
    checkFile_t _checkFn;
    decryptFile_t _decryptFn;
};

}  // namespace vpux
#endif
