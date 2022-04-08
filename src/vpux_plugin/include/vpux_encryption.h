//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#if defined(__arm__) || defined(__aarch64__)

#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux {

/**
 * @brief This class provides functionality for working with encrypted models. Only for KMB ARM platform!
 *
 */
class Encryption {
public:
    explicit Encryption(const std::string& nameOfLib = "/usr/lib/libSecureDeviceDataEncrypt.so");

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
    using checkFile_t = int (*)(uint8_t const* cipher_input, size_t cipher_input_size, bool* is_encrypted);
    using decryptFile_t = int (*)(uint8_t const* cipher_input, size_t cipher_input_size, uint8_t const* plain_output,
                                  size_t* plain_output_size);

    void* _sharedLibHandle = nullptr;
    checkFile_t _checkFn = nullptr;
    decryptFile_t _decryptFn = nullptr;
    Logger _logger;
};

}  // namespace vpux

#endif
