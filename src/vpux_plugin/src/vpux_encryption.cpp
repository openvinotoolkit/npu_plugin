//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#if defined(__arm__) || defined(__aarch64__)

#include "vpux_encryption.h"
#include <dlfcn.h>
#include <file_reader.h>
#include <ie_common.h>
#include <fstream>

namespace vpux {
namespace IE = InferenceEngine;

Encryption::Encryption(const std::string& nameOfLib): _logger("Encryption", LogLevel::Warning) {
    try {
        _sharedLibHandle = dlopen(nameOfLib.data(), RTLD_LAZY | RTLD_LOCAL);

        if (!_sharedLibHandle) {
            IE_THROW() << "Could not find data encryption library";
        }

        dlerror();
        _checkFn = (checkFile_t)dlsym(_sharedLibHandle, "secure_device_data_is_encrypted");
        const char* dlsym_error = dlerror();
        if (dlsym_error) {
            dlclose(_sharedLibHandle);
            _sharedLibHandle = nullptr;
            IE_THROW() << "dlsym_error";
        }

        dlerror();
        _decryptFn = (decryptFile_t)dlsym(_sharedLibHandle, "secure_device_data_decrypt");
        dlsym_error = dlerror();
        if (dlsym_error) {
            dlclose(_sharedLibHandle);
            _sharedLibHandle = nullptr;
            IE_THROW() << "dlsym_error";
        }
    } catch (const std::exception& ex) {
        _logger.info("{0}", ex.what());
    }
}

std::istream& Encryption::getDecryptedStream(std::ifstream& blobStream, std::stringstream& sstream) {
    std::vector<char> mBuffer;
    if (blobStream.is_open()) {
        const auto current_pos = blobStream.tellg();
        blobStream.seekg(0, blobStream.end);
        const auto last_char_pos = blobStream.tellg();
        const auto bytes_to_read = last_char_pos - current_pos;
        blobStream.seekg(current_pos, blobStream.beg);
        mBuffer.resize(bytes_to_read);
        blobStream.read(mBuffer.data(), bytes_to_read);
    } else {
        IE_THROW() << "Unable to open model file";
    }

    // Now check whether the buffer is encrypted or not
    auto buffer = reinterpret_cast<uint8_t*>(mBuffer.data());
    bool isEncrypted = false;
    size_t bufferSize = mBuffer.size();

    if (_checkFn(buffer, bufferSize, &isEncrypted)) {
        IE_THROW() << "Model Encryption Check failed";
    }

    /* If the model is encrypted decrypt to the same buffer */
    if (isEncrypted) {
        size_t outBufferSize = 0;

        if (_decryptFn(buffer, bufferSize, buffer, &outBufferSize)) {
            IE_THROW() << "Model Decryption failed";
        }
        bufferSize = outBufferSize;
    } else {
        return blobStream;
    }

    sstream.write(reinterpret_cast<const char*>(buffer), bufferSize);
    return sstream;
}

bool Encryption::isLibraryFound() {
    return _sharedLibHandle != nullptr;
}

Encryption::~Encryption() {
    if (_sharedLibHandle) {
        dlclose(_sharedLibHandle);
        _sharedLibHandle = nullptr;
    }
}

}  // namespace vpux

#endif
