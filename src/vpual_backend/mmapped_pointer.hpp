//
// Copyright 2019-2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

enum class MappingMode {
    read_write,
    readonly,
};

// physical pointer in another process, points to certain data size
template <class T>
class  MMappedPtr final {
protected:
    uint8_t  * mapped_ptr = nullptr;
    size_t mapsize = 0;
    size_t page_offset = 0;
    int fd;
    size_t  page_size = 4096;
public:
    /**
     * @ptr to physically stable address - in any process you need to map
     * @sz - size in bytes of the object
     * @page_size - default page is 4096 bytes
     */
    MMappedPtr(uint64_t phy_addr
               , size_t sz = sizeof(T)
               , size_t page_size = 4096
               , MappingMode acces_type = MappingMode::readonly)
        : page_size(page_size) {

        auto openMode = (acces_type == MappingMode::readonly) ? O_RDONLY : O_RDWR;
        fd = open("/dev/mem", openMode | O_SYNC);
        if (fd < 0) {
            throw std::runtime_error("/dev/mem Open failed");
        }
        // phy_addr - might be not page aligned - so map from page boundary
        auto phy_addr_page_aligned = pageAlignDown(phy_addr);
        page_offset = phy_addr - phy_addr_page_aligned;

        auto mapMode = (acces_type == MappingMode::readonly) ? PROT_READ : (PROT_READ | PROT_WRITE);
        mapsize = pageAlignUp(page_offset + sz);
        mapped_ptr = reinterpret_cast<uint8_t*>(mmap(NULL, mapsize, mapMode, MAP_SHARED, fd, phy_addr_page_aligned));

        if(mapped_ptr == MAP_FAILED) {
            close(fd);
            std::stringstream err;
            err << "failed to map header at : mmap(offset=0x" << std::hex << phy_addr_page_aligned << " failed";
            throw std::runtime_error(err.str());
        }
    }
    T* operator *() const noexcept {
        return  reinterpret_cast<T*>(mapped_ptr + page_offset);
    }
    T* operator ->() const noexcept {
        return  reinterpret_cast<T*>(mapped_ptr + page_offset);
    }
    T* get() const noexcept {
        return  reinterpret_cast<T*>(mapped_ptr + page_offset);
    }
    virtual ~MMappedPtr() {
        if (mapped_ptr != nullptr) {
            munmap(mapped_ptr, mapsize);
        }
        if (fd >= 0) {
            close(fd);
        }
    }
private:
    MMappedPtr(const MMappedPtr &) = delete;
    MMappedPtr(MMappedPtr &&) = delete;
    MMappedPtr & operator =(const MMappedPtr &) = delete;
    MMappedPtr & operator =(MMappedPtr &&) = delete;

    constexpr size_t pageAlignDown(size_t val) noexcept {
        return val & (~(page_size-1));
    }
    constexpr size_t pageAlignUp(size_t val) noexcept {
        return (((val + page_size - 1) / page_size) * page_size);
    }
};
