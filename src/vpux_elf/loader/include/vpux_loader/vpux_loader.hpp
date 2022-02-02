//
// Copyright Intel Corporation.
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

#include <map>
#include <string>
#include <functional>
#include <vector>

#include <vpux_elf/reader.hpp>
#include <vpux_elf/types/relocation_entry.hpp>
#include <vpux_elf/types/symbol_entry.hpp>
#include <vpux_elf/types/vpu_extensions.hpp>
#include <vpux_elf/utils/error.hpp>

namespace elf {
namespace details{

//quick immutable array ref
template <typename T>
class ArrayRef {
public:
    using value_type = T;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using reference = value_type &;
    using const_reference = const value_type &;
    using iterator = const_pointer;
    using const_iterator = const_pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;

public:
    inline size_t size() const {return m_size;}
    inline const T* data() const { return m_data;}
    inline const T& operator[](size_t index) const {return m_data[index];}

    ArrayRef() : m_data(nullptr), m_size(0) {};
    ArrayRef(const T* data, size_t size) : m_data(data), m_size(size) {};

    template<typename A>
    ArrayRef(const std::vector<T, A> &vector)
      : m_data(vector.data()), m_size(vector.size()) {}

    template <size_t N>
    constexpr ArrayRef(const std::array<T, N> &array)
        : m_data(array.data()), m_size(N) {}


    iterator begin() const { return m_data; }
    iterator end() const { return m_data + m_size; }

    reverse_iterator rbegin() const { return reverse_iterator(end()); }
    reverse_iterator rend() const { return reverse_iterator(begin()); }

    bool empty() const { return m_size == 0; }

    std::vector<T> vec() const {
        return std::vector<T>(m_data, m_data + m_size);
    }

private:
    const T* m_data;
    const size_t m_size;

};

template <typename T>
class FixedVector {
public:
    inline size_t size() const { return m_size; }
    inline T* data() { return m_data; }
    inline const T* data() const { return m_data; }
    inline T& operator[](int index) { return m_data[index]; }
    inline const T& operator[](int index) const { return m_data[index]; }

    FixedVector()
        : m_size(0)
        , m_data(nullptr) {}

    FixedVector(size_t size) : m_size(size), m_data(new T[m_size]) {
        VPUX_ELF_THROW_UNLESS(m_data != nullptr, "Failed to allocate memory for internal FixedVector");
    }

    void resize(size_t size) {
        if(m_data) {
            delete m_data;
        }
        m_size = size;
        m_data = new T[m_size];

        VPUX_ELF_THROW_UNLESS(m_data != nullptr, "Failed to allocate memory for internal FixedVector");
        return;
    }

    ~FixedVector() {
        if(m_data){
            delete m_data;
        }
    }

private:
    size_t m_size;
    T* m_data;
};

} //namespace details

/*
Abstraction class to encapsulate device addressing logic. We have 2 addresses, that specify the same physical location.
Object does not own any of the pointed regions
in memory, but from 2 different view-points
@cpu_addr - Defines cpu visible address, aka, the physical address that is visible from the host perspective.
            Any access to the contents that the "host" does, will use this address
@vpu_addr - defines the vpu visible address, aka, the physical address that is visible from device perspective.
            Any access to the contents that the "vpu" does, will use this address
*/

class DeviceBuffer {
public:
    DeviceBuffer() :
         m_cpuAddr(nullptr), m_vpuAddr(0), m_size(0) {};

    DeviceBuffer(uint8_t* cpu_addr, uint64_t vpu_addr, size_t size) :
        m_cpuAddr(cpu_addr), m_vpuAddr(vpu_addr), m_size(size) {};

    DeviceBuffer(const DeviceBuffer& other) :
        m_cpuAddr(other.m_cpuAddr), m_vpuAddr(other.m_vpuAddr), m_size(other.m_size) {};

    DeviceBuffer(DeviceBuffer&& other) :
        m_cpuAddr(other.m_cpuAddr), m_vpuAddr(other.m_vpuAddr), m_size(other.m_size) {};

    DeviceBuffer& operator=(const DeviceBuffer& other) {
        m_cpuAddr = other.m_cpuAddr;
        m_vpuAddr = other.m_vpuAddr;
        m_size = other.m_size;

        return *this;
    }

    DeviceBuffer& operator=(const DeviceBuffer&& other) {
        m_cpuAddr = other.m_cpuAddr;
        m_vpuAddr = other.m_vpuAddr;
        m_size = other.m_size;

        return *this;
    }

    uint8_t* cpu_addr() {return m_cpuAddr;}
    const uint8_t* cpu_addr() const {return m_cpuAddr;}
    uint64_t vpu_addr() const {return m_vpuAddr;}
    size_t size() const {return m_size;}

private:
    uint8_t* m_cpuAddr;
    uint64_t m_vpuAddr;
    size_t m_size;

};

class BufferManager {
public:
    virtual DeviceBuffer allocate(size_t alignment, size_t size) = 0;
    virtual void deallocate(DeviceBuffer& devAddress) = 0;
    virtual size_t copy(DeviceBuffer& to, const uint8_t* from, size_t count) = 0;
    virtual ~BufferManager() {};
};

class VPUXLoader {
private:
    using RelocationFunc = std::function<void(void*, const elf::SymbolEntry&,const Elf_Sxword)>;
    using RelocationType = Elf_Word;

    enum class Action {
        None,
        AllocateAndLoad,
        Allocate,
        Relocate,
        RegisterUserIO,
        Error
    };

    static const std::map<Elf_Word, Action> actionMap;
    static const std::map<RelocationType, RelocationFunc> relocationMap;

public:

    explicit VPUXLoader(void* elf, size_t elfSize, details::ArrayRef<SymbolEntry> runtimeSymTabs, BufferManager* bufferManager);
    ~VPUXLoader();

    uint64_t getEntry() ;

    void applyJitRelocations(std::vector<DeviceBuffer>& inputs, std::vector<DeviceBuffer>& outputs);

    details::ArrayRef<DeviceBuffer> getAllocatedBuffers();
    details::ArrayRef<DeviceBuffer> getInputBuffers();
    details::ArrayRef<DeviceBuffer> getOutputBuffers();

private:
    bool checkSectionType(const elf::SectionHeader* section, Elf_Word secType) const;
    void registerUserIO(details::FixedVector<DeviceBuffer>& io,const elf::SymbolEntry* symbols, size_t symbolCount) const;
    void load();
    void clean();

    void applyRelocations(details::ArrayRef<int> relocationSectionIndexes);

    Reader m_reader;
    details::ArrayRef<SymbolEntry> m_runtimeSymTabs;
    BufferManager* m_bufferManager;

    std::vector<DeviceBuffer> m_allocatedZones;
    std::vector<DeviceBuffer> m_sectionToAddr;
    std::vector<int> m_jitRelocations;

    details::FixedVector<DeviceBuffer> m_userInputs;
    details::FixedVector<DeviceBuffer> m_userOutputs;

};

}
