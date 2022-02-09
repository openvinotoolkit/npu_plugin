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

#if !defined(_ICV_TEST_SUITE_H_)
#define _ICV_TEST_SUITE_H_

//#include <DrvLeonL2C.h>

#include <sw_tensor_ref.h>
#include <mvTensorUtil.h>

#ifndef TOTAL_NUM_SHAVES
# define TOTAL_NUM_SHAVES MVTENSOR_MAX_SHAVES /* AE: temporary. what is MVTENSOR_MAX_SHAVES for MTL? */
#endif

using namespace nn::shave_lib;
/* Note that external variable "__l2_config" declared twice with different linkage:
 *
 * app_config.h:        "extern "C" { extern u32 __l2_config }"  ("C" linkage)
 * OsDrvShaveL2Cache.h:              "extern u32 __l2_config"    (default linkage)
 *
 * So, to get these includes compiled successfully, app_config.h MUST be included first.
 */

#if defined(MA2480)
//# include <OsDrvShaveL2c.h>
#else
//# include <OsDrvShaveL2Cache.h>
#endif

// #include <DrvTimer.h>

#include <Fp16Convert.h>
//#include <UnitTestApi.h>
#include <VcsHooksApi.h>

# ifdef __cplusplus
#  include    <atomic>
# else
#  include    <stdatomic.h>
# endif

# include <rtems/rtems/intr.h> // libcpu/cache_.h is missing rtems_interrupt_lock
#include <libcpu/cache_.h>

#include <mvTensor.h>
//#include <mvTensor_cpp.h>
#include <mvTensorDebug.h>
//#include <mvTensorInternal.h>
#include <mvTensorTimer.h>
#include <opManager.h>
#include <rtems.h>
//#include <swcLeonUtils.h>
#include <string.h>
#include <stdarg.h>

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <utility>
#include <vector>
#include <deque>
#include <type_traits>
#include <set>
#include <memory>

#include <fnmatch.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "mvSubspaces.h"

#include <sstream>

namespace icv_tests
{

//== Macros ====================================================================

#define ICV_TESTS_MEMORY_ALIGN(_Boundary, _Value) (((_Value) + ((_Boundary) - 1)) & ~((_Boundary) - 1))

#define ICV_TESTS_REPORT_PERF_CYCLES (1) /* report performance data in clock cycles (ms otherwise) */

#define ICV_TESTS_MEMORY_BUFFER_ALIGN (32)
#define ICV_TESTS_MEMORY_BUFFER_SIZE ICV_TESTS_MEMORY_ALIGN(ICV_TESTS_MEMORY_BUFFER_ALIGN, 100*1024*1024)

#define ICV_TESTS_REGISTER_SUITE(_Class) const SuiteHandle tests[] = { SuiteHandle().registerSuite<_Class>() };

#define ICV_TESTS_STRINGIFY(_text) _ICV_TESTS_STRINGIFY(_text)
#define _ICV_TESTS_STRINGIFY(_text) #_text

#define ICV_TESTS_NAMESPACE(_name) ICV_TESTS_PASTE2(Test_, _name)

#define ICV_TESTS_PASTE2(_a, _b) _ICV_TESTS_PASTE2(_a, _b)
#define _ICV_TESTS_PASTE2(_a, _b) _a ## _b

//#define ICV_TESTS_CALL_MVTENSOR_ONCE      (1) /* uncomment it only for debug purposes */
//#define ICV_TESTS_GENERATE_DATA_PER_SHAVE (1) /* use old SHAVE loop behaviour, if defined */
//#define ICV_TEST_DO_CHECK_TENSOR_INDICES  (1) /* do check range of tensor indices upon addressing */

//#define ICV_TEST_DURATION_TIMINGS /* internal: do suite/generate/check timings */

#define ICV_TEST_ITER_NAME_STR_MAXSIZE     (64) /* max size of buffer to store iterator name string */
#define ICV_TEST_TEST_FILTER_STR_MAXSIZE (1024) /* max size of buffer to store test filter string */
#define ICV_TEST_TEST_NAME_STR_MAXSIZE    (256) /* max size of buffer to store test name string */
#define ICV_TEST_RUN_PARAMS_STR_MAXSIZE    (64) /* max size of buffer to store formatted runParams string */
#define ICV_TEST_TEST_PARAMS_STR_MAXSIZE  (256) /* max size of buffer to store formatted testParams string */

//== Utilities =================================================================

#if defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64)
# if defined(snprintf)
#  undef snprintf
# endif
# define snprintf _snprintf_s
#endif

inline void update_append(char*& str, int& maxLength)
{
    str[maxLength] = 0;
    const int len = strlen(str);
    str += len;
    maxLength -= len;
}

inline void strncpy_append(char*& str, int& maxLength, const char* src)
{
    strncpy(str, src, maxLength);
    update_append(str, maxLength);
}

inline void snprintf_append(char*& str, int& maxLength, const char* fmt, ...)
{
    va_list argPtr;
    va_start(argPtr, fmt);
    vsnprintf(str, maxLength, fmt, argPtr);
    va_end(argPtr);
    update_append(str, maxLength);
}

inline void rand_seed()
{
#ifdef USE_RANDOM_SEED
    u64 ticks_for_seed;
    DrvTimerGetSystemTicks64(&ticks_for_seed);
    srand(ticks_for_seed);
#else
    srand(1);
#endif

}

//== Debugging =================================================================

//#define DEBUG 0
#if defined(DEBUG) && (DEBUG != 0)
# include <stdarg.h>
# include <stdio.h>
#endif // DEBUG

class Logger
{
public:
    Logger(const char* aName, bool state = true)
    #if defined(DEBUG) && (DEBUG != 0)
        : name(aName)
        , instanceEnabled(state)
    #endif // DEBUG
        {
        #if defined(DEBUG) && (DEBUG != 0)
            if (globalEnabled && instanceEnabled)
            {
                indent(); printf(">> %s\n", name); fflush(stdout); ++level;
            }
        #else // DEBUG
            (void)aName;
            (void)state;
        #endif // DEBUG
        }
    ~Logger()
        {
        #if defined(DEBUG) && (DEBUG != 0)
            if (globalEnabled && instanceEnabled)
            {
                --level; indent(); printf("<< %s\n", name); fflush(stdout);
            }
        #endif // DEBUG
        }
    void print(const char* fmt, ...)
        {
        #if defined(DEBUG) && (DEBUG != 0)
            if (globalEnabled && instanceEnabled)
            {
                indent(); printf("-- %s: ", name);
                va_list argPtr;
                va_start(argPtr, fmt);
                vprintf(fmt, argPtr);
                va_end(argPtr);
                printf("\n"); fflush(stdout);
            }
        #else // DEBUG
            (void)fmt;
        #endif // DEBUG
        }
    void enable(bool state)
        {
        #if defined(DEBUG) && (DEBUG != 0)
            instanceEnabled = state;
        #else // DEBUG
            (void)state;
        #endif // DEBUG
        }
    void global(bool state)
        {
        #if defined(DEBUG) && (DEBUG != 0)
            globalEnabled = state;
        #else // DEBUG
            (void)state;
        #endif // DEBUG
        }
protected:
private:
#if defined(DEBUG) && (DEBUG != 0)
    void indent()
        {
            for (int i = 0; i < level; ++i)
                printf("|  ");
        }
    const char* name;
    bool instanceEnabled;
#endif // DEBUG
    static bool globalEnabled;
    static int level;
};

//== Drivers ===================================================================

class MyriadDrivers
{
public:
    MyriadDrivers()
        {}
    ~MyriadDrivers()
        {}
    static void invalidateAllCaches()
        {
            rtems_cache_writeback_l2();
            rtems_cache_invalidate_l2();
            rtems_cache_invalidate_entire_l1_data();
        }
    static void leonFlushL2Range(void* ptr, size_t sz)
        { // TEST_END_CACHE_RESET_COMMON;
            if ((ptr != nullptr) && (sz > 0))
            {
                u32 startAddress = reinterpret_cast<u32>(ptr);
                rtems_cache_flush_multiple_data_lines((void *)startAddress, sz);
            }
        }
    static void leonFlushL2Cache()
        {
            rtems_cache_flush_entire_data();
        }
protected:
private:
};

//== Iterators =================================================================

class Iterator
{
public:
    typedef std::function<void(bool)> Action;
    typedef std::function<void(char*, int, const Iterator*)> Formatter;
    void append(Iterator* link, Action action = Action())
        {
            m_link = link;
            m_action = action;
        }
    Iterator* link() const
        { return m_link; }
    const char* name() const
        { return m_name; }
    Formatter formatter() const
        { return m_formatter; }
    virtual int count() const = 0;
    virtual int size() const = 0;
    virtual void reset() = 0;
    virtual bool next() = 0;
    void setAction(Action action = Action())
        { m_action = action; }
    void action(bool begin)
        {
            if (m_action)
                m_action(begin);
        }
protected:
    explicit Iterator(Formatter formatter)
        : m_link(nullptr)
        , m_name("")
        , m_formatter(formatter ? formatter : defaultFormatter)
        {}
    explicit Iterator(const char* name = nullptr)
        : m_link(nullptr)
        , m_name(name ? name : "")
        , m_formatter(defaultFormatter)
        {}
    virtual ~Iterator()
        { m_link = nullptr; }
    static void defaultFormatter(char* str, int maxLength, const Iterator* i)
        {
            str[0] = 0;
            const char* name = i->name();
            if ((name != nullptr) && (name[0] != 0))
            {
                snprintf(str, maxLength, "%s%d", name, i->count());
                str[maxLength] = 0;
            }
        }
private:
    Iterator* m_link;
    Action m_action;
    const char* m_name;
    Formatter m_formatter;
};

template<class CountType>
class CounterIterator: public Iterator
{
    typedef Iterator Inherited;
protected:
    using typename Inherited::Formatter;
public:
    CounterIterator(CountType start, CountType end, Formatter formatter)
        : Inherited(formatter)
        , m_start(start)
        , m_end(end)
        , m_cur(start)
        {}
    CounterIterator(CountType start, CountType end, const char* name = nullptr)
        : Inherited(name)
        , m_start(start)
        , m_end(end)
        , m_cur(start)
        {}
    explicit CounterIterator(Formatter formatter)
        : Inherited(formatter)
        , m_start(0)
        , m_end(0)
        , m_cur(0)
        {}
    explicit CounterIterator(const char* name = nullptr)
        : Inherited(name)
        , m_start(0)
        , m_end(0)
        , m_cur(0)
        {}
    virtual ~CounterIterator()
        {}
    void init(CountType start, CountType end)
        {
            m_start = start;
            m_end = end;
            m_cur = start;
        }
    CountType start() const
        { return m_start; }
    CountType end() const
        { return m_end; }
    CountType value() const
        { return m_cur; }
    int count() const override
        { return static_cast<int>(value()); }
    int size() const override
        { return int(m_end - m_start); }
    void reset() override
        { m_cur = m_start; }
    bool next() override
        {
            if (m_cur < m_end-1)
            {
                ++m_cur;
                return true;
            }
            return false;
        }
protected:
private:
    CountType m_start;
    CountType m_end;
    CountType m_cur;
};

template<class CountType>
class MappedIterator: public CounterIterator<CountType>
{
    typedef CounterIterator<CountType> Inherited;
protected:
    using typename Inherited::Formatter;
public:
    typedef std::function<CountType(CountType)> Map;
    MappedIterator(CountType start, CountType end, Formatter formatter, Map map = Map())
        : Inherited(start, end, formatter)
        , m_map(map)
        {}
    MappedIterator(CountType start, CountType end, const char* name = nullptr, Map map = Map())
        : Inherited(start, end, name)
        , m_map(map)
        {}
    explicit MappedIterator(Formatter formatter)
        : Inherited(formatter)
        {}
    explicit MappedIterator(const char* name = nullptr)
        : Inherited(name)
        {}
    virtual ~MappedIterator()
        {}
    void init(CountType start, CountType end, Map map = Map())
        {
            Inherited::init(start, end);
            m_map = map;
        }
    void setMap(Map map = Map())
        { m_map = map; }
    CountType value() const
        {
            CountType val = Inherited::value();
            if (m_map)
            {
                const CountType start = Inherited::start();
                val = start + m_map(val - start);
            }
            return val;
        }
protected:
private:
    Map m_map;
};

template<class SampleType, class CountType>
class SampleIterator: public MappedIterator<CountType>
{
    typedef MappedIterator<CountType> Inherited;
protected:
    using typename Inherited::Formatter;
public:
    typedef typename Inherited::Map Map;
    SampleIterator(SampleType start, SampleType end, CountType count, Formatter formatter, Map map = Map())
        : Inherited(0, count, formatter, map)
        , m_start(start)
        , m_delta((end - start) / SampleType(count - 1))
        {}
    SampleIterator(SampleType start, SampleType end, CountType count, const char* name = nullptr, Map map = Map())
        : Inherited(0, count, name, map)
        , m_start(start)
        , m_delta((end - start) / SampleType(count - 1))
        {}
    explicit SampleIterator(Formatter formatter)
        : Inherited(0, 0, formatter)
        , m_start(0)
        , m_delta(0)
        {}
    explicit SampleIterator(const char* name = nullptr)
        : Inherited(0, 0, name)
        , m_start(0)
        , m_delta(0)
        {}
    virtual ~SampleIterator()
        {}
    void init(SampleType start, SampleType end, CountType count, Map map = Map())
        {
            Inherited::init(0, count);
            m_start = start;
            m_delta = ((end - start) / SampleType(count - 1));
        }
    SampleType value() const
        { return m_start + m_delta * SampleType(Inherited::value()); }
protected:
private:
    SampleType m_start;
    SampleType m_delta;
};

class FlipIterator: public CounterIterator<int>
{
    typedef CounterIterator<int> Inherited;
protected:
    using typename Inherited::Formatter;
public:
    explicit FlipIterator(Formatter formatter)
        : Inherited(0, 2, formatter)
        {}
    explicit FlipIterator(const char* name = nullptr)
        : Inherited(0, 2, name)
        {}
    virtual ~FlipIterator()
        {}
    bool value() const
        { return bool(Inherited::value() != 0); }
protected:
private:
};


template<class Container>
class ListIteratorBase: public CounterIterator<typename Container::size_type>
{
    typedef CounterIterator<typename Container::size_type> Inherited;
protected:
    using typename Inherited::Formatter;
public:
    ListIteratorBase(const Container& list, Formatter formatter)
        : Inherited(0, list.size(), formatter)
        , m_list(list)
        , m_cur(m_list.begin())
        {}
    ListIteratorBase(const Container& list, const char* name = nullptr)
        : Inherited(0, list.size(), name)
        , m_list(list)
        , m_cur(m_list.begin())
        {}
    explicit ListIteratorBase(Formatter formatter)
        : Inherited(0, 0, formatter)
        , m_list(Container())
        , m_cur(m_list.begin())
        {}
    explicit ListIteratorBase(const char* name = nullptr)
        : Inherited(0, 0, name)
        , m_list(Container())
        , m_cur(m_list.begin())
        {}
    virtual ~ListIteratorBase()
        {}
    void init(const Container& list)
        {
            Inherited::init(0, list.size());
            m_list = list;
            m_cur = m_list.begin();
        }
    const typename Container::value_type& value() const
        { return *m_cur; }
    void reset() override
        {
            Inherited::reset();
            m_cur = m_list.begin();
        }
    bool next() override
        {
            if (Inherited::next())
            {
                ++m_cur;
                return true;
            }
            return false;
        }
protected:
private:
    Container m_list;
    typename Container::const_iterator m_cur;
};

template <typename ItemType>
using ListIterator = ListIteratorBase<const std::initializer_list<ItemType> >;
template <typename ItemType>
using ListIteratorVec = ListIteratorBase<std::vector<ItemType> >;

template<class ItemType>
class DependentListIterator: public ListIteratorVec<ItemType>
{
    typedef ListIteratorVec<ItemType> Inherited;
    using CheckFunc = std::function<bool(ItemType)>;
    using PreprocessFunc = std::function<ItemType(ItemType, ItemType, size_t, size_t)>;
public:
    DependentListIterator(ItemType start, ItemType end, size_t count,
                          CheckFunc cfunc = defaultChecker,
                          PreprocessFunc pfunc = defaultPreprocessor,
                          const char* name = nullptr)
        : Inherited(name)
        , m_start(start)
        , m_end(end)
        , m_count(count)
        , checker(cfunc)
        , preprocessor(pfunc)
        { init(start, end, count); }
    void setChecker(CheckFunc func)
        { checker = func; }
    void setPreprocessor(PreprocessFunc func)
        { preprocessor = func; }
    void reset() override
        { regenerateValues(); }
    size_t regenerateValues() //return number of generated points
        {
            ItemType temp_start = m_start;
            ItemType temp_end = m_end;

            while (!checker(temp_start) && temp_start<=temp_end)
            {
                temp_start++;
            }
            while (!checker(temp_end) && temp_start<=temp_end)
            {
                temp_end--;
            }
            if (temp_start >= temp_end)
            {
                // Error
                temp_start = temp_end = m_start;
            }

            return init(temp_start, temp_end, m_count);
        }
    size_t init(ItemType start, ItemType end, size_t count) //return number of generated points
        {
            std::set<ItemType> list;
            for (size_t i = 0; i < count; i++)
            {
                ItemType val = preprocessor(start, end, m_count, i);
                while ( !checker(val) || list.find(val) != list.end() )
                {
                    val++;
                    if (val > m_end)
                        break;
                }
                if (val <= m_end)
                {
                    list.insert(val);
                }
            }
            auto itemsVec = std::vector<ItemType>(list.begin(), list.end());
            Inherited::init(itemsVec);
            return list.size();
        }
    static bool defaultChecker(ItemType)
        { return true; }
    static ItemType defaultPreprocessor(ItemType start, ItemType end, size_t count, size_t i)
        {
            float step = count>1 ? static_cast<float>((end-start))/(count-1) : 0;
            return static_cast<ItemType>(start+i*step);
        }
protected:
private:
    ItemType m_start;
    ItemType m_end;
    size_t m_count;
    CheckFunc checker;
    PreprocessFunc preprocessor;
};

template<class ItemType>
class MultipleListIterator: public Iterator
{
public:
    using ValueType = std::vector<ItemType>;
    using SubIterator = DependentListIterator<ItemType>;
    using SubIteratorList = std::vector<SubIterator*>;
    MultipleListIterator(const SubIteratorList& list, const char* name = nullptr)
        : Iterator(name)
        , m_iterators(list)
        , iter_num(0)
        {}
    explicit MultipleListIterator(const char* name = nullptr)
        : Iterator(name)
        , iter_num(0)
        {}
    virtual ~MultipleListIterator()
        {}
    int size() const override
        {
            int res = 1;
            for (size_t i = 0; i < m_iterators.size(); i++)
            {
                res *= m_iterators[i]->size();
            }
            return res;
        }
    const ValueType value() const
        {
            ValueType res;
            for (size_t i = 0; i < m_iterators.size(); i++)
            {
                res.push_back(m_iterators[i]->value());
            }
            return res;
        }
    void reset() override
        {
            for (size_t i = 0; i < m_iterators.size(); i++)
            {
                m_iterators[i]->regenerateValues();
            }
        }
    bool next() override
        {
            iter_num++;
            size_t i = 0;
            for (; i < m_iterators.size(); i++)
            {
                if (m_iterators[i]->next())
                {
                    break;
                }
            }
            if (i < m_iterators.size())
            {
                return resetIteratorsInRange(0, i-1);
            }
            return false;
        }
    int count() const override
        { return static_cast<int>(iter_num); }
protected:
private:
    bool resetIteratorsInRange(int start, int end)
        {
            for (int i = end; i >= start; i--)
            {
                while ((m_iterators[static_cast<size_t>(i)]->regenerateValues() == 0))
                {
                    int k = i+1;
                    for (; k < (int)m_iterators.size(); k++)
                    {
                        if (m_iterators[k]->next())
                        {
                            break;
                        }
                    }
                    if (k >= (int)m_iterators.size())
                    {
                        return false;
                    }
                    else
                    {
                        if (!resetIteratorsInRange(start+1, k-1))
                            return false;
                    }
                }
            }
            return true;
        }
private:
    SubIteratorList m_iterators;
    size_t iter_num;
};

class LoopManager
{
public:
    LoopManager()
        : m_iterators(nullptr)
        {}
    virtual ~LoopManager()
        { m_iterators = nullptr; }
    void addLoop(Iterator& i, Iterator::Action action = Iterator::Action())
        {
            i.append(m_iterators, action);
            m_iterators = &i;
        }
    int totalIterations()
    {
        int total = 1;
        for (Iterator* i = m_iterators; i != nullptr; i = i->link())
            total *= i->size();
        return total;
    }
    void firstIteration()
        { firstIteration(m_iterators); }
    bool nextIteration()
        { return nextIteration(m_iterators); }
    void formatIterationPath(char* str, int maxLength) const
        {
            str[0] = 0;
            FormatIterationPathContext context = { str, maxLength };
            formatIterationPath(m_iterators, context);
        }
protected:
private:
    static void firstIteration(Iterator* i)
        {
            if (i)
            {
                firstIteration(i->link());
                i->action(true);
                i->reset();
            }
        }
    static bool nextIteration(Iterator* i)
        {
            if (i)
            {
                if (i->next())
                    return true;
                i->action(false);
                if (nextIteration(i->link()))
                {
                    i->reset();
                    i->action(true);
                    return true;
                }
            }
            return false;
        }
    struct FormatIterationPathContext
        {
            char* str;
            int maxLength;
        };
    static bool formatIterationPath(Iterator* i, FormatIterationPathContext& context)
        {
            bool first = true;
            if (i != nullptr)
            {
                first &= formatIterationPath(i->link(), context);
                auto formatter = i->formatter();
                if (formatter)
                {
                    char str[ICV_TEST_ITER_NAME_STR_MAXSIZE] = "";
                    formatter(str, sizeof(str)-1, i);
                    if (str[0] != 0)
                    {
                        snprintf_append(context.str, context.maxLength, "%s%s", (first ? "/" : "."), str);
                        first = false;
                    }
                }
            }
            return first;
        }
private:
    Iterator* m_iterators;
};

//== Buffers ===================================================================

// Notes
//   1) data buffer, pointed by buffer/data, is not owned by Tensor,
//      so we don't have to define move/copy constructors/assignment operators
//   2) push/pop are designed in 'inverse' notation here,
//      i.e. item . push/pop onto/from stack

class DataStorage
{
public:
    DataStorage()
        : m_buffer(nullptr)
        , m_permanent(false)
        , m_next(nullptr)
        {}
    virtual ~DataStorage()
        {
            m_buffer = nullptr;
            m_next = nullptr;
        }
    void push(DataStorage*& stack)
        {
            m_next = stack;
            stack = this;
        }
    void pop(DataStorage*& stack)
        {
            stack = m_next;
            m_next = nullptr;
        }
    virtual size_t bufferSize() const = 0;
    virtual void setBuffer(void* buffer, bool permanent = false)
        {
            m_buffer = buffer;
            m_permanent = permanent;
        }
    void* buffer() const
        { return m_buffer; }
    bool permanent() const
        { return m_permanent; }
    DataStorage* next() const
        { return m_next; }
protected:
private:
    void* m_buffer;
    bool m_permanent;
    DataStorage* m_next;
};

class StorageManager
{
public:
protected:
    StorageManager(u8* buffer, size_t bufferSize, size_t bufferAlign)
        : m_buffer(buffer)
        , m_bufferSize(bufferSize)
        , m_bufferAlign(bufferAlign)
        , m_allocatedSize(0)
        , m_bufferedList(nullptr)
        {}
    virtual ~StorageManager()
        {
            freeStorage(true);
            m_buffer = nullptr;
            m_bufferSize = 0;
        }
    template<class DataType>
    DataType* allocData(int count = 1)
        {
            size_t size = sizeof(DataType) * count;
            if (m_allocatedSize + size > m_bufferSize)
                allocError(size, m_allocatedSize, m_bufferSize);
            DataType* data = reinterpret_cast<DataType*>(m_buffer + m_allocatedSize);
            m_allocatedSize = ICV_TESTS_MEMORY_ALIGN(m_bufferAlign, m_allocatedSize + size);
            return data;
        }
    void allocBuffer(DataStorage& t, bool permanent = false)
        {
            if (!(t.buffer() || t.next()))
            {
                size_t size = t.bufferSize();
                if (m_allocatedSize + size > m_bufferSize)
                    allocError(size, m_allocatedSize, m_bufferSize);
                t.setBuffer(m_buffer + m_allocatedSize, permanent);
                m_allocatedSize = ICV_TESTS_MEMORY_ALIGN(m_bufferAlign, m_allocatedSize + size);
                t.push(m_bufferedList);
            }
        }
    void freeStorage(bool full = false)
        {
            DataStorage* b = m_bufferedList;
            while (b && (!full || !b->permanent()))
            {
                b->setBuffer(nullptr, false);
                b->pop(b);
            }
            m_bufferedList = b;
            m_allocatedSize = 0;
        }
    void setBuffer(DataStorage& t, void* buffer, bool permanent = false)
        {
            t.setBuffer(buffer, permanent);
            t.push(m_bufferedList);
        }
    virtual void allocError(int requestedSize, int allocatedSize, int bufferSize) = 0;
private:
    u8* m_buffer;
    size_t m_bufferSize;
    size_t m_bufferAlign;
    size_t m_allocatedSize;
    DataStorage* m_bufferedList;
};

//== Tensors ===================================================================

const int MaxTensorDims = subspace::MAX_DIMS;
const int DefTensorAlign = 16;
const int DefTensorDim = -1;

static const char DimNames[MaxTensorDims] = { 'W', 'H', 'C', 'N', '5', '6', '7', '8'};

struct TensorDims;
struct MemoryDims;

struct Map { enum { W, H, C, N, N5, N6, N7, N8 }; };

struct TensorDims
{
    TensorDims(int w = 0, int h = 0, int c = 0, int n = 0, int n5 = 0, int n6 = 0, int n7 = 0, int n8 = 0)
        : width(w), height(h), channels(c), batch(n), dimn5(n5), dimn6(n6), dimn7(n7), dimn8(n8)
        {}
    // creates regular dims using array data
    TensorDims(const int32_t dims_array[], int size)
    {
        mvTensorAssert(size > 0, "Size of TensorDims must be greater than zero");
        size = std::min(size, MaxTensorDims);
        for (int i = 0; i < size; ++i)
        {
            dims[i] = dims_array[i];
        }
    }

    MemoryDims toMemory(int ndims, const int32_t mapTo[]) const;
    union {
        struct {
            int width;
            int height;
            int channels;
            int batch;
            int dimn5;
            int dimn6;
            int dimn7;
            int dimn8;
        };
        int dims[8]{};
    };
};

struct MemoryDims
{
    MemoryDims(int d0 = 0, int d1 = 0, int d2 = 0, int d3 = 0, int d4 = 0, int d5 = 0, int d6 = 0, int d7 = 0)
        : dims{ d0, d1, d2, d3, d4, d5, d6, d7 }
        {}
    // creates regular dims using array data
    MemoryDims(const int32_t dims_array[], int size)
        {
            size = (size > MaxTensorDims) ? MaxTensorDims : size;
            int i = 0;
            for (; i < size; ++i)
                dims[i] = dims_array[i];
            for (; i < MaxTensorDims; ++i)
                dims[i] = DefTensorDim;
        }
    // creates dims using array data, and replacing it by 1 for broadcasted dim
    MemoryDims(const int32_t dims_array[], int size, const int32_t broadcast_flags[])
        {
            size = (size > MaxTensorDims) ? MaxTensorDims : size;
            int i = 0;
            for (; i < size; ++i)
                dims[i] = (broadcast_flags[i] ? 1 : dims_array[i]);
            for (; i < MaxTensorDims; ++i)
                dims[i] = DefTensorDim;
        }
    // creates dims as {1, 1, ... , n, ... 1, 1} where size is taken from order, and n = dims_array[dim mapped by order]
    MemoryDims(const int32_t dims_array[], t_MvTensorStorageOrder storageOrder, int dim)
        {
            int32_t map[MaxTensorDims];
            int size = orderToIndices(storageOrder, map);
            int i = 0;
            for (; i < size; ++i)
                dims[i] = 1;
            for (; i < MaxTensorDims; ++i)
                dims[i] = DefTensorDim;
            if ((dim >= 0) && (dim < size) && ((dim = map[dim]) >= 0))
                dims[dim] = dims_array[dim];
        }
    // computes # of data elements described by dims
    int total(int ndims) const
        {
            int size = dims[0];
            for (int i = 1; i < ndims; ++i)
                size *= dims[i];
            return size;
        }
    // converts to tensor-order dims
    TensorDims toTensor(int ndims, const int32_t mapFrom[]) const;
    // converts to memory-order dims
    MemoryDims toMemory(int ndims, const int32_t mapFrom[], const int32_t mapTo[]) const;
    // iterate over dims // TODO: consider generalization (iteration over subset);
    template <class Action>
    static void forEach(int ndims, const int32_t ranges[], Action action)
        {
            MemoryDims indices;
            int total = 1;
            for (int i = 0; i < ndims; ++i)
            {
                indices.dims[i] = 0;
                total *= ranges[i];
            }
            for (int n = 0; n < total; ++n)
            {
                action(indices);
                for (int i = 0; i < ndims; ++i)
                {
                    ++indices.dims[i];
                    if (indices.dims[i] < ranges[i]) break;
                    indices.dims[i] = 0;
                }
            }
        #if defined(ICV_TEST_DO_CHECK_TENSOR_INDICES)
            for (int i = 0; i < ndims; ++i)
            {
                mvTensorAssert(indices.dims[i] == 0, "Internal Tensor::forEach() iterations count mismatch");
            }
        #endif // ICV_TEST_DO_CHECK_TENSOR_INDICES
        }
    // iterate over dims
    template <class Action>
    void forEach(int ndims, Action action) const
        { forEach(ndims, dims, action); }
    // check if dims is zero or non-zero
    bool isZero(int ndims = MaxTensorDims) const
        {
            for (int i = 0; i < ndims; ++i)
                if (dims[i] != 0)
                    return false;
            return true;
        }
    bool isNonZero(int ndims = MaxTensorDims) const
        { return !isZero(ndims); }
    int32_t dims[MaxTensorDims];
};

inline MemoryDims TensorDims::toMemory(int ndims, const int32_t mapTo[]) const
{
    MemoryDims memory;
    if (ndims >= 1) memory.dims[mapTo[Map::W]]  = this->width;
    if (ndims >= 2) memory.dims[mapTo[Map::H]]  = this->height;
    if (ndims >= 3) memory.dims[mapTo[Map::C]]  = this->channels;
    if (ndims >= 4) memory.dims[mapTo[Map::N]]  = this->batch;
    if (ndims >= 5) memory.dims[mapTo[Map::N5]] = this->dimn5;
    if (ndims >= 6) memory.dims[mapTo[Map::N6]] = this->dimn6;
    if (ndims >= 7) memory.dims[mapTo[Map::N7]] = this->dimn7;
    if (ndims >= 8) memory.dims[mapTo[Map::N8]] = this->dimn8;
    return memory;
}

inline TensorDims MemoryDims::toTensor(int ndims, const int32_t mapFrom[]) const
{
    (void)ndims;
    TensorDims tensor;
    tensor.width    = mapFrom[Map::W]  != UNDEF ? this->dims[mapFrom[Map::W]]  : tensor.width;
    tensor.height   = mapFrom[Map::H]  != UNDEF ? this->dims[mapFrom[Map::H]]  : tensor.height;
    tensor.channels = mapFrom[Map::C]  != UNDEF ? this->dims[mapFrom[Map::C]]  : tensor.channels;
    tensor.batch    = mapFrom[Map::N]  != UNDEF ? this->dims[mapFrom[Map::N]]  : tensor.batch;
    tensor.dimn5    = mapFrom[Map::N5] != UNDEF ? this->dims[mapFrom[Map::N5]] : tensor.dimn5;
    tensor.dimn6    = mapFrom[Map::N6] != UNDEF ? this->dims[mapFrom[Map::N6]] : tensor.dimn6;
    tensor.dimn7    = mapFrom[Map::N7] != UNDEF ? this->dims[mapFrom[Map::N7]] : tensor.dimn7;
    tensor.dimn8    = mapFrom[Map::N8] != UNDEF ? this->dims[mapFrom[Map::N8]] : tensor.dimn8;
    return tensor;
}

inline MemoryDims MemoryDims::toMemory(int ndims, const int32_t mapFrom[], const int32_t mapTo[]) const
{
    MemoryDims memory;
    if (ndims >= 1) memory.dims[mapTo[Map::W]]  = this->dims[mapFrom[Map::W]];
    if (ndims >= 2) memory.dims[mapTo[Map::H]]  = this->dims[mapFrom[Map::H]];
    if (ndims >= 3) memory.dims[mapTo[Map::C]]  = this->dims[mapFrom[Map::C]];
    if (ndims >= 4) memory.dims[mapTo[Map::N]]  = this->dims[mapFrom[Map::N]];
    if (ndims >= 5) memory.dims[mapTo[Map::N5]] = this->dims[mapFrom[Map::N5]];
    if (ndims >= 6) memory.dims[mapTo[Map::N6]] = this->dims[mapFrom[Map::N6]];
    if (ndims >= 7) memory.dims[mapTo[Map::N7]] = this->dims[mapFrom[Map::N7]];
    if (ndims >= 8) memory.dims[mapTo[Map::N8]] = this->dims[mapFrom[Map::N8]];
    return memory;
}

struct TensorAlign
{
    TensorAlign(int a0 = 0, int a1 = 0, int a2 = 0, int a3 = 0, int a4 = 0, int a5 = 0, int a6 = 0, int a7 = 0)
        : align{ std::max(a0, 0), std::max(a1, 0), std::max(a2, 0), std::max(a3, 0),
                 std::max(a4, 0), std::max(a5, 0), std::max(a6, 0), std::max(a7, 0) }
        {}
    int align[MaxTensorDims];
};

inline TensorDims operator+(const TensorDims& d1, const TensorDims& d2)
{
    return TensorDims(d1.width + d2.width, d1.height + d2.height, d1.channels + d2.channels, d1.batch + d2.batch,
                      d1.dimn5 + d2.dimn5, d1.dimn6 + d2.dimn6, d1.dimn7 + d2.dimn7, d1.dimn8 + d2.dimn8);
}

inline MemoryDims operator+(const MemoryDims& d1, const MemoryDims& d2)
{
    MemoryDims ret;
    for (int i = 0; i < MaxTensorDims; i++)
        ret.dims[i] = d1.dims[i] + d2.dims[i];
    return ret;
}

#define ICV_TESTS_TENSOR_ERROR(_this, _code, _info) \
    TensorBase::error((const void*)(_this), (TensorBase::ErrorCode)(_code), (int)(_info), __FILE__, __LINE__)

class TensorBase
{
public:
    // Unique Tensor data type for unused template parameters
    enum class Unused { Dummy = -1 };
    // Error handling
    enum class ErrorCode { BadAlign=-1, BadIndex=-2, BadOrder=-3 };
    typedef void ErrorHandler(const void* thisPtr, ErrorCode code, int info, const char* file, int line);
    static void error(const void* thisPtr, ErrorCode code, int info, const char* file, int line)
        {
            if (errorHandler)
                errorHandler(thisPtr, code, info, file, line);
        }
    static void defaultHandler(const void* thisPtr, ErrorCode code, int info, const char* file, int line)
        {
            const char* msg = nullptr;
            switch (code)
            {
            case ErrorCode::BadAlign: msg = "align boundary must be a multiple of sizeof(data type)"; break;
            case ErrorCode::BadIndex: msg = "tensor index out of range"; break;
            case ErrorCode::BadOrder: msg = "tensor storage order is invalid"; break;
            default:                  msg = "<unknown>"; break;
            }
            printf("\nFATAL: Tensor error: this = %p, code = %d [%s], info = %d\n" "at: %s:%d\n\n",
                   thisPtr, static_cast<int>(code), msg, info, file, line); fflush(stdout);
            exit(EXIT_FAILURE);
        }
    static ErrorHandler* errorHandler;
    // Data type to code mapping
    template<class DataType>
    static t_MvTensorDataType type2code(const DataType*);
};

template<> inline t_MvTensorDataType TensorBase::type2code<u8>(const u8*) { return t_u8f; }
template<> inline t_MvTensorDataType TensorBase::type2code<int32_t>(const int32_t*) { return t_int; }
template<> inline t_MvTensorDataType TensorBase::type2code<half>(const half*) { return t_fp16; }
template<> inline t_MvTensorDataType TensorBase::type2code<float>(const float*) { return t_fp32; }
template<> inline t_MvTensorDataType TensorBase::type2code<TensorBase::Unused>(const TensorBase::Unused*) { return t_MvTensorDataType(-1); }

template<class DataType>
class Tensor : public DataStorage, public TensorBase
{
public:
    Tensor()
        : m_storageOrder(t_MvTensorStorageOrder(0))
        , m_data(nullptr)
        , m_padSize(0)
        , m_ndims(0)
        {
            for (int i = 0; i < MaxTensorDims; ++i)
                m_map[i] = UNDEF;
        }
    Tensor(Tensor &tensor)
        {
            this->m_data = tensor.m_data;
            this->m_storageOrder = tensor.m_storageOrder;
            this->m_dims = tensor.m_dims;
            this->m_steps = tensor.m_steps;
            this->m_limits = tensor.m_limits;
            this->m_padSize = tensor.m_padSize;
            this->m_ndims = tensor.m_ndims;
            {
                for (int i = 0; i < MaxTensorDims; ++i)
                    this->m_map[i] = tensor.m_map[i];
            }
        }
    Tensor &operator=(const Tensor &tensor)
        {
            if (this != &tensor)
            {
                this->m_data = tensor.m_data;
                this->m_storageOrder = tensor.m_storageOrder;
                this->m_dims = tensor.m_dims;
                this->m_steps = tensor.m_steps;
                this->m_limits = tensor.m_limits;
                this->m_padSize = tensor.m_padSize;
                this->m_ndims = tensor.m_ndims;
                {
                    for (int i = 0; i < MaxTensorDims; ++i)
                        this->m_map[i] = tensor.m_map[i];
                }
            }
            return *this;
        }
    virtual ~Tensor()
        {
            DataStorage::setBuffer(nullptr);
            m_data = nullptr;
        }
    void init(t_MvTensorStorageOrder storageOrder, const TensorDims& dims,
              const TensorAlign& align = TensorAlign(), int padSize = 0, int innerStep = 1)
        {
            // storage order
            m_storageOrder = storageOrder;
            if (!isOrderValid(storageOrder))
                ICV_TESTS_TENSOR_ERROR(this, ErrorCode::BadOrder, storageOrder);
            m_ndims = orderToIndices(storageOrder, m_map);
            // dimensions
            m_dims = dims.toMemory(m_ndims, m_map);
            // limits & steps(strides)
            int step = innerStep;
            for (int i = 0; i < m_ndims; ++i) //MaxTensorDims
            {
                m_limits.dims[i] = m_dims.dims[i];
                int size = align.align[i];
                if (size > 0) // must be aligned
                {
                    if ((size % sizeof(DataType)) != 0)
                        ICV_TESTS_TENSOR_ERROR(this, ErrorCode::BadAlign, size);
                    size /= sizeof(DataType);
                    m_limits.dims[i] = ((m_dims.dims[i] + (size - 1)) / size) * size;
                }
                m_steps.dims[i] = step;
                step *= m_limits.dims[i];
            }
            // pad
            m_padSize = std::max(padSize, 0);
        }
    void init(t_MvTensorStorageOrder storageOrder, const TensorDims& dims,
              const TensorDims& limits, int padSize = 0, int innerStep = 1)
        {
            // storage order
            m_storageOrder = storageOrder;
            if (!isOrderValid(storageOrder))
                ICV_TESTS_TENSOR_ERROR(this, ErrorCode::BadOrder, storageOrder);
            m_ndims = orderToIndices(storageOrder, m_map);
            // dimensions
            m_dims = dims.toMemory(m_ndims, m_map);
            m_limits = limits.toMemory(m_ndims, m_map);
            // limits & strides
            m_steps.dims[0] = innerStep;
            for (int i = 1; i < m_ndims; ++i)
                m_steps.dims[i] = m_limits.dims[i - 1] * m_steps.dims[i - 1];
            // pad
            m_padSize = std::max(padSize, 0);
        }
    void init(t_MvTensorStorageOrder storageOrder, const MemoryDims& dims,
              const MemoryDims& limits, int padSize = 0, int innerStep = 1)
        {
            // storage order
            m_storageOrder = storageOrder;
            if (!isOrderValid(storageOrder))
                ICV_TESTS_TENSOR_ERROR(this, ErrorCode::BadOrder, storageOrder);
            m_ndims = orderToIndices(storageOrder, m_map);
            int ndims = 0;
            for (; (ndims < subspace::MAX_DIMS) && (dims.dims[ndims] > 0); ++ndims)
                ;
            if ((ndims == 0) && (m_ndims == 1)) // enable {0} case
                ndims = m_ndims;
            if (ndims != m_ndims)
                ICV_TESTS_TENSOR_ERROR(this, ErrorCode::BadOrder, storageOrder);
            // dimensions
            m_dims = dims;
            m_limits = limits;
            // limits & strides
            m_steps.dims[0] = innerStep;
            for (int i = 1; i < m_ndims; ++i)
                m_steps.dims[i] = m_limits.dims[i - 1] * m_steps.dims[i - 1];
            // pad
            m_padSize = std::max(padSize, 0);
        }
    void init(t_MvTensorStorageOrder storageOrder, const MemoryDims& dims,
              int padSize = 0)
        {
            const MemoryDims& limits = dims;
            init(storageOrder, dims, limits, padSize);
        }
    // Geometry
    t_MvTensorStorageOrder storageOrder() const
        { return m_storageOrder; }
    int ndims() const
        { return m_ndims; }
    TensorDims tensorDims() const
        { return m_dims.toTensor(m_ndims, m_map); }
    MemoryDims memoryDims() const
        { return m_dims; }
    TensorDims tensorSteps() const
        { return m_steps.toTensor(m_ndims, m_map); }
    MemoryDims memorySteps() const
        { return m_steps; }
    TensorDims tensorLimits() const
        { return m_limits.toTensor(m_ndims, m_map); }
    MemoryDims memoryLimits() const
        { return m_limits; }
    void geometry(TensorDims& dims, TensorDims& steps, TensorDims& limits) const
        {
            dims = m_dims.toTensor(m_ndims, m_map);
            steps = m_steps.toTensor(m_ndims, m_map);
            limits = m_limits.toTensor(m_ndims, m_map);
        }
    MemoryDims toMemory(const TensorDims& tdims) const
        { return tdims.toMemory(m_ndims, m_map); }
    TensorDims toTensor(const MemoryDims& mdims) const
        { return mdims.toTensor(m_ndims, m_map); }
    // Access by indices
    int index(const MemoryDims& indices, bool allowPadding = false) const
        {
        #if defined(ICV_TEST_DO_CHECK_TENSOR_INDICES)
            const int32_t* dims = allowPadding ? m_limits.dims : m_dims.dims;
        #else // ICV_TEST_DO_CHECK_TENSOR_INDICES
            UNUSED(allowPadding);
        #endif // ICV_TEST_DO_CHECK_TENSOR_INDICES
            int res = 0;
            for (int i = 0; i < m_ndims; ++i)
            {
                int index = indices.dims[i];
        #if defined(ICV_TEST_DO_CHECK_TENSOR_INDICES)
                if ((index < 0) || (index >= m_dims.dims[i]))
                    ICV_TESTS_TENSOR_ERROR(this, ErrorCode::BadIndex);
        #endif // ICV_TEST_DO_CHECK_TENSOR_INDICES
                res += index * m_steps.dims[i];
            }
            return res;
        }
    int index(const TensorDims& indices, bool allowPadding = false) const
        {
        #if defined(ICV_TEST_DO_CHECK_TENSOR_INDICES)
            const int32_t* dims = allowPadding ? m_limits.dims : m_dims.dims;
        #else // ICV_TEST_DO_CHECK_TENSOR_INDICES
            UNUSED(allowPadding);
        #endif // ICV_TEST_DO_CHECK_TENSOR_INDICES
        #if defined(ICV_TEST_DO_CHECK_TENSOR_INDICES)
            if ((indices.width < 0) || (indices.width >= dims[m_map[Map::W]]))
                ICV_TESTS_TENSOR_ERROR(this, ErrorCode::BadIndex, indices.width);
        #endif // ICV_TEST_DO_CHECK_TENSOR_INDICES
            int res = indices.width * m_steps.dims[m_map[Map::W]];
        #if defined(ICV_TEST_DO_CHECK_TENSOR_INDICES)
            if ((indices.height < 0) || (indices.height >= dims[m_map[Map::H]]))
                ICV_TESTS_TENSOR_ERROR(this, ErrorCode::BadIndex, indices.height);
        #endif // ICV_TEST_DO_CHECK_TENSOR_INDICES
            res += indices.height * m_steps.dims[m_map[Map::H]];
        #if defined(ICV_TEST_DO_CHECK_TENSOR_INDICES)
            if ((indices.channels < 0) || (indices.channels >= dims[m_map[Map::C]]))
                ICV_TESTS_TENSOR_ERROR(this, ErrorCode::BadIndex, indices.channels);
        #endif // ICV_TEST_DO_CHECK_TENSOR_INDICES
            res += indices.channels * m_steps.dims[m_map[Map::C]];
            if (m_ndims >= 4)
            {
        #if defined(ICV_TEST_DO_CHECK_TENSOR_INDICES)
                if ((indices.batch < 0) || (indices.batch >= dims[m_map[Map::N]]))
                    ICV_TESTS_TENSOR_ERROR(this, ErrorCode::BadIndex, indices.batch);
        #endif // ICV_TEST_DO_CHECK_TENSOR_INDICES
                res += indices.batch * m_steps.dims[m_map[Map::N]];
            }
            if (m_ndims >= 5)
            {
        #if defined(ICV_TEST_DO_CHECK_TENSOR_INDICES)
                if ((indices.dimn5 < 0) || (indices.dimn5 >= dims[m_map[Map::N5]]))
                    ICV_TESTS_TENSOR_ERROR(this, ErrorCode::BadIndex, indices.dimn5);
        #endif // ICV_TEST_DO_CHECK_TENSOR_INDICES
                res += indices.dimn5 * m_steps.dims[m_map[Map::N5]];
            }
            if (m_ndims >= 6)
            {
        #if defined(ICV_TEST_DO_CHECK_TENSOR_INDICES)
                if ((indices.dimn6 < 0) || (indices.dimn6 >= dims[m_map[Map::N6]]))
                    ICV_TESTS_TENSOR_ERROR(this, ErrorCode::BadIndex, indices.dimn6);
        #endif // ICV_TEST_DO_CHECK_TENSOR_INDICES
                res += indices.dimn6 * m_steps.dims[m_map[Map::N6]];
            }
            if (m_ndims >= 7)
            {
        #if defined(ICV_TEST_DO_CHECK_TENSOR_INDICES)
                if ((indices.dimn7 < 0) || (indices.dimn7 >= dims[m_map[Map::N7]]))
                    ICV_TESTS_TENSOR_ERROR(this, ErrorCode::BadIndex, indices.dimn7);
        #endif // ICV_TEST_DO_CHECK_TENSOR_INDICES
                res += indices.dimn7 * m_steps.dims[m_map[Map::N7]];
            }
            if (m_ndims >= 8)
            {
        #if defined(ICV_TEST_DO_CHECK_TENSOR_INDICES)
                if ((indices.dimn8 < 0) || (indices.dimn8 >= dims[m_map[Map::N8]]))
                    ICV_TESTS_TENSOR_ERROR(this, ErrorCode::BadIndex, indices.dimn8);
        #endif // ICV_TEST_DO_CHECK_TENSOR_INDICES
                res += indices.dimn8 * m_steps.dims[m_map[Map::N8]];
            }
            return res;
        }
    DataType& at(const MemoryDims& indices, bool allowPadding = false)
        { return m_data[index(indices, allowPadding)]; }
    const DataType& at(const MemoryDims& indices, bool allowPadding = false) const
        { return m_data[index(indices, allowPadding)]; }
    DataType& at(const TensorDims& indices, bool allowPadding = false)
        { return m_data[index(indices, allowPadding)]; }
    const DataType& at(const TensorDims& indices, bool allowPadding = false) const
        { return m_data[index(indices, allowPadding)]; }
    // Data buffer
    size_t fullSize() const // overall (allocated) data size, in items, by axes limits (i.e. with strides)
        { return m_steps.dims[m_ndims - 1] * m_limits.dims[m_ndims - 1]; }
    size_t dataSize() const // overall (used) data size, in items, by axes dimensions (i.e. without strides)
        { return m_dims.total(m_ndims); }
    size_t bufferSize() const override // overall allocated buffer size, in bytes
        { return m_padSize * 2 + sizeof(DataType) * fullSize(); }
    void setBuffer(void* buf, bool permanent = false) override
        {
            DataStorage::setBuffer(buf, permanent);
            m_data = buffer() ? reinterpret_cast<DataType*>(reinterpret_cast<u8*>(buffer()) + m_padSize) : nullptr;
        }
    void confirmBufferData()
        { MyriadDrivers::leonFlushL2Range(buffer(), bufferSize()); }
    DataType* data() const
        { return m_data; }
    // Padding
    void fillPad(const void* patternData, int patternSize) const
        {
            if (m_buffer)
            {
                fillBytes(pad1(), m_padSize, patternData, patternSize);
                fillBytes(pad2(), m_padSize, patternData, patternSize);
            }
        }
    bool checkPad(const void* patternData, int patternSize) const
        {
            if (m_buffer)
            {
                if (!checkBytes(pad1(), m_padSize, patternData, patternSize))
                    return false;
                if (!checkBytes(pad2(), m_padSize, patternData, patternSize))
                    return false;
            }
            return true;
        }
    // Simple iterator
    template<class Action>
    void forEach(bool fullRange, Action action) const
        {
            const int32_t* ranges = fullRange ? m_limits.dims : m_dims.dims;
            MemoryDims::forEach(m_ndims, ranges, action);
        }
    // MvTensor interface
    void exportToBuffer(OpTensor& b) const
        {
            int32_t d[MaxTensorDims];
            int32_t s[MaxTensorDims];
            for (int i = 0; i < MaxTensorDims; ++i)
            {
                d[i] = m_dims.dims[i];
                s[i] = sizeof(DataType) * m_steps.dims[i];
            }
            b.set(m_data, type2code(data()), m_storageOrder, d, s);
        }
    void exportToBuffer4Dto3D(OpTensor& b) const
        {
            mvTensorAssert(ndims() == 4, "exportToBuffer4Dto3D works only with 4d tensors");
            int32_t d[3];
            int32_t s[3];
            for (int i = 0; i < 2; ++i)
            {
                d[i] = m_dims.dims[i];
                s[i] = sizeof(DataType) * m_steps.dims[i];
            }
            d[2] = m_dims.dims[2] * m_dims.dims[3];
            s[2] = sizeof(DataType) * m_steps.dims[2];
            t_MvTensorStorageOrder newOrder = maskOrder(m_storageOrder, 3);
            b.set(m_data, type2code(data()), newOrder, d, s);
        }

    void print_HWC(const char * tname, int iw0 = 0, int iw1 = -1, int ih0 = 0, int ih1 = -1, int ic0 = 0, int ic1 = -1 ) const;
    void print_CHW(const char * tname, int iw0 = 0, int iw1 = -1, int ih0 = 0, int ih1 = -1, int ic0 = 0, int ic1 = -1 ) const;
    void print_NCHW(const char * tname, int iw0 = 0, int iw1 = -1, int ih0 = 0, int ih1 = -1, int ic0 = 0, int ic1 = -1
            , int in0 = 0, int in1 = -1, int in5_0 = 0, int in5_1 = -1, int in6_0 = 0, int in6_1 = -1
            , int in7_0 = 0, int in7_1 = -1, int in8_0 = 0, int in8_1 = -1) const;
    char* dimsToStringNCHW(char dimsString[]) const
        {
            char * s = dimsString;
            s[0] = 0;
            for (int i = m_ndims - 1; i > 0; --i)
            {
                sprintf(s, "%lux", m_dims.dims[i]);
                s += strlen(s);
            }
            sprintf(s, "%lu (limits: ", m_dims.dims[0]);
            s += strlen(s);
            for (int i = m_ndims - 1; i > 0; --i)
            {
                sprintf(s, "%lux", m_limits.dims[i]);
                s += strlen(s);
            }
            sprintf(s, "%lu)", m_limits.dims[0]);
            return dimsString;
        }
    std::string dimsToStringNCHW() const
    {
        std::ostringstream stream;

        for (int i = m_ndims - 1; i >= 0; --i)
        {
            stream << m_dims.dims[i];
            if (i > 0)
            {
                stream << "x";
            }
        }
        stream << " ";

        stream << "(limits: ";
        for (int i = m_ndims - 1; i >= 0; --i)
        {
            stream << m_limits.dims[i];
            if (i > 0)
            {
                stream << "x";
            }
        }
        stream << ")";

        return stream.str();
    }
    char* indicesToString(const MemoryDims& indices, char indicesString[]) const
        {
            char* s = indicesString;
            s[0] = 0;
            for (int i = m_ndims - 1; i >= 0; --i)
            {
                sprintf(s, ":%ld", indices.dims[i]);
                s += strlen(s);
            }
            return (indicesString + 1);
        }

    void incrementLine(MemoryDims& ind, int axis)
        {
            subspace::incrementLine(ind.dims, m_dims.dims, m_ndims, axis);
        }
    void incrementPlane(MemoryDims& ind, int axis0, int axis1)
        {
            subspace::incrementPlane(ind.dims, m_dims.dims, m_ndims, axis0, axis1);
        }

    int totalLines(int axis)
        {
            return subspace::getTotalLines(m_dims.dims, m_ndims, axis);
        }
    int totalPlanes(int axis0, int axis1)
        {
            return subspace::getTotalPlanes(m_dims.dims, m_ndims, axis0, axis1);
        }

protected:
    void* pad1() const
        { return m_buffer; }
    void* pad2() const
        { return reinterpret_cast<u8*>(m_buffer + bufferSize() - m_padSize); }
    void fillBytes(void* dst, int size, const void* patternData, int patternSize) const
        {
            u8* d = reinterpret_cast<u8*>(dst);
            const u8* p = reinterpret_cast<const u8*>(patternData);
            int sz = patternSize;
            for (int i = 0; i < size; ++i)
            {
                *d++ = *p++;
                if (--sz <= 0)
                {
                    p = reinterpret_cast<const u8*>(patternData);
                    sz = patternSize;
                }
            }
        }
    bool checkBytes(const void* dst, int size, const void* patternData, int patternSize) const
        {
            const u8* d = reinterpret_cast<const u8*>(dst);
            const u8* p = reinterpret_cast<const u8*>(patternData);
            int sz = patternSize;
            for (int i = 0; i < size; ++i)
            {
                if (*d++ != *p++)
                    return false;
                if (--sz <= 0)
                {
                    p = reinterpret_cast<const u8*>(patternData);
                    sz = patternSize;
                }
            }
            return true;
        }
private:
    t_MvTensorStorageOrder m_storageOrder;
    DataType* m_data;
    MemoryDims m_dims;             // actual dimensions, least to most
    MemoryDims m_steps;            // aligned steps
    MemoryDims m_limits;           // aligned dimensions
    int m_padSize;
    int m_ndims;                   // actual # of tensor dimensions used
    int32_t m_map[MaxTensorDims];  // dimensions mapping
};

class TensorMap
{
public:
    template<class DataType>
    TensorMap(const Tensor<DataType>& tensor)
        { init(tensor.storageOrder()); }
    TensorMap(t_MvTensorStorageOrder storageOrder)
        { init(storageOrder); }
    t_MvTensorStorageOrder storageOrder() const
        { return m_storageOrder; }
    TensorDims toTensor(const MemoryDims& mdims)
        { return mdims.toTensor(m_ndims, m_map); }
    MemoryDims toMemory(const TensorDims& tdims)
        { return tdims.toMemory(m_ndims, m_map); }
protected:
    void init(t_MvTensorStorageOrder storageOrder)
        {
            m_storageOrder = storageOrder;
            if (!isOrderValid(storageOrder))
                ICV_TESTS_TENSOR_ERROR(this, TensorBase::ErrorCode::BadOrder, storageOrder);
            m_ndims = orderToIndices(storageOrder, m_map);
        }
private:
    t_MvTensorStorageOrder m_storageOrder;
    int m_ndims;
    int32_t m_map[MaxTensorDims];
};

class MemoryMap
{
public:
    template<class TypeFrom, class TypeTo>
    MemoryMap(const Tensor<TypeFrom>& tensorFrom, const Tensor<TypeTo>& tensorTo)
        { init(tensorFrom.storageOrder(), tensorTo.storageOrder()); }
    MemoryMap(t_MvTensorStorageOrder orderFrom, t_MvTensorStorageOrder orderTo)
        { init(orderFrom, orderTo); }
    t_MvTensorStorageOrder orderFrom() const
        { return m_orderFrom; }
    t_MvTensorStorageOrder orderTo() const
        { return m_orderTo; }
    MemoryDims remap(const MemoryDims& mdims) const
        { return mdims.toMemory(m_ndims, m_mapFrom, m_mapTo); }
protected:
    void init(t_MvTensorStorageOrder orderFrom, t_MvTensorStorageOrder orderTo)
        {
            m_orderFrom = orderFrom;
            m_orderTo = orderTo;
            if (!isOrderValid(orderFrom))
                ICV_TESTS_TENSOR_ERROR(this, TensorBase::ErrorCode::BadOrder, orderFrom);
            if (!isOrderValid(orderTo))
                ICV_TESTS_TENSOR_ERROR(this, TensorBase::ErrorCode::BadOrder, orderTo);
            int fromDims = orderToIndices(m_orderFrom, m_mapFrom);
            m_ndims = orderToIndices(m_orderTo, m_mapTo);
            // according to current storage order rules, higher dimensions cannot be used
            // in permutation without using of lower; so it's enough to compare their sizes
            if (fromDims != m_ndims)
            {
                ICV_TESTS_TENSOR_ERROR(this, TensorBase::ErrorCode::BadOrder, orderFrom);
                ICV_TESTS_TENSOR_ERROR(this, TensorBase::ErrorCode::BadOrder, orderTo);
            }
        }
private:
    t_MvTensorStorageOrder m_orderFrom;
    t_MvTensorStorageOrder m_orderTo;
    int m_ndims;
    int32_t m_mapFrom[MaxTensorDims];
    int32_t m_mapTo[MaxTensorDims];
};

template <class T>
void print_element(T e);

template <>
inline void print_element<fp16>(fp16 e)
{
    printf("%8.4f ", f16Tof32(e));
}

template <>
inline void print_element<float>(float e)
{
    printf("%8.4f ", e);
}

template <>
inline void print_element<int32_t>(int32_t e)
{
    printf("%6d ", (int)e);
}

template <class DataType>
void Tensor<DataType>::print_HWC(const char * tname, int iw0, int iw1, int ih0, int ih1, int ic0, int ic1) const
{
    const Tensor<DataType>& t = *this;
    int IH = t.tensorDims().height;
    int IW = t.tensorDims().width;
    int IC = t.tensorDims().channels;
    iw1 = (iw1 == -1) ? IW-1 : iw1;
    ih1 = (ih1 == -1) ? IH-1 : ih1;
    ic1 = (ic1 == -1) ? IC-1 : ic1;

    printf("%s: H=%i, W=%i, C=%i\n", tname, IH, IW, IC);
    for (int ih = ih0; ih <= ih1; ih++)
    {
        printf("h %i: ", ih);
        for (int iw = iw0; iw <= iw1 ; iw++)
        {
            printf("(");
            for (int ic = ic0; ic <= ic1; ic++)
            {
                print_element<DataType>(t.at(TensorDims(iw, ih, ic, 0)));
            }
            printf("), ");
        }
        printf("\n"); fflush(stdout);
    }
}

template <class DataType>
void Tensor<DataType>::print_CHW(const char * tname, int iw0, int iw1, int ih0, int ih1, int ic0, int ic1) const
{
    const Tensor<DataType>& t = *this;
    int IH = t.tensorDims().height;
    int IW = t.tensorDims().width;
    int IC = t.tensorDims().channels;
    iw1 = (iw1 == -1) ? IW-1 : iw1;
    ih1 = (ih1 == -1) ? IH-1 : ih1;
    ic1 = (ic1 == -1) ? IC-1 : ic1;

    printf("%s: H=%i, W=%i, C=%i\n", tname, IH, IW, IC);
    for (int ic = ic0; ic <= ic1; ic++)
    {
        printf("ic %i: \n", ic);
        for (int ih = ih0; ih <= ih1; ih++)
        {
            for (int iw = iw0; iw <= iw1 ; iw++)
            {
                {
                    print_element<DataType>(t.at(TensorDims(iw, ih, ic, 0)));
                }
                printf(", ");
            }
            printf("\n"); fflush(stdout);
        }
    }
}

namespace {

// Functions just to decrease cyclomatic code complexity
// to satisfy less than 25 'cyclomatic complexity' scores requirement
inline void print_if(bool condition, const char * title, int val)
{
    if (condition)
        printf("%s %i: \n", title, val);
}

template <class T>
inline T iif(bool condition, T forTrue, T forFalse)
{
    return (condition) ? forTrue : forFalse;
}

}; // namespace

template <class DataType>
void Tensor<DataType>::print_NCHW(const char * tname, int iw0, int iw1, int ih0, int ih1, int ic0, int ic1
                                , int in0, int in1, int in5_0, int in5_1, int in6_0, int in6_1
                                , int in7_0, int in7_1, int in8_0, int in8_1) const
{
    const Tensor<DataType>& t = *this;
    int IH = t.tensorDims().height;
    int IW = t.tensorDims().width;
    int IC = t.tensorDims().channels;
    int IN = t.tensorDims().batch;
    int IN5 = t.tensorDims().dimn5;
    int IN6 = t.tensorDims().dimn6;
    int IN7 = t.tensorDims().dimn7;
    int IN8 = t.tensorDims().dimn8;

    iw1 = iif(iw1 == -1, IW-1, iw1);
    ih1 = iif(ih1 == -1, IH-1, ih1);
    ic1 = iif(ic1 == -1, IC-1, ic1);

    in1 = iif(in1 == -1, IN-1, in1);
    in1 = iif(IN == 0, 0, in1);
    in0 = iif(IN == 0, 0, in0);

    in5_1 = iif(in5_1 == -1, IN5-1, in5_1);
    in5_1 = iif(IN5 == 0, 0, in5_1);
    in5_0 = iif(IN5 == 0, 0, in5_0);

    in6_1 = iif(in6_1 == -1, IN6-1, in6_1);
    in6_1 = iif(IN6 == 0, 0, in6_1);
    in6_0 = iif(IN6 == 0, 0, in6_0);

    in7_1 = iif(in7_1 == -1, IN7-1, in7_1);
    in7_1 = iif(IN7 == 0, 0, in7_1);
    in7_0 = iif(IN7 == 0, 0, in7_0);

    in8_1 = iif(in8_1 == -1, IN8-1, in8_1);
    in8_1 = iif(IN8 == 0, 0, in8_1);
    in8_0 = iif(IN8 == 0, 0, in8_0);

    printf("%s: N8=%i, N7=%i, N6=%i, N5=%i, N=%i, C=%i, H=%i, W=%i\n", tname, IN8, IN7, IN6, IN5, IN, IC, IH, IW);
    for (int in8 = in8_0; in8 <= in8_1; in8++)
    {
        print_if(IN8 > 0, "in8", in8);
        for (int in7 = in7_0; in7 <= in7_1; in7++)
        {
            print_if(IN7 > 0, "in7", in7);
            for (int in6 = in6_0; in6 <= in6_1; in6++)
            {
                print_if(IN6 > 0, "in6", in6);
                for (int in5 = in5_0; in5 <= in5_1; in5++)
                {
                    print_if(IN5 > 0, "in5", in5);
                    for (int in = in0; in <= in1; in++)
                    {
                        print_if(IN > 0, "in", in);
                        for (int ic = ic0; ic <= ic1; ic++)
                        {
                            printf("ic %i: \n", ic);
                            for (int ih = ih0; ih <= ih1; ih++)
                            {
                                for (int iw = iw0; iw <= iw1 ; iw++)
                                {
                                    {
                                        TensorDims ind(iw, ih, ic, in, in5, in6, in7, in8);
                                        print_element<DataType>(t.at(ind));
                                    }
                                    printf(", ");
                                }
                                printf("\n"); fflush(stdout);
                            }
                        }
                    }
                }
            }
        }
    }
}

//== SuiteRunner ===============================================================

class SuiteRunner
{
public:
    virtual const char* suiteName() const = 0;
    virtual int run() = 0;
protected:
private:
};

class TestSuite;

class SuiteRegistry
{
    using TestSuitePtr = std::unique_ptr<TestSuite>;
    using TestSuiteFactory = std::function<TestSuitePtr()>;
public:
    static SuiteRegistry& getInstance();

    static void registerSuite(const TestSuiteFactory& suite);

    template<class Action>
    static void forEach(Action action)
        {
            auto& registry = SuiteRegistry::getInstance();
            auto& suites = registry.m_suiteBuilders;

            for (const auto& suite : suites)
            {
                const TestSuitePtr suiteInstance = suite();
                SuiteRunner* suiteRunner = static_cast<SuiteRunner*>(suiteInstance.get());
                action(suiteRunner);
            }
        }
protected:
private:
    std::deque<TestSuiteFactory> m_suiteBuilders;
};

class SuiteHandle
{
public:
    SuiteHandle()
        {}
    ~SuiteHandle()
        {}

    template<class Suite>
    SuiteHandle& registerSuite()
        {
            static_assert(std::is_base_of<TestSuite, Suite>::value);
            SuiteRegistry::registerSuite([](){return std::unique_ptr<Suite>(new Suite());});
            return *this;
        }
protected:
private:
};

//== UnitTest local replacement ================================================

int unitTestInit();
int unitTestLogFail();
int unitTestLogPass();
int unitTestFinalReport();

//== Test Suite ================================================================

enum class RunMode { Run=0, ListSuites=1, ListTests=2 }; // RUN_MODE
enum class CheckMode { No=0, Cancel=1, Continue=2, Success=3 }; // CHECK_RESULT

class GlobalData
{
public:
    static void init();
public:
    static const int dataPartitionNo;
    static const int instrPartitionNo;
    static const int bypassPartitionNo;
    static const int maxShaves;
    static int startShave;
    static int numShavesToBegin;
    static int numShavesToEnd;
    static int numRepeats;

    static bool doPrintDiffs;
    static bool doPrintDiffRange;
    static bool doPrintDiffMax;

    static RunMode runMode;     // RUN_MODE
    static bool doPrintName;    // PRINT_NAME
    static bool doPrintTime;    // PRINT_TIME
    static bool doPrintParams;  // PRINT_PARAMS
    static CheckMode checkMode; // CHECK_RESULT
    static bool doCallOnce;     // CALL_ONCE
    static bool doPrintPerfCounters; // PRINT_PERF_COUNTERS

    static char testFilter[];
//    static MvTensorDmaDescriptor dmaTask[];

    static nn::shave_lib::ShavePerfCounters shavePerfCounters;
    static u8 memoryBuffer[];

    static u8 *getMemoryBuffer();
protected:
private:
};

class TestSuite: public StorageManager, public LoopManager, public SuiteRunner
{

#if defined(ICV_TESTS_REPORT_PERF_CYCLES)
    using DurationTime = uint32_t;
#else // ICV_TESTS_REPORT_PERF_CYCLES
    using DurationTime = float;
#endif // ICV_TESTS_REPORT_PERF_CYCLES

public:
    TestSuite()
        : StorageManager(GlobalData::getMemoryBuffer(), ICV_TESTS_MEMORY_BUFFER_SIZE, ICV_TESTS_MEMORY_BUFFER_ALIGN)
        , m_repeatsLoop() // force empty name - ".repeats" excluded from formatted test names
        , m_shavesLoop("sh")
        , m_numShavesToBegin(1)
        , m_numShavesToEnd(MVTENSOR_MAX_SHAVES)
        , m_totalDuration(0)
        , m_numCalls(0)
        , m_op(nullptr)
        , m_totalTests(0)
        , m_testNumber(0)
        , m_exitStatus(EXIT_FAILURE)
    #if defined(ICV_TEST_DURATION_TIMINGS)
        , m_suiteDuration(0)
        , m_generateDuration(0)
        , m_checkDuration(0)
    #endif // ICV_TEST_DURATION_TIMINGS
        {}
    virtual ~TestSuite()
        { freeStorage(true); }
    int exitStatus() const
        { return m_exitStatus; }
protected:
    // Tests filtering
    int run() override
        {
            const char* nameString = suiteName();
            bool match = filterName(nameString, GlobalData::testFilter);
            if (match)
            {
                if (GlobalData::runMode == RunMode::ListSuites)
                {
                    printf("%s\n", suiteName());
                    return EXIT_SUCCESS;
                }
                else
                {
            #if defined(ICV_TEST_DURATION_TIMINGS)
                    m_suiteDuration = m_generateDuration = m_checkDuration = 0;
                    mv::tensor::Timer suiteTimer;
            #endif // ICV_TEST_DURATION_TIMINGS
                    execute();
            #if defined(ICV_TEST_DURATION_TIMINGS)
                    m_suiteDuration += static_cast<float>(suiteTimer.elapsed());
                    const float others = m_suiteDuration - (m_generateDuration + m_checkDuration);
                    printf("\nStatistics (suite, generate, check, others): %f %f %f %f\n",
                           m_suiteDuration/1000, m_generateDuration/1000, m_checkDuration/1000, others/1000);
            #endif // ICV_TEST_DURATION_TIMINGS
                    return exitStatus();
                }
            } else {
                return EXIT_SUCCESS;
            }
        }
    virtual void formatTestName(char* str, int maxLength) const
        {
            snprintf_append(str, maxLength, "%s.%s", suiteName(), getOpName(opType()));
            formatIterationPath(str, maxLength);
            str[maxLength] = 0;
        }
    virtual void formatRunParams(char* str, int maxLength) const
        { snprintf_append(str, maxLength, "running on S%d:S%d", firstShave(), lastShave()); }
    virtual void formatTestParams(char* /*str*/, int /*maxLength*/) const
        {}
    virtual void formatCallDuration(char* /*str*/, int /*maxLength*/) const
        {}
    virtual bool filterName(const char* nameString, const char* filterString) const
        {
            // empty filter matches any
            if (filterString[0] == 0)
                return true;

            // strtok functions do modify string to be parsed, so we copy it into temp buffer
            char tempFilter[ICV_TEST_TEST_FILTER_STR_MAXSIZE];
            strcpy(tempFilter, filterString);
            char* filter = tempFilter;

            // match + - cases separately. TODO consider +- to be 'sticky'
            bool matchPositive = false, matchNegative = false;

            char *token, *saveptr;
            while((token = strtok_r(filter, ":", &saveptr)) != nullptr)
            {
                bool positive = true;
                switch (token[0])
                {
                case '+':
                    ++token; positive = true;
                    break;
                case '-':
                    ++token; positive = false;
                    break;
                default:
                    positive = true;
                    break;
                }
                if (token[0] != 0)
                {
                    bool match = bool(0 == fnmatch(token, nameString, 0));
                    if (positive)
                        matchPositive |= match;
                    else
                        matchNegative |= match;
                }
                filter = nullptr;
            }

            return (matchPositive && !matchNegative);
        }
    // Test execution
    enum class ExecutionStage { BeforeLoops, NextRepeat, BeforeCall, AfterCall, AfterLoops };
    virtual void execute()
        {
            setExitStatus(EXIT_SUCCESS);

            bool pending_testAndMvTensorInit    = true;
            bool pending_printReportBeforeLoops = true;
            bool pending_printReportNextRepeat  = false;
            bool pending_initAndGenerateData    = true;

            m_totalDuration = 0;
            m_numCalls = 0;

            limitNumShaves(GlobalData::numShavesToBegin, GlobalData::numShavesToEnd);

            m_repeatsLoop.init(0, (GlobalData::runMode == RunMode::Run) ? GlobalData::numRepeats : 1);
            m_shavesLoop.init(m_numShavesToBegin, m_numShavesToEnd + 1);

            initResources();
            initLoops();

            auto dataControl = [this, &pending_initAndGenerateData] (bool begin)
                {
                    if (begin)
                    {
                        pending_initAndGenerateData = true;
                    }
                    else
                    {
                        if (GlobalData::runMode == RunMode::Run)
                        {
                            if (!pending_initAndGenerateData)
                                freeStorage(false);
                        }
                    }
                };

            auto exitControl = [this, &pending_testAndMvTensorInit, &pending_printReportBeforeLoops] ()
                {
                    if (!pending_printReportBeforeLoops)
                        printReport(ExecutionStage::AfterLoops);
                    if (!pending_testAndMvTensorInit)
                    {
                        setExitStatus((int) unitTestFinalReport());
                    }
                };

        #if defined(ICV_TESTS_GENERATE_DATA_PER_SHAVE)
            m_shavesLoop.setAction();
        #else // ICV_TESTS_GENERATE_DATA_PER_SHAVE
            m_shavesLoop.setAction(dataControl);
        #endif // ICV_TESTS_GENERATE_DATA_PER_SHAVE

            m_totalTests = totalIterations();
            if (m_totalTests <= 0)
            {
                printf("\n%s: WARNING: there are no test(s) defined, exiting\n\n", suiteName()); fflush(stdout);
                printReport(ExecutionStage::AfterLoops);
                return;
            }

            m_testNumber = 0;

            firstIteration();
            auto lastRep = m_repeatsLoop.value() - 1;
            do {
                if ((m_repeatsLoop.size() > 1) && (m_repeatsLoop.value() != lastRep))
                {
                    pending_printReportNextRepeat = true;
                    lastRep = m_repeatsLoop.value();
                }

            #if defined(ICV_TESTS_GENERATE_DATA_PER_SHAVE)
                dataControl(true);
            #endif // ICV_TESTS_GENERATE_DATA_PER_SHAVE

                formatTestName(m_testName, sizeof(m_testName)-1);
                bool match = filterName(m_testName, GlobalData::testFilter);

                if (match)
                {
                    if (GlobalData::runMode == RunMode::ListTests)
                    {
                        printf("%s\n", m_testName);
                    }
                    else if (GlobalData::runMode == RunMode::Run)
                    {
                        if (pending_testAndMvTensorInit)
                        {
                            pending_testAndMvTensorInit = false;
                            unitTestInit();
                        }
                        if (pending_printReportBeforeLoops)
                        {
                            pending_printReportBeforeLoops = false;
                            printReport(ExecutionStage::BeforeLoops);
                        }
                        if (pending_printReportNextRepeat)
                        {
                            pending_printReportNextRepeat = false;
                            printReport(ExecutionStage::NextRepeat);
                        }
                        if (pending_initAndGenerateData)
                        {
                            pending_initAndGenerateData = false;
                            initData();
                        #if defined(ICV_TEST_DURATION_TIMINGS)
                            mv::tensor::Timer generateTimer;
                        #endif // ICV_TEST_DURATION_TIMINGS
//                            printf("generateData+\n");
                            generateData();
//                            printf("generateData-\n");
                        #if defined(ICV_TEST_DURATION_TIMINGS)
                            m_generateDuration += static_cast<float>(generateTimer.elapsed());
                        #endif // ICV_TEST_DURATION_TIMINGS
                        }

                        resetOutputData();

                        MyriadDrivers::leonFlushL2Cache();

                        createParserRunner();

                        initParserRunner();

                        printReport(ExecutionStage::BeforeCall);

                        MyriadDrivers::invalidateAllCaches();

                        // twice to get more reliable timing results
                        const int count = (GlobalData::doCallOnce ? 1 : 2);

                        for (int i = 0; i < count; ++i)
                        {
//                            printf("callParserRuner=%d\n", i);
                            m_callDuration = callParserRunner(m_op);

                        }

                        ++m_numCalls;

                        destroyParserRunner();

                        m_testFailed = false;
                        if (GlobalData::checkMode != CheckMode::No)
                        {
                        #if defined(ICV_TEST_DURATION_TIMINGS)
                            mv::tensor::Timer checkTimer;
                        #endif // ICV_TEST_DURATION_TIMINGS
                            m_testFailed = !checkResult();
                        #if defined(ICV_TEST_DURATION_TIMINGS)
                            m_checkDuration += static_cast<float>(checkTimer.elapsed());
                        #endif // ICV_TEST_DURATION_TIMINGS
                            if (m_testFailed && !GlobalData::doPrintName)
                                printf("%s\n", m_testName);
                            logTestStatus(m_testFailed);
                            if (m_testFailed && (GlobalData::checkMode == CheckMode::Cancel))
                            {
                                exitControl();
                                return;
                            }
                        }

                        printReport(ExecutionStage::AfterCall);

                        m_totalDuration += m_callDuration;

                    #if defined(ICV_TESTS_GENERATE_DATA_PER_SHAVE)
                        dataControl(false);
                    #endif // ICV_TESTS_GENERATE_DATA_PER_SHAVE
                    }
                }

                ++m_testNumber;

            } while (nextIteration());

            if (GlobalData::runMode == RunMode::Run)
                exitControl();
        }
    // execute() subfunctions
    virtual void createParserRunner()
        {
            t_MvTensorOpType type = opType();
            m_op = m_opFactory.createOp(type, opManager::primaryOperation);
            mvTensorAssert(m_op != nullptr, "Can not create kernel operation");
        }
    virtual void destroyParserRunner()
        {
            if (m_op)
            {
                delete m_op;
                m_op = nullptr;
            }
        }
    virtual void initResources()
        {}
    virtual void initLoops()
        {
            if (m_repeatsLoop.size() > 1)
                addLoop(m_repeatsLoop);
            userLoops();
            addLoop(m_shavesLoop);
        }
    virtual void userLoops()
        {}
    virtual void initData()
        {}
    virtual void generateData()
        {}
    virtual void resetOutputData() = 0;
    template<class DataType>
    void resetTensorBuffer(Tensor<DataType>& t, uint8_t value = 0xaa)
        { memset(t.buffer(), value, t.bufferSize()); }
    virtual void initParserRunner()
        {
            // by default test suite calls kernels using MvTensor structures and callbacks
            // than it does not need parser and this method returns false
            // it is necessary to override this method in real test for calling using parser`s run method
            // so than it initializes corresponding parser`s structures and returns true
            destroyParserRunner();
        }
    virtual DurationTime callParserRunner(Op* op)
        {
            mvTensorAssert(op != nullptr, "Operation undefined");
            using namespace mv::tensor;
            mv::tensor::Processor proc(m_myriadResources, &m_debugInfo);

            op->perfData.perfCounters = &GlobalData::shavePerfCounters;
            op->run(proc, m_myriadResources, m_debugInfo);
            op->perfData.perfCounters = nullptr;

#if defined(ICV_TESTS_REPORT_PERF_CYCLES)
            DurationTime callDuration = GlobalData::shavePerfCounters.cycles; // clock cycles
#else // ICV_TESTS_REPORT_PERF_CYCLES
            DurationTime callDuration = static_cast<float>(op->perfData.elapsedTimeNs) / 1.0e+6f; // Ns to Ms
#endif // ICV_TESTS_REPORT_PERF_CYCLES

            return callDuration;
        }
    virtual bool checkResult()
        { return true; }
    struct ReportContext
        {
            bool need_nl_1;
            bool need_nl_2;
        };
    virtual void printReport(ExecutionStage stage)
        {
            switch (stage)
            {
            case ExecutionStage::BeforeLoops:
                printf("%s: INFO: startShave %d, numShavesToBegin %d, numShavesToEnd %d, numRepeats %d; max test runs %d\n%s",
                       suiteName(), GlobalData::startShave, m_numShavesToBegin, m_numShavesToEnd, GlobalData::numRepeats, m_totalTests,
                       (GlobalData::doPrintParams ? "" : "\n"));
                fflush(stdout);
                break;
            case ExecutionStage::NextRepeat:
                printf("\n=== %s: Repeating %d ===\n",
                       suiteName(), m_repeatsLoop.value());
                break;
            case ExecutionStage::BeforeCall:
                m_reportContext.need_nl_1 = GlobalData::doPrintParams;
                m_reportContext.need_nl_2 = false;
                if (GlobalData::doPrintName)
                {
                    printf("%s%s",
                           (m_reportContext.need_nl_1 ? "\n" : ""), m_testName);
                    m_reportContext.need_nl_1 = false;
                    m_reportContext.need_nl_2 = true;
                }
                if (GlobalData::doPrintParams)
                {
                    char runParams[ICV_TEST_RUN_PARAMS_STR_MAXSIZE] = "";
                    formatRunParams(runParams, sizeof(runParams)-1);
                    runParams[sizeof(runParams)-1] = 0;

                    char testParams[ICV_TEST_TEST_PARAMS_STR_MAXSIZE] = "";
                    formatTestParams(testParams, sizeof(testParams)-1);
                    testParams[sizeof(testParams)-1] = 0;

                    printf("%s%s%s\nParams: %s\n",
                           (m_reportContext.need_nl_1 ? "\n" : ""),
                           (m_reportContext.need_nl_2 ? ", " : ""),
                           runParams, testParams);
                    m_reportContext.need_nl_1 = false;
                    m_reportContext.need_nl_2 = false;
                }
                break;
            case ExecutionStage::AfterCall:
                if (GlobalData::doPrintTime)
                {
#if defined(ICV_TESTS_REPORT_PERF_CYCLES)
                    printf("%s%sMvTensor done in %lu clock cycles",
#else // ICV_TESTS_REPORT_PERF_CYCLES
                    printf("%s%sMvTensor done in %f ms",
#endif // ICV_TESTS_REPORT_PERF_CYCLES
                            (m_reportContext.need_nl_1 ? "\n" : ""),
                            (m_reportContext.need_nl_2 ? "; " : ""),
                            m_callDuration);
                    if (GlobalData::doPrintPerfCounters)
                    {
                        const auto &perf = GlobalData::shavePerfCounters;
                        printf("; CLK/INS/ST/BR: %d : %d : %d : %d",
                               perf.cycles, perf.instrs, perf.stalls, perf.branches);
                    }
                    if (GlobalData::checkMode != CheckMode::No)
                        printf("; test *** %s ***", (m_testFailed ? "FAILED" : "PASSED"));
                    printf("\n");
                    m_reportContext.need_nl_1 = false;
                    m_reportContext.need_nl_2 = false;
                }
                if (m_reportContext.need_nl_2)
                {
                    printf("\n");
                    m_reportContext.need_nl_2 = false;
                }
                break;
            case ExecutionStage::AfterLoops:
#if defined(ICV_TESTS_REPORT_PERF_CYCLES)
                printf("\n%s: Total of %d mvTensor call(s) took %lu clock cycles\n",
#else // ICV_TESTS_REPORT_PERF_CYCLES
                printf("\n%s: Total of %d mvTensor call(s) took %f ms\n",
#endif // ICV_TESTS_REPORT_PERF_CYCLES
                       suiteName(), m_numCalls, m_totalDuration);
                break;
            default:
                break;
            }
        }
    virtual t_MvTensorOpType opType() const = 0;
    virtual void initMyriadResources()
        {
            memset(&m_myriadResources, 0, sizeof(m_myriadResources));
            m_myriadResources.firstShave       = firstShave();
            m_myriadResources.lastShave        = lastShave();
            m_myriadResources.dmaLinkAgent     = 1;
            m_myriadResources.dataPartitionNo  = GlobalData::dataPartitionNo;
            m_myriadResources.instrPartitionNo = GlobalData::instrPartitionNo;
            m_myriadResources.bypassPartitionNo = GlobalData::bypassPartitionNo;
          //  m_myriadResources.dmaTransactions  = &GlobalData::dmaTask[0];
        }
    virtual void initDebugInfo()
        {
            memset(&m_debugInfo, 0, sizeof(m_debugInfo));
            m_debugInfo.debugMsg = m_debugMessage;
        }
    // Shaves access
    static int firstShave()
        { return GlobalData::startShave; }
    int lastShave() const
        { return GlobalData::startShave + m_shavesLoop.value() - 1; }
    int numShaves() const
        { return m_shavesLoop.value(); }
    void limitNumShaves(int toBegin, int toEnd)
        {
            const int limit = GlobalData::maxShaves - GlobalData::startShave;
            m_numShavesToBegin = std::max(0, std::min(toBegin, limit));
            m_numShavesToEnd   = std::max(0, std::min(toEnd, limit));
            const int delta = GlobalData::numShavesToEnd - GlobalData::numShavesToBegin;
            m_numShavesToBegin = std::max(m_numShavesToBegin, m_numShavesToEnd - delta);
        }
    // Misc
    static void logTestStatus(bool testFailed)
        {
            if (testFailed)
                unitTestLogFail();
            else if (GlobalData::checkMode == CheckMode::Success)
                unitTestLogPass();
        }
    static const char* layoutString(t_MvTensorStorageOrder storageOrder)
        { // note that you can't call layoutString() twice simultaneously, since they share the same static string buffer
            static char str[MaxTensorDims + 1];
            char* s = &str[sizeof(str) - 1]; *s = '\0';
            for (int i = 0; i < MaxTensorDims; ++i)
            {
                static const char chars[16] = { 'x','W','H','C','N','5','6','7','8','?','?','?','?','?','?','?' };
                int index = ((unsigned)storageOrder >> (i << 2)) & 0x0f;
                if (index == 0) break;
                *--s = chars[index];
            }
            return s;
        }
    static const char* layoutString8(t_MvTensorStorageOrder storageOrder, char layout[])
        {
            int32_t perm[MaxTensorDims];
            int ndims = orderToPermutation(storageOrder, perm);
            layout[ndims] = 0;
            for (int i = 0; i < ndims; ++i)
            {
                layout[i] = DimNames[perm[ndims - 1 - i]];
            }
            return layout;
        }
    static const char* strideString(bool hasInput, bool hasOutput)
        {
            static const char* strings[4] =
            {
                "no strides",    "input stride",
                "output stride", "in/out strides"
            };
            const int index = (hasInput ? 1 : 0) | (hasOutput ? 2 : 0);
            return strings[index];
        }
    // Error handling
    void allocError(int requestedSize, int allocatedSize, int totalSize) override
        {
            printf("\n%s: FATAL: memory allocation failure: requested %d bytes, but only %d is available (%d - %d)\n\n",
                   suiteName(), requestedSize, totalSize - allocatedSize, totalSize, allocatedSize); fflush(stdout);
            exit(EXIT_FAILURE);
        }
    void setExitStatus(int exitStatus)
        { m_exitStatus = exitStatus; }
    static void abortTests()
        { ::exit(EXIT_FAILURE); }
protected:
    MyriadDrivers m_drivers;
    CounterIterator<int> m_repeatsLoop;
    CounterIterator<int> m_shavesLoop;
    int m_numShavesToBegin;
    int m_numShavesToEnd;
    t_MvTensorMyriadResources m_myriadResources;
    t_MvTensorDebugInfo m_debugInfo;
    char m_debugMessage[MV_TENSOR_DBG_MSG_SIZE];
    bool m_testFailed;
    DurationTime m_totalDuration;
    DurationTime m_callDuration;
    unsigned m_numCalls;
    opManager m_opFactory;
    Op* m_op;
    int m_totalTests;
    int m_testNumber;
    char m_testName[ICV_TEST_TEST_NAME_STR_MAXSIZE];
private:
    int m_exitStatus;
#if defined(ICV_TEST_DURATION_TIMINGS)
    float m_suiteDuration;
    float m_generateDuration;
    float m_checkDuration;
#endif // ICV_TEST_DURATION_TIMINGS
    ReportContext m_reportContext;
};

template <class T>
struct TypeNameTrait;

#define REGISTER_TYPE(_type, _name)                                      \
    template <>                                                          \
    struct TypeNameTrait<_type>                                          \
    {                                                                    \
        static std::string name() { return ICV_TESTS_STRINGIFY(_name); } \
    }

REGISTER_TYPE(fp16, fp16);
REGISTER_TYPE(uint8_t, uint8);
REGISTER_TYPE(int32_t, int32);
REGISTER_TYPE(float, float);
REGISTER_TYPE(int8_t, int8);

//==============================================================================

}; // namespace icv_tests

#endif // _ICV_TEST_SUITE_H_
