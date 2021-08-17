// {% copyright %}

//#define ICV_TESTS_CALL_MVTENSOR_ONCE      (1) /* uncomment it only for debug purposes */
//#define ICV_TESTS_GENERATE_DATA_PER_SHAVE (1) /* use old SHAVE loop behaviour, if defined */
//#define ICV_TEST_DO_CHECK_TENSOR_INDICES  (1) /* do check range of tensor indices upon addressing */

#define ICV_TEST_SUITE_NAME AvgPooling

#include "icv_test_suite.h"

#include "Pooling.h"

using namespace icv_tests;

namespace ICV_TESTS_NAMESPACE(ICV_TEST_SUITE_NAME)
{

//#define ALL_TESTS_SET
//#define ALL_STRIDES_SET /* defined: use additional strides loop; else use 'rover' strategy */

#define WITH_EXCLUDE_PAD

#define DO_TEST_HWC_POOLING
#define DO_TEST_CHW_POOLING

#define DO_TEST_POOLING_3x3
#define DO_TEST_POOLING_MxN_AVG
#define DO_TEST_POOLING_MxN_ZERO

//#define GENERATE_SEQ_DATA

const float test_threshold = 0.000196; //0.00013;
const bool save_to_file = false;

#if defined(WITH_EXCLUDE_PAD)
unsigned int excludePad = 0;
#endif // WITH_EXCLUDE_PAD

struct Dimensions
{
    int width;
    int height;
    int channels;
};

struct KernelStride
{
    int x;
    int y;
};

struct KernelSize
{
    int width;
    int height;
};

struct KernelPad
{
    int x;
    int y;
};

struct Test
{
    Dimensions   idim;
    KernelSize   ksize;
    KernelStride kstride;
    KernelPad    kpad;
    Dimensions   odim;
};

const std::initializer_list<Test> tests_list =
{

#if !defined(ALL_TESTS_SET)
  /*     ts       ps    ch / i{    w     h     c }     ksize      kstride     kpad   o{    w     h     c }*/
  /*2097152:    2048: 1024*/ {{   64,   32, 1024 }, { 32, 16 }, { 32, 16 }, { 0, 0 }, {    2,    2, 1024 }}, // MxN Z NET
  /* 233472:   29184:    8*/ {{  128,  228,    8 }, {  4,  2 }, {  1,  1 }, { 1, 1 }, {  127,  229,    8 }}, // MxN Z IE  pad = 1 1 | 1 1
  /* 233472:   29184:    8*/ {{  128,  228,    8 }, {  2,  4 }, {  2,  2 }, { 0, 0 }, {   64,  113,    8 }}, // MxN Z IE
  /* 221952:     289:  768*/ {{   17,   17,  768 }, {  3,  3 }, {  1,  1 }, { 1, 1 }, {   17,   17,  768 }}, // 3x3   NET pad = 1 1 | 1 1
  /* 206976:     196: 1056*/ {{   14,   14, 1056 }, {  2,  2 }, {  2,  2 }, { 0, 0 }, {    7,    7, 1056 }}, // MxN Z NET
  /* 150528:   50176:    3*/ {{  224,  224,    3 }, {  4,  4 }, {  1,  1 }, { 1, 2 }, {  225,  225,    3 }}, // IE        pad = 1 2 | 3 2 [MaxPooling data: AVG HWC MxN hung on 1 SHAVE]
  /* 100352:     196:  512*/ {{   14,   14,  512 }, {  2,  2 }, {  2,  2 }, { 0, 0 }, {    7,    7,  512 }}, // MxN Z NET
  /*  98304:      64: 1536*/ {{    8,    8, 1536 }, {  3,  3 }, {  1,  1 }, { 1, 1 }, {    8,    8, 1536 }}, // 3x3   NET pad = 1 1 | 1 1
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  4,  2 }, {  1,  1 }, { 2, 1 }, {   65,   33,   16 }}, // MxN Z IE  pad = 2 1 | 2 1
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  2,  4 }, {  2,  2 }, { 1, 1 }, {   33,   16,   16 }}, // MxN Z IE  pad = 1 1 | 1 1
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  2,  2 }, {  1,  1 }, { 0, 0 }, {   63,   31,   16 }}, // MxN Z IE
  /*  13552:     484:   28*/ {{   22,   22,   28 }, {  3,  3 }, {  2,  2 }, { 0, 0 }, {   11,   11,   28 }}, // NET       pad = 0 0 | 1 1 [MaxPooling data: AVG HWC MxN failed on 1-4 SHAVEs]
  /*   3072:      24:  128*/ {{   24,    1,  128 }, {  5,  5 }, {  3,  1 }, { 0, 0 }, {    8,    1,  128 }}, // MxN A IE  pad = 0 0 | 2 4
  /*    256:     256:    1*/ {{   16,   16,    1 }, {  4,  2 }, {  1,  1 }, { 2, 1 }, {   17,   17,    1 }}, // MxN Z IE  pad = 2 1 | 2 1
  /*     25:      25:    1*/ {{    5,    5,    1 }, {  2,  2 }, {  3,  2 }, { 1, 1 }, {    2,    3,    1 }}, // CVS-11517
#endif // !ALL_TESTS_SET

#if defined(ALL_TESTS_SET) && defined(DO_TEST_POOLING_3x3)
  /*     ts       ps    ch / i{    w     h     c }     ksize      kstride     kpad   o{    w     h     c }*/
  /* 470400:    1225:  384*/ {{   35,   35,  384 }, {  3,  3 }, {  1,  1 }, { 1, 1 }, {   35,   35,  384 }}, // NET pad = 1 1 | 1 1
  /* 352800:    1225:  288*/ {{   35,   35,  288 }, {  3,  3 }, {  1,  1 }, { 1, 1 }, {   35,   35,  288 }}, // NET pad = 1 1 | 1 1
  /* 313600:    1225:  256*/ {{   35,   35,  256 }, {  3,  3 }, {  1,  1 }, { 1, 1 }, {   35,   35,  256 }}, // NET pad = 1 1 | 1 1
  /* 295936:     289: 1024*/ {{   17,   17, 1024 }, {  3,  3 }, {  1,  1 }, { 1, 1 }, {   17,   17, 1024 }}, // NET pad = 1 1 | 1 1
  /* 235200:    1225:  192*/ {{   35,   35,  192 }, {  3,  3 }, {  1,  1 }, { 1, 1 }, {   35,   35,  192 }}, // NET pad = 1 1 | 1 1
  /* 233472:   29184:    8*/ {{  128,  228,    8 }, {  3,  3 }, {  1,  1 }, { 1, 1 }, {  128,  228,    8 }}, // IE  pad = 1 1 | 1 1
  /* 221952:     289:  768*/ {{   17,   17,  768 }, {  3,  3 }, {  1,  1 }, { 1, 1 }, {   17,   17,  768 }}, // NET pad = 1 1 | 1 1
  /* 200704:     784:  256*/ {{   28,   28,  256 }, {  3,  3 }, {  1,  1 }, { 1, 1 }, {   28,   28,  256 }}, // NET pad = 1 1 | 1 1
  /* 150528:     784:  192*/ {{   28,   28,  192 }, {  3,  3 }, {  1,  1 }, { 1, 1 }, {   28,   28,  192 }}, // NET pad = 1 1 | 1 1
  /* 112896:     196:  576*/ {{   14,   14,  576 }, {  3,  3 }, {  1,  1 }, { 1, 1 }, {   14,   14,  576 }}, // NET pad = 1 1 | 1 1
  /*  98304:      64: 1536*/ {{    8,    8, 1536 }, {  3,  3 }, {  1,  1 }, { 1, 1 }, {    8,    8, 1536 }}, // NET pad = 1 1 | 1 1
  /*  81920:      64: 1280*/ {{    8,    8, 1280 }, {  3,  3 }, {  1,  1 }, { 1, 1 }, {    8,    8, 1280 }}, // NET pad = 1 1 | 1 1
  /*  50176:      49: 1024*/ {{    7,    7, 1024 }, {  3,  3 }, {  1,  1 }, { 1, 1 }, {    7,    7, 1024 }}, // NET pad = 1 1 | 1 1
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  3,  3 }, {  1,  1 }, { 1, 1 }, {   64,   32,   16 }}, // IE  pad = 1 1 | 1 1
  /*    256:     256:    1*/ {{   16,   16,    1 }, {  3,  3 }, {  1,  1 }, { 1, 1 }, {   16,   16,    1 }}, // IE  pad = 1 1 | 1 1
  /*     36:      36:    1*/ {{    6,    6,    1 }, {  3,  3 }, {  3,  3 }, { 0, 0 }, {    2,    2,    1 }}, // CVS-12295
  /*     30:      30:    1*/ {{    6,    5,    1 }, {  3,  3 }, {  3,  2 }, { 0, 0 }, {    2,    2,    1 }}, // CVS-12295
#endif // ALL_TESTS_SET & DO_TEST_POOLING_3x3

#if defined(ALL_TESTS_SET) && defined(DO_TEST_POOLING_MxN_AVG)
  /*     ts       ps    ch / i{    w     h     c }     ksize      kstride     kpad   o{    w     h     c }*/
  /*2097152:    2048: 1024*/ {{   64,   32, 1024 }, { 22, 12 }, { 22, 11 }, { 0, 0 }, {    3,    3, 1024 }}, // NET pad = 0 0 | 2 2
  /*2097152:    2048: 1024*/ {{   64,   32, 1024 }, { 12,  6 }, { 11,  6 }, { 0, 0 }, {    6,    6, 1024 }}, // NET pad = 0 0 | 3 4
  /*   4608:       9:  512*/ {{    3,    3,  512 }, {  2,  2 }, {  2,  2 }, { 0, 0 }, {    2,    2,  512 }}, // NET pad = 0 0 | 1 1
  /*   3072:      24:  128*/ {{   24,    1,  128 }, {  5,  5 }, {  3,  1 }, { 0, 0 }, {    8,    1,  128 }}, // IE  pad = 0 0 | 2 4
  /*   3072:      24:  128*/ {{   24,    1,  128 }, {  5,  1 }, {  3,  1 }, { 0, 0 }, {    8,    1,  128 }}, // NET pad = 0 0 | 2 0
#endif // ALL_TESTS_SET & DO_TEST_POOLING_MxN_AVG

#if defined(ALL_TESTS_SET) && defined(DO_TEST_POOLING_MxN_ZERO)
  /*     ts       ps    ch / i{    w     h     c }     ksize      kstride     kpad   o{    w     h     c }*/
  /*6291456: 2097152:    3*/ {{ 2048, 1024,    3 }, {  2,  2 }, {  2,  2 }, { 0, 0 }, { 1024,  512,    3 }}, // NET
  /*2097152:    8192:  256*/ {{  128,   64,  256 }, {  2,  2 }, {  2,  2 }, { 0, 0 }, {   64,   32,  256 }}, // NET
  /*2097152:    2048: 1024*/ {{   64,   32, 1024 }, { 32, 16 }, { 32, 16 }, { 0, 0 }, {    2,    2, 1024 }}, // NET
  /* 602112:    3136:  192*/ {{   56,   56,  192 }, {  2,  2 }, {  2,  2 }, { 0, 0 }, {   28,   28,  192 }}, // NET
  /* 401408:    3136:  128*/ {{   56,   56,  128 }, {  2,  2 }, {  2,  2 }, { 0, 0 }, {   28,   28,  128 }}, // NET
  /* 344064:    7168:   48*/ {{  112,   64,   48 }, {  4,  4 }, {  4,  4 }, { 0, 0 }, {   28,   16,   48 }}, // NET
  /* 344064:    7168:   48*/ {{  112,   64,   48 }, {  2,  2 }, {  2,  2 }, { 0, 0 }, {   56,   32,   48 }}, // NET
  /* 301056:     784:  384*/ {{   28,   28,  384 }, {  2,  2 }, {  2,  2 }, { 0, 0 }, {   14,   14,  384 }}, // NET
  /* 233472:   29184:    8*/ {{  128,  228,    8 }, {  4,  2 }, {  2,  2 }, { 2, 1 }, {   65,  115,    8 }}, // IE  pad = 2 1 | 2 1
  /* 233472:   29184:    8*/ {{  128,  228,    8 }, {  4,  2 }, {  2,  2 }, { 1, 1 }, {   64,  115,    8 }}, // IE  pad = 1 1 | 1 1
  /* 233472:   29184:    8*/ {{  128,  228,    8 }, {  4,  2 }, {  2,  2 }, { 0, 0 }, {   63,  114,    8 }}, // IE
  /* 233472:   29184:    8*/ {{  128,  228,    8 }, {  4,  2 }, {  1,  1 }, { 2, 1 }, {  129,  229,    8 }}, // IE  pad = 2 1 | 2 1
  /* 233472:   29184:    8*/ {{  128,  228,    8 }, {  4,  2 }, {  1,  1 }, { 1, 1 }, {  127,  229,    8 }}, // IE  pad = 1 1 | 1 1
  /* 233472:   29184:    8*/ {{  128,  228,    8 }, {  4,  2 }, {  1,  1 }, { 0, 0 }, {  125,  227,    8 }}, // IE
  /* 233472:   29184:    8*/ {{  128,  228,    8 }, {  2,  4 }, {  2,  2 }, { 1, 2 }, {   65,  115,    8 }}, // IE  pad = 1 2 | 1 2
  /* 233472:   29184:    8*/ {{  128,  228,    8 }, {  2,  4 }, {  2,  2 }, { 1, 1 }, {   65,  114,    8 }}, // IE  pad = 1 1 | 1 1
  /* 233472:   29184:    8*/ {{  128,  228,    8 }, {  2,  4 }, {  2,  2 }, { 0, 0 }, {   64,  113,    8 }}, // IE
  /* 233472:   29184:    8*/ {{  128,  228,    8 }, {  2,  4 }, {  1,  1 }, { 1, 2 }, {  129,  229,    8 }}, // IE  pad = 1 2 | 1 2
  /* 233472:   29184:    8*/ {{  128,  228,    8 }, {  2,  4 }, {  1,  1 }, { 1, 1 }, {  129,  227,    8 }}, // IE  pad = 1 1 | 1 1
  /* 233472:   29184:    8*/ {{  128,  228,    8 }, {  2,  4 }, {  1,  1 }, { 0, 0 }, {  127,  225,    8 }}, // IE
  /* 233472:   29184:    8*/ {{  128,  228,    8 }, {  2,  2 }, {  2,  2 }, { 1, 1 }, {   65,  115,    8 }}, // IE  pad = 1 1 | 1 1
  /* 233472:   29184:    8*/ {{  128,  228,    8 }, {  2,  2 }, {  2,  2 }, { 0, 0 }, {   64,  114,    8 }}, // IE
  /* 233472:   29184:    8*/ {{  128,  228,    8 }, {  2,  2 }, {  1,  1 }, { 1, 1 }, {  129,  229,    8 }}, // IE  pad = 1 1 | 1 1
  /* 233472:   29184:    8*/ {{  128,  228,    8 }, {  2,  2 }, {  1,  1 }, { 0, 0 }, {  127,  227,    8 }}, // IE
  /* 206976:     196: 1056*/ {{   14,   14, 1056 }, {  2,  2 }, {  2,  2 }, { 0, 0 }, {    7,    7, 1056 }}, // NET
  /* 200704:     784:  256*/ {{   28,   28,  256 }, {  2,  2 }, {  2,  2 }, { 0, 0 }, {   14,   14,  256 }}, // NET
  /* 175616:     196:  896*/ {{   14,   14,  896 }, {  2,  2 }, {  2,  2 }, { 0, 0 }, {    7,    7,  896 }}, // NET
  /* 125440:     196:  640*/ {{   14,   14,  640 }, {  2,  2 }, {  2,  2 }, { 0, 0 }, {    7,    7,  640 }}, // NET
  /* 100352:     196:  512*/ {{   14,   14,  512 }, {  2,  2 }, {  2,  2 }, { 0, 0 }, {    7,    7,  512 }}, // NET
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  4,  2 }, {  2,  2 }, { 2, 1 }, {   33,   17,   16 }}, // IE  pad = 2 1 | 2 1
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  4,  2 }, {  2,  2 }, { 1, 1 }, {   32,   17,   16 }}, // IE  pad = 1 1 | 1 1
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  4,  2 }, {  2,  2 }, { 0, 0 }, {   31,   16,   16 }}, // IE
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  4,  2 }, {  1,  1 }, { 2, 1 }, {   65,   33,   16 }}, // IE  pad = 2 1 | 2 1
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  4,  2 }, {  1,  1 }, { 1, 1 }, {   63,   33,   16 }}, // IE  pad = 1 1 | 1 1
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  4,  2 }, {  1,  1 }, { 0, 0 }, {   61,   31,   16 }}, // IE
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  2,  4 }, {  2,  2 }, { 1, 2 }, {   33,   17,   16 }}, // IE  pad = 1 2 | 1 2
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  2,  4 }, {  2,  2 }, { 1, 1 }, {   33,   16,   16 }}, // IE  pad = 1 1 | 1 1
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  2,  4 }, {  2,  2 }, { 0, 0 }, {   32,   15,   16 }}, // IE
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  2,  4 }, {  1,  1 }, { 1, 2 }, {   65,   33,   16 }}, // IE  pad = 1 2 | 1 2
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  2,  4 }, {  1,  1 }, { 1, 1 }, {   65,   31,   16 }}, // IE  pad = 1 1 | 1 1
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  2,  4 }, {  1,  1 }, { 0, 0 }, {   63,   29,   16 }}, // IE
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  2,  2 }, {  2,  2 }, { 1, 1 }, {   33,   17,   16 }}, // IE  pad = 1 1 | 1 1
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  2,  2 }, {  2,  2 }, { 0, 0 }, {   32,   16,   16 }}, // IE
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  2,  2 }, {  1,  1 }, { 1, 1 }, {   65,   33,   16 }}, // IE  pad = 1 1 | 1 1
  /*  32768:    2048:   16*/ {{   64,   32,   16 }, {  2,  2 }, {  1,  1 }, { 0, 0 }, {   63,   31,   16 }}, // IE
  /*    256:     256:    1*/ {{   16,   16,    1 }, {  4,  2 }, {  2,  2 }, { 2, 1 }, {    9,    9,    1 }}, // IE  pad = 2 1 | 2 1
  /*    256:     256:    1*/ {{   16,   16,    1 }, {  4,  2 }, {  2,  2 }, { 1, 1 }, {    8,    9,    1 }}, // IE  pad = 1 1 | 1 1
  /*    256:     256:    1*/ {{   16,   16,    1 }, {  4,  2 }, {  2,  2 }, { 0, 0 }, {    7,    8,    1 }}, // IE
  /*    256:     256:    1*/ {{   16,   16,    1 }, {  4,  2 }, {  1,  1 }, { 2, 1 }, {   17,   17,    1 }}, // IE  pad = 2 1 | 2 1
  /*    256:     256:    1*/ {{   16,   16,    1 }, {  4,  2 }, {  1,  1 }, { 1, 1 }, {   15,   17,    1 }}, // IE  pad = 1 1 | 1 1
  /*    256:     256:    1*/ {{   16,   16,    1 }, {  4,  2 }, {  1,  1 }, { 0, 0 }, {   13,   15,    1 }}, // IE
  /*    256:     256:    1*/ {{   16,   16,    1 }, {  2,  4 }, {  2,  2 }, { 1, 2 }, {    9,    9,    1 }}, // IE  pad = 1 2 | 1 2
  /*    256:     256:    1*/ {{   16,   16,    1 }, {  2,  4 }, {  2,  2 }, { 1, 1 }, {    9,    8,    1 }}, // IE  pad = 1 1 | 1 1
  /*    256:     256:    1*/ {{   16,   16,    1 }, {  2,  4 }, {  2,  2 }, { 0, 0 }, {    8,    7,    1 }}, // IE
  /*    256:     256:    1*/ {{   16,   16,    1 }, {  2,  4 }, {  1,  1 }, { 1, 2 }, {   17,   17,    1 }}, // IE  pad = 1 2 | 1 2
  /*    256:     256:    1*/ {{   16,   16,    1 }, {  2,  4 }, {  1,  1 }, { 1, 1 }, {   17,   15,    1 }}, // IE  pad = 1 1 | 1 1
  /*    256:     256:    1*/ {{   16,   16,    1 }, {  2,  4 }, {  1,  1 }, { 0, 0 }, {   15,   13,    1 }}, // IE
  /*    256:     256:    1*/ {{   16,   16,    1 }, {  2,  2 }, {  2,  2 }, { 1, 1 }, {    9,    9,    1 }}, // IE  pad = 1 1 | 1 1
  /*    256:     256:    1*/ {{   16,   16,    1 }, {  2,  2 }, {  2,  2 }, { 0, 0 }, {    8,    8,    1 }}, // IE
  /*    256:     256:    1*/ {{   16,   16,    1 }, {  2,  2 }, {  1,  1 }, { 1, 1 }, {   17,   17,    1 }}, // IE  pad = 1 1 | 1 1
  /*    256:     256:    1*/ {{   16,   16,    1 }, {  2,  2 }, {  1,  1 }, { 0, 0 }, {   15,   15,    1 }}, // IE
  /*     25:      25:    1*/ {{    5,    5,    1 }, {  2,  2 }, {  3,  2 }, { 1, 1 }, {    2,    3,    1 }}, // CVS-11517
#endif // ALL_TESTS_SET & DO_TEST_POOLING_MxN_ZERO

}; // const std::initializer_list<Test> tests_list

struct LayoutStrides
{
    t_MvTensorStorageOrder storageOrder;
    bool                   withInputStride;
    bool                   withOutputStride;
};

const std::initializer_list<LayoutStrides> layoutStrides_list
{

#if defined(DO_TEST_HWC_POOLING)
    { orderNYXZ, false, false }, // HWC
# if defined(ALL_STRIDES_SET)
    { orderNYXZ, false, true  }, // HWC
# endif // ALL_STRIDES_SET
#endif // DO_TEST_HWC_POOLING

#if defined(DO_TEST_CHW_POOLING)
    { orderNZYX, false, false }, // CHW
# if defined(ALL_STRIDES_SET)
    { orderNZYX, false, true  }, // CHW
    { orderNZYX, true,  false }, // CHW
    { orderNZYX, true,  true  }, // CHW
# endif // ALL_STRIDES_SET
#endif // DO_TEST_CHW_POOLING

};

class Tests: public TestSuite
{
public:
    explicit Tests()
        : m_testLoop(tests_list, "test")
        , m_layoutStridesLoop(layoutStrides_list, "ordstride")
#if defined(WITH_EXCLUDE_PAD)
        , m_excludePadLoop("excl")
#endif
        {}
    virtual ~Tests()
        {}
protected:
    const char* suiteName() const override
        { return ICV_TESTS_STRINGIFY(ICV_TEST_SUITE_NAME); }
    void userLoops() override
        {
            addLoop(m_testLoop);
            addLoop(m_layoutStridesLoop);
#if defined(WITH_EXCLUDE_PAD)
            addLoop(m_excludePadLoop);
#endif // WITH_EXCLUDE_PAD
        }
    void initData() override
        {
            const auto layoutStrides = m_layoutStridesLoop.value();

#if defined(WITH_EXCLUDE_PAD)
            excludePad = m_excludePadLoop.value() ? 1 : 0;
#endif // WITH_EXCLUDE_PAD

#if defined(ALL_STRIDES_SET)
            m_withInputStride = layoutStrides.withInputStride;
            m_withOutputStride = layoutStrides.withOutputStride;
#else // ALL_STRIDES_SET
            {
                static int rover = 0;

                m_withInputStride = (layoutStrides.storageOrder == orderNYXZ) ? false : bool((rover & 2) != 0);
                m_withOutputStride = bool((rover & 1) != 0);

            #if defined(DO_TEST_HWC_POOLING) && defined(DO_TEST_CHW_POOLING)
                const int step = (layoutStrides.storageOrder == orderNYXZ) ? 3 : 1;
                const int range = 5;
            #else // DO_TEST_HWC_POOLING && DO_TEST_CHW_POOLING
                const int step = 1;
                const int range = 4;
            #endif // DO_TEST_HWC_POOLING && DO_TEST_CHW_POOLING

                rover = (rover + step) % range;
            }
#endif // ALL_STRIDES_SET

            const auto& test = m_testLoop.value();

            if (test.idim.channels != test.odim.channels)
            {
                printf("\nFATAL: test data: input & output tensors' number of channels not equal: %d %d\n\n",
                       test.idim.channels, test.odim.channels);
                abortTests();
            }

            TensorDims dims3in(test.idim.width, test.idim.height, test.idim.channels, 1);
            TensorDims dims3out(test.odim.width, test.odim.height, test.odim.channels, 1);

            TensorAlign align3in((m_withInputStride ? 16 : 0), 0, 0, 0);
            TensorAlign align3out((m_withOutputStride ? 16 : 0), 0, 0, 0);
            TensorAlign align3ref(0, 0, 0, 0);

            int pad = mv::tensor::util::convPoolSizesRPadBySizeOutput(test.idim.width, test.odim.width,
                    test.ksize.width, test.kstride.x, test.kpad.x);
            pad = std::max(pad, 0);
            bool sizesCorrectness = mv::tensor::util::convPoolSizesCheck(test.idim.width, test.odim.width,
                    test.ksize.width, test.kstride.x, test.kpad.x, pad);
            mvTensorAssert(sizesCorrectness, "avg_pooling test: Horizontal sizes incorrect");

            pad = mv::tensor::util::convPoolSizesRPadBySizeOutput(test.idim.height, test.odim.height,
                    test.ksize.height, test.kstride.y, test.kpad.y);
            pad = std::max(pad, 0);
            sizesCorrectness = mv::tensor::util::convPoolSizesCheck(test.idim.height, test.odim.height,
                    test.ksize.height, test.kstride.y, test.kpad.y, pad);
            mvTensorAssert(sizesCorrectness, "avg_pooling test: Vertical sizes incorrect");

            m_inputTensor.init(layoutStrides.storageOrder, dims3in, align3in);
            m_outputTensor.init(layoutStrides.storageOrder, dims3out, align3out);
            m_referenceTensor.init(layoutStrides.storageOrder, dims3out, align3ref);

            allocBuffer(m_inputTensor);
            allocBuffer(m_outputTensor);
            allocBuffer(m_referenceTensor);
        }
    void formatTestParams(char* str, int maxLength) const override
        {
            const auto layoutStrides = m_layoutStridesLoop.value();

            const auto& test = m_testLoop.value();

            const auto& inDims = m_inputTensor.tensorDims();
            const auto& outDims = m_outputTensor.tensorDims();

            const char* layout_text = layoutString(layoutStrides.storageOrder);
            const char* stride_text = strideString(m_withInputStride, m_withOutputStride);
            const char* excludePad_text = excludePadString();

            snprintf_append(str, maxLength, "H W C = %u %u %u : %u %u %u, %s, %s, exclude-pad=%s (size %dx%d stride %d:%d pad %d:%d)",
                            inDims.height, inDims.width, inDims.channels,
                            outDims.height, outDims.width, outDims.channels,
                            layout_text, stride_text, excludePad_text, test.ksize.width, test.ksize.height,
                            test.kstride.x, test.kstride.y, test.kpad.x, test.kpad.y);
        }
    t_MvTensorOpType opType() const override
        { return kPool; }
    void initParserRunner()  override
        {
            const auto& test = m_testLoop.value();

            initMyriadResources();
            initDebugInfo();

            Pooling* poolingOp = static_cast<Pooling*>(m_op);

            m_inputTensor.exportToBuffer(poolingOp->input);
            m_outputTensor.exportToBuffer(poolingOp->output);

            poolingOp->radixX       = test.ksize.width;
            poolingOp->radixY       = test.ksize.height;
            poolingOp->radixStrideX = test.kstride.x;
            poolingOp->radixStrideY = test.kstride.y;
            poolingOp->padX         = test.kpad.x;
            poolingOp->padY         = test.kpad.y;
            poolingOp->pool_method = "avg";

#if defined(WITH_EXCLUDE_PAD)
            poolingOp->ops.excludePad = excludePad;
#else // WITH_EXCLUDE_PAD
            poolingOp->ops.excludePad = 0;
#endif // WITH_EXCLUDE_PAD
        }
    void generateData() override
        {
            // input
#if defined(GENERATE_SEQ_DATA)
            int val = 0;
#endif
            printf("m_inputTensor.forEach+\n");
            m_inputTensor.forEach(false, [&](const MemoryDims& indices)
            {
#if defined(GENERATE_SEQ_DATA)
                float tmp = float(val++);
#else
                const TensorDims ti = m_inputTensor.toTensor(indices);
                float tmp = float( 1 + ((ti.height * 61 + ti.width * 17 + ti.channels * 3) % 19) ) / 17.0f;
#endif
                printf("h=%d, w=%d, c=%d \n", ti.height, ti.width, ti.channels);
                m_inputTensor.at(indices) = f32Tof16(tmp);
            });

            // reference output
            printf("calcReferenceOutput+\n");
            calcReferenceOutput();
        }
    void calcReferenceOutput()
        {
            const auto& test = m_testLoop.value();
            const auto& inDims = m_inputTensor.tensorDims();
            const auto& refDims = m_referenceTensor.tensorDims();

            mvTensorAssert(refDims.channels == inDims.channels, "refDims.channels == inDims.channels");

            const float kernelArea = float(test.ksize.height * test.ksize.width);

#if defined(WITH_EXCLUDE_PAD)
            const bool zeroFill = bool( ((test.kpad.x != 0) || (test.kpad.y != 0)) && (excludePad == 0) );
#else // WITH_EXCLUDE_PAD
            const bool zeroFill = bool((test.kpad.x != 0) || (test.kpad.y != 0));
#endif // WITH_EXCLUDE_PAD

            m_referenceTensor.forEach(false, [&](const MemoryDims& indices)
            {
                const TensorDims ti = m_referenceTensor.toTensor(indices);
                float out_ref = 0.0f;
                int count = 0;
                for (int kh = 0; kh < test.ksize.height; ++kh)
                {
                    for (int kw = 0; kw < test.ksize.width; ++kw)
                    {
                        int32_t iw = ti.width * test.kstride.x - test.kpad.x + kw;
                        int32_t ih = ti.height * test.kstride.y - test.kpad.y + kh;
                        if (iw < 0 || iw >= inDims.width || ih < 0 || ih >= inDims.height) continue;

                        fp16 dh = m_inputTensor.at(TensorDims(iw, ih, ti.channels, ti.batch));
                        out_ref = f16Tof32(f32Tof16(out_ref + f16Tof32(dh)));
                        ++count;
                    }
                }

                mvTensorAssert(count > 0, "invalid kernel parameters");

                out_ref /= zeroFill ? kernelArea : float(count);
                m_referenceTensor.at(indices) = f32Tof16(out_ref);
            });
        }
    void resetOutputData() override
        { resetTensorBuffer(m_outputTensor); }
    bool checkResult() override
        {
            m_outputTensor.confirmBufferData();

            // save output data
            if (save_to_file)
            {
                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(), "outMyriad.bin");
            }

            const auto& test = m_testLoop.value();
            const float kernelArea = float(test.ksize.height * test.ksize.width);

            const auto& outDims = m_outputTensor.tensorDims();
            int wmin = outDims.width, wmax = -1;
            int hmin = outDims.height, hmax = -1;
            int cmin = outDims.channels, cmax = -1;

            float max_abs_diff = 0;
            bool nan_abs_diff = false;

            bool threshold_test_failed = false;

            m_outputTensor.forEach(false, [&](const MemoryDims& indices)
            {
                float value = f16Tof32(m_outputTensor.at(indices));
                float gt_value = f16Tof32(m_referenceTensor.at(indices));
                float abs_diff = fabs(value - gt_value);
                bool differ = bool(!(abs_diff <= test_threshold*kernelArea));

                bool diff_is_nan = bool(abs_diff != abs_diff);
                nan_abs_diff |= diff_is_nan;
                max_abs_diff = std::max(max_abs_diff, abs_diff);

                threshold_test_failed |= differ;

                const TensorDims ti = m_outputTensor.toTensor(indices);

                if (differ && GlobalData::doPrintDiffs)
                    printf("DIFF HWC [%d:%d:%d] %f %f %f\n", ti.height, ti.width, ti.channels, value, gt_value, abs_diff);

                if (differ && GlobalData::doPrintDiffRange)
                {
                    wmin = std::min(wmin, ti.width); wmax = std::max(wmax, ti.width);
                    hmin = std::min(hmin, ti.height); hmax = std::max(hmax, ti.height);
                    cmin = std::min(cmin, ti.channels); cmax = std::max(cmax, ti.channels);
                }
            });

            if (threshold_test_failed && GlobalData::doPrintDiffRange)
                printf("DIFF RANGE HWC: %d:%d %d:%d %d:%d\n", hmin, hmax, wmin, wmax, cmin, cmax);

            if (GlobalData::doPrintDiffMax)
                printf("DIFF MAX ABS = %f / %f%s\n", max_abs_diff, max_abs_diff/kernelArea, (nan_abs_diff ? " : NaN detected" : ""));

            return !threshold_test_failed;
        }
    static const char* excludePadString()
        {
#if defined(WITH_EXCLUDE_PAD)
            if (excludePad != 0)
                return "yes";
            else
#endif // WITH_EXCLUDE_PAD
                return "no";
        }
protected:
    ListIterator<Test> m_testLoop;
    ListIterator<LayoutStrides> m_layoutStridesLoop;
#if defined(WITH_EXCLUDE_PAD)
    FlipIterator m_excludePadLoop;
#endif
    Tensor<fp16> m_inputTensor;
    Tensor<fp16> m_outputTensor;
    Tensor<fp16> m_referenceTensor;
    bool m_withInputStride;
    bool m_withOutputStride;
};

ICV_TESTS_REGISTER_SUITE(Tests)

} // namespace ICV_TESTS_NAMESPACE(ICV_TEST_SUITE_NAME)
