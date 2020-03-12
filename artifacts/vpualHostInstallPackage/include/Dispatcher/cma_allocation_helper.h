#ifndef __CMA_ALLOCATION_HELPER_H__
#define __CMA_ALLOCATION_HELPER_H__

#include <string>

/** CMA allocation helper class. */
class CmaData {
  public:
    int            fd;          					/*< File descriptor. */
    unsigned char* buf;         					/*< Buffer for use in virtual address space. */
    unsigned long  phys_addr;   					/*< Physical address of allocation. */
    unsigned int   size;        					/*< Size of allocation. */

    int Create(std::string);    					/*< Open and map the allocation files. */
    int Create(size_t size);
    CmaData()
        : fd { -1 }
        , buf { nullptr }
        , phys_addr { 0 }
        , size { 0 }
        , b_use_vpusmm { false }
        {}
    ~CmaData();                 					/*< Close the file if opened. */

    // Delete copy constructor and assignment operator.
    CmaData(const CmaData&) = delete;
    CmaData& operator=(const CmaData&) = delete;
private:
	bool           b_use_vpusmm;

};

#endif // __CMA_ALLOCATION_HELPER_H__
