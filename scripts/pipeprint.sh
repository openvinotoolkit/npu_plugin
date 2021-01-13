#!/bin/bash
cat << EOF_PIPE_PRINT_SRC | gcc -xc -o pipeprint - && chmod +x pipeprint && ./pipeprint $1
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/time.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t u8;

#define CONFIGURED_PIPEPRINT_BUFFER_SIZE_MAX (1024*1024)

struct tyMvConsoleQueue {
	volatile u32  canaryStart;
	volatile u32  in;
	volatile u32  out;
	volatile u32  queueSize;
	volatile u32  canaryEnd;
	volatile u8   buffer[CONFIGURED_PIPEPRINT_BUFFER_SIZE_MAX];
};

#define VPU_CACHE_LINE_SIZE 64L
#define ALIGN_DOWN(a) ((a)&(~(VPU_CACHE_LINE_SIZE-1)))

#define PAGE_SIZE (4096)

int main(int argc, const char * argv[])
{
	struct timespec ts;
	int fd;
	struct tyMvConsoleQueue * p;
	u64	phy_addr = 0x94500000;
	u32 in, next_in;
	u32 cnt, cnt_caligned;
	u32 no_data_ticks;
	u32 buffer_base;
	u32 queueSize;
	u32 mapsize;

	if (argc > 1)
		phy_addr = strtoll(argv[1], NULL, 0);

	if (phy_addr & (PAGE_SIZE-1)) {
		printf("phy_addr 0x%x is not page-aligned!\n", phy_addr);
		return 1;
	}

	fd = open("/dev/mem", O_RDONLY | O_SYNC);
	if (fd < 0) {
		printf("/dev/mem Open failed\n");
		return 1;
	}

	mapsize = (sizeof(struct tyMvConsoleQueue) + PAGE_SIZE - 1) / PAGE_SIZE * PAGE_SIZE;
	p = mmap(NULL, mapsize, PROT_READ, MAP_SHARED, fd, phy_addr);
	if(p == MAP_FAILED){
		printf("mmap(offset=0x%x ... ) failed\n", phy_addr);
		close(fd);
		return 1;
	}

	// 1ms sleep when no logs are presented.
	ts.tv_sec = 0;
	ts.tv_nsec = 1 * 1000000;

	buffer_base = (u32)(((struct tyMvConsoleQueue*)(phy_addr))->buffer);
	in = 0;
	while(1) {
		queueSize = p->queueSize;
		next_in = p->in;
		cnt = (next_in - in) % queueSize;

		if (cnt > 0) {
			//printf("queueSize=%d, next_in=%d, in=%d, cnt=%d\n",queueSize, next_in, in, cnt);fflush(stdout);

			// only 64bit aligned part is flushed from cache to RAM in time
			// the rest part will be flushed later by subsequent logs or forcely with timeout
			cnt_caligned = (ALIGN_DOWN(buffer_base + next_in) - (buffer_base + in)) % queueSize;

			if (cnt_caligned)
				cnt = cnt_caligned;
			else if (no_data_ticks < 10000)
				cnt = 0;

			if (cnt) {
				write(1, (p->buffer) + in, cnt);
				in = (in + cnt) % queueSize;
				no_data_ticks = 0;
				continue;
			}
		}
		nanosleep(&ts, &ts);
		no_data_ticks ++;
	}

	munmap(p, mapsize);
	close(fd);

	return 0;
}
EOF_PIPE_PRINT_SRC
