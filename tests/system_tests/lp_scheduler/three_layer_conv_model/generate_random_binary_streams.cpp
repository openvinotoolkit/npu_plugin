#include<cassert>
#include<cstdio>
#include<stdlib.h>
#include<unistd.h>

#include<string>

#define DEFAULT_RAND_SEED 19875636

struct InputParams {

  InputParams() : output_file_prefix_("input-"), stream_count_(1UL),
    rand_seed_(DEFAULT_RAND_SEED) {}

  bool parse_args(int argc, char **argv) {
    int opt;
    char const * const options = "o:n:r:";

    while ( (opt = getopt(argc, argv, options)) != -1 ) {
      switch (opt) {
        case 'o':
          output_file_prefix_ = optarg;
          break;

        case 'r':
          {
            unsigned int seed;
            if (sscanf(optarg, "%u", &seed) !=1) {
              rand_seed_ = DEFAULT_RAND_SEED;
              fprintf(stderr, "WARNING: unable to scan random see defaulting to"
                  " %u\n", rand_seed_);
            } else {
              rand_seed_ = seed;
            }
          }
          break;

        case 'n':
          {
            size_t stream_count;
            if (sscanf(optarg, "%lu", &stream_count) != 1) {
              fprintf(stderr, "WARNING: unable to scan random see defaulting to"
                  " %u\n", rand_seed_);
            } else {
              stream_count_ = stream_count;
            }
          }
          break;

        default:
          help();
          return false;
      }
    }
    return true;
  }

  void help() const {
    fprintf(stderr, "./generate_random_binary_streams \n"
                    "[-o] specify output file prefix (default \"input-*.bin\""
                    "\n"
                    "[-n] specify number of binary streams (default 1)\n"
                    "[-r] specify random seed (default %u)\n"
                    "[-h] print this message\n", DEFAULT_RAND_SEED);
  }

  std::string output_file_prefix_;
  size_t stream_count_;
  unsigned int rand_seed_;
  static size_t current_stream_count_;

}; // struct InputParams //
size_t InputParams::current_stream_count_ = 0UL;


size_t GetTensorSize(FILE *fptr) {
  size_t ret = 1UL, scan;
  char c = 'x';

  while ((c == 'x') && (fscanf(fptr, "%lu", &scan) == 1)) {
    ret*=scan;
    c = EOF;
    while (((c = fgetc(fptr)) != EOF) && (c != 'x')) {}
  }
  return ret;
}


int main(int argc, char **argv) {
  InputParams params;

  if (!params.parse_args(argc, argv)) { return -1; }

  printf("Enter tensor input dimensions (e.g. 1x56x56x3) and CTRL-D:\n");
  size_t len = GetTensorSize(stdin);

  printf("Generating a %lu random streams of %lu bytes into file \n",
        params.stream_count_, len);
  srand(params.rand_seed_);

  for (size_t s=0; s<params.stream_count_; ++s) {
    char buf[4096];
    int ret = snprintf(buf, 4096, "%s%lu.bin",
        params.output_file_prefix_.c_str(), (params.current_stream_count_)++); 
    assert((ret > 0) && (ret <= 4096));
    FILE *out = fopen(buf, "w");
    assert(out);

    int fd = fileno(out);
    for (size_t i=0; i<len; i++) {
      char b = rand()%8;
      write(fd, &b, 1UL);
    }
    fclose(out);
  }
}
