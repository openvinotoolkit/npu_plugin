** Post training quantization script for ICV networks **

Additional notes to use Post training quantization script for quantizing of ICV networks
* Learn [common part of instruction](scripts/post_training_quantization/README.md)
* Additional repository is necessary:
    - [annotation_converters_ext](https://gitlab-icv.inn.intel.com/algo/annotation_converters_ext)
* Additional environment variable:
  - export ANNOTATION_CONVERTERS_EXT=<path to annotation convertors extension for accuracy checker> or
  - export DEFINITIONS_FILE==<path to annotation convertors extension definition file>
