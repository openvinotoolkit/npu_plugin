Requirements:
 - swig3.0
 - g++ >= 4.7
 - python3
 - pip3
    - xmlrunner
 - numpy

Build the general mcmCompiler before running anything in this folder.

This Python3 API's primary aim is to facilitate transformation from the old python-based compiler to this mcmCompiler. Therefore the functionality has been limited to these requirements.

To run, just call make.
If you wish to run the python outside of the make file, ensure your LD_LIBRARY_PATH is correctly pointing to the folder for libcm.so
