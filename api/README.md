Requirements:
 - swig3.0
 - g++ >= 4.7
 - python3 (note: only 3.6 currently supported)
 - numpy

To add the shared .so for use, after running make, ensure your LD_LIBRARY_PATH enviroment variable is pointing to the correct folder. 

This Python3 API's primary aim is to facilitate transformation from the old python-based compiler to this mcmCompiler. Therefore the functionality has been limited to these requirements.
