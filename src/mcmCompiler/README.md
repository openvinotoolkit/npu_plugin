# mcmCompiler
Movidius C++ Computational Model Compiler

#### Python CompositionAPI requirements
- numpy 1.16.4
- tensorflow 1.13.1

## Building
```
git clone --recursive https://github.com/movidius/mcmCompiler.git
cd mcmCompiler
git submodule update --init
mkdir build && cd build && cmake ..
make -j8
```

## Troubleshooting

#### Thrown `OverflowError: in method 'conv2D', argument 6 of type 'unsigned short'` in the Python CompositionAPI bridge

Invalid numpy version, must be 1.16.4
