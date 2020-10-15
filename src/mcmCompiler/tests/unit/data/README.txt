conv_01.py   1 2D-convolution
   input: N=1 h=32 w=32 c=1
   kernel =3x3 , stride=x:4, y:4
   bias = 0.0

conv_02.py   1 2D-convolution
   input: N=1 h=32 w=32 c=3
   kernel =3x3 , stride=x:4, y:4
   bias = 0.0

conv_03.py   1 2D-convolution
   input: N=1 h=256 w=256 c=3
   kernel =3x3 , stride=2, y:2
   bias = 0.0

conv_04.py   1 2D-convolution
   input: N=1 h=256 w=256 c=3
   kernel =5x5 , stride=2, y:2
   bias = 0.0

conv_05.py   2 2D-convolutions
   input: N=1 h=256 w=256 c=3
   kernel =3x3 , stride=1,1
   kernel =3x3 , stride=1,1
   bias = 0.0

conv_06.py   conv1->maxpool->conv2->maxpool2
   input: N=1 h=64 w=64 c=3
   t1: [1,30,30,1] pad x=2,y=2 
   t2: [1,10,10,1] pad x=0,y=0 
   t3: [1,8,8,1]   pad x=2,y=2 
   output [1,4,4,1] 

