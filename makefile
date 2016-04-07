EXEC   = lucy

OPTIMIZE =  -O2  


OBJS   = main.o lucy.o cuda_convolve.o global.o rng.o

CXX    = /usr/bin/g++
CC     = /usr/bin/gcc
NVCC   = nvcc

INCL   = -I./ -I/usr/local/cuda/include

NVLIBS = -L/usr/local/cuda/lib -lcuda -lcudart -lcufft
LIBS   = -lm -lgsl

PRECISION = -DPRECISION=1


.SUFFIXES : .c .cpp. .cu .o

FLAGS = $(PRECISION) -DCUDA
CFLAGS = $(OPTIMIZE) -m64  $(FLAGS)
CXXFLAGS = $(OPTIMIZE) -m64 $(FLAGS)
NVCCFLAGS = $(FLAGS) -m64 -arch=sm_12 --compiler-bindir=/usr/bin/
LDFLAGS = -m64

%.o:  %.c
	$(CC) $(CFLAGS) $(INCL) -c $< -o $@ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS)  $(INCL) -c $< -o $@ $(LIBS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS)  $(INCL)  -c $< -o $@ $(NVLIBS)


$(EXEC): $(OBJS) 
	 $(CXX) $(CXXFLAGS) $(OBJS) $(LIBS) $(NVLIBS) -o $(EXEC) $(INCL)

#$(OBJS): $(INCL) 

.PHONY : clean

clean:
	 rm -f $(OBJS) $(EXEC)

