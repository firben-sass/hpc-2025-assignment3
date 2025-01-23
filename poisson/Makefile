# Makefile
#
TARGET    = bin/poisson # Poisson

# Object files
OBJECTS   = bin/main.o bin/print.o bin/alloc3d.o bin/define_u_f.o bin/poisson.o poisson_dalloc.o

# Compiler and flags
CC        = nvc
CXX       = nvc++
DEFS      =
OPT       = -g -fast -Msafeptr -Minfo -acc -mp=gpu -gpu=mem:separate:pinnedalloc -gpu=cc90 -gpu=lineinfo -cuda -mp=noautopar
CFLAGS    = $(DEFS) $(OPT)
LDFLAGS   = -lm


all: bin $(TARGET)

bin:
	@mkdir -p bin

$(TARGET): $(OBJECTS)
	$(CXX) -o $@ $(CFLAGS) $(OBJECTS) $(LDFLAGS)

bin/%.o: %.cpp
	$(CXX) -o $@ $(CFLAGS) -c $<

clean:
	@/bin/rm -f core bin/*.o bin/poisson *~

clean-out:
	@/bin/rm -f *.out

realclean: clean
	@rmdir bin || true

# Dependency information
bin/main.o: main.cpp print.h poisson.h define_u_f.h poisson_dalloc.h
bin/print.o: print.cpp print.h
bin/alloc3d.o: alloc3d.cpp alloc3d.h
bin/define_u_f.o: define_u_f.cpp define_u_f.h
