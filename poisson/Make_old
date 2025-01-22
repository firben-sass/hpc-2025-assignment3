# Makefile
#
TARGET_J  = bin/poisson_j        # Jacobi
TARGET_GS = bin/poisson_gs       # Gauss-Seidel
# COMPARE   = bin/compare_outputs  # Comparison tool

SOURCES = main.c print.c alloc3d.c
OBJECTS = bin/print.o bin/alloc3d.o 
MAIN_J  = bin/main_j.o
MAIN_GS = bin/main_gs.o
OBJS_J  = $(MAIN_J) bin/jacobi.o
OBJS_GS = $(MAIN_GS) bin/gauss_seidel.o
# TEST_SRC = compare_outputs.c

# options and settings for the GCC compilers
#
CC      = gcc
DEFS    =
OPT     = -g -fopenmp
IPO     =
ISA     =
CHIP    =
ARCH    =
PARA    =
CFLAGS  = $(DEFS) $(ARCH) $(OPT) $(ISA) $(CHIP) $(IPO) $(PARA) $(XOPTS)
LDFLAGS = -lm

# Ensure bin directory exists before compiling
# all: bin $(TARGET_J) $(TARGET_GS) $(COMPARE)

# bin:
# 	@mkdir -p bin

$(TARGET_J): $(OBJECTS) $(OBJS_J)
	$(CC) -o $@ $(CFLAGS) $(OBJS_J) $(OBJECTS) $(LDFLAGS)

$(TARGET_GS): $(OBJECTS) $(OBJS_GS)
	$(CC) -o $@ $(CFLAGS) $(OBJS_GS) $(OBJECTS) $(LDFLAGS)

bin/main_j.o: main.c
	$(CC) -o $@ -D_JACOBI $(CFLAGS) -c main.c

bin/main_gs.o: main.c
	$(CC) -o $@ -D_GAUSS_SEIDEL $(CFLAGS) -c main.c

bin/%.o: %.c
	$(CC) -o $@ $(CFLAGS) -c $<

$(COMPARE): $(TEST_SRC)
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(LDFLAGS)

# test: $(TARGET_J) $(TARGET_GS) $(COMPARE)
# 	@echo "Running Jacobi binary..."
# 	./$(TARGET_J) > /dev/null
# 	@echo "Jacobi output written to outputs/output_j.bin."
# 	@echo "Running Gauss-Seidel binary..."
# 	./$(TARGET_GS) > /dev/null
# 	@echo "Gauss-Seidel output written to outputs/output_gs.bin."
# 	@echo "Comparing outputs..."
# 	./$(COMPARE) outputs/poisson_gs_5.bin outputs/poisson_j_5.bin

clean:
	@/bin/rm -f core bin/*.o *~

# @/bin/rm -f $(TARGET_J) $(TARGET_GS) $(COMPARE)

realclean: clean
	@rmdir bin || true

# DO NOT DELETE

bin/main_j.o: main.c print.h jacobi.h
bin/main_gs.o: main.c print.h gauss_seidel.h
bin/print.o: print.h