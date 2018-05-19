# Matrix-Vector Multiplication
# Author: Jimmy Nguyen
# Email: jimmy@jimmyworks.net
#
# Makefile
#
# Description:
# This C program performs Matrix-Vector Multiplication of a M x N matrix
# using up to 32 nodes using MPI and OpenMP parallelization for performance
# optimization.

# Set up info for C implicit rule
CC = mpicc
CCFLAGS = -lm -fopenmp
CPPFLAGS =
LDFLAGS =
PROJECTNAME = Matrix_Vector_Multiplication
SRCS = matrix_vector_rowwise.c
EXEC = matrix_vector_rowwise

MPI = mpirun
MPIFLAGS = -n 6
ARGS =

# For Testing

N1  = -n 1
N2  = -n 2
N3  = -n 3
N4  = -n 4
N5  = -n 5
N6  = -n 6
N7  = -n 7
N8  = -n 8
N9  = -n 9
N10 = -n 10

M1  = 100
M2  = 1000
M3  = 5000
M4  = 10000
M5  = 20000
M6  = 30000

# ======================================================
# ======================================================

OBJS = $(SRCS:c=o)

all: $(EXEC) $(EXEO3)

test: all all_tests

debug:
	clear
	$(MPI) $(N4) ./$(EXEC) 4 4 DEBUG

clean:
	rm -f $(OBJS) *.d *~  $(EXEC)

Makefile: $(SRCS:.c=.d)

# Pattern for .d files.
%.d:%.c
	@echo Updating .d Dependency File
	@set -e; rm -f $@; \
	$(CC) -MM $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$



#  This is a rule to link the files.  Pretty standard
$(EXEC) : $(OBJS)
	$(CC) -o $(EXEC) $(CCFLAGS) $(OBJS)

# Backup Target
backup: clean
	@mkdir -p ~/backups; chmod 700 ~/backups
	@$(eval CURDIRNAME := $(shell basename `pwd`))
	@$(eval MKBKUPNAME := ~/backups/$(PROJECTNAME)-$(shell date +'%Y.%m.%d-%H:%M:%S').tar.gz)
	@echo
	@echo Writing Backup file to: $(MKBKUPNAME)
	@echo
	@-tar zcfv $(MKBKUPNAME) ../$(CURDIRNAME)
	@chmod 600 $(MKBKUPNAME)
	@echo
	@echo Done!


# Include the dependency files
-include $(SRCS:.c=.d)

all_tests:
	$(MPI) $(N1) ./$(EXEC) $(M1) $(M1)
	$(MPI) $(N1) ./$(EXEC) $(M2) $(M2)
	$(MPI) $(N1) ./$(EXEC) $(M3) $(M3)
	$(MPI) $(N1) ./$(EXEC) $(M4) $(M4)
	$(MPI) $(N1) ./$(EXEC) $(M5) $(M5)
	$(MPI) $(N1) ./$(EXEC) $(M6) $(M6)

proc_tests:
	$(MPI) $(N10) ./$(EXEC) 30000 16000
	$(MPI) $(N9) ./$(EXEC) 30000 16000
	$(MPI) $(N8) ./$(EXEC) 30000 16000
	$(MPI) $(N7) ./$(EXEC) 30000 16000
	$(MPI) $(N6) ./$(EXEC) 30000 16000
	$(MPI) $(N5) ./$(EXEC) 30000 16000
	$(MPI) $(N4) ./$(EXEC) 30000 16000
	$(MPI) $(N3) ./$(EXEC) 30000 16000
	$(MPI) $(N2) ./$(EXEC) 30000 16000
	$(MPI) $(N1) ./$(EXEC) 30000 16000
