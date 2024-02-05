# Make minimum chips

CXX=g++

RM=rm -f

FLAGS=-Og -Wall -fopenmp -DMC_OMP
DEFINES=-DMC_TEST_SIZE=100000
LDFLAGS=-lm -lboost_unit_test_framework

COVFLAGS=--coverage -Og -Wall -Wno-unknown-pragmas -gno-inline-points -fno-inline
COVDEFINES=-DMC_TEST_SIZE=3
COVLDFLAGS=-lgcov -lm -lboost_unit_test_framework

hh := $(wildcard *.hh)
tt := $(wildcard *.tt)

all: serial

serial: mc.x test.pack.x coverage.pack.x

mc.x: mc.cc $(hh) $(tt)
	$(CXX) $(FLAGS) $(DEFINES) $< $(LDFLAGS) -o $@

test.pack.x: test.pack.cc $(hh) $(tt)
	$(CXX) $(FLAGS) $(DEFINES) $< $(LDFLAGS) -o $@

coverage.pack.x: test.pack.cc $(hh) $(tt)
	$(CXX) $(COVFLAGS) $(COVDEFINES) $< $(COVLDFLAGS) -o $@

clean:
	$(RM) *.x *.gcda *.gcno *err *out *.memcheck *.tsv
