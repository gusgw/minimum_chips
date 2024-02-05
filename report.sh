#! /bin/bash

# Do not try threaded operation when checking
# coverage and leaks in serial execution
export OMP_NUM_THREADS=1

# Get debug symbols from a server.
# See https://wiki.archlinux.org/title/Debuginfod for information.
# valgrind needs this.
export DEBUGINFOD_URLS="https://debuginfod.elfutils.org/"

# Check coverage after coverage_pack.x is built
nice -n 19 ./coverage.pack.x 1> coverage.pack.out 2> coverage.pack.err
ln -s coverage.pack.x-test.pack.gcda test.pack.gcda
ln -s coverage.pack.x-test.pack.gcno test.pack.gcno
rm -rf coverage/
mkdir coverage/
gcovr -r . --exclude-unreachable-branches \
		   --exclude-throw-branches |\
		   grep -v test.pack.cc \
		   > coverage/coverage.pack.txt
gcovr -r . --branches \
		   --exclude-unreachable-branches \
		   --exclude-throw-branches |\
		   grep -v test.pack.cc \
		   > coverage/coverage.branches.pack.txt
gcovr -r . --exclude-unreachable-branches \
		   --exclude-throw-branches  \
		   --html-details coverage/coverage.pack.html
rm test.pack.gcda
rm test.pack.gcno
rsync -av --progress coverage/ ${HOME}/gaol/coverage/

# Run some checks on the main program and unit tests
valgrind --leak-check=full \
		 --show-error-list=yes \
		 --show-reachable=yes \
		 --leak-check-heuristics=all \
		 --num-callers=24 \
		 ./mc.x 1000 \
		 1> mc.out 2> mc.memcheck
