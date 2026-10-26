MPICC ?= mpicc
CFLAGS ?= -O2 -fPIC -Wall

all: libcagepmpi.so selftest

libcagepmpi.so: cage_pmpi.c
	$(MPICC) $(CFLAGS) -shared -o $@ $<

selftest: selftest.c
	$(MPICC) -O2 -o $@ $<

# Runs the self-test under the shim and checks the parsed trace with pytest.
test: all
	rm -rf selftest_out
	CAGE_OUT=selftest_out LD_PRELOAD=$(CURDIR)/libcagepmpi.so mpirun -np 4 ./selftest
	cd ../.. && CAGE_PMPI_SELFTEST=tools/pmpi_shim/selftest_out python -m pytest tests/test_pmpi_selftest.py -v

clean:
	rm -rf libcagepmpi.so selftest selftest_out

.PHONY: all test clean
