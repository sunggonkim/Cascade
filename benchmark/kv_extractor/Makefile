# KV Cache Aggregator Build
#
# For Perlmutter:
#   module load PrgEnv-gnu cray-mpich
#   make

CXX = CC
# Fallback for non-Cray systems
ifeq ($(CXX),)
    CXX = mpicxx
endif

CXXFLAGS = -O3 -fopenmp -std=c++17 -Wall
LDFLAGS = -lcrypto

# Check for Cray compiler wrapper
ifeq ($(shell which CC 2>/dev/null),)
    CXX = mpicxx
endif

TARGET = kv_aggregator

.PHONY: all clean test

all: $(TARGET)

$(TARGET): kv_aggregator.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	mkdir -p /tmp/kv_test_$$$$
	./$(TARGET) --output /tmp/kv_test_$$$$ --sessions 10 --blocks 2
	ls -la /tmp/kv_test_$$$$/
	rm -rf /tmp/kv_test_$$$$
