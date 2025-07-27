# Native compilation settings
CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall

# Cross-compilation settings
MINGW_PREFIX = x86_64-w64-mingw32
CROSS_CXX = $(MINGW_PREFIX)-g++
CROSS_CXXFLAGS = -std=c++11 -O3 -Wall -static-libgcc -static-libstdc++

# Paths for Windows dependencies
WIN_DEPS_DIR = ./windows-deps
WIN_INCLUDE_DIR = $(WIN_DEPS_DIR)/include
WIN_LIB_DIR = $(WIN_DEPS_DIR)/lib

# OpenCL library paths
OPENCL_LIBS = -lOpenCL
WIN_OPENCL_LIBS = -L$(WIN_LIB_DIR) -lOpenCL

# For macOS
ifeq ($(shell uname), Darwin)
    OPENCL_LIBS = -framework OpenCL
endif

# BOINC support (uncomment if needed)
# CXXFLAGS += -DBOINC
# CROSS_CXXFLAGS += -DBOINC
# BOINC_LIBS = -lboinc_api -lboinc
# WIN_BOINC_LIBS = -L$(WIN_LIB_DIR) -lboinc_api -lboinc

TARGET = seed_search
WIN_TARGET = seed_search.exe
SOURCES = main.cpp

.PHONY: all clean windows native

all: native

native: $(TARGET)

windows: $(WIN_TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(OPENCL_LIBS) $(BOINC_LIBS)

$(WIN_TARGET): $(SOURCES)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -I$(WIN_INCLUDE_DIR) -o $(WIN_TARGET) $(SOURCES) $(WIN_OPENCL_LIBS) $(WIN_BOINC_LIBS)

clean:
	rm -f $(TARGET) $(WIN_TARGET)