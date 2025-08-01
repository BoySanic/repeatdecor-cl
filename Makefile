# Native compilation settings
CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall

# Cross-compilation settings
MINGW_PREFIX = x86_64-w64-mingw32
CROSS_CXX = $(MINGW_PREFIX)-g++
CROSS_CXXFLAGS = -std=c++11 -O3 -Wall -static-libgcc -static-libstdc++ 

# Paths for Windows dependencies
WIN_LIBS_DIR = ./lib/opencl/win

# OpenCL library paths
OPENCL_LIBS = -lOpenCL
WIN_OPENCL_LIBS = -L$(WIN_LIBS_DIR) -lOpenCL

# BOINC library paths
BOINC_LIBS = ./lib/boinc/
BOINC_WIN = ./lib/boinc/win
BOINC_LIN = ./lib/boinc/lin

INCLUDE_DIR = ./include
BOINC_INCLUDE = $(INCLUDE_DIR)/boinc
BOINC_INCLUDE_WIN = $(INCLUDE_DIR)/boinc/win

# For macOS
ifeq ($(shell uname), Darwin)
    OPENCL_LIBS = -framework OpenCL
endif


LIN_TARGET = repeatdecor-cl
LIN_BOINC_TARGET = repeatdecor-cl_boinc
WIN_TARGET = repeatdecor-cl.exe
WIN_BOINC_TARGET = repeatdecor-cl_boinc.exe
MAC_TARGET = repeatdecor-cl
MAC_BOINC_TARGET = repeatdecor-cl_boinc
SOURCES = main.cpp

.PHONY: all clean windows native boinc linux boinc_win boinc_lin

all: linux windows boinc_win boinc_lin
windows: $(WIN_TARGET) $(WIN_BOINC_TARGET)
native: $(LIN_TARGET) $(LIN_BOINC_TARGET)
mac: $(MAC_TARGET)
linux: $(LIN_TARGET)

windows: $(WIN_TARGET)

boinc_win: $(WIN_BOINC_TARGET)

boinc_lin: $(LIN_BOINC_TARGET)

$(LIN_TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(LIN_TARGET) $(SOURCES) $(OPENCL_LIBS)

$(WIN_TARGET): $(SOURCES)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -I$(INCLUDE_DIR) -o $(WIN_TARGET) $(SOURCES) $(WIN_OPENCL_LIBS)

$(WIN_BOINC_TARGET): $(SOURCES)
	$(CROSS_CXX) $(CROSS_CXXFLAGS) -I$(INCLUDE_DIR) -I$(BOINC_INCLUDE_WIN) -L$(BOINC_WIN) -o $(WIN_BOINC_TARGET) $(SOURCES) -D_WIN32 -DBOINC -lboinc_api -lboinc -luser32 -lpthread -lboinc_opencl $(WIN_OPENCL_LIBS)

$(LIN_BOINC_TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -L$(BOINC_LIN) -o $(LIN_BOINC_TARGET) $(SOURCES) -DBOINC -lboinc_api -lboinc -lpthread -lboinc_opencl $(OPENCL_LIBS) 

$(MAC_TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -o $(MAC_TARGET) $(SOURCES) $(OPENCL_LIBS) -lpthread $(OPENCL_LIBS)

clean:
	rm -f $(LIN_TARGET) $(WIN_TARGET) $(WIN_BOINC_TARGET) $(LIN_BOINC_TARGET)