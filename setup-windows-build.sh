#!/bin/bash

set -e

echo "Setting up Windows cross-compilation environment..."

# Create directories
mkdir -p windows-deps/{include,lib,bin}

# Download OpenCL headers
echo "Downloading OpenCL headers..."
cd windows-deps/include
if [ ! -d "CL" ]; then
    git clone https://github.com/KhronosGroup/OpenCL-Headers.git CL
fi
cd ../..

# Download Intel OpenCL runtime (contains OpenCL.lib)
echo "Downloading Intel OpenCL runtime..."
cd windows-deps
if [ ! -f "intel_opencl_runtime.zip" ]; then
    # You'll need to manually download this from Intel's website
    # or use an alternative method
    echo "Please download Intel OpenCL runtime and extract OpenCL.lib to windows-deps/lib/"
    echo "Alternative: Download from https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases"
fi
cd ..

echo "Setup complete. You can now run 'make windows' to cross-compile."