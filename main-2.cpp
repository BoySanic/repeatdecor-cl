#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <climits>
#include <cstring>
#include <vector>
#include <chrono>
#include <inttypes.h>
#include <iostream>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

using namespace std::chrono;

#ifdef BOINC
constexpr int RUNS_PER_CHECKPOINT = 16;
#include "boinc/boinc_api.h"
#if defined _WIN32 || defined _WIN64
#include "boinc/boinc_win.h"
#endif
#endif

// for boinc checkpointing
struct checkpoint_vars {
    int32_t range_min;
    int32_t range_max;
    uint32_t stored_checksum;
    uint64_t elapsed_chkpoint;
};

// Constants
constexpr unsigned long long THREAD_SIZE = 512;
constexpr unsigned long long BLOCK_SIZE = 1ULL << 23;
constexpr unsigned long long BATCH_SIZE = BLOCK_SIZE * THREAD_SIZE;
constexpr int RESULTS_BUFFER_SIZE = 8;
constexpr int SCORE_CUTOFF = 50;

constexpr int HASH_BATCH_SIZE = 4;
constexpr uint64_t XL = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t XH = 0x6A09E667F3BCC909ULL;
constexpr uint64_t XL_BASE = XL * HASH_BATCH_SIZE;

// Structs
struct Result {
    int64_t  score;
    uint64_t seed;
    int64_t  a, b;
};

// OpenCL error checking with more detailed messages
const char* getErrorString(cl_int error) {
    switch(error) {
        case CL_SUCCESS: return "Success";
        case CL_DEVICE_NOT_FOUND: return "Device not found";
        case CL_DEVICE_NOT_AVAILABLE: return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE: return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES: return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY: return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP: return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH: return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE: return "Build program failure";
        case CL_MAP_FAILURE: return "Map failure";
        case CL_INVALID_VALUE: return "Invalid value";
        case CL_INVALID_DEVICE_TYPE: return "Invalid device type";
        case CL_INVALID_PLATFORM: return "Invalid platform";
        case CL_INVALID_DEVICE: return "Invalid device";
        case CL_INVALID_CONTEXT: return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES: return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE: return "Invalid command queue";
        case CL_INVALID_HOST_PTR: return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT: return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE: return "Invalid image size";
        case CL_INVALID_SAMPLER: return "Invalid sampler";
        case CL_INVALID_BINARY: return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS: return "Invalid build options";
        case CL_INVALID_PROGRAM: return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME: return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION: return "Invalid kernel definition";
        case CL_INVALID_KERNEL: return "Invalid kernel";
        case CL_INVALID_ARG_INDEX: return "Invalid argument index";
        case CL_INVALID_ARG_VALUE: return "Invalid argument value";
        case CL_INVALID_ARG_SIZE: return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS: return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION: return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE: return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE: return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET: return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST: return "Invalid event wait list";
        case CL_INVALID_EVENT: return "Invalid event";
        case CL_INVALID_OPERATION: return "Invalid operation";
        case CL_INVALID_GL_OBJECT: return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE: return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL: return "Invalid mip-map level";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "Invalid global work size";
        case -1001: return "Platform not found (CL_PLATFORM_NOT_FOUND_KHR)";
        default: return "Unknown error";
    }
}

void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL error during %s: %d (%s)\n", operation, err, getErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Load OpenCL kernel source
std::string loadKernelSource(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open kernel file: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void printPlatformInfo() {
    cl_uint numPlatforms;
    cl_int err = clGetPlatformIDs(0, NULL, &numPlatforms);
    
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get number of platforms: %d (%s)\n", err, getErrorString(err));
        fprintf(stderr, "\nThis usually means:\n");
        fprintf(stderr, "1. OpenCL drivers are not installed\n");
        fprintf(stderr, "2. OpenCL runtime is not available\n");
        fprintf(stderr, "3. No OpenCL-capable devices are present\n\n");
        
        fprintf(stderr, "To fix this:\n");
        fprintf(stderr, "- On Ubuntu/Debian: sudo apt install ocl-icd-opencl-dev\n");
        fprintf(stderr, "- On NVIDIA systems: Install CUDA toolkit\n");
        fprintf(stderr, "- On AMD systems: Install AMD APP SDK or ROCm\n");
        fprintf(stderr, "- On Intel systems: Install Intel OpenCL runtime\n");
        return;
    }
    
    if (numPlatforms == 0) {
        fprintf(stderr, "No OpenCL platforms found!\n");
        return;
    }
    
    fprintf(stderr, "Found %u OpenCL platform(s):\n", numPlatforms);
    
    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Failed to get platform IDs: %d (%s)\n", err, getErrorString(err));
        return;
    }
    
    for (cl_uint i = 0; i < numPlatforms; i++) {
        char platformName[256];
        char platformVendor[256];
        char platformVersion[256];
        
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(platformVendor), platformVendor, NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(platformVersion), platformVersion, NULL);
        
        fprintf(stderr, "  Platform %u: %s (%s) - %s\n", i, platformName, platformVendor, platformVersion);
        
        // Get devices for this platform
        cl_uint numDevices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        if (err == CL_SUCCESS && numDevices > 0) {
            std::vector<cl_device_id> devices(numDevices);
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL);
            
            for (cl_uint j = 0; j < numDevices; j++) {
                char deviceName[256];
                cl_device_type deviceType;
                clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
                clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL);
                
                const char* typeStr = "Unknown";
                if (deviceType & CL_DEVICE_TYPE_GPU) typeStr = "GPU";
                else if (deviceType & CL_DEVICE_TYPE_CPU) typeStr = "CPU";
                else if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) typeStr = "Accelerator";
                
                fprintf(stderr, "    Device %u: %s (%s)\n", j, deviceName, typeStr);
            }
        }
    }
}
size_t detectOptimalBatchSize(cl_context context, cl_device_id device, cl_command_queue queue, 
                             cl_kernel kernel, cl_mem d_results, cl_mem d_results_count, cl_mem d_checksum) {
    cl_int err;
    
    // Comprehensive test sizes from 8K to 64M
    size_t testSizes[] = {
        // Small sizes (8K - 128K)
        8192,      // 8K
        16384,     // 16K
        24576,     // 24K
        32768,     // 32K
        49152,     // 48K
        65536,     // 64K
        98304,     // 96K
        131072,    // 128K
        
        // Medium sizes (192K - 1M)
        196608,    // 192K
        262144,    // 256K
        393216,    // 384K
        524288,    // 512K
        786432,    // 768K
        1048576,   // 1M
        
        // Large sizes (1.5M - 8M)
        1572864,   // 1.5M
        2097152,   // 2M
        3145728,   // 3M
        4194304,   // 4M
        6291456,   // 6M
        8388608,   // 8M
        
        // Very large sizes (12M - 64M)
        12582912,  // 12M
        16777216,  // 16M
        25165824,  // 24M
        33554432,  // 32M
        50331648,  // 48M
        67108864,  // 64M
        
        // Extreme sizes (using ULL for correct literals)
        100663296, // 96M
        134217728, // 128M
        201326592, // 192M
        268435456, // 256M
        (1ULL<<29),
        (1ULL<<30),
        (1ULL<<31),
        (1ULL<<32),
        (1ULL<<33),
        (1ULL<<34),
        (1ULL<<35)
    };
    
    int numSizes = sizeof(testSizes) / sizeof(testSizes[0]);
    
    size_t bestSize = 524288;
    double bestThroughput = 0;
    
    fprintf(stderr, "Comprehensive batch size optimization (throughput-based)...\n");
    fprintf(stderr, "Testing %d different batch sizes from 8K to 256M work items\n\n", numSizes);
    
    // In detectOptimalBatchSize, the start_seed for testing is always 0.
    // The "nValidSeeds" for this test will be the 'testSize' itself.
    cl_ulong test_start_seed_cl = 0ULL; 

    // Query device capabilities once for this function
    size_t maxWorkGroupSize;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);

    for (int i = 0; i < numSizes; i++) {
        size_t testSize = testSizes[i];
        size_t localWorkSize = std::min(512UL, maxWorkGroupSize); // Use maxWorkGroupSize here
        
        // Skip if not divisible by local work size or if testSize is too small for localWorkSize
        if (testSize == 0 || testSize % localWorkSize != 0) {
            fprintf(stderr, "  Size %9zu: SKIPPED (not divisible by %zu or too small)\n", testSize, localWorkSize);
            continue;
        }
        
        // Reset buffers
        uint32_t checksum_val = 0;
        int results_count_val = 0; // Use a distinct variable name for clarity here
        
        err = clEnqueueWriteBuffer(queue, d_checksum, CL_TRUE, 0, sizeof(uint32_t), &checksum_val, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "  Size %9zu: FAILED (checksum write: %s)\n", testSize, getErrorString(err));
            continue; // Skip this test size if buffer write fails
        }
        
        err = clEnqueueWriteBuffer(queue, d_results_count, CL_TRUE, 0, sizeof(int), &results_count_val, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "  Size %9zu: FAILED (results_count write: %s)\n", testSize, getErrorString(err));
            continue; // Skip this test size if buffer write fails
        }
        
        // Set kernel arguments
        // IMPORTANT: The arguments MUST match the searchKernel signature:
        // __kernel void searchKernel(ulong start_seed, ulong nValidSeeds, __global Result* results, volatile __global int* result_idx, volatile __global uint* checksum)
        cl_ulong cl_nValidSeeds = (cl_ulong)testSize; // The "valid" number of work items for this test
        
        err  = clSetKernelArg(kernel, 0, sizeof(cl_ulong), &test_start_seed_cl); // start_seed
        err |= clSetKernelArg(kernel, 1, sizeof(cl_ulong), &cl_nValidSeeds);     // nValidSeeds (the actual size to process)
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem),   &d_results);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem),   &d_results_count);
        err |= clSetKernelArg(kernel, 4, sizeof(cl_mem),   &d_checksum);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "  Size %9zu: FAILED (set kernel args: %s)\n", testSize, getErrorString(err));
            continue; // Skip this test size if setting arguments fails
        }
        
        // Test with multiple iterations for accuracy
        const int iterations = (testSize < 1048576) ? 5 : 3; // More iterations for smaller sizes
        double totalTime = 0;
        bool success = true;
        
        for (int iter = 0; iter < iterations && success; iter++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            // The global work size passed to EnqueueNDRangeKernel must be rounded up
            // to be a multiple of localWorkSize. The kernel itself handles the actual
            // number of valid seeds using the nValidSeeds argument (cl_nValidSeeds).
            size_t adjustedGlobalSize = ((testSize + localWorkSize - 1) / localWorkSize) * localWorkSize;

            err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &adjustedGlobalSize, &localWorkSize, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "  Size %9zu: FAILED (kernel launch: %s)\n", testSize, getErrorString(err));
                success = false;
                break;
            }
            
            err = clFinish(queue);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "  Size %9zu: FAILED (execution: %s)\n", testSize, getErrorString(err));
                success = false;
                break;
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            totalTime += duration.count() / 1000000.0; // Convert to ms
        }
        
        if (!success) continue;
        
        double avgTime = totalTime / iterations;
        double throughput = testSize / avgTime * 1000.0; // items per second
        double throughputGHz = throughput / 1e9; // Giga items per second
        
        // Format size nicely
        const char* sizeUnit;
        double displaySize;
        if (testSize >= 1048576) {
            displaySize = testSize / 1048576.0;
            sizeUnit = "M";
        } else if (testSize >= 1024) {
            displaySize = testSize / 1024.0;
            sizeUnit = "K";
        } else {
            displaySize = testSize;
            sizeUnit = "";
        }
        
        fprintf(stderr, "  Size %6.1f%s: %7.2fms -> %6.2f Gitems/sec", 
               displaySize, sizeUnit, avgTime, throughputGHz);
        
        if (throughput > bestThroughput) {
            bestThroughput = throughput;
            bestSize = testSize;
            fprintf(stderr, " <- NEW BEST");
        }
        fprintf(stderr, "\n");
        
        // Early exit if we hit memory limits (very slow performance)
        if (avgTime > 1000.0) { // If a single batch takes over 1 second
            fprintf(stderr, "  Stopping test - batch time too high (likely memory limit reached)\n");
            break;
        }
    }
    
    double bestThroughputGHz = bestThroughput / 1e9;
    fprintf(stderr, "\nFinal optimal batch size: %zu (%.2f Gitems/sec)\n", bestSize, bestThroughputGHz);
    
    return bestSize;
}
int main(int argc, char **argv) {
    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t time_elapsed = 0;
    uint64_t start_seed = 0;
    uint64_t end_seed = 211414360895ULL / HASH_BATCH_SIZE;
    uint64_t device_id = 0;
    uint32_t checksum = 0;
    
    FILE* seed_output = fopen("seeds.txt", "wa");
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-s") && i + 1 < argc)
            start_seed = strtoull(argv[++i], nullptr, 0);
        else if (!strcmp(argv[i], "-e") && i + 1 < argc)
            end_seed = strtoull(argv[++i], nullptr, 0);
        else if (!strcmp(argv[i], "-d") && i + 1 < argc)
            device_id = strtoull(argv[++i], nullptr, 0);
        else {
            printf("Usage: %s [-s start_seed] [-e end_seed] [-d device_id]\n", argv[0]);
            return 0;
        }
    }

#ifdef BOINC
    BOINC_OPTIONS options;
    boinc_options_defaults(options);
    options.normal_thread_priority = true;
    boinc_init_options(&options);
    APP_INIT_DATA aid;
    boinc_get_init_data(aid);
    if (aid.gpu_device_num >= 0) {
        device_id = aid.gpu_device_num;
        fprintf(stderr, "boinc gpu %i gpuindex: %i \n", aid.gpu_device_num, device_id);
    }
    else {
        device_id = -5;
        for (int i = 1; i < argc; i += 2) {
            if (strcmp(argv[i], "-d") == 0) {
                sscanf(argv[i + 1], "%i", &device_id);
            }
        }
        if (device_id == -5) {
            fprintf(stderr, "Error: No --device parameter provided! Defaulting to device 0...\n");
            device_id = 0;
        }
        fprintf(stderr, "stndalone gpuindex %i (aid value: %i)\n", device_id, aid.gpu_device_num);
    }
    
    FILE* checkpoint_data = boinc_fopen("checkpoint.txt", "rb");
    if (!checkpoint_data) {
        fprintf(stderr, "No checkpoint to load\n");
    }
    else {
        boinc_begin_critical_section();
        struct checkpoint_vars data_store;
        fread(&data_store, sizeof(data_store), 1, checkpoint_data);
        start_seed = data_store.range_min;
        end_seed = data_store.range_max;
        time_elapsed = data_store.elapsed_chkpoint;
        checksum = data_store.stored_checksum;
        fprintf(stderr, "Checkpoint loaded, task time %llu us, seed pos: %llu\n", time_elapsed, start_seed);
        fclose(checkpoint_data);
        boinc_end_critical_section();
    }
#endif

    // Print platform information for debugging
    printPlatformInfo();

    // Initialize OpenCL
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    // Get platform
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) {
        fprintf(stderr, "No OpenCL platforms available. Exiting.\n");
        return EXIT_FAILURE;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), NULL);
    checkError(err, "clGetPlatformIDs");
    
    platform = platforms[0]; // Use first platform

    // Get device - try GPU first, then fall back to any available device
    cl_uint numDevices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if (err != CL_SUCCESS || numDevices == 0) {
        fprintf(stderr, "No GPU devices found, trying all device types...\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        if (err != CL_SUCCESS || numDevices == 0) {
            fprintf(stderr, "No OpenCL devices found!\n");
            return EXIT_FAILURE;
        }
        
        std::vector<cl_device_id> devices(numDevices);
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL);
        checkError(err, "clGetDeviceIDs");
        device = devices[0];
    } else {
        std::vector<cl_device_id> devices(numDevices);
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), NULL);
        checkError(err, "clGetDeviceIDs");
        device = devices[0];
    }

    // Print selected device info
    char deviceName[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    fprintf(stderr, "Using device: %s\n", deviceName);

    // Create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "clCreateContext");

    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "clCreateCommandQueue");

    // Load and build kernel
    std::string kernelSource = loadKernelSource("kernel.cl");
    const char* source = kernelSource.c_str();
    size_t sourceSize = kernelSource.length();
    
    program = clCreateProgramWithSource(context, 1, &source, &sourceSize, &err);
    checkError(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char* log = new char[logSize];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        delete[] log;
        exit(EXIT_FAILURE);
    }

    kernel = clCreateKernel(program, "searchKernel", &err);
    checkError(err, "clCreateKernel");

    // Create buffers
    cl_mem d_results = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                                     RESULTS_BUFFER_SIZE * sizeof(Result), NULL, &err);
    checkError(err, "clCreateBuffer results");

    cl_mem d_results_count = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                           sizeof(int), NULL, &err);
    checkError(err, "clCreateBuffer results_count");

    cl_mem d_checksum = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                      sizeof(uint32_t), NULL, &err);
    checkError(err, "clCreateBuffer checksum");

    // Initialize checksum buffer
    err = clEnqueueWriteBuffer(queue, d_checksum, CL_TRUE, 0, sizeof(uint32_t), 
                              &checksum, 0, NULL, NULL);
    checkError(err, "clEnqueueWriteBuffer checksum");

    Result h_results[RESULTS_BUFFER_SIZE];
    uint64_t checkpointTemp = 0;

    const size_t OPTIMAL_WORK_SIZE = detectOptimalBatchSize(context, device, queue, kernel, d_results, d_results_count, d_checksum);
    fprintf(stderr, "Starting computation from seed %llu to %llu\n", start_seed, end_seed);
    fprintf(stderr, "Using optimized batch size: %zu work items per sub-batch\n", OPTIMAL_WORK_SIZE);

   for (uint64_t curr_seed = start_seed; curr_seed <= end_seed; curr_seed += BATCH_SIZE) {
        int results_count = 0;
        
        err = clEnqueueWriteBuffer(queue, d_results_count, CL_TRUE, 0, sizeof(int), 
                                &results_count, 0, NULL, NULL);
        checkError(err, "clEnqueueWriteBuffer results_count");

        // Calculate number of sub-batches needed for this BATCH_SIZE
        size_t totalWorkItems = BATCH_SIZE;
        size_t numSubBatches = (totalWorkItems + OPTIMAL_WORK_SIZE - 1) / OPTIMAL_WORK_SIZE;
        // Query device capabilities (add this after device selection)
        size_t maxWorkGroupSize;
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);

        for (size_t subBatch = 0; subBatch < numSubBatches; subBatch++) {
            // compute the true start and size of this sub-batch
            uint64_t sbStart = curr_seed
                            + subBatch * OPTIMAL_WORK_SIZE;
            size_t   sbCount = std::min(
                OPTIMAL_WORK_SIZE,
                totalWorkItems - subBatch * OPTIMAL_WORK_SIZE
            );

            // don't overshoot the end_seed
            if (sbStart > end_seed) break;
            if (sbStart + sbCount > end_seed)
                sbCount = end_seed - sbStart + 1;

            // cast to cl_ulong to match kernel's `ulong`
            cl_ulong clStart = (cl_ulong)sbStart;
            cl_ulong clCount = (cl_ulong)sbCount;

            // *** set exactly five arguments ***
            err  = clSetKernelArg(kernel, 0, sizeof(cl_ulong), &clStart);
            err |= clSetKernelArg(kernel, 1, sizeof(cl_ulong), &clCount);
            err |= clSetKernelArg(kernel, 2, sizeof(cl_mem),   &d_results);
            err |= clSetKernelArg(kernel, 3, sizeof(cl_mem),   &d_results_count);
            err |= clSetKernelArg(kernel, 4, sizeof(cl_mem),   &d_checksum);
            checkError(err, "clSetKernelArg");

            // choose local size and round up global size
            size_t localWorkSize      = std::min<size_t>(512, maxWorkGroupSize);
            size_t adjustedGlobalSize =
                ((sbCount + localWorkSize - 1) / localWorkSize)
                * localWorkSize;

            // launch!
            err = clEnqueueNDRangeKernel(
                queue, kernel, 1,
                nullptr,
                &adjustedGlobalSize,
                &localWorkSize,
                0, nullptr, nullptr
            );
            checkError(err, "clEnqueueNDRangeKernel");
        }
        // Wait for all sub-batches to complete
        err = clFinish(queue);
        checkError(err, "clFinish");

        // Read results count
        err = clEnqueueReadBuffer(queue, d_results_count, CL_TRUE, 0, sizeof(int), 
                                &results_count, 0, NULL, NULL);
        checkError(err, "clEnqueueReadBuffer results_count");

    #ifdef BOINC
        if (checkpointTemp >= RUNS_PER_CHECKPOINT-1 || boinc_time_to_checkpoint()) {
            auto checkpoint_time = std::chrono::high_resolution_clock::now();
            time_elapsed = duration_cast<milliseconds>(checkpoint_time - start_time).count() + time_elapsed;
            
            // Read current checksum
            err = clEnqueueReadBuffer(queue, d_checksum, CL_TRUE, 0, sizeof(uint32_t), 
                                    &checksum, 0, NULL, NULL);
            checkError(err, "clEnqueueReadBuffer checksum");
            
            boinc_begin_critical_section();
            checkpointTemp = 0;
            boinc_delete_file("checkpoint.txt");
            FILE* checkpoint_data = boinc_fopen("checkpoint.txt", "wb");

            struct checkpoint_vars data_store;
            data_store.range_min = curr_seed + 1;
            data_store.range_max = end_seed;
            data_store.elapsed_chkpoint = time_elapsed;
            data_store.stored_checksum = checksum;
            fwrite(&data_store, sizeof(data_store), 1, checkpoint_data);
            fclose(checkpoint_data);

            boinc_end_critical_section();
            boinc_checkpoint_completed();
        }
        
        double frac = (double)(curr_seed - start_seed + 1) / (double)(end_seed - start_seed);
        boinc_fraction_done(frac);
    #endif

        if (results_count > 0) {
            err = clEnqueueReadBuffer(queue, d_results, CL_TRUE, 0, 
                                    results_count * sizeof(Result), h_results, 0, NULL, NULL);
            checkError(err, "clEnqueueReadBuffer results");

            for (int i = 0; i < results_count; i++) {
                Result result = h_results[i];
                fprintf(seed_output, "seed: %lld score: %lld\n", result.seed, result.score);
            }
        }
        checkpointTemp++;
        
    // Progress indicator
    if ((curr_seed - start_seed) % (BATCH_SIZE * 100) == 0) {
        double progress = (double)(curr_seed - start_seed) / (double)(end_seed - start_seed) * 100.0;
        auto progress_time = std::chrono::high_resolution_clock::now();
        uint64_t current_elapsed = duration_cast<milliseconds>(progress_time - start_time).count() + time_elapsed;
        
        // Use minimum 1ms to avoid division by zero
        uint64_t safe_elapsed = (current_elapsed > 0) ? current_elapsed : 1;
        uint64_t seeds_processed = (curr_seed - start_seed) * HASH_BATCH_SIZE;
        double seconds_elapsed = safe_elapsed / 1000.0;
        double gsps = (seeds_processed / seconds_elapsed) / 1e9;
        
        fprintf(stderr, "Progress: %.2f%%, Gsps: %.2f\r", progress, gsps);
        fflush(stderr);
    }
    }

    // Read final checksum
    err = clEnqueueReadBuffer(queue, d_checksum, CL_TRUE, 0, sizeof(uint32_t), 
                             &checksum, 0, NULL, NULL);
    checkError(err, "clEnqueueReadBuffer final checksum");

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    fprintf(seed_output, "Checksum: %u\n", checksum);
    fprintf(stderr, "\nSeeds checked: %llu\n", (end_seed - start_seed) * HASH_BATCH_SIZE);
    fprintf(stderr, "Time taken: %lldms\n", duration.count() + time_elapsed);
    fprintf(stderr, "GSPS: %llu\n", (end_seed - start_seed) * HASH_BATCH_SIZE / duration.count() * 1000 / 1000000000);

    // Cleanup
    clReleaseMemObject(d_results);
    clReleaseMemObject(d_results_count);
    clReleaseMemObject(d_checksum);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    fclose(seed_output);

#ifdef BOINC
    boinc_finish(0);
#endif

    return 0;
}