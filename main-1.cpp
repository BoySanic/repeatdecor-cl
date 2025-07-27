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

// OpenCL error checking
void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL error during %s: %d\n", operation, err);
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

    // Initialize OpenCL
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    // Get platform
    err = clGetPlatformIDs(1, &platform, NULL);
    checkError(err, "clGetPlatformIDs");

    // Get device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);
    }
    checkError(err, "clGetDeviceIDs");

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

    for (uint64_t curr_seed = start_seed; curr_seed <= end_seed; curr_seed += BATCH_SIZE) {
        int results_count = 0;
        
        // Write results count
        err = clEnqueueWriteBuffer(queue, d_results_count, CL_TRUE, 0, sizeof(int), 
                                  &results_count, 0, NULL, NULL);
        checkError(err, "clEnqueueWriteBuffer results_count");

        // Set kernel arguments
        err = clSetKernelArg(kernel, 0, sizeof(uint64_t), &curr_seed);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_results);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_results_count);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_checksum);
        checkError(err, "clSetKernelArg");

        // Execute kernel
        size_t globalWorkSize = BLOCK_SIZE * THREAD_SIZE;
        size_t localWorkSize = THREAD_SIZE;
        
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, 
                                    &localWorkSize, 0, NULL, NULL);
        checkError(err, "clEnqueueNDRangeKernel");

        // Wait for completion
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
    }

    // Read final checksum
    err = clEnqueueReadBuffer(queue, d_checksum, CL_TRUE, 0, sizeof(uint32_t), 
                             &checksum, 0, NULL, NULL);
    checkError(err, "clEnqueueReadBuffer final checksum");

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    fprintf(seed_output, "Checksum: %u\n", checksum);
    fprintf(stderr, "Seeds checked: %llu\n", (end_seed - start_seed) * HASH_BATCH_SIZE);
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
