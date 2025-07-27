#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>           // strcmp()
#include <vector>            // std::vector
#include <chrono>
using namespace std::chrono;
#include <inttypes.h>
#include <iostream>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

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

// OpenCL kernel source
const char* kernel_source = R"(
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

typedef struct {
    long score;
    ulong seed;
    long a, b;
} Result;

inline ulong rotl64(ulong x, uint r) {
    return (x << r) | (x >> (64u - r));
}

inline ulong mix64(ulong z) {
    const ulong M1 = 0xBF58476D1CE4E5B9UL;
    const ulong M2 = 0x94D049BB133111EBUL;
    z = (z ^ (z >> 30)) * M1;
    z = (z ^ (z >> 27)) * M2;
    return z ^ (z >> 31);
}

typedef struct {
    ulong lo, hi;
} PRNG128;

inline PRNG128 prng_init(ulong s) {
    const ulong XL = 0x9E3779B97F4A7C15UL;
    PRNG128 prng;
    prng.lo = mix64(s);
    prng.hi = mix64(s + XL);
    return prng;
}

inline PRNG128 prng_init2(ulong _lo, ulong _hi) {
    PRNG128 prng;
    prng.lo = _lo;
    prng.hi = _hi;
    return prng;
}

inline ulong prng_next64(PRNG128* prng) {
    ulong res = rotl64(prng->lo + prng->hi, 17) + prng->lo;
    ulong t   = prng->hi ^ prng->lo;
    prng->lo = rotl64(prng->lo, 49) ^ t ^ (t << 21);
    prng->hi = rotl64(t, 28);
    return res;
}

inline uint prng_nextLongLower32(PRNG128* prng) {
    ulong t = prng->hi ^ prng->lo;
    prng->lo = rotl64(prng->lo, 49) ^ t ^ (t << 21);
    prng->hi = rotl64(t, 28);
    t = prng->hi ^ prng->lo;
    return (uint)((rotl64(prng->lo + prng->hi, 17) + prng->lo) >> 32);
}

inline void prng_advance(PRNG128* prng) {
    ulong t = prng->hi ^ prng->lo;
    prng->lo = rotl64(prng->lo, 49) ^ t ^ (t << 21);
    prng->hi = rotl64(t, 28);
}

inline long prng_nextLong(PRNG128* prng) {
    int high = (int)(prng_next64(prng) >> 32);
    int low  = (int)(prng_next64(prng) >> 32);
    return ((long)high << 32) + (long)low;
}

inline void compute_ab(ulong seed, long* a, long* b) {
    PRNG128 rng = prng_init(seed);
    *a = prng_nextLong(&rng) | 1L;
    *b = prng_nextLong(&rng) | 1L;
}

inline bool goodLower32(PRNG128* rng) {
    uint al = prng_nextLongLower32(rng) | 1U;
    prng_advance(rng);
    uint bl = prng_nextLongLower32(rng) | 1U;

    return 
        al == bl || al + bl == 0 ||
        3*al == bl || 3*al + bl == 0 ||
        al == 3*bl || al + 3*bl == 0 ||
        5*al == bl || 5*al + bl == 0 ||
        al == 5*bl || al + 5*bl == 0 ||
        3*al == 5*bl || 3*al + 5*bl == 0 ||
        5*al == 3*bl || 5*al + 3*bl == 0 ||
        7*al == bl || 7*al + bl == 0 ||
        al == 7*bl || al + 7*bl == 0 ||
        7*al == 3*bl || 7*al + 3*bl == 0 ||
        7*al == 5*bl || 7*al + 5*bl == 0 ||
        5*al == 7*bl || 5*al + 7*bl == 0 ||
        7*al == 3*bl || 7*al + 3*bl == 0;
}

inline void processFullPrngState(ulong xseed, __global Result* results, 
                                __global volatile int* result_idx, 
                                __global volatile uint* checksum) {
    const ulong XH = 0x6A09E667F3BCC909UL;
    const int SCORE_CUTOFF = 50;
    const int RESULTS_BUFFER_SIZE = 8;
    
    long a, b;
    compute_ab(xseed, &a, &b);

    long score = 0;
    ulong x;
    int tz;

    x = (ulong)a ^ (ulong)b;
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-a) ^ (ulong)b;
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)a ^ (ulong)(3 * b);
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-a) ^ (ulong)(3 * b);
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(3 * a) ^ (ulong)b;
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-3 * a) ^ (ulong)b;
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)a ^ (ulong)(5 * b);
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-a) ^ (ulong)(5 * b);
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(5 * a) ^ (ulong)b;
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-5 * a) ^ (ulong)b;
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(3 * a) ^ (ulong)(5 * b);
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-3 * a) ^ (ulong)(5 * b);
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(5 * a) ^ (ulong)(3 * b);
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-5 * a) ^ (ulong)(3 * b);
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)a ^ (ulong)(7 * b);
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-a) ^ (ulong)(7 * b);
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(7 * a) ^ (ulong)b;
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-7 * a) ^ (ulong)b;
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(3 * a) ^ (ulong)(7 * b);
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-3 * a) ^ (ulong)(7 * b);
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(7 * a) ^ (ulong)(3 * b);
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-7 * a) ^ (ulong)(3 * b);
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(5 * a) ^ (ulong)(7 * b);
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-5 * a) ^ (ulong)(7 * b);
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(7 * a) ^ (ulong)(5 * b);
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-7 * a) ^ (ulong)(5 * b);
    tz = x ? (clz(x & (-x)) ^ 63) : 64;
    score = tz > score ? tz : score;

    if (score < SCORE_CUTOFF)
        return;

    ulong seed = xseed ^ XH;
    int this_result_idx = atomic_add(result_idx, 1);
    
    if (this_result_idx < RESULTS_BUFFER_SIZE) {
        results[this_result_idx].score = score;
        results[this_result_idx].seed = seed;
        results[this_result_idx].a = a;
        results[this_result_idx].b = b;
        atomic_add(checksum, 1);
    }
}

__kernel void searchKernel(ulong start_seed, __global Result* results, 
                          __global volatile int* result_idx, 
                          __global volatile uint* checksum) {
    const ulong XL = 0x9E3779B97F4A7C15UL;
    const ulong XL_BASE = XL * 4; // HASH_BATCH_SIZE = 4
    const int HASH_BATCH_SIZE = 4;
    
    ulong gid = get_global_id(0);
    ulong seed_base = (start_seed + gid) * XL_BASE;

    ulong hashes[5]; // HASH_BATCH_SIZE + 1
    for (int i = 0; i <= HASH_BATCH_SIZE; i++)
        hashes[i] = mix64(seed_base + i*XL);

    for (int i = 0; i < HASH_BATCH_SIZE; i++) {
        PRNG128 prng = prng_init2(hashes[i], hashes[i+1]);
        if (!goodLower32(&prng))
            continue;
        ulong curr_s = seed_base + i * XL;
        processFullPrngState(curr_s, results, result_idx, checksum);
        atomic_add(checksum, 1);
    }
}
)";

// OpenCL error checking
void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL error during %s: %d\n", operation, err);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv) {
    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t time_elapsed = 0;
    uint64_t start_seed = 0;
    uint64_t end_seed   = 211414360895ULL / HASH_BATCH_SIZE;
    uint64_t device_id  = 0;
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
    #endif // BOINC

    // Initialize OpenCL
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // Get platform
    err = clGetPlatformIDs(1, &platform, NULL);
    checkError(err, "getting platform");

    // Get device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    checkError(err, "getting device");

    // Create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "creating context");

    // Create command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "creating command queue");

    // Create program
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    checkError(err, "creating program");

    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build error:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }

    // Create kernel
    kernel = clCreateKernel(program, "searchKernel", &err);
    checkError(err, "creating kernel");

    // Create buffers
    cl_mem d_results = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                     RESULTS_BUFFER_SIZE * sizeof(Result), NULL, &err);
    checkError(err, "creating results buffer");

    cl_mem d_result_idx = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                        sizeof(int), NULL, &err);
    checkError(err, "creating result_idx buffer");

    cl_mem d_checksum = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                      sizeof(uint32_t), NULL, &err);
    checkError(err, "creating checksum buffer");

    // Initialize checksum
    err = clEnqueueWriteBuffer(queue, d_checksum, CL_TRUE, 0, sizeof(uint32_t), 
                              &checksum, 0, NULL, NULL);
    checkError(err, "writing checksum");

    Result h_results[RESULTS_BUFFER_SIZE];
    size_t global_work_size = BATCH_SIZE;
    size_t local_work_size = THREAD_SIZE;

    uint64_t checkpointTemp = 0;
    for (uint64_t curr_seed = start_seed; curr_seed <= end_seed; curr_seed += BATCH_SIZE) {
        int results_count = 0;
        
        // Reset result count
        err = clEnqueueWriteBuffer(queue, d_result_idx, CL_TRUE, 0, sizeof(int), 
                                  &results_count, 0, NULL, NULL);
        checkError(err, "writing result_idx");

        // Set kernel arguments
        err = clSetKernelArg(kernel, 0, sizeof(uint64_t), &curr_seed);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_results);
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_result_idx);
        err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_checksum);
        checkError(err, "setting kernel arguments");

        // Execute kernel
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, 
                                    &local_work_size, 0, NULL, NULL);
        checkError(err, "executing kernel");

        // Wait for completion
        err = clFinish(queue);
        checkError(err, "waiting for kernel completion");

        // Read results count
        err = clEnqueueReadBuffer(queue, d_result_idx, CL_TRUE, 0, sizeof(int), 
                                 &results_count, 0, NULL, NULL);
        checkError(err, "reading result count");

        #ifdef BOINC
            if (checkpointTemp >= RUNS_PER_CHECKPOINT-1 || boinc_time_to_checkpoint()) {
                auto checkpoint_time = std::chrono::high_resolution_clock::now();
                time_elapsed = duration_cast<milliseconds>(checkpoint_time - start_time).count() + time_elapsed;
                
                // Read current checksum
                err = clEnqueueReadBuffer(queue, d_checksum, CL_TRUE, 0, sizeof(uint32_t), 
                                         &checksum, 0, NULL, NULL);
                checkError(err, "reading checksum for checkpoint");
                
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
        #endif // BOINC

        if (results_count > 0) {
            err = clEnqueueReadBuffer(queue, d_results, CL_TRUE, 0, 
                                     results_count * sizeof(Result), h_results, 0, NULL, NULL);
            checkError(err, "reading results");

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
    checkError(err, "reading final checksum");

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    fprintf(seed_output, "Checksum: %u\n", checksum);
    fprintf(stderr, "Seeds checked: %lld\n", (end_seed - start_seed) * HASH_BATCH_SIZE);
    fprintf(stderr, "Time taken: %lldms\n", duration.count() + time_elapsed);
    fprintf(stderr, "GSPS: %lld\n", (end_seed - start_seed) * HASH_BATCH_SIZE / duration.count() * 1000 / 1000000000);

    // Cleanup
    clReleaseMemObject(d_results);
    clReleaseMemObject(d_result_idx);
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