#define CL_TARGET_OPENCL_VERSION 200  // For OpenCL 2.0
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <chrono>

#define min(a, b) ((a) < (b) ? (a) : (b))
#define RESULTS_BUFFER_SIZE 1024
#define HASH_BATCH_SIZE 4
// In your host code, replace the #defines with:
#define THREAD_SIZE 512
#define BLOCK_SIZE (1ULL << 23)  // 8,388,608 - same as CUDA
#define BATCH_SIZE (BLOCK_SIZE * THREAD_SIZE)
#define SCORE_CUTOFF 50

typedef struct {
    int64_t  score;
    uint64_t seed;
    int64_t  a, b;
} Result;

typedef struct {
    size_t work_size;
    double items_per_second;
    double time_ms;
    bool valid;
} BenchmarkResult;

const char* get_cl_error_string(cl_int error) {
    switch(error) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
        default: return "UNKNOWN_ERROR";
    }
}

void print_device_info(cl_device_id device, int device_index) {
    char device_name[256];
    char vendor_name[256];
    char driver_version[256];
    char device_version[256];
    cl_device_type device_type;
    cl_uint compute_units;
    size_t max_work_group_size;
    cl_ulong global_mem_size;
    cl_ulong local_mem_size;
    cl_uint max_clock_frequency;
    
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, NULL);
    clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(driver_version), driver_version, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(device_version), device_version, NULL);
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_clock_frequency), &max_clock_frequency, NULL);
    // Add this in your device info printing function:
    char opencl_version[256];
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(opencl_version), opencl_version, NULL);
    printf("  OpenCL Version: %s\n", opencl_version);

    char opencl_c_version[256];
    clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(opencl_c_version), opencl_c_version, NULL);
    printf("  OpenCL C Version: %s\n", opencl_c_version);
    printf("Device %d:\n", device_index);
    printf("  Name: %s\n", device_name);
    printf("  Vendor: %s\n", vendor_name);
    printf("  Driver Version: %s\n", driver_version);
    printf("  Device Version: %s\n", device_version);
    printf("  Type: %s\n", (device_type & CL_DEVICE_TYPE_GPU) ? "GPU" : 
                           (device_type & CL_DEVICE_TYPE_CPU) ? "CPU" : "Other");
    printf("  Compute Units: %u\n", compute_units);
    printf("  Max Work Group Size: %zu\n", max_work_group_size);
    printf("  Global Memory: %.2f GB\n", (double)global_mem_size / (1024*1024*1024));
    printf("  Local Memory: %.2f KB\n", (double)local_mem_size / 1024);
    printf("  Max Clock Frequency: %u MHz\n", max_clock_frequency);
    printf("\n");
}

size_t benchmark_work_sizes(cl_context context, cl_device_id device, 
                                       cl_command_queue queue, cl_kernel kernel,
                                       cl_mem results_buf, cl_mem result_idx_buf, 
                                       cl_mem checksum_buf) {
    printf("Testing conservative work sizes...\n");
    
    // Get device info
    cl_uint compute_units;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL);
    
    // Test only reasonable sizes based on compute units
    size_t test_sizes[] = {
        compute_units * 64,
        compute_units * 128,
        compute_units * 256,
        compute_units * 512,
        compute_units * 1024,
        BATCH_SIZE / 4,
        BATCH_SIZE / 2,
        BATCH_SIZE,
        BATCH_SIZE * 2  // Don't go beyond 2x original
    };
    
    size_t num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    size_t best_size = BATCH_SIZE;
    double best_throughput = 0.0;
    
    const uint64_t test_seed = 12345;
    
    for (size_t i = 0; i < num_tests; i++) {
        size_t work_size = test_sizes[i];
        if (work_size > BATCH_SIZE * 2) continue; // Safety limit
        
        printf("Testing work size: %zu... ", work_size);
        
        // Reset buffers
        int zero_count = 0;
        uint32_t zero_checksum = 0;
        clEnqueueWriteBuffer(queue, result_idx_buf, CL_TRUE, 0, sizeof(int), &zero_count, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, checksum_buf, CL_TRUE, 0, sizeof(uint32_t), &zero_checksum, 0, NULL, NULL);
        
        // Set kernel args
        clSetKernelArg(kernel, 0, sizeof(uint64_t), &test_seed);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &results_buf);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &result_idx_buf);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &checksum_buf);
        
        // Benchmark
        auto start_time = std::chrono::high_resolution_clock::now();
        
        cl_int err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_size, NULL, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            printf("FAILED\n");
            continue;
        }
        
        clFinish(queue);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        double time_ms = duration.count() / 1000.0;
        double items_per_second = (work_size * 4.0 / time_ms) * 1000.0;
        
        printf("%.2f ms, %.0f items/sec\n", time_ms, items_per_second);
        
        if (items_per_second > best_throughput) {
            best_throughput = items_per_second;
            best_size = work_size;
        }
    }
    
    printf("Selected work size: %zu (%.0f items/sec)\n", best_size, best_throughput);
    return best_size;
}

char* load_kernel_source(const char* filename, size_t* length_out) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    size_t len = ftell(f);
    rewind(f);
    char* src = (char*)malloc(len + 1);
    if (!src) { fclose(f); return NULL; }
    fread(src, 1, len, f);
    src[len] = '\0';
    fclose(f);
    if (length_out) *length_out = len;
    return src;
}

int main(int argc, char **argv) {
    using namespace std::chrono;
    auto start_time = high_resolution_clock::now();
    uint64_t time_elapsed = 0;
    uint64_t start_seed = 0;
    uint64_t end_seed   = 211414360895ULL / HASH_BATCH_SIZE;
    uint64_t device_id  = 0;
    uint32_t checksum = 0;
    FILE* seed_output = fopen("seeds.txt", "w");
    if (!seed_output) { fprintf(stderr, "Failed to open seeds.txt\n"); return 1; }

    // Parse arguments
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

    // Enhanced OpenCL setup with verbose logging
    cl_int err;
    cl_uint num_platforms;
    cl_platform_id platforms[10];
    
    printf("=== OpenCL Platform Detection ===\n");
    err = clGetPlatformIDs(10, platforms, &num_platforms);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error getting platform IDs: %s\n", get_cl_error_string(err));
        return 1;
    }
    printf("Found %u OpenCL platform(s)\n\n", num_platforms);
    
    // Print platform information
    for (cl_uint i = 0; i < num_platforms; i++) {
        char platform_name[256];
        char platform_vendor[256];
        char platform_version[256];
        
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(platform_vendor), platform_vendor, NULL);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, NULL);
        
        printf("Platform %u:\n", i);
        printf("  Name: %s\n", platform_name);
        printf("  Vendor: %s\n", platform_vendor);
        printf("  Version: %s\n", platform_version);
        printf("\n");
    }
    
    // Use first platform
    cl_platform_id platform = platforms[0];
    printf("Using platform 0\n\n");
    
    printf("=== OpenCL Device Detection ===\n");
    cl_uint num_devices;
    cl_device_id devices[10];
    
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 10, devices, &num_devices);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error getting device IDs: %s\n", get_cl_error_string(err));
        return 1;
    }
    printf("Found %u device(s)\n\n", num_devices);
    
    // Print all device information
    for (cl_uint i = 0; i < num_devices; i++) {
        print_device_info(devices[i], i);
    }
    
    // Select device
    if (device_id >= num_devices) {
        fprintf(stderr, "Error: Device ID %lu not found. Available devices: 0-%u\n", device_id, num_devices - 1);
        return 1;
    }
    
    cl_device_id device = devices[device_id];
    printf("Selected device %lu\n\n", device_id);
    // Add this after device selection:
    size_t max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    printf("Max work group size: %zu\n", max_work_group_size);

    printf("=== OpenCL Context and Queue Creation ===\n");
    cl_context context;
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating context: %s\n", get_cl_error_string(err));
        return 1;
    }
    printf("Context created successfully\n");
    
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating command queue: %s\n", get_cl_error_string(err));
        return 1;
    }
    printf("Command queue created successfully\n\n");

    // Load and build kernel
    printf("=== Kernel Loading and Compilation ===\n");
    size_t kernel_length;
    char* kernelSource = load_kernel_source("kernel.cl", &kernel_length);
    if (!kernelSource) {
        fprintf(stderr, "Failed to load kernel.cl\n");
        return 1;
    }
    printf("Kernel source loaded successfully (%zu bytes)\n", kernel_length);
    
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, &kernel_length, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating program: %s\n", get_cl_error_string(err));
        return 1;
    }
    printf("Program created successfully\n");
    
    printf("Building program...\n");
    const char* build_options = "-cl-std=CL2.0 -cl-mad-enable -cl-no-signed-zeros";
    err = clBuildProgram(program, 1, &device, build_options, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error building program: %s\n", get_cl_error_string(err));
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build log:\n%s\n", log);
        free(log);
        return 1;
    }
    printf("Program built successfully\n");
    
    cl_kernel kernel = clCreateKernel(program, "searchKernel", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating kernel: %s\n", get_cl_error_string(err));
        return 1;
    }
    printf("Kernel created successfully\n\n");

    // Buffer creation
    printf("=== Buffer Creation ===\n");
    cl_mem d_results = clCreateBuffer(context, CL_MEM_WRITE_ONLY, RESULTS_BUFFER_SIZE * sizeof(Result), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating results buffer: %s\n", get_cl_error_string(err));
        return 1;
    }
    printf("Results buffer created (%zu bytes)\n", RESULTS_BUFFER_SIZE * sizeof(Result));
    
    cl_mem results_count_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating results count buffer: %s\n", get_cl_error_string(err));
        return 1;
    }
    printf("Results count buffer created (%zu bytes)\n", sizeof(int));
    
    cl_mem checksum_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(uint32_t), NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating checksum buffer: %s\n", get_cl_error_string(err));
        return 1;
    }
    printf("Checksum buffer created (%zu bytes)\n", sizeof(uint32_t));

    Result h_results[RESULTS_BUFFER_SIZE];

    // Use original batch size
    const size_t work_size = BATCH_SIZE;
    
    // Simple progress tracking with ETA
    uint64_t total_iterations = (end_seed - start_seed + BATCH_SIZE - 1) / BATCH_SIZE;
    uint64_t completed_iterations = 0;
    
    printf("\n=== Starting Processing ===\n");
    printf("Work size: %zu\n", work_size);
    printf("Total iterations: %lu\n", total_iterations);
    printf("Batch size: %llu\n", (unsigned long long)BATCH_SIZE);
    printf("Hash batch size: %d\n\n", HASH_BATCH_SIZE);

    auto main_start_time = std::chrono::high_resolution_clock::now();
    auto last_update_time = main_start_time;
    // Calculate optimal work sizes (add this before the main loop)

    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);

    size_t local_work_size = min(256, max_work_group_size);
    size_t work_chunk_size = local_work_size * 1024; // e.g., 256K work items per chunk

    printf("Using local_work_size: %zu, work_chunk_size: %zu\n", local_work_size, work_chunk_size);
    printf("Will launch %zu kernel chunks per iteration\n", (BATCH_SIZE + work_chunk_size - 1) / work_chunk_size);

    // Modified main processing loop
    for (uint64_t curr_seed = start_seed; curr_seed <= end_seed; curr_seed += BATCH_SIZE) {
        // Reset results count once per batch
        int results_count = 0;
        err = clEnqueueWriteBuffer(queue, results_count_buf, CL_TRUE, 0, sizeof(int), &results_count, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error writing results count buffer: %s\n", get_cl_error_string(err));
            break;
        }
        
        // Launch multiple kernel chunks to cover the full BATCH_SIZE
        for (uint64_t chunk_start = 0; chunk_start < BATCH_SIZE; chunk_start += work_chunk_size) {
            uint64_t chunk_seed = curr_seed + chunk_start;
            size_t this_chunk_size = (chunk_start + work_chunk_size <= BATCH_SIZE) ? 
                                    work_chunk_size : (BATCH_SIZE - chunk_start);
            
            // Round up to multiple of local_work_size
            size_t padded_chunk_size = ((this_chunk_size + local_work_size - 1) / local_work_size) * local_work_size;
            
            // Set kernel arguments for this chunk
            err = clSetKernelArg(kernel, 0, sizeof(uint64_t), &chunk_seed);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error setting kernel arg 0 (seed): %s\n", get_cl_error_string(err));
                break;
            }
            
            err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_results);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error setting kernel arg 1 (results): %s\n", get_cl_error_string(err));
                break;
            }
            
            err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &results_count_buf);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error setting kernel arg 2 (results_count): %s\n", get_cl_error_string(err));
                break;
            }
            
            err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &checksum_buf);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error setting kernel arg 3 (checksum): %s\n", get_cl_error_string(err));
                break;
            }
            
            // Launch kernel chunk with proper work group size
            err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &padded_chunk_size, &local_work_size, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error launching kernel chunk at seed %llu, chunk %llu: %s\n", 
                        curr_seed, chunk_start, get_cl_error_string(err));
                
                // Additional debugging info for common errors
                if (err == CL_INVALID_WORK_GROUP_SIZE) {
                    fprintf(stderr, "Chunk size: %zu, Padded: %zu, Local: %zu, Max work group: %zu\n", 
                            this_chunk_size, padded_chunk_size, local_work_size, max_work_group_size);
                }
                break;
            }
        }
        
        // Check if we broke out of the inner loop due to error
        if (err != CL_SUCCESS) {
            break;
        }
        
        // Wait for all kernel chunks to complete
        err = clFinish(queue);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error waiting for kernel completion: %s\n", get_cl_error_string(err));
            break;
        }
        
        // Read back results (same as before)
        err = clEnqueueReadBuffer(queue, results_count_buf, CL_TRUE, 0, sizeof(int), &results_count, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error reading results count: %s\n", get_cl_error_string(err));
            break;
        }
        
        if (results_count > 0) {
            if (results_count > RESULTS_BUFFER_SIZE) {
                fprintf(stderr, "Warning: Results count (%d) exceeds buffer size (%d), clamping\n", 
                        results_count, RESULTS_BUFFER_SIZE);
                results_count = RESULTS_BUFFER_SIZE;
            }
            
            err = clEnqueueReadBuffer(queue, d_results, CL_TRUE, 0, results_count * sizeof(Result), h_results, 0, NULL, NULL);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "Error reading results: %s\n", get_cl_error_string(err));
                break;
            }
            
            for (int i = 0; i < results_count; i++) {
                Result result = h_results[i];
                fprintf(seed_output, "seed: %lld score: %lld\n", result.seed, result.score);
            }
            fflush(seed_output);
        }
        
        // Progress update (same as before)
        completed_iterations++;
        auto current_time = std::chrono::high_resolution_clock::now();
        auto time_since_update = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_update_time);
        
        if (time_since_update.count() >= 5000 || completed_iterations == total_iterations) {
            double progress_percent = (double)completed_iterations / total_iterations * 100.0;
            
            // Calculate ETA
            auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(current_time - main_start_time);
            double eta_seconds = 0.0;
            if (completed_iterations > 0) {
                double avg_time_per_iteration = (double)elapsed_time.count() / completed_iterations;
                eta_seconds = avg_time_per_iteration * (total_iterations - completed_iterations);
            }
            
            // Format ETA
            printf("Progress: %.1f%% (%llu/%llu)", progress_percent, completed_iterations, total_iterations);
            
            if (eta_seconds > 0 && completed_iterations < total_iterations) {
                int eta_hours = (int)(eta_seconds / 3600);
                int eta_minutes = (int)((eta_seconds - eta_hours * 3600) / 60);
                int eta_secs = (int)(eta_seconds - eta_hours * 3600 - eta_minutes * 60);
                
                if (eta_hours > 0) {
                    printf(" ETA: %dh %02dm %02ds", eta_hours, eta_minutes, eta_secs);
                } else if (eta_minutes > 0) {
                    printf(" ETA: %dm %02ds", eta_minutes, eta_secs);
                } else {
                    printf(" ETA: %ds", eta_secs);
                }
            }
            
            printf("    \r"); // Extra spaces to clear previous line
            fflush(stdout);
            
            last_update_time = current_time;
        }
    }
    // Read final checksum
    err = clEnqueueReadBuffer(queue, checksum_buf, CL_TRUE, 0, sizeof(uint32_t), &checksum, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading final checksum: %s\n", get_cl_error_string(err));
    }

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    
    printf("\n"); // New line after progress
    printf("\n=== Final Results ===\n");
    fprintf(seed_output, "Checksum: %lld\n", (long long)checksum);
    fprintf(stderr, "Seeds checked: %lld\n", (long long)((end_seed - start_seed) * HASH_BATCH_SIZE));
    fprintf(stderr, "Time taken: %lldms\n", (long long)duration.count());
    fprintf(stderr, "GSPS: %lld\n", (long long)((end_seed - start_seed) * HASH_BATCH_SIZE / duration.count() * 1000 / 1000000000));

    // Cleanup
    printf("\n=== Cleanup ===\n");
    clReleaseMemObject(d_results);
    clReleaseMemObject(results_count_buf);
    clReleaseMemObject(checksum_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    fclose(seed_output);
    free(kernelSource);
    printf("Cleanup completed\n");

    return 0;
}