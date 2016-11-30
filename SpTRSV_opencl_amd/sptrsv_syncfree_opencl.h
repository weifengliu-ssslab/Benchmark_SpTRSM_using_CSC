#ifndef _SPTRSV_SYNCFREE_OPENCL_
#define _SPTRSV_SYNCFREE_OEPNCL_

#include "common.h"
#include "utils.h"
#include "basiccl.h"

int sptrsv_syncfree_opencl (const int           *cscColPtrTR,
                            const int           *cscRowIdxTR,
                            const VALUE_TYPE    *cscValTR,
                            const int            m,
                            const int            n,
                            const int            nnzTR,
                            const int            device_id,
                            const int            substitution,
                            const int            rhs,
                            const int            opt,
                                  VALUE_TYPE    *x,
                            const VALUE_TYPE    *b,
                            const VALUE_TYPE    *x_ref,
                                  double        *gflops)
{
    if (m != n)
    {
        printf("This is not a square matrix, return.\n");
        return -1;
    }

    int err = 0;

    // set device
    BasicCL basicCL;
    cl_event            ceTimer;                 // OpenCL event
    cl_ulong            queuedTime;
    cl_ulong            submitTime;
    cl_ulong            startTime;
    cl_ulong            endTime;

    char platformVendor[CL_STRING_LENGTH];
    char platformVersion[CL_STRING_LENGTH];

    char gpuDeviceName[CL_STRING_LENGTH];
    char gpuDeviceVersion[CL_STRING_LENGTH];
    int  gpuDeviceComputeUnits;
    cl_ulong  gpuDeviceGlobalMem;
    cl_ulong  gpuDeviceLocalMem;

    cl_uint             numPlatforms;           // OpenCL platform
    cl_platform_id*     cpPlatforms;

    cl_uint             numGpuDevices;          // OpenCL Gpu device
    cl_device_id*       cdGpuDevices;

    cl_context          cxGpuContext;           // OpenCL Gpu context
    cl_command_queue    ocl_command_queue;      // OpenCL Gpu command queues

    bool profiling = true;

    // platform
    err = basicCL.getNumPlatform(&numPlatforms);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    printf("platform number: %i.\n", numPlatforms);

    cpPlatforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);

    err = basicCL.getPlatformIDs(cpPlatforms, numPlatforms);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    for (unsigned int i = 0; i < numPlatforms; i++)
    {
        err = basicCL.getPlatformInfo(cpPlatforms[i], platformVendor, platformVersion);
        if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

        // Gpu device
        err = basicCL.getNumGpuDevices(cpPlatforms[i], &numGpuDevices);

        if (numGpuDevices > 0)
        {
            cdGpuDevices = (cl_device_id *)malloc(numGpuDevices * sizeof(cl_device_id) );

            err |= basicCL.getGpuDeviceIDs(cpPlatforms[i], numGpuDevices, cdGpuDevices);

            err |= basicCL.getDeviceInfo(cdGpuDevices[device_id], gpuDeviceName, gpuDeviceVersion,
                                         &gpuDeviceComputeUnits, &gpuDeviceGlobalMem,
                                         &gpuDeviceLocalMem, NULL);
            if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

            printf("Platform [%i] Vendor: %s Version: %s\n", i, platformVendor, platformVersion);
            printf("Using GPU device: %s ( %i CUs, %lu kB local, %lu MB global, %s )\n",
                   gpuDeviceName, gpuDeviceComputeUnits,
                   gpuDeviceLocalMem / 1024, gpuDeviceGlobalMem / (1024 * 1024), gpuDeviceVersion);

            break;
        }
        else
        {
            continue;
        }
    }

    // Gpu context
    err = basicCL.getContext(&cxGpuContext, cdGpuDevices, numGpuDevices);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // Gpu commandqueue
    if (profiling)
        err = basicCL.getCommandQueueProfilingEnable(&ocl_command_queue, cxGpuContext, cdGpuDevices[device_id]);
    else
        err = basicCL.getCommandQueue(&ocl_command_queue, cxGpuContext, cdGpuDevices[device_id]);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    const char *ocl_source_code_sptrsv =
    "    #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable                                          \n"
    "    #pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable                                      \n"
    "    #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable                                                 \n"
    "    #pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable                                             \n"
    "                                                                                                                \n"
    "    #pragma OPENCL EXTENSION cl_khr_fp64 : enable                                                               \n"
    "                                                                                                                \n"
    "    #ifndef VALUE_TYPE                                                                                          \n"
    "    #define VALUE_TYPE float                                                                                    \n"
    "    #endif                                                                                                      \n"
    "    #define WARP_SIZE 64                                                                                        \n"
    "                                                                                                                \n"
    "    #define SUBSTITUTION_FORWARD 0                                                                              \n"
    "    #define SUBSTITUTION_BACKWARD 1                                                                             \n"
    "                                                                                                                \n"
    "    #define OPT_WARP_NNZ   1                                                                                    \n"
    "    #define OPT_WARP_RHS   2                                                                                    \n"
    "    #define OPT_WARP_AUTO  3                                                                                    \n"
    "                                                                                                                \n"
    "    inline                                                                                                      \n"
    "    void atom_add_d_fp32(volatile __global float *val,                                                          \n"
    "                       float delta)                                                                             \n"
    "    {                                                                                                           \n"
    "        union { float f; unsigned int i; } old;                                                                 \n"
    "        union { float f; unsigned int i; } new;                                                                 \n"
    "        do                                                                                                      \n"
    "        {                                                                                                       \n"
    "            old.f = *val;                                                                                       \n"
    "            new.f = old.f + delta;                                                                              \n"
    "        }                                                                                                       \n"
    "        while (atomic_cmpxchg((volatile __global unsigned int *)val, old.i, new.i) != old.i);                   \n"
    "    }                                                                                                           \n"
    "                                                                                                                \n"
    "    inline                                                                                                      \n"
    "    void atom_add_d_fp64(volatile __global double *val,                                                         \n"
    "                       double delta)                                                                            \n"
    "    {                                                                                                           \n"
    "        union { double f; ulong i; } old;                                                                       \n"
    "        union { double f; ulong i; } new;                                                                       \n"
    "        do                                                                                                      \n"
    "        {                                                                                                       \n"
    "            old.f = *val;                                                                                       \n"
    "            new.f = old.f + delta;                                                                              \n"
    "        }                                                                                                       \n"
    "        while (atom_cmpxchg((volatile __global ulong *)val, old.i, new.i) != old.i);                            \n"
    "    }                                                                                                           \n"
    "    inline                                                                                                      \n"
    "    void atom_add_s_fp32(volatile __local float *val,                                                           \n"
    "                       float delta)                                                                             \n"
    "    {                                                                                                           \n"
    "        union { float f; unsigned int i; } old;                                                                 \n"
    "        union { float f; unsigned int i; } new;                                                                 \n"
    "        do                                                                                                      \n"
    "        {                                                                                                       \n"
    "            old.f = *val;                                                                                       \n"
    "            new.f = old.f + delta;                                                                              \n"
    "        }                                                                                                       \n"
    "        while (atomic_cmpxchg((volatile __local unsigned int *)val, old.i, new.i) != old.i);                    \n"
    "    }                                                                                                           \n"
    "                                                                                                                \n"
    "    inline                                                                                                      \n"
    "    void atom_add_d_fp(volatile __global VALUE_TYPE *address,                                                   \n"
    "                       VALUE_TYPE val)                                                                          \n"
    "    {                                                                                                           \n"
    "        if (sizeof(VALUE_TYPE) == 8)                                                                            \n"
    "            atom_add_d_fp64(address, val);                                                                      \n"
    "        else                                                                                                    \n"
    "            atom_add_d_fp32(address, val);                                                                      \n"
    "    }                                                                                                           \n"
    "                                                                                                                \n"
    "    inline                                                                                                      \n"
    "    void atom_add_s_fp64(volatile __local double *val,                                                          \n"
    "                       double delta)                                                                            \n"
    "    {                                                                                                           \n"
    "        union { double f; ulong i; } old;                                                                       \n"
    "        union { double f; ulong i; } new;                                                                       \n"
    "        do                                                                                                      \n"
    "        {                                                                                                       \n"
    "            old.f = *val;                                                                                       \n"
    "            new.f = old.f + delta;                                                                              \n"
    "        }                                                                                                       \n"
    "        while (atom_cmpxchg((volatile __local ulong *)val, old.i, new.i) != old.i);                             \n"
    "    }                                                                                                           \n"
    "                                                                                                                \n"
    "    __kernel                                                                                                    \n"
    "    void sptrsv_syncfree_opencl_analyser(__global const int      *d_cscRowIdx,                                  \n"
    "                                         const int                m,                                            \n"
    "                                         const int                nnz,                                          \n"
    "                                         __global int            *d_graphInDegree)                              \n"
    "    {                                                                                                           \n"
    "        const int global_id = get_global_id(0);                                                                 \n"
    "        if (global_id < nnz)                                                                                    \n"
    "        {                                                                                                       \n"
    "            atomic_fetch_add_explicit((atomic_int*)&d_graphInDegree[d_cscRowIdx[global_id]], 1,                 \n"
    "                                      memory_order_acq_rel, memory_scope_device);                               \n"
    "        }                                                                                                       \n"
    "    }                                                                                                           \n"
    "                                                                                                                \n"
    "    __kernel                                                                                                    \n"
    "    void sptrsv_syncfree_opencl_executor(__global const int            *d_cscColPtr,                            \n"
    "                                         __global const int            *d_cscRowIdx,                            \n"
    "                                         __global const VALUE_TYPE     *d_cscVal,                               \n"
    "                                         __global volatile int         *d_graphInDegree,                        \n"
    "                                         __global volatile VALUE_TYPE  *d_left_sum,                             \n"
    "                                         const int                      m,                                      \n"
    "                                         const int                      substitution,                           \n"
    "                                         __global const VALUE_TYPE     *d_b,                                    \n"
    "                                         __global VALUE_TYPE           *d_x,                                    \n"
    "                                         __local volatile int          *s_graphInDegree,                        \n"
    "                                         __local volatile VALUE_TYPE   *s_left_sum,                             \n"
    "                                         const int                      warp_per_block)                         \n"
    "    {                                                                                                           \n"
    "        const int global_id = get_global_id(0);                                                                 \n"
    "        const int local_id = get_local_id(0);                                                                   \n"
    "        int global_x_id = global_id / WARP_SIZE;                                                                \n"
    "        if (global_x_id >= m) return;                                                                           \n"
    "                                                                                                                \n"
    "        // substitution is forward or backward                                                                  \n"
    "        global_x_id = substitution == SUBSTITUTION_FORWARD ?                                                    \n"
    "                      global_x_id : m - 1 - global_x_id;                                                        \n"
    "                                                                                                                \n"
    "        // Initialize                                                                                           \n"
    "        const int local_warp_id = local_id / WARP_SIZE;                                                         \n"
    "        const int lane_id = (WARP_SIZE - 1) & local_id;                                                         \n"
    "        int starting_x = (global_id / (warp_per_block * WARP_SIZE)) * warp_per_block;                           \n"
    "        starting_x = substitution == SUBSTITUTION_FORWARD ?                                                     \n"
    "                      starting_x : m - 1 - starting_x;                                                          \n"
    "                                                                                                                \n"
    "        // Prefetch                                                                                             \n"
    "        const int pos = substitution == SUBSTITUTION_FORWARD ?                                                  \n"
    "                        d_cscColPtr[global_x_id] : d_cscColPtr[global_x_id+1]-1;                                \n"
    "        const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];                                                  \n"
    "                                                                                                                \n"
    "        if (local_id < warp_per_block) { s_graphInDegree[local_id] = 1; s_left_sum[local_id] = 0; }             \n"
    "        barrier(CLK_LOCAL_MEM_FENCE);                                                                           \n"
    "                                                                                                                \n"
    "        // Consumer                                                                                             \n"
    "        int loads, loadd;                                                                                       \n"
    "        do {                                                                                                    \n"
    "            // busy-wait                                                                                        \n"
    "        }                                                                                                       \n"
    "        while ((loads = atomic_load_explicit((atomic_int*)&s_graphInDegree[local_warp_id],                      \n"
    "                                             memory_order_acquire, memory_scope_work_group)) !=                 \n"
    "               (loadd = atomic_load_explicit((atomic_int*)&d_graphInDegree[global_x_id],                        \n"
    "                                             memory_order_acquire, memory_scope_device)) );                     \n"
    "                                                                                                                \n"
    "        VALUE_TYPE xi = d_left_sum[global_x_id] + s_left_sum[local_warp_id];                                    \n"
    "        xi = (d_b[global_x_id] - xi) * coef;                                                                    \n"
    "                                                                                                                \n"
    "        // Producer                                                                                             \n"
    "        const int start_ptr = substitution == SUBSTITUTION_FORWARD ?                                            \n"
    "                              d_cscColPtr[global_x_id]+1 : d_cscColPtr[global_x_id];                            \n"
    "        const int stop_ptr  = substitution == SUBSTITUTION_FORWARD ?                                            \n"
    "                              d_cscColPtr[global_x_id+1] : d_cscColPtr[global_x_id+1]-1;                        \n"
    "        for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE) {                                    \n"
    "            const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);          \n"
    "            const int rowIdx = d_cscRowIdx[j];                                                                  \n"
    "            const bool cond = substitution == SUBSTITUTION_FORWARD ?                                            \n"
    "                            (rowIdx < starting_x + warp_per_block) : (rowIdx > starting_x - warp_per_block);    \n"
    "            if (cond) {                                                                                         \n"
    "                const int pos = substitution == SUBSTITUTION_FORWARD ?                                          \n"
    "                                rowIdx - starting_x : starting_x - rowIdx;                                      \n"
    "                if (sizeof(VALUE_TYPE) == 8)                                                                    \n"
    "                    atom_add_s_fp64(&s_left_sum[pos], xi * d_cscVal[j]);                                        \n"
    "                else                                                                                            \n"
    "                    atom_add_s_fp32(&s_left_sum[pos], xi * d_cscVal[j]);                                        \n"
    "                mem_fence(CLK_LOCAL_MEM_FENCE);                                                                 \n"
    "                atomic_fetch_add_explicit((atomic_int*)&s_graphInDegree[pos], 1,                                \n"
    "                                          memory_order_acquire, memory_scope_work_group);                       \n"
    "            }                                                                                                   \n"
    "            else {                                                                                              \n"
    "                atom_add_d_fp(&d_left_sum[rowIdx], xi * d_cscVal[j]);                                           \n"
    "                mem_fence(CLK_GLOBAL_MEM_FENCE);                                                                \n"
    "                atomic_fetch_sub_explicit((atomic_int*)&d_graphInDegree[rowIdx], 1,                             \n"
    "                                           memory_order_acquire, memory_scope_device);                          \n"
    "            }                                                                                                   \n"
    "        }                                                                                                       \n"
    "                                                                                                                \n"
    "        // Finish                                                                                               \n"
    "        if (!lane_id) d_x[global_x_id] = xi ;                                                                   \n"
    "    }                                                                                                           \n"
    "                                                                                                                \n"
    "    __kernel                                                                                                    \n"
    "    void sptrsm_syncfree_opencl_executor(__global const int            *d_cscColPtr,                            \n"
    "                                         __global const int            *d_cscRowIdx,                            \n"
    "                                         __global const VALUE_TYPE     *d_cscVal,                               \n"
    "                                         __global volatile int         *d_graphInDegree,                        \n"
    "                                         __global volatile VALUE_TYPE  *d_left_sum,                             \n"
    "                                         const int                      m,                                      \n"
    "                                         const int                      substitution,                           \n"
    "                                         const int                      rhs,                                    \n"
    "                                         const int                      opt,                                    \n"
    "                                         __global const VALUE_TYPE     *d_b,                                    \n"
    "                                         __global VALUE_TYPE           *d_x,                                    \n"
    "                                         const int                      warp_per_block)                         \n"
    "    {                                                                                                           \n"
    "        const int global_id = get_global_id(0);                                                                 \n"
    "        int global_x_id = global_id / WARP_SIZE;                                                                \n"
    "        if (global_x_id >= m) return;                                                                           \n"
    "                                                                                                                \n"
    "        // substitution is forward or backward                                                                  \n"
    "        global_x_id = substitution == SUBSTITUTION_FORWARD ?                                                    \n"
    "                      global_x_id : m - 1 - global_x_id;                                                        \n"
    "                                                                                                                \n"
    "        // Initialize                                                                                           \n"
    "        const int lane_id = (WARP_SIZE - 1) & get_local_id(0);                                                  \n"
    "                                                                                                                \n"
    "        // Prefetch                                                                                             \n"
    "        const int pos = substitution == SUBSTITUTION_FORWARD ?                                                  \n"
    "                        d_cscColPtr[global_x_id] : d_cscColPtr[global_x_id+1]-1;                                \n"
    "        const VALUE_TYPE coef = (VALUE_TYPE)1 / d_cscVal[pos];                                                  \n"
    "                                                                                                                \n"
    "        // Consumer                                                                                             \n"
    "        int loadd;                                                                                              \n"
    "        do {                                                                                                    \n"
    "            // busy-wait                                                                                        \n"
    "        }                                                                                                       \n"
    "        while (1 != (loadd = atomic_load_explicit((atomic_int*)&d_graphInDegree[global_x_id],                   \n"
    "                                             memory_order_acquire, memory_scope_device)) );                     \n"
    "                                                                                                                \n"
    "       for (int k = lane_id; k < rhs; k += WARP_SIZE)                                                           \n"
    "       {                                                                                                        \n"
    "           const int pos = global_x_id * rhs + k;                                                               \n"
    "           d_x[pos] = (d_b[pos] - d_left_sum[pos]) * coef;                                                      \n"
    "       }                                                                                                        \n"
    "                                                                                                                \n"
    "       // Producer                                                                                              \n"
    "       const int start_ptr = substitution == SUBSTITUTION_FORWARD ?                                             \n"
    "                             d_cscColPtr[global_x_id]+1 : d_cscColPtr[global_x_id];                             \n"
    "       const int stop_ptr  = substitution == SUBSTITUTION_FORWARD ?                                             \n"
    "                              d_cscColPtr[global_x_id+1] : d_cscColPtr[global_x_id+1]-1;                        \n"
    "                                                                                                                \n"
    "       if (opt == OPT_WARP_NNZ)                                                                                 \n"
    "       {                                                                                                        \n"
    "           for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)                                   \n"
    "           {                                                                                                    \n"
    "               const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);       \n"
    "               const int rowIdx = d_cscRowIdx[j];                                                               \n"
    "               for (int k = 0; k < rhs; k++)                                                                    \n"
    "                   atom_add_d_fp(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);      \n"
    "               mem_fence(CLK_GLOBAL_MEM_FENCE);                                                                 \n"
    "               atomic_fetch_sub_explicit((atomic_int*)&d_graphInDegree[rowIdx], 1,                              \n"
    "                                          memory_order_acquire, memory_scope_device);                           \n"
    "           }                                                                                                    \n"
    "       }                                                                                                        \n"
    "       else if (opt == OPT_WARP_RHS)                                                                            \n"
    "       {                                                                                                        \n"
    "           for (int jj = start_ptr; jj < stop_ptr; jj++)                                                        \n"
    "           {                                                                                                    \n"
    "               const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);       \n"
    "               const int rowIdx = d_cscRowIdx[j];                                                               \n"
    "               for (int k = lane_id; k < rhs; k+=WARP_SIZE)                                                     \n"
    "                   atom_add_d_fp(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);      \n"
    "               mem_fence(CLK_GLOBAL_MEM_FENCE);                                                                 \n"
    "               if (!lane_id)                                                                                    \n"
    "                   atomic_fetch_sub_explicit((atomic_int*)&d_graphInDegree[rowIdx], 1,                          \n"
    "                                             memory_order_acquire, memory_scope_device);                        \n"
    "           }                                                                                                    \n"
    "       }                                                                                                        \n"
    "       else if (opt == OPT_WARP_AUTO)                                                                           \n"
    "       {                                                                                                        \n"
    "           const int len = stop_ptr - start_ptr;                                                                \n"
    "                                                                                                                \n"
    "           if ((len <= rhs || rhs > 16) && len < 2048)                                                          \n"
    "           {                                                                                                    \n"
    "               for (int jj = start_ptr; jj < stop_ptr; jj++)                                                    \n"
    "               {                                                                                                \n"
    "                   const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);   \n"
    "                   const int rowIdx = d_cscRowIdx[j];                                                           \n"
    "                   for (int k = lane_id; k < rhs; k+=WARP_SIZE)                                                 \n"
    "                       atom_add_d_fp(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);  \n"
    "                   mem_fence(CLK_GLOBAL_MEM_FENCE);                                                             \n"
    "                   if (!lane_id)                                                                                \n"
    "                       atomic_fetch_sub_explicit((atomic_int*)&d_graphInDegree[rowIdx], 1,                      \n"
    "                                             memory_order_acquire, memory_scope_device);                        \n"
    "               }                                                                                                \n"
    "           }                                                                                                    \n"
    "           else                                                                                                 \n"
    "           {                                                                                                    \n"
    "               for (int jj = start_ptr + lane_id; jj < stop_ptr; jj += WARP_SIZE)                               \n"
    "               {                                                                                                \n"
    "                   const int j = substitution == SUBSTITUTION_FORWARD ? jj : stop_ptr - 1 - (jj - start_ptr);   \n"
    "                   const int rowIdx = d_cscRowIdx[j];                                                           \n"
    "                   for (int k = 0; k < rhs; k++)                                                                \n"
    "                       atom_add_d_fp(&d_left_sum[rowIdx * rhs + k], d_x[global_x_id * rhs + k] * d_cscVal[j]);  \n"
    "                   mem_fence(CLK_GLOBAL_MEM_FENCE);                                                             \n"
    "                   atomic_fetch_sub_explicit((atomic_int*)&d_graphInDegree[rowIdx], 1,                          \n"
    "                                             memory_order_acquire, memory_scope_device);                        \n"
    "               }                                                                                                \n"
    "           }                                                                                                    \n"
    "        }                                                                                                       \n"
    "    }                                                                                                           \n"
    "                                                                                                                \n";

    // Create the program
    cl_program          ocl_program_sptrsv;

    size_t source_size_sptrsv[] = { strlen(ocl_source_code_sptrsv)};

    ocl_program_sptrsv = clCreateProgramWithSource(cxGpuContext, 1, &ocl_source_code_sptrsv, source_size_sptrsv, &err);

    if(err != CL_SUCCESS) {printf("OpenCL clCreateProgramWithSource ERROR CODE = %i\n", err); return err;}

    // Build the program

    if (sizeof(VALUE_TYPE) == 8)
        err = clBuildProgram(ocl_program_sptrsv, 0, NULL, "-cl-std=CL2.0 -D VALUE_TYPE=double", NULL, NULL);
    else
        err = clBuildProgram(ocl_program_sptrsv, 0, NULL, "-cl-std=CL2.0 -D VALUE_TYPE=float", NULL, NULL);
    
    // Create kernels
    cl_kernel  ocl_kernel_sptrsv_analyser;
    cl_kernel  ocl_kernel_sptrsv_executor;
    cl_kernel  ocl_kernel_sptrsm_executor;
    ocl_kernel_sptrsv_analyser = clCreateKernel(ocl_program_sptrsv, "sptrsv_syncfree_opencl_analyser", &err);
    if(err != CL_SUCCESS) {printf("OpenCL clCreateKernel ERROR CODE = %i\n", err); return err;}
    ocl_kernel_sptrsv_executor = clCreateKernel(ocl_program_sptrsv, "sptrsv_syncfree_opencl_executor", &err);
    if(err != CL_SUCCESS) {printf("OpenCL clCreateKernel ERROR CODE = %i\n", err); return err;}
    ocl_kernel_sptrsm_executor = clCreateKernel(ocl_program_sptrsv, "sptrsm_syncfree_opencl_executor", &err);
    if(err != CL_SUCCESS) {printf("OpenCL clCreateKernel ERROR CODE = %i\n", err); return err;}

    // transfer host mem to device mem
    // Define pointers of matrix L, vector x and b
    cl_mem      d_cscColPtrTR;
    cl_mem      d_cscRowIdxTR;
    cl_mem      d_cscValTR;
    cl_mem      d_b;
    cl_mem      d_x;

    // Matrix L
    d_cscColPtrTR = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, (n+1) * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    d_cscRowIdxTR = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, nnzTR  * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    d_cscValTR    = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, nnzTR  * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    err = clEnqueueWriteBuffer(ocl_command_queue, d_cscColPtrTR, CL_TRUE, 0, (n+1) * sizeof(int), cscColPtrTR, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    err = clEnqueueWriteBuffer(ocl_command_queue, d_cscRowIdxTR, CL_TRUE, 0, nnzTR  * sizeof(int), cscRowIdxTR, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    err = clEnqueueWriteBuffer(ocl_command_queue, d_cscValTR, CL_TRUE, 0, nnzTR  * sizeof(VALUE_TYPE), cscValTR, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // Vector b
    d_b    = clCreateBuffer(cxGpuContext, CL_MEM_READ_ONLY, m * rhs * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    err = clEnqueueWriteBuffer(ocl_command_queue, d_b, CL_TRUE, 0, m * rhs * sizeof(VALUE_TYPE), b, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // Vector x
    d_x    = clCreateBuffer(cxGpuContext, CL_MEM_READ_WRITE, n * rhs * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    memset(x, 0, m  * sizeof(VALUE_TYPE));
    err = clEnqueueWriteBuffer(ocl_command_queue, d_x, CL_TRUE, 0, n * rhs * sizeof(VALUE_TYPE), x, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    //  - opencl syncfree sptrsv analysis start!
    printf(" - opencl syncfree sptrsv analysis start!\n");

    // malloc tmp memory to simulate atomic operations
    cl_mem d_graphInDegree;
    cl_mem d_graphInDegree_backup;
    d_graphInDegree = clCreateBuffer(cxGpuContext, CL_MEM_READ_WRITE, m * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    d_graphInDegree_backup = clCreateBuffer(cxGpuContext, CL_MEM_READ_WRITE, m * sizeof(int), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // memset d_graphInDegree to 0
    int *graphInDegree = (int *)malloc(m * sizeof(int));
    memset(graphInDegree, 0, m * sizeof(int));
    err = clEnqueueWriteBuffer(ocl_command_queue, d_graphInDegree, CL_TRUE, 0, m  * sizeof(int), graphInDegree, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = 128;
    int num_blocks = ceil ((double)nnzTR / (double)num_threads);

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    err  = clSetKernelArg(ocl_kernel_sptrsv_analyser, 0, sizeof(cl_mem), (void*)&d_cscRowIdxTR);
    err |= clSetKernelArg(ocl_kernel_sptrsv_analyser, 1, sizeof(cl_int), (void*)&m);
    err |= clSetKernelArg(ocl_kernel_sptrsv_analyser, 2, sizeof(cl_int), (void*)&nnzTR);
    err |= clSetKernelArg(ocl_kernel_sptrsv_analyser, 3, sizeof(cl_mem), (void*)&d_graphInDegree);

    double time_opencl_analysis = 0;
    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        // memset d_graphInDegree to 0
        err = clEnqueueWriteBuffer(ocl_command_queue, d_graphInDegree, CL_TRUE, 0, m  * sizeof(int), graphInDegree, 0, NULL, NULL);
        if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

        err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_sptrsv_analyser, 1,
                                     NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
        if(err != CL_SUCCESS) { printf("ocl_kernel_sptrsv_analyser kernel run error = %i\n", err); return err; }

        err = clWaitForEvents(1, &ceTimer);
        if(err != CL_SUCCESS) { printf("event error = %i\n", err); return err; }

        basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
        time_opencl_analysis += double(endTime - startTime) / 1000000.0;
    }

    time_opencl_analysis /= BENCH_REPEAT;

    printf("opencl syncfree SpTRSV analysis on L used %4.2f ms\n", time_opencl_analysis);

    //  - opencl syncfree sptrsv solve start!
    printf(" - opencl syncfree SpTRSV solve start!\n");

    // malloc tmp memory to collect a partial sum of each row
    cl_mem d_left_sum;
    d_left_sum = clCreateBuffer(cxGpuContext, CL_MEM_READ_WRITE, m * rhs * sizeof(VALUE_TYPE), NULL, &err);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // memset d_left_sum to 0
    int *left_sum = (int *)malloc(m * rhs * sizeof(VALUE_TYPE));
    memset(left_sum, 0, m * rhs * sizeof(VALUE_TYPE));
    err = clEnqueueWriteBuffer(ocl_command_queue, d_left_sum, CL_TRUE, 0, m * rhs * sizeof(VALUE_TYPE), left_sum, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // backup in-degree array, only used for benchmarking multiple runs
    err = clEnqueueCopyBuffer(ocl_command_queue, d_graphInDegree, d_graphInDegree_backup, 0, 0, m * sizeof(int), 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // step 5: solve L*y = x
    const int wpb = WARP_PER_BLOCK;

    err  = clSetKernelArg(ocl_kernel_sptrsv_executor, 0,  sizeof(cl_mem), (void*)&d_cscColPtrTR);
    err |= clSetKernelArg(ocl_kernel_sptrsv_executor, 1,  sizeof(cl_mem), (void*)&d_cscRowIdxTR);
    err |= clSetKernelArg(ocl_kernel_sptrsv_executor, 2,  sizeof(cl_mem), (void*)&d_cscValTR);
    err |= clSetKernelArg(ocl_kernel_sptrsv_executor, 3,  sizeof(cl_mem), (void*)&d_graphInDegree);
    err |= clSetKernelArg(ocl_kernel_sptrsv_executor, 4,  sizeof(cl_mem), (void*)&d_left_sum);
    err |= clSetKernelArg(ocl_kernel_sptrsv_executor, 5,  sizeof(cl_int), (void*)&m);
    err |= clSetKernelArg(ocl_kernel_sptrsv_executor, 6,  sizeof(cl_int), (void*)&substitution);
    err |= clSetKernelArg(ocl_kernel_sptrsv_executor, 7,  sizeof(cl_mem), (void*)&d_b);
    err |= clSetKernelArg(ocl_kernel_sptrsv_executor, 8,  sizeof(cl_mem), (void*)&d_x);
    err |= clSetKernelArg(ocl_kernel_sptrsv_executor, 9,  sizeof(cl_int) * WARP_PER_BLOCK, NULL);
    err |= clSetKernelArg(ocl_kernel_sptrsv_executor, 10, sizeof(VALUE_TYPE) * WARP_PER_BLOCK, NULL);
    err |= clSetKernelArg(ocl_kernel_sptrsv_executor, 11, sizeof(cl_int), (void*)&wpb);

    err  = clSetKernelArg(ocl_kernel_sptrsm_executor, 0,  sizeof(cl_mem), (void*)&d_cscColPtrTR);
    err |= clSetKernelArg(ocl_kernel_sptrsm_executor, 1,  sizeof(cl_mem), (void*)&d_cscRowIdxTR);
    err |= clSetKernelArg(ocl_kernel_sptrsm_executor, 2,  sizeof(cl_mem), (void*)&d_cscValTR);
    err |= clSetKernelArg(ocl_kernel_sptrsm_executor, 3,  sizeof(cl_mem), (void*)&d_graphInDegree);
    err |= clSetKernelArg(ocl_kernel_sptrsm_executor, 4,  sizeof(cl_mem), (void*)&d_left_sum);
    err |= clSetKernelArg(ocl_kernel_sptrsm_executor, 5,  sizeof(cl_int), (void*)&m);
    err |= clSetKernelArg(ocl_kernel_sptrsm_executor, 6,  sizeof(cl_int), (void*)&substitution);
    err |= clSetKernelArg(ocl_kernel_sptrsm_executor, 7,  sizeof(cl_int), (void*)&rhs);
    err |= clSetKernelArg(ocl_kernel_sptrsm_executor, 8,  sizeof(cl_int), (void*)&opt);
    err |= clSetKernelArg(ocl_kernel_sptrsm_executor, 9,  sizeof(cl_mem), (void*)&d_b);
    err |= clSetKernelArg(ocl_kernel_sptrsm_executor, 10, sizeof(cl_mem), (void*)&d_x);
    err |= clSetKernelArg(ocl_kernel_sptrsm_executor, 11, sizeof(cl_int), (void*)&wpb);

    double time_opencl_solve = 0;
    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        // set d_graphInDegree to initial values
        err = clEnqueueCopyBuffer(ocl_command_queue, d_graphInDegree_backup, d_graphInDegree, 0, 0, m * sizeof(int), 0, NULL, NULL);
        if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

        // memset d_left_sum to 0
        err = clEnqueueWriteBuffer(ocl_command_queue, d_left_sum, CL_TRUE, 0, m * rhs * sizeof(VALUE_TYPE), left_sum, 0, NULL, NULL);
        if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

        if (rhs == 1)
        {
            num_threads = WARP_PER_BLOCK * WARP_SIZE;
            num_blocks = ceil ((double)m / (double)(num_threads/WARP_SIZE));
            szLocalWorkSize[0]  = num_threads;
            szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

            err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_sptrsv_executor, 1,
                                         NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
            if(err != CL_SUCCESS) { printf("ocl_kernel_sptrsv_executor kernel run error = %i\n", err); return err; }
        }
        else
        {
            num_threads = 1 * WARP_SIZE;
            num_blocks = ceil ((double)m / (double)(num_threads/WARP_SIZE));
            szLocalWorkSize[0]  = num_threads;
            szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

            err = clEnqueueNDRangeKernel(ocl_command_queue, ocl_kernel_sptrsm_executor, 1,
                                         NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, &ceTimer);
            if(err != CL_SUCCESS) { printf("ocl_kernel_sptrsm_executor kernel run error = %i\n", err); return err; }
        }

        err = clWaitForEvents(1, &ceTimer);
        if(err != CL_SUCCESS) { printf("event error = %i\n", err); return err; }

        basicCL.getEventTimer(ceTimer, &queuedTime, &submitTime, &startTime, &endTime);
        time_opencl_solve += double(endTime - startTime) / 1000000.0;
    }

    time_opencl_solve /= BENCH_REPEAT;
    double flop = 2*(double)rhs*(double)nnzTR;

    printf("opencl syncfree SpTRSV solve used %4.2f ms, throughput is %4.2f gflops\n",
           time_opencl_solve, flop/(1e6*time_opencl_solve));
    *gflops = flop/(1e6*time_opencl_solve);

    err = clEnqueueReadBuffer(ocl_command_queue, d_x, CL_TRUE, 0, n * rhs * sizeof(VALUE_TYPE), x, 0, NULL, NULL);
    if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    // validate x
    double accuracy = 1e-4;
    double ref = 0.0;
    double res = 0.0;

    for (int i = 0; i < n * rhs; i++)
    {
        ref += abs(x_ref[i]);
        res += abs(x[i] - x_ref[i]);
        //if (x_ref[i] != x[i]) printf ("[%i, %i] x_ref = %f, x = %f\n", i/rhs, i%rhs, x_ref[i], x[i]);
    }
    res = ref == 0 ? res : res / ref;

    if (res < accuracy)
        printf("opencl syncfree SpTRSV executor passed! |x-xref|/|xref| = %8.2e\n", res);
    else
        printf("opencl syncfree SpTRSV executor _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);

    // step 6: free resources
    free(graphInDegree);

    if(d_graphInDegree) err = clReleaseMemObject(d_graphInDegree); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_graphInDegree_backup) err = clReleaseMemObject(d_graphInDegree_backup); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_left_sum) err = clReleaseMemObject(d_left_sum); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    if(d_cscColPtrTR) err = clReleaseMemObject(d_cscColPtrTR); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_cscRowIdxTR) err = clReleaseMemObject(d_cscRowIdxTR); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_cscValTR)    err = clReleaseMemObject(d_cscValTR); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_b) err = clReleaseMemObject(d_b); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}
    if(d_x) err = clReleaseMemObject(d_x); if(err != CL_SUCCESS) {printf("OpenCL ERROR CODE = %i\n", err); return err;}

    return 0;
}

#endif



