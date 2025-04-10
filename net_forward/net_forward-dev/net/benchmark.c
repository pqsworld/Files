#include "alog.h"
#include "benchmark.h"
#if BENCHMARK

#include <stdio.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else // _WIN32
#include <sys/time.h>
#endif // _WIN32

double get_current_time()
{
#ifdef _WIN32
    LARGE_INTEGER freq;
    LARGE_INTEGER pc;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&pc);

    return pc.QuadPart * 1000.0 / freq.QuadPart;
#else  // _WIN32
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
#endif // _WIN32
}


void benchmark(const char* name, double start, double end)
{
    SL_LOGD("%-30s %8.2lfms", name, end - start);
}


#endif // BENCHMARK