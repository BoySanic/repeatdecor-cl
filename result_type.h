/* result_type.h  (host *and* kernel) */

#ifndef RESULT_TYPE_H
#define RESULT_TYPE_H

/* 64-bit aliases that behave identically on every platform */
typedef long long          k_long;   // signed  64-bit
typedef unsigned long long k_ulong;  // unsigned 64-bit

/* exactly 32 bytes – identical layout on host and device */
#pragma pack(push, 1)
struct Result {
    k_long  score;
    k_ulong seed;
    k_long  a;
    k_long  b;
};
#pragma pack(pop)

static_assert(sizeof(Result) == 32, "Result must be 32 bytes");

#endif