#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

typedef struct {
    long score;
    ulong seed;
    long a, b;
} Result;

#define HASH_BATCH_SIZE 4
#define SCORE_CUTOFF 50
#define XL 0x9E3779B97F4A7C15UL
#define XH 0x6A09E667F3BCC909UL
#define XL_BASE (XL * HASH_BATCH_SIZE)

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
    ulong t = prng->hi ^ prng->lo;
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
    int low = (int)(prng_next64(prng) >> 32);
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

inline void processFullPrngState(ulong xseed, __global Result* results, volatile __global int* result_idx) {
    long a, b;
    compute_ab(xseed, &a, &b);

    long score = 0;
    ulong x;
    int tz;

    x = (ulong)a ^ (ulong)b;
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-a) ^ (ulong)b;
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)a ^ (ulong)(3 * b);
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-a) ^ (ulong)(3 * b);
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(3 * a) ^ (ulong)b;
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-3 * a) ^ (ulong)b;
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)a ^ (ulong)(5 * b);
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-a) ^ (ulong)(5 * b);
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(5 * a) ^ (ulong)b;
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-5 * a) ^ (ulong)b;
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(3 * a) ^ (ulong)(5 * b);
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-3 * a) ^ (ulong)(5 * b);
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(5 * a) ^ (ulong)(3 * b);
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-5 * a) ^ (ulong)(3 * b);
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)a ^ (ulong)(7 * b);
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-a) ^ (ulong)(7 * b);
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(7 * a) ^ (ulong)b;
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-7 * a) ^ (ulong)b;
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(3 * a) ^ (ulong)(7 * b);
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-3 * a) ^ (ulong)(7 * b);
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(7 * a) ^ (ulong)(3 * b);
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-7 * a) ^ (ulong)(3 * b);
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(5 * a) ^ (ulong)(7 * b);
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-5 * a) ^ (ulong)(7 * b);
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(7 * a) ^ (ulong)(5 * b);
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    x = (ulong)(-7 * a) ^ (ulong)(5 * b);
    tz = x ? (64 - clz(x & (-x))) - 1 : 64;
    score = tz > score ? tz : score;

    
    if (score < SCORE_CUTOFF)
        return;

    ulong seed = xseed ^ XH;
    int this_result_idx = atomic_add(result_idx, 1);
    if (this_result_idx < 8) {  // Prevent buffer overflow
        results[this_result_idx].score = score;
        results[this_result_idx].seed = seed;
        results[this_result_idx].a = a;
        results[this_result_idx].b = b;
    }
}

__kernel void searchKernel(ulong start_seed, __global Result* results, 
                          volatile __global int* result_idx, volatile __global uint* checksum) {
    ulong gid = get_global_id(0);
    ulong seed_base = (start_seed + gid) * XL_BASE;

    ulong hashes[HASH_BATCH_SIZE + 1];
    for (int i = 0; i <= HASH_BATCH_SIZE; i++)
        hashes[i] = mix64(seed_base + i*XL);

    for (int i = 0; i < HASH_BATCH_SIZE; i++) {
        PRNG128 prng = prng_init2(hashes[i], hashes[i+1]);
        if (!goodLower32(&prng))
            continue;
        ulong curr_s = seed_base + i * XL;
        processFullPrngState(curr_s, results, result_idx);
        atomic_add(checksum, 1);  // Changed from atomic_inc
    }
}