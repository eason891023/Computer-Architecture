#include <stdio.h>
#include <stdint.h>
#include <math.h>

typedef struct {
    uint16_t bits;
} bf16_t;


static inline bf16_t fp32_to_bf16(float s)
{
    bf16_t h;
    union {
        float f;
        uint32_t i;
    } u = {.f = s};
    if ((u.i & 0x7fffffff) > 0x7f800000) { /* NaN /
        h.bits = (u.i >> 16) | 64;         / force to quiet */
        return h;                                                                                                                                             
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 0x10) & 1))) >> 0x10;
    return h;
}

void print_bits(float f, bf16_t bf) {
    union {
        float f;
        uint32_t i;
    } u = {.f = f};
    
    printf("Float: %f\n", f);
    printf("Float bits:  0x%08x\n", u.i);
    printf("BF16 bits:   0x%04x\n\n", bf.bits);
}

int main() {
    float test_cases[] = {
        0.0f,                    // Zero
        1.0f,                    // One
        -1.0f,                   // Negative one
        3.14159f,                // Pi
        INFINITY,                // Positive infinity
        -INFINITY,               // Negative infinity
        NAN,                     // Not a Number
        1.0e-20f,                // Very small number
        1.0e20f,                 // Very large number
        0x1.0p-126f,             // Smallest normalized float32
        0x1.0p-127f,             // Largest subnormal float32
        nextafterf(0.0f, 1.0f),  // Smallest positive float32
    };

    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    
    printf("Running %d test cases:\n\n", num_tests);
    
    for (int i = 0; i < num_tests; i++) {
        printf("Test case %d:\n", i + 1);
        bf16_t result = fp32_to_bf16(test_cases[i]);
        print_bits(test_cases[i], result);
    }
    
    return 0;
}