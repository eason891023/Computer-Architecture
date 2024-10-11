#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

// 定義 bfloat16 結構
typedef struct {
    uint16_t bits;
} bf16_t;


// transfer fp32 to bf16
static inline bf16_t fp32_to_bf16(float s) {
    bf16_t h;
    union {
        float f;
        uint32_t i;
    } u = {.f = s};
    if ((u.i & 0x7fffffff) > 0x7f800000) { /* NaN */
        h.bits = (u.i >> 16) | 64;         /* force to quiet */
        return h;                                                                                                                                             
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 0x10) & 1))) >> 0x10;
    return h;
}

// floating point single number
float findSingleNumber(float* nums, int numsSize) {
    uint32_t res = 0;
    for (int i = 0; i < numsSize; i++) {
        bf16_t bf_num = fp32_to_bf16(nums[i]); 
        res ^= bf_num.bits;
    }

    union {
        uint32_t i;
        float f;
    } result = {.i = res << 16}; //return FP32 value
    return result.f;
}

void print_float_bits(float f) {
    union {
        float f;
        uint32_t i;
    } u = {.f = f};
    printf("Float: %f, Bits: 0x%08x, ", f, u.i);
    bf16_t bf = fp32_to_bf16(f);
    printf("BF16 bits: 0x%04x\n", bf.bits);
}

void run_test_case(float* nums, int size, float expected) {
    printf("\nTest Case:\n");
    printf("Input array: ");
    for (int i = 0; i < size; i++) {
        printf("%f ", nums[i]);
    }
    printf("\n\nDetailed bits for each number:\n");
    for (int i = 0; i < size; i++) {
        print_float_bits(nums[i]);
    }
    
    float result = findSingleNumber(nums, size);
    printf("\nExpected single number: %f\n", expected);
    printf("Actual result: %f\n", result);
    
    // Compare bits instead of float values directly
    bf16_t expected_bf = fp32_to_bf16(expected);
    bf16_t result_bf = fp32_to_bf16(result);
    assert(expected_bf.bits == result_bf.bits);
    printf("Test passed!\n");
}

int main() {
    // Test Case 1: Simple case with integers
    {
        float test1[] = {1.0f, 2.0f, 1.0f, 2.0f, 3.0f};
        printf("\n=== Test Case 1: Simple integers ===");
        run_test_case(test1, 5, 3.0f);
    }

    // Test Case 2: Fractional numbers
    {
        float test2[] = {1.5f, 2.5f, 1.5f, 2.5f, 3.75f};
        printf("\n=== Test Case 2: Fractional numbers ===");
        run_test_case(test2, 5, 3.75f);
    }

    // Test Case 3: Negative numbers
    {
        float test3[] = {-1.0f, -2.0f, -1.0f, -2.0f, -3.0f};
        printf("\n=== Test Case 3: Negative numbers ===");
        run_test_case(test3, 5, -3.0f);
    }

    // Test Case 4: Mixed positive and negative
    {
        float test4[] = {1.0f, -1.0f, 1.0f, -1.0f, 2.5f};
        printf("\n=== Test Case 4: Mixed positive and negative ===");
        run_test_case(test4, 5, 2.5f);
    }

    // Test Case 5: Larger numbers
    {
        float test5[] = {1000.0f, 2000.0f, 1000.0f, 2000.0f, 3000.0f};
        printf("\n=== Test Case 5: Larger numbers ===");
        run_test_case(test5, 5, 3000.0f);
    }

    // Test Case 6: Small numbers
    {
        float test6[] = {0.001f, 0.002f, 0.001f, 0.002f, 0.003f};
        printf("\n=== Test Case 6: Small numbers ===");
        run_test_case(test6, 5, 0.003f);
    }

    // Test Case 7: Special cases
    {
        float test7[] = {INFINITY, 1.0f, INFINITY, 1.0f, 2.0f};
        printf("\n=== Test Case 7: Special case with infinity ===");
        run_test_case(test7, 5, 2.0f);
    }

    printf("\nAll tests passed successfully!\n");
    return 0;
}
