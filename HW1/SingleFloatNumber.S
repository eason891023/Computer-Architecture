.data
# Fractional numbers
test_data0:
    .word 0x3fc00000 # Float: 1.500000, BF16 bits: 0x3fc0
    .word 0x40200000 # Float: 2.500000, BF16 bits: 0x4020
    .word 0x3fc00000 # Float: 1.500000, BF16 bits: 0x3fc0
    .word 0x40200000 # Float: 2.500000, BF16 bits: 0x4020
    .word 0x40700000 # Float: 3.500000, BF16 bits: 0x4070

# Negative numbers
test_data1:
    .word 0xc0200000 # Float: -2.500000, BF16 bits: 0xc020
    .word 0xc0700000 # Float: -3.500000, BF16 bits: 0xc070
    .word 0xc0200000 # Float: -2.500000, BF16 bits: 0xc020
    .word 0xc0700000 # Float: -3.500000, BF16 bits: 0xc070
    .word 0xc0a00000 # Float: -5.000000, BF16 bits: 0xc0a0

test_data2:
    .word 0x3727c5ac # Float: 0.000010, BF16 bits: 0x3728
    .word 0x37a7c5ac # Float: 0.000020, BF16 bits: 0x37a8
    .word 0x3727c5ac # Float: 0.000010, BF16 bits: 0x3728
    .word 0x37a7c5ac # Float: 0.000020, BF16 bits: 0x37a8
    .word 0x37fba882 # Float: 0.000030, BF16 bits: 0x37fc
    .word 0x37fba882 # Float: 0.000030, BF16 bits: 0x37fc
    .word 0x3851b717 # Float: 0.000050, BF16 bits: 0x3852

test_data_size:
    .word 5
    .word 5
    .word 7

golden_data0:
    .word 0x40700000 # Float:  3.500000, BF16 bits: 0x4070
golden_data1:
    .word 0xc0a00000 # Float: -5.000000, BF16 bits: 0xc0a0
golden_data2:
    .word 0x38520000 # Float:  0.000050, BF16 bits: 0x3852

test_data_count: .word 2

str0:    .string "=========test case:=========\n"
str1:    .string "Expected Answer: "
str2:    .string "Actual Answer: "
str3:    .string "=========Wrong Answer========\n"
str4:    .string "=========test passed=========\n"
newline: .string "\n"

.text
main:
    la s0, test_data_count
    lw s0, 0,(s0)       # s0 = total num of test data
    
    addi s1, x0, -1     # s1 = counter of data index

    la s2, test_data_size # s2 = address of test_data_size

    la s3, test_data0  # s3 = address of test_data

    la s4, golden_data0 # s4 = address of golden_data0

test_loop:

    addi s1, s1, 1      # s1 += 1

    la t0, test_data_size # t0 = address of test_data_size
    slli t1, s1, 2      # t1 = 4 * counter
    add s2, t0, t1      # t0 = address of test_data_size + 4 * counter
    lw a1, 0(s2)        # a1 = numsSize (array size)

    mv a0, s3          # a0 = address of testdata

    jal ra, findSingleNumber

    mv t1, a0          # t1 = my answer

    la a0, str0        # print : test case:
    li a7, 4           # System call code for printing a string
    ecall              # Print the string

    la a0, str1        # print : Expected Answer:
    li a7, 4           # System call code for printing a string
    ecall              # Print the string

    slli t2, s1, 2     # t1 = 4 * counter
    add  t0, s4, t2    # t0 = address of golden_data0 + 4 * counter
    lw t0, 0,(t0)      # t0 = golden_data
    mv a0, t0          # print : golden data
    li a7, 34          # System call code for printing a string with hexadecimal
    ecall              # Print the string

    la a0, newline     # change line
    li a7, 4           # System call code for printing a string
    ecall              # Print the string

    la a0, str2        # print : Actual Answer:
    li a7, 4           # System call code for printing a string
    ecall              # Print the string

    mv a0, t1          # print : my answer
    li a7, 34          # System call code for printing a string with hexadecimal
    ecall

    la a0, newline     # change line
    li a7, 4           # System call code for printing a string
    ecall              # Print the string

    bne t0,t1, test_wrong # if golden data != my answer, go to test_wrong, and shutdown
    la a0, str4        # print : Wrong Answer
    li a7, 4           # System call code for printing a string
    ecall              # Print the string

    # update test_data address
    slli t1, a1, 2      # t1 = 4 * total num of test data
    add  s3, s3, t1     # s3 = address of test_data + 4 * total num of test data

    blt s1, s0, test_loop   # if counter of data index < total num of test data, go to loop

    li a7, 10          # System call code for exiting the program
    ecall

findSingleNumber:
    # main function use s0, s1, s2, s3, ra
    addi sp, sp, -20          # push stack space
    sw ra, 16(sp)             # save return address
    sw s0, 12(sp)             # save s0
    sw s1, 8(sp)              # save s1
    sw s2, 4(sp)              # save s2
    sw s3, 0(sp)              # save s3

    li s2, 0                  # initialize res = 0
    mv s1, a1                 # s1 = numsSize (array size)
    mv s0, a0                 # s0 = inputs array address

loop_start:
    beqz s1, loop_end         # if numsSize == 0, end loop

    lw a0, 0(s0)              # load current float from nums[i]
    jal ra, fp32_to_bf16      # call fp32_to_bf16 to convert to bf16
    xor s2, s2, a0            # res ^= bf_num.bits
    # print answer for debugging
    # mv a0, a0          # print : my answer
    # li a7, 34          # System call code for printing a string with hexadecimal
    # ecall

    # la a0, newline     # change line
    # li a7, 4           # System call code for printing a string
    # ecall              # Print the string

    addi s0, s0, 4            # move to next element (float is 4 bytes)
    addi s1, s1, -1           # decrement numsSize
    j loop_start              # repeat the loop

loop_end:
    slli s2, s2, 16           # shift result left to form FP32
    mv a0, s2                 # move result to a0

    lw s3, 0(sp)              # restore s3
    lw s2, 4(sp)              # restore s2
    lw s1, 8(sp)              # restore s1
    lw s0, 12(sp)             # restore s0
    lw ra, 16(sp)             # restore return address
    addi sp, sp, 20           # restore stack pointer
    ret

fp32_to_bf16:
    addi sp, sp, -8           # push stack space
    sw ra, 4(sp)              # save return address
    sw s0, 0(sp)              # save s0

    mv s0, a0                 # move input (float s) to s0

    li t1, 0x7fffffff         # load 0x7fffffff to t1
    li t2, 0x7f800000         # load 0x7f800000 (FP32 INF) to t2
    and t3, s0, t1            # t3 = s0 & 0x7fffffff
    bgtu t3, t2, is_nan       # if (s0 & 0x7fffffff > 0x7f800000), go to NaN handler

    srli t2, s0, 0x10         # t2 = s0 >> 16  (u.i >> 0x10)
    andi t2, t2, 1            # t2 = t2 & 1    ((u.i >> 0x10) & 1)
    li t4, 0x7fff             # t4 = 0x7FFF    
    add t2, t2, t4            # t2 = t2 + 0x7FFF (0x7fff + ((u.i >> 0x10) & 1))
    add s0, s0, t2            # t2 = t2 + s0 (u.i + (0x7fff + ((u.i >> 0x10) & 1)))
    srli s0, s0, 0x10         # get the upper 16 bits
    mv a0, s0                 # move result to a0

    lw s0, 0(sp)              # restore s0
    lw ra, 4(sp)              # restore return address
    addi sp, sp, 8            # restore stack pointer
    ret

is_nan:
    srli s0, s0, 16           # s0 = s0 >> 16 (keep upper 16 bits)
    ori s0, s0, 0x0040        # force quiet NaN by setting bit 6
    mv a0, s0                 # move result to a0

    lw s0, 0(sp)              # restore s0
    lw ra, 4(sp)              # restore return address
    addi sp, sp, 8            # restore stack pointer
    ret

test_wrong:
    la a0, str3        # print : test passed
    li a7, 4           # System call code for printing a string
    ecall              # Print the string