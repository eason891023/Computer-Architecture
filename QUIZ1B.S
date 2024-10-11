.data
test_data: 
    .word 0x00000000 # 0
    .word 0x3f800000 # 1
    .word 0xbf800000 # -1
    .word 0x7f800000 # INF
    .word 0xff800000 # -INF
    .word 0x07fc0000 # NAN
    .word 0x7f7fffff # Largest Number 
    .word 0xff7fffff # Smallist Number
    .word 0x00800000 # Minimum normalized positive value
    .word 0x00000001 # Minimum subnormal positive value


test_data_count: .word 9

str0:    .string "test case:\n"
str1:    .string "fp32 value is "
str2:    .string "bf16 value is "
newline: .string "\n"

.text
main:
    la t4, test_data_count
    lw t4, 0,(t4) # t4 = total num of test data
    
    addi t3, x0, -1 # t3 = counter of data index
    
loop:
    addi t3, t3, 1     # t3 += 1
    la t0, test_data   # t0 = address of test_data
    slli t6, t3, 2     # t6 = 4 * counter
    add t0, t0, t6     # t0 = address of test_data + 4 * counter
    lw a0, 0(t0)       # a0 = testdata
    mv t5, a0          # t5 = origin fp32 testdata
    jal ra, fp32_to_bf16
    
    mv t0, a0          # t0 = ans:bf16
    
    la a0, str0        # print : test case:
    li a7, 4           # System call code for printing a string
    ecall              # Print the string
    
    la a0, str1        # print : fp32 value is:
    li a7, 4           # System call code for printing a string
    ecall              # Print the string
    
    mv a0,t5           # print : origin fp32 testdata
    li a7, 34          # System call code for printing a string with hexadecimal
    ecall              # Print the string
    
    la a0, newline     # change line
    li a7, 4           # System call code for printing a string
    ecall              # Print the string
    
    la a0, str2        # print : bf16 value is:
    li a7, 4           # System call code for printing a string
    ecall              # Print the string
    
    mv a0,t0           # print : bf16 data
    li a7, 34          # System call code for printing a string with hexadecimal
    ecall
    
    la a0, newline     # change line
    li a7, 4           # System call code for printing a string
    ecall              # Print the string    
    
    blt t3, t4, loop   # if counter of data index < total num of test data, go to loop
    
    li a7, 10          # System call code for exiting the program
    ecall  
    
fp32_to_bf16:    
    addi sp, sp, -8     # push stack pointer
    sw ra, 4(sp)        # store return address
    sw s0, 0(sp)        # store s0

    mv s0, a0           # s0 = s (input)
    # Check if s is NAN
    li t1, 0x7f800000   # FP32 NAN
    li t2, 0x7fffffff
    and t2, s0, t2      # t2 = s & 0x7fffffff
    bgtu t2, t1, is_nan # if (s & 0x7fffffff > 0x7f800000), go to NAN
    
    
    srli t1, s0, 16     # t1 = s0 >> 16 
    li t2, 0x00008000  
    and t2, s0,t2        
    add s0, s0, t2      
    srli s0, s0, 16     
    
    mv a0, s0      
    
    lw s0, 0(sp)
    lw ra, 4(sp)       
    addi sp, sp, 8       
    ret

is_nan:
    srli s0, s0, 16         # s0 = s0 >> 16
    ori s0, s0, 0x0040      # s0 = s0 | 64
    
    mv a0, s0               # ao = s0

    lw s0, 0(sp)
    lw ra, 4(sp)           
    addi sp, sp, 8   
    ret
