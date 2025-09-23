# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2025
# Assignment 0
#
# Do not rename/delete any functions or global variables provided in this template. Write your implementation
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that test code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages not specified in the
# assignment, you will need to obtain approval from the course staff.
# This part of the assignment will be graded automatically using Gradescope.
# =========================================================================================================


# Function to return the sum of the first n positive odd numbers
# n: the number of initial odd numbers to sum
# Returns: sum as an integer
def sum_odd(n):
    if n <= 0:
        return 0
    total = 0
    for i in range(n):
        total += 2 * i + 1
    return total

# Function to calculate the sum of the first N Fibonacci numbers
# n: the number of initial Fibonacci numbers to sum
# Returns: sum as an integer
def sum_fib(n):
    if n <= 0:
        return 0
    a, b = 0, 1
    total = 0
    for _ in range(n):
        total += a
        a, b = b, a + b
    return total


# Use this main function to test your code. Sample code is provided to assist with the assignment;
# feel free to change/remove it. If you want, you may run the code from the terminal as:
# python loops.py
# It should produce the following output (with correct solution):
# 	    $ python loops.py
#       The sum of the first 5 positive odd numbers is: 25
#       The sum of the first 5 fibonacci numbers is: 7

def main():
    # Call the function to calculate sum
    osum = sum_odd(5) 

    # Print it out
    print(f'The sum of the first 5 positive odd numbers is: {osum}')

    # Call the function to calculate sum of fibonacci numbers
    fsum = sum_fib(5)
    print(f'The sum of the first 5 fibonacci numbers is: {fsum}')

################ Do not make any changes below this line ################
if __name__ == '__main__':
    main()