# FizzBuzz

## Overview
This program is a simple implementation of the classic **FizzBuzz** problem in JavaScript.  
It loops through the numbers **1 to 100** and prints either the number itself or a specific word depending on divisibility rules.

## How It Works
- The program uses a `for` loop to iterate from **1 through 100**.
- For each number (`i`), it checks the following conditions in order:

1. **Divisible by both 3 and 5**  
   - If `i % 3 === 0 && i % 5 === 0`  
   - Prints `"FizzBuzz"`

2. **Divisible by only 3**  
   - If `i % 3 === 0`  
   - Prints `"Fizz"`

3. **Divisible by only 5**  
   - If `i % 5 === 0`  
   - Prints `"Buzz"`

4. **Not divisible by 3 or 5**  
   - Prints the number `i`

## DOCUMENTATION:
I gave chatgpt this prompt: 
turn this js code into a .readme explaining what the code does: for (let i = 1; i <= 100; i++) { if (i % 3 === 0 && i % 5 === 0) { console.log("FizzBuzz"); } else if (i % 3 === 0) { console.log("Fizz"); } else if (i % 5 === 0) { console.log("Buzz"); } else { console.log(i); } }
