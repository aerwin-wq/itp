# Eloquent JavaScript Ch. 1-3 Notes

### Chapter 1: Values, Types, and Operators

* **What I knew:** I was already aware of JavaScript's aggressive type coercion. For example, how `false == 0` and `"" == 0` both evaluate to true because the operands are coerced to the same type before comparison. It's something you learn to be careful with, which is why I always default to using `===`.

* **What I didn't know:** I didn't realize that `||` and `&&` don't necessarily produce a boolean. They actually return the value of the operand that determined the outcome. For instance, `"cat" && "dog"` returns `"dog"`. This "short-circuiting" behavior is a clever way to set default values, like `let name = providedName || "default";`.

### Chapter 2: Program Structure

* **What I knew:** I was comfortable with binding and updating variables, and how a variable's scope is generally tied to the block (`{}`) it's defined in. The distinction between `let`/`const` (block-scoped) and the older `var` (function-scoped) was already clear to me.

* **What I didn't know:** I hadn't paid much attention to the "fall-through" behavior in `switch` statements. I learned that if you omit the `break` keyword, the program will just continue executing the code in the *next* case, regardless of whether it matches. It seems like a potential source of bugs, but I can see how it could be useful for situations where multiple cases should execute the same block of code.

### Chapter 3: Functions

* **What I knew:** I understood the difference between function declarations and function expressions. Declarations are hoisted to the top of their scope, so you can call them before they appear in the code. Expressions, on the other hand, are not, so they're only defined when the program execution reaches them.

* **What I didn't know:** The concept of **closure** was new to me. I hadn't fully grasped that an inner function "closes over" its parent function's environment, retaining access to its parent's variables even after the parent function has returned. This explains how a function can "remember" its lexical scope, which is a really powerful feature for more advanced patterns.
