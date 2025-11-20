# Python Reading — Observations  
_Readings from **Think Python** by Allen Downey_  
Chapters 1, 3, 5, and 10  

For each chapter:  
- **One thing I already knew**  
- **One thing I didn’t know / hadn’t thought about**

---

## **Chapter 1 — The Way of the Program**

**Knew:**  
Most of the chapter was extremely familiar — basics like variables, expressions, and how Python runs code. I’ve been using Python long enough that the core “how a program executes” model wasn’t new.

**Didn’t Know:**  
The part that actually made me pause was the comparison between natural language ambiguity and the intentional rigidity of formal languages. I’ve never really thought about programming language design framed around eliminating ambiguity at all costs, even though I rely on that property constantly.

---

## **Chapter 3 — Functions**

**Knew:**  
Function definitions, parameters, returns, scoping rules — all standard stuff I already use daily. The mechanical parts of writing and calling functions were nothing new.

**Didn’t Know:**  
His early distinction between *pure* functions and *functions with side effects* was more emphasized than I expected. I’m used to thinking about that in the context of larger architectures or functional programming, but not at the “intro” level — it made me rethink how often I mix computation with side effects out of convenience.

---

## **Chapter 5 — Conditionals and Recursion**

**Knew:**  
Conditional flow and Boolean logic are things I’m already fluent with. I’ve also written recursive functions before, especially for tree/graph problems and certain math functions.

**Didn’t Know:**  
What I hadn’t noticed before is how Downey connects recursion to the idea of defining a concept *in terms of itself* rather than as a control-flow trick. The mathematical framing (base case + reduction step as a logical definition) clarified why certain recursive structures feel “natural” while others feel forced.

---

## **Chapter 10 — Lists**

**Knew:**  
List creation, slicing, mutability, and iteration are things I use constantly. Python’s list methods (`append`, `sort`, `pop`, etc.) were all familiar.

**Didn’t Know:**  
The section on aliasing — specifically how easily two references can unintentionally point to the same underlying list — was actually a good reminder. I knew it technically, but I hadn’t really thought about how many subtle bugs come from forgetting when Python copies vs. when it references.
