# What Is This and Why Does It Exist?

An accessible introduction to TensorFizzBuzz for the curious, the skeptical, and the confused.

---

## First: What is FizzBuzz?

FizzBuzz is a simple programming problem that goes like this:

> Write a program that prints the numbers from 1 to 100. But for multiples of 3, print "Fizz" instead of the number. For multiples of 5, print "Buzz". For numbers that are multiples of both 3 and 5, print "FizzBuzz".

So the output looks like:
```
1
2
Fizz
4
Buzz
Fizz
7
8
Fizz
Buzz
11
Fizz
13
14
FizzBuzz
16
...
```

Most programmers can solve this in about 30 seconds using if-statements:

```python
for i in range(1, 101):
    if i % 15 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
```

Done. Problem solved.

---

## Why is FizzBuzz Famous as an Interview Question?

FizzBuzz became infamous in tech hiring for a surprising reason: **a significant number of job candidates couldn't solve it**.

In 2007, a programmer named Imran Ghory wrote about using FizzBuzz in interviews and found that many candidates with impressive resumes struggled with this simple problem. It became a litmus test: "Can this person actually write basic code?"

Over time, FizzBuzz became:
- **A screening tool**: Quick way to filter out people who can't program at all
- **A meme**: The quintessential "too easy" interview question that somehow still trips people up
- **A creative challenge**: How many different ways can you solve the simplest problem?

People have solved FizzBuzz in hundreds of programming languages, without conditionals, using regex, with CSS, in assembly language, and even by exploiting CPU branch prediction. It's become a playground for creative problem-solving.

---

## So... Why Tensors?

That's where this project comes in.

I came across Susam Pal's article ["Fizz Buzz With Cosines"](https://susam.net/fizz-buzz-with-cosines.html), where he solved FizzBuzz using trigonometric functions instead of if-statements. The key insight: **divisibility is periodic**. Multiples of 3 repeat every 3 numbers, multiples of 5 repeat every 5 numbers.

That made me wonder: if FizzBuzz is fundamentally about periodicity, why not represent it as a **periodic signal** using tensors?

So I tried it. And it worked. And then it got interesting.

---

## What Makes This Approach Fun/Interesting?

### 1. **It Reveals Hidden Structure**

The tensor approach exposes something that's not obvious from the if-statement version: **FizzBuzz has a period of 15**.

Because 15 is the least common multiple (LCM) of 3 and 5, the pattern repeats exactly every 15 numbers. This means the entire infinite sequence can be represented as a single 15-element vector:

```python
PATTERN = [0, 0, 1, 0, 2, 1, 0, 0, 1, 2, 0, 1, 0, 0, 3]
```

That's it. That's all of FizzBuzz, forever.

### 2. **It Changes How You Think About the Problem**

Traditional approach: "For each number, check conditions and decide what to print."

Tensor approach: "FizzBuzz is a pattern. Find the pattern, then just look it up."

This shift from **algorithmic thinking** (step-by-step instructions) to **declarative thinking** (describe the structure) is fundamental to many areas:
- Database queries (SQL describes what you want, not how to get it)
- Functional programming (describe transformations, not mutations)
- Machine learning (learn the pattern, don't hardcode the rules)

### 3. **It's Actually Useful for Understanding Other Concepts**

This isn't just an intellectual exercise. The tensor representation demonstrates:

- **Representation learning**: How you represent data changes what patterns you can extract
- **Vectorization**: How to compute many values in parallel instead of one-by-one loops
- **Periodicity analysis**: How to identify and exploit repeating patterns
- **Frequency domain thinking**: Viewing problems as signals with spectral properties

These concepts show up everywhere:
- Audio processing (signals and frequencies)
- Time series analysis (periodic patterns in data)
- Computer graphics (repeating textures)
- Compression algorithms (finding patterns to encode efficiently)

### 4. **It Makes Cool Visualizations**

When you view FizzBuzz as a signal, you can:
- Plot it as a waveform and see the pattern repeat
- Apply FFT and see its frequency spectrum (fundamental at 1/15 Hz)
- Render it as a 2D texture and see spatial patterns emerge

These visualizations aren't just pretty—they reveal the **two interfering frequencies** (divisibility by 3 and 5) that combine to create FizzBuzz.

---

## Why This Is Kind of Ridiculous

Let's be honest:

1. **It's massive overkill** for a problem that takes 5 lines of code
2. **It's slower to explain** than just writing the if-statements
3. **It uses advanced concepts** (tensors, FFT, signal processing) for a beginner problem
4. **Nobody would actually use this** to solve FizzBuzz in production

The traditional if-statement solution is:
- Easier to understand
- Easier to modify (what if we want different divisors?)
- More explicit about what's happening
- Perfectly fine

Using signal processing and tensor operations for FizzBuzz is like using a Formula 1 race car to go to the grocery store. Yes, it works. No, it's not practical.

---

## Why This Is Actually Not Ridiculous

But here's the thing:

1. **It demonstrates fundamental concepts** in an accessible context
   - Anyone can understand FizzBuzz
   - Not everyone knows how tensors/FFT work
   - This bridges the gap: "Here's a complex concept applied to something simple"

2. **Over-engineering is a learning tool**
   - You learn more by exploring than by stopping at "good enough"
   - Taking the "wrong" approach often teaches you why the "right" approach works
   - Constraints breed creativity; removing constraints breeds understanding

3. **It reveals connections between domains**
   - Programming problem ↔ Signal processing
   - Discrete math ↔ Continuous functions (Fourier series)
   - Algorithmic thinking ↔ Mathematical representation

   These connections aren't obvious until you look for them.

4. **The pattern recognition skill transfers**
   - Recognizing that FizzBuzz has period 15 is the same skill as:
     - Noticing that a database query repeats the same joins
     - Seeing that an algorithm recomputes the same values
     - Identifying that user behavior follows daily/weekly cycles

   **Finding patterns is a meta-skill** that applies everywhere.

5. **It's genuinely novel**
   - Thousands of FizzBuzz solutions exist
   - Very few approach it as a signal processing problem
   - Exploring unusual angles leads to unexpected insights

---

## The Real Point

This project isn't really about FizzBuzz. FizzBuzz is just the vehicle.

It's about:
- **Curiosity**: "What if I tried this differently?"
- **Exploration**: "Let me see where this goes..."
- **Connection**: "Oh, this is related to that other thing!"
- **Communication**: "Here's what I found, isn't this neat?"

The best learning often comes from taking something simple and asking "what if?" The best discoveries come from applying knowledge from one domain to problems in another.

Is it ridiculous to analyze FizzBuzz with Fourier transforms? Yes.

Is it ridiculous to practice creative problem-solving, learn new tools, and explore connections between concepts? No.

---

## Who This Is For

- **Students**: See how "useless" math (FFT, tensors) applies to real problems
- **Developers**: Exercise creative thinking and learn about vectorization
- **Data scientists**: See representation learning in a trivial context before applying it to complex ones
- **Curious people**: Enjoy the exploration of a simple idea taken seriously
- **Interviewers**: Remember that the best candidates ask "why?" not just "how?"

---

## The Bottom Line

FizzBuzz is trivial. The insights from exploring it differently are not.

The pattern vector approach is ridiculous and educational, over-engineered and elegant, unnecessary and insightful—all at once.

And sometimes that's exactly the point.

---

**Next**: Check out the [technical paper](tensor-fizzbuzz-paper.md) for the full mathematical treatment, or the [LinkedIn article](linkedin-article.md) for a social-media-friendly overview.

**Code**: https://github.com/aaronsb/fizzbuzztensor
