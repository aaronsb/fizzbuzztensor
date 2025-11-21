# I Solved FizzBuzz With Tensors and Accidentally Did Signal Processing

*What happens when you take a simple coding interview question way too seriously*

---

## The Question Everyone Loves to Hate

If you've ever interviewed for a software engineering role, you've probably encountered FizzBuzz. It's that classic problem: print numbers 1 to 100, but replace multiples of 3 with "Fizz", multiples of 5 with "Buzz", and multiples of both with "FizzBuzz".

Most people solve it with if-statements in about 30 seconds.

I spent an afternoon turning it into a **signal processing problem** and visualizing it as a waveform. Here's why that matters—whether you're hiring developers, learning AI, or just curious about how engineers think.

---

## It Started With a Question: "Why Not Tensors?"

I came across Susam Pal's brilliant article ["Fizz Buzz With Cosines"](https://susam.net/fizz-buzz-with-cosines.html) where he solved FizzBuzz using trigonometric functions. No loops, no conditionals—just pure math exploiting the periodic nature of divisibility.

That got me thinking: **If FizzBuzz is fundamentally a periodic function, why not represent it as a first-class tensor?**

So I did. And what I found was way more interesting than I expected.

---

## The "Aha!" Moment: FizzBuzz is Just a Pattern

Here's the core insight: FizzBuzz repeats every 15 numbers (that's the LCM of 3 and 5). This means the *entire infinite sequence* can be represented as a single 15-element vector:

```
[0, 0, 1, 0, 2, 1, 0, 0, 1, 2, 0, 1, 0, 0, 3]
```

That's it. That's FizzBuzz.

Every number from 1 to infinity maps to one of these 15 positions. Position 1? Print "1". Position 3? Print "Fizz". Position 15? Print "FizzBuzz". Then it repeats.

**From infinite complexity to 15 numbers.** That's the power of recognizing patterns.

---

## Why This Matters (For Different Audiences)

### For Hiring Managers & Recruiters

When evaluating developers, look for people who can:
- Recognize patterns and simplify complexity
- Think in multiple paradigms (procedural, functional, mathematical)
- Ask "why?" instead of just "how?"

Someone who looks at FizzBuzz and thinks "this is a periodic signal" is someone who will find optimization opportunities and architectural improvements in your codebase.

### For Data Scientists & AI Engineers

This demonstrates a fundamental concept in AI/ML: **representation learning**.

The way you represent data determines what patterns you can extract. FizzBuzz as if-statements is just procedural code. FizzBuzz as a tensor is a periodic signal with:
- Defined frequency spectrum (1/15 Hz fundamental)
- Component waveforms (divisibility by 3 and 5 are separate frequencies)
- Perfect compressibility (15 elements, infinite sequence)

This is similar to what transformers do with language, CNNs do with images, and embeddings do with categorical data—finding representations that make patterns explicit.

### For Developers & Learners

Cross-pollinating ideas from different domains (signal processing + basic programming) leads to unexpected insights:

1. **Periodicity is everywhere** - look for patterns that repeat
2. **Tensors aren't just for AI** - they're a useful abstraction for structured data
3. **Visualization reveals structure** - plotting FizzBuzz as a waveform made the two interfering frequencies (div by 3, div by 5) obvious
4. **Over-engineering as exploration** - sometimes taking the "wrong" approach teaches you more than the "right" one

### For the Curious

I literally turned FizzBuzz into:
- **A waveform** showing five periods of the pattern
- **An FFT spectrum** revealing its fundamental frequency
- **A 2D texture** that tiles perfectly

Because sometimes the best way to understand something is to look at it from a completely different angle.

---

## The Visualizations

Here's what FizzBuzz looks like as a signal:

![FizzBuzz Waveform](images/fizzbuzz_waveform.png)
*The pattern signal (top), component frequencies for divisibility by 3 and 5 (middle), and the binary divisibility matrix (bottom)*

![Frequency Spectrum](images/fizzbuzz_fft.png)
*Time domain (top) and frequency spectrum (bottom) showing the fundamental frequency at 1/15*

![2D Heatmap](images/fizzbuzz_2d.png)
*FizzBuzz as a 20×20 texture—notice the diagonal patterns where multiples of 15 align*

These aren't just pretty pictures. They **reveal** the mathematical structure:
- The two component signals interfere to create FizzBuzz
- The frequency spectrum proves it's periodic with period 15
- The 2D view shows spatial patterns invisible in 1D

**Good visualizations don't just display data—they create understanding.**

---

## What I Learned (Beyond FizzBuzz)

### 1. **Patterns Are Power**
Recognizing that FizzBuzz has period 15 turns an O(N) conditional problem into an O(1) lookup. In production code, pattern recognition can mean the difference between linear and constant time.

### 2. **Representation Matters**
As if-statements: FizzBuzz is procedural code.
As a pattern vector: FizzBuzz is a mathematical object with well-defined properties.
Same problem, completely different insights.

### 3. **Cross-Domain Thinking**
Applying signal processing concepts (periodicity, frequency analysis, Fourier transforms) to a programming problem revealed structure that pure algorithmic thinking missed.

### 4. **Overengineering Has Value**
Yes, this is massive overkill for FizzBuzz. But it taught me about:
- NumPy broadcasting
- Discrete Fourier transforms
- Tensor representations
- Signal visualization
- Pattern compression

**Sometimes the journey matters more than the destination.**

---

## The Technical Deep Dive

For those interested in the full mathematical treatment, I wrote a complete paper with:
- Formal proofs of periodicity and complexity
- Fourier analysis and trigonometric representations
- Generalization to arbitrary divisor sets (FizzBuzzBazz anyone?)
- Connection to embedding theory and representation learning

**Full paper**: [TensorFizzBuzz: A Signal Processing Approach](https://github.com/aaronsb/fizzbuzztensor/blob/main/docs/tensor-fizzbuzz-paper.md)

**Code repository**: https://github.com/aaronsb/fizzbuzztensor

All code is open source, fully commented, and includes visualization tools.

---

## The Bottom Line

**FizzBuzz isn't about printing numbers.**

It's about how you think about problems. Do you see:
- A sequence of if-statements?
- A periodic function?
- A discrete signal?
- A tensor lookup?
- A Fourier series?

**All of these are correct.** The question is: which representation gives you insight?

In software engineering, in data science, in AI—the ability to see the same problem from multiple angles is what separates good solutions from great ones.

---

## Your Turn

**Question for developers**: What's a problem you've solved where changing the representation completely changed the solution?

**Question for non-technical folks**: What's something in your field that looks complex but is actually just a pattern repeating?

**Question for everyone**: What interview question should I over-engineer next?

---

**Stats:**
- Pattern period: 15
- Space complexity: O(15) = O(1)
- Time per lookup: O(1)
- Fun had: Immeasurable

**Links:**
- GitHub: https://github.com/aaronsb/fizzbuzztensor
- Inspiration: https://susam.net/fizz-buzz-with-cosines.html
- Full paper: [tensor-fizzbuzz-paper.md](https://github.com/aaronsb/fizzbuzztensor/blob/main/docs/tensor-fizzbuzz-paper.md)

---

*"The most elemental solution to FizzBuzz is not an algorithm, but a number: 15."*

---

**#SoftwareEngineering #DataScience #MachineLearning #AI #ArtificialIntelligence #Programming #Python #TensorFlow #NumPy #SignalProcessing #TechInnovation #DeveloperLife #CodingLife #SoftwareDevelopment #TechCareers #HiringDevelopers #TechRecruiting #LearnToCode #DataVisualization #ComputerScience #AlgorithmDesign #TechEducation #DevCommunity #EngineeringExcellence #ProblemSolving #ThinkDifferent #TechTrends #FutureOfWork #Innovation #STEM #TechLeadership**

---

**About the Author:**
Sometimes you learn the most by taking a simple problem and asking "what if?" This is one of those times.

If you found this interesting, check out the full repository with working code, visualizations, and a complete mathematical analysis. And if you're hiring, remember: the best developers aren't the ones who know all the answers—they're the ones who ask interesting questions.

---

**P.S.** Yes, I know this is complete overkill for FizzBuzz. That was the point.
