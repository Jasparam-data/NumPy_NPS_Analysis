# Linear Algebra for Machine Learning — Complete Notes

---

## 1. What is Linear Algebra?

Linear algebra is the branch of mathematics that deals with **vectors**, **matrices**, and **linear transformations**.

The word **"linear"** means every variable has power 1 — no x², no curves, always straight lines.

The word
 **"algebra"** means manipulating equations to find unknowns.

So **linear algebra = solving equations with vectors and matrices where all variables stay power 1**.

---

## 2. The Four Building Blocks

| Name | What it is | Example | In ML |
|---|---|---|---|
| Scalar | one number | 3.14 | a single weight, learning rate |
| Vector | list of numbers | [0.9, 0.2, 0.7] | word embedding |
| Matrix | grid of numbers | [[1,2],[3,4]] | weight matrix W |
| Tensor | any dimensions | shape (224,224,3) | colour image |

**Tensor dimensions:**
- 0D = scalar (just a number)
- 1D = vector (list)
- 2D = matrix (grid)
- 3D = cube — e.g. colour image (height × width × 3 RGB channels)
- 4D = batch of images (batch × height × width × channels)
- 5D = batch of videos (batch × frames × height × width × channels)

---

## 3. Straight Lines and Equations

### y = mx + c (slope-intercept form)
- **m** = slope (steepness of the line)
- **c** = y-intercept (where line crosses y-axis)
- **slope = rise / run = change in y / change in x**

### Slope and angle connection
```
slope = tan(θ)   where θ is angle with x-axis

tan(0°)  = 0        → flat line
tan(30°) = 0.577    → gentle
tan(45°) = 1        → equal rise and run (45° diagonal)
tan(60°) = 1.732    → steep
tan(90°) = undefined → vertical line (division by zero!)
```

### Important: slope = 1 means
- rise = run
- 45° angle exactly
- In ML: weight = 1 means "pass input through unchanged" (identity)

### Special cases
```
normal line:     y = mx + c   (m is any number)
horizontal line: y = c        (m = 0)
vertical line:   x = k        (slope undefined — no m or c can describe it)
```

### Python for tan and arctan
```python
import math

# angle → slope
slope = math.tan(math.radians(45))   # = 1.0

# slope → angle
angle = math.degrees(math.atan(3))   # = 71.57°
angle = math.degrees(math.atan(9))   # = 83.66°

# numpy version (works on whole arrays)
import numpy as np
angles = np.degrees(np.arctan(np.array([1, 3, 9])))
# = [45.0, 71.57, 83.66]
```

---

## 4. Parallel and Perpendicular Lines

### Parallel lines
- Same slope (m₁ = m₂), different intercept
- Never meet, run forever side by side
- Example: y = 3x + 4 and y = 3x + 3 are parallel

### Perpendicular lines
- Slopes multiply to give −1: m₁ × m₂ = −1
- Meet at 90°

### How to check: L1: y = 3x + 4 and L2: 3y = 9x + 9
```
Simplify L2: divide by 3 → y = 3x + 3

L1: m = 3, c = 4
L2: m = 3, c = 3

Same slope (3 = 3), different intercept (4 ≠ 3) → PARALLEL
```

---

## 5. Vector Operations

### Addition
```
[1,2] + [3,4] = [4,6]   (add element by element)
```

### Scaling
```
2 × [1,2] = [2,4]   (multiply every element)
```

### Dot Product — THE most important operation in ML
```
[1,2] · [3,4] = (1×3) + (2×4) = 3 + 8 = 11

general: a · b = a1×b1 + a2×b2 + ... + an×bn
```

The dot product is:
- Every neuron firing
- Every attention score
- Every similarity check

### Magnitude (length of a vector)
```
|[3,4]| = √(3² + 4²) = √(9+16) = √25 = 5
```

---

## 6. Matrix Operations

### Multiply: A × B
**Rule: inner dimensions must match**
```
(m × n) × (n × p) = (m × p)
inner two numbers must be equal — they cancel
result = outer two numbers
```

Example:
```
(2×3) × (3×2) = (2×2)   ✓
(2×3) × (2×3) = ERROR   ✗  inner dims don't match
```

How each cell is calculated — **dot product of row × column**:
```
A = [[1,2,3],[4,5,6]]    B = [[7,8],[9,10],[11,12]]

C[0,0] = row0 of A · col0 of B
       = (1×7) + (2×9) + (3×11) = 7+18+33 = 58

C[0,1] = row0 of A · col1 of B
       = (1×8) + (2×10) + (3×12) = 8+20+36 = 64
```

### Transpose: Aᵀ
Flip rows and columns. Shape (m×n) becomes (n×m).
```
A = [[1,2,3],    →    Aᵀ = [[1,4],
     [4,5,6]]               [2,5],
                             [3,6]]

A[i][j] becomes Aᵀ[j][i]
```
Used in: Q×Kᵀ in attention, backpropagation

### Identity Matrix: I
```
I = [[1,0,0],
     [0,1,0],
     [0,0,1]]

A × I = A   (like multiplying by 1 — nothing changes)
```

### Inverse: A⁻¹
```
A × A⁻¹ = I   (like dividing — undoes the multiplication)

For [[a,b],[c,d]]:
A⁻¹ = 1/(ad-bc) × [[d,-b],[-c,a]]

Example: A = [[4,7],[2,6]]
det = (4×6)-(7×2) = 24-14 = 10
A⁻¹ = 1/10 × [[6,-7],[-2,4]] = [[0.6,-0.7],[-0.2,0.4]]
```

---

## 7. Linear Equations — from 1 to n

### One equation — infinite solutions
```
y = 2x + 3   → infinite (x,y) pairs satisfy this
```

### Two equations — one unique solution
```
2x + y = 5
x + 3y = 7

Substitution: y = 5-2x → x + 3(5-2x) = 7 → x = 1.6, y = 1.8
```

### n equations — matrix form: Ax = b
Pack all equations into matrices:
```
[[2,1],[1,3]] × [x,y] = [5,7]
      A       ×   x   =   b

Solve: x = A⁻¹ × b
```

### Hyperplane — the ML version
```
line (2D):       w1×x1 + w2×x2 + b = 0
plane (3D):      w1×x1 + w2×x2 + w3×x3 + b = 0
hyperplane (nD): w1×x1 + w2×x2 + ... + wn×xn + b = 0
```

**This IS the neuron equation. One neuron = one hyperplane.**

```
output > 0 → class 1
output < 0 → class 2
```

### Naming confusion — same equation, different letters
```
maths:   A  × x  = b       (x is the unknown)
ML:      X  × w  = y       (w = weights is the unknown)
         data   labels
```

---

## 8. Neurons and Neural Networks

### What is a neuron?
A neuron takes inputs, multiplies by weights, sums up, applies activation:

```
step 1 — multiply:  each input × its weight
step 2 — sum:       add everything up (+ bias)
step 3 — activate:  apply ReLU

output = ReLU(w1×x1 + w2×x2 + ... + wn×xn + b)
```

**One neuron = one row of a weight matrix = one hyperplane equation**

### ReLU activation
```
ReLU(x) = max(0, x)

positive numbers → pass through unchanged
negative numbers → become 0 (neuron is "silent")
```

Why needed: Without activation, stacking layers collapses to one linear equation. ReLU adds non-linearity → model can learn curves and complex patterns.

### A layer = matrix multiplication
```
neuron 1: w11×x1 + w12×x2 + w13×x3 = out1
neuron 2: w21×x1 + w22×x2 + w23×x3 = out2
...
packed into: W × input = output
each ROW of W = one neuron
```

### Neural network naming
| Word | What it actually means |
|---|---|
| Neural network | stack of matrix multiplications |
| Neuron | one row of a weight matrix |
| Layer | one full matrix multiplication |
| Firing | outputting a positive number |
| Silent | outputting zero (after ReLU) |

---

## 9. Embeddings

An **embedding** is a list of numbers that represents something — a word, sentence, image — where **similar things get similar numbers**.

### Why not just use word IDs?
```
"cat" → 1    (meaningless number)
"dog" → 2    (looks similar to cat in math but means nothing)
"car" → 3    (looks similar to cat — wrong!)
```

### Embeddings solve this
```
"cat" → [0.9, 0.2, 0.7, 0.4, ...]   ~1536 numbers
"dog" → [0.8, 0.3, 0.6, 0.5, ...]   similar to cat ✓
"car" → [0.1, 0.8, 0.2, 0.9, ...]   very different ✓
```

### The embedding table
```
shape = (50,000 × 1536)
50,000 rows = one per word in vocabulary
1536 columns = the numbers for that word

to get word's embedding → just look up its row
```

### Three types of embeddings
1. **Word embedding** — raw lookup from table ("cat" in isolation)
2. **Contextual embedding** — after attention ("cat that is sitting")
3. **Sentence embedding** — whole sentence as one vector

### Famous example — vector arithmetic
```
king − man + woman ≈ queen

the numbers for "king" and "queen" differ in exactly
the same way as "man" and "woman"
```

### Who assigns the numbers?
Nobody. The model learns them automatically by reading billions of sentences. Words that appear in similar sentences get similar numbers.

---

## 10. How Data Becomes Numbers

| Real thing | How stored |
|---|---|
| Image | Grid of pixel brightness values (0–255) |
| Word | Vector of ~1536 numbers (embedding) |
| Sound | List of air pressure values over time |
| Video | Many image matrices per second |

### Image example
A 4×4 pixel image:
```
255  80  60  255
 40  20  20   40
 40  10  10   40
255  80  60  255

→ flatten → [255, 80, 60, 255, 40, 20, 20, 40, ...]
→ feed into neural network as input vector
```

---

## 11. The Transformer — Attention Mechanism

### Input matrix X
Stack word embeddings as rows:
```
"The cat sat on mat"

X = (5 words × 1536 dims)
each row = one word's embedding
```

### Three weight matrices: W_Q, W_K, W_V
All have shape (1536 × 64). All are neurons. All learned during training.

```
Q = X × W_Q   shape (5×64)  "what am I looking for?"
K = X × W_K   shape (5×64)  "what do I contain?"
V = X × W_V   shape (5×64)  "what do I pass forward?"
```

### Attention scores: Q × Kᵀ
```
K shape:  (5×64)
Kᵀ shape: (64×5)   ← transpose to fix dimensions

Q × Kᵀ = (5×64) × (64×5) = (5×5)

result = attention score matrix
each cell = how much that word attends to every other word
```

### Full attention formula
```
Attention(Q, K, V) = softmax( Q×Kᵀ / √d ) × V

Q×Kᵀ  → dot products between all word pairs
/ √d   → scale down (d=64, so ÷8) to prevent softmax going extreme
softmax → turn scores into probabilities (sum to 1)
× V    → blend value vectors by attention weights → updated embeddings
```

### Why divide by √d?
- Dot product grows larger with more dimensions
- Without scaling, softmax becomes extreme (99%/1%/0%)
- Dividing by √d normalises scores regardless of dimension size
- d = 64, √64 = 8

### What happens to the embedding
```
before attention: "cat" = [0.9, 0.2, 0.7, ...]  ← just "furry animal"
after attention:  "cat" = [0.85, 0.5, 0.7, ...]  ← "cat that is sitting"
```

Context from other words gets baked into each word's numbers.

---

## 12. A Complete Transformer Layer

Each layer has two parts, each using weight matrices:

### Part 1 — Attention (W_Q, W_K, W_V)
Figures out relationships between words.

### Part 2 — Feedforward network (W_1, W_2)
Stores knowledge, detects patterns.
```
W_1: expands embedding (1536 → 6144)
W_2: compresses back  (6144 → 1536)
```

### Per layer — 5 weight matrices total
```
W_Q, W_K, W_V  →  attention (relationships)
W_1, W_2       →  feedforward (knowledge)
```

96 layers × 5 matrices = 480 weight matrices total (simplified).

### KV Cache
K and V matrices for already-processed words are saved (cached) so they don't need recomputing. Only the new word gets fully processed. This makes generation much faster.

---

## 13. Word Prediction — Output Layer

After 96 layers, the final vector gets multiplied by the output matrix:

```
final vector (1×1536) × output matrix (1536×50000) = scores (1×50000)
```

One score per word in vocabulary.

### Softmax
Converts raw scores (logits) into probabilities:
```
softmax: e^score / sum(all e^scores)

raw:      [9.54, 8.21, 4.12, ...]
softmax:  [0.91, 0.05, 0.01, ...]   ← all add to 1.0
```

### Temperature
Controls how "creative" the model is:
```
temperature = 0   → always pick highest probability word (predictable)
temperature = 0.7 → usually top word, sometimes others (natural)
temperature = 2.0 → very random (creative but sometimes nonsense)
```

---

## 14. Training — How the Model Learns

### The process
```
step 1: fill all weight matrices with random numbers
step 2: forward pass → predict next word
step 3: calculate loss (how wrong?)
step 4: backpropagation → calculate gradient for every weight
step 5: update every weight: W = W − α × gradient
step 6: repeat billions of times
```

### Loss function — cross entropy
```
loss = −log(probability given to correct word)

correct word got 1%  → loss = −log(0.01) = 4.6  ← huge penalty
correct word got 90% → loss = −log(0.90) = 0.10  ← tiny penalty
```

The log curve punishes being confidently wrong enormously more than being slightly wrong.

### Gradient = blame
```
gradient of W = how much does loss change if W changes slightly?

gradient = +2.1  → this weight caused lots of error → change it a lot
gradient = +0.0  → this weight didn't affect loss → don't touch it
```

Gradient is computed using calculus (chain rule) — automatically.

### Backpropagation
Works backwards from the loss through all 96 layers:
```
compute loss at output
→ layer 96: calculate gradients for all weights
→ layer 95: pass blame backwards, calculate gradients
→ ...
→ layer 1: calculate gradients
→ now every weight has its gradient
→ update all weights simultaneously
```

### Self-supervised learning
The correct answer is hidden in the text itself — no human labelling needed:
```
text: "The cat sat on the mat"
quiz: "The cat sat on the ___"  → answer: "mat"  (already in text!)
```

### RLHF — making it helpful
Second phase of training:
1. Model generates several responses
2. Human raters rank them
3. Model trained to score higher on human preferences

Humans give direction ("be helpful") — math does the weight nudging.

---

## 15. GPUs — Why They're Used

### CPU vs GPU
```
CPU: 4–32 powerful cores → one task at a time
GPU: 10,000+ tiny cores  → all tasks simultaneously
```

### Why GPU is perfect for ML
Matrix multiplication: every cell is independent — can all be calculated at the same time.
```
CPU: 1 million cells = 1 million steps
GPU: 1 million cells = 1 step (parallel)

Speedup: ~100× faster than CPU
```

### Real numbers
A modern NVIDIA H100 GPU: 2,000 trillion matrix operations per second.
Training GPT-4: ~25,000 GPUs running for months.

### Why GPUs originally?
Built for video games — rendering millions of pixels simultaneously. Same parallel problem as ML matrix multiplication. NVIDIA accidentally became the most important company in AI.

---

## 16. Classification and Decision Boundaries

### The task
Given data points with labels, find a boundary that separates them.

### Boundary shape depends on data dimensions
```
1D data → point
2D data → line     (y = mx + c)
3D data → plane    (w1x + w2y + w3z + b = 0)
nD data → hyperplane
```

Always one dimension less than your data.

### How to check if a line classifies correctly
```
for vertical line (x = k):
  left side (x < k) vs right side (x > k)

for diagonal line (y = mx + c):
  plug each x → get y_line
  compare y_point vs y_line:
    above (y_point > y_line) → one class
    below (y_point < y_line) → other class

check: all label 0 on same side AND all label 1 on other side?
  yes → line works ✓
  no  → line fails ✗
```

---

## 17. The Full ML Pipeline

```
words
  ↓ look up embedding table
[0.9, 0.2, 0.7, ...]       raw embedding (1536 numbers)
  ↓ stack all words into matrix X  (n words × 1536)
  ↓ layer 1:
    → W_Q, W_K, W_V (attention)    → updated embedding
    → W_1, W_2 (feedforward)       → richer embedding
  ↓ layer 2 ... layer 96  (same structure, different weights)
  ↓ × output matrix (1536 × 50,000)
scores for all 50,000 words
  ↓ softmax
probabilities
  ↓ sample (with temperature)
predicted next word
  ↓ append and repeat
full response generated
```

---

## 18. Linear Algebra Topics Roadmap

| # | Topic | Covered? | ML Use |
|---|---|---|---|
| 1 | Scalars, Vectors, Matrices, Tensors | ✅ | storing all data |
| 2 | Vector ops + dot product | ✅ | every neuron |
| 3 | Matrix multiply + transpose + inverse | ✅ | every layer |
| 4 | Linear equations + hyperplanes | ✅ | decision boundaries |
| 5 | Vector spaces + span | — | embedding space |
| 6 | Norms + cosine similarity | — | word similarity |
| 7 | Eigenvalues + eigenvectors | — | PCA, dimensionality reduction |
| 8 | SVD | — | recommendations, compression |

---

## 19. Key Formulas Summary

```
line:          y = mx + c
slope:         m = (y2-y1)/(x2-x1) = tan(θ)
angle:         θ = arctan(m)
parallel:      m1 = m2 (same slope)
perpendicular: m1 × m2 = -1

dot product:   a·b = a1b1 + a2b2 + ... + anbn
matrix shape:  (m×n) × (n×p) = (m×p)
transpose:     A[i][j] → Aᵀ[j][i]
inverse:       A × A⁻¹ = I
system:        Ax = b → x = A⁻¹b

neuron:        output = ReLU(W × input + b)
attention:     Attention(Q,K,V) = softmax(Q×Kᵀ/√d) × V
loss:          L = −log(p_correct)
update:        W = W − α × gradient
```

---

## 20. The Big Picture — One Sentence Each

**Linear algebra** = the math of lists and grids of numbers, and transforming them.

**Embedding** = a word converted to a list of numbers where meaning is encoded in the values.

**Neuron** = one row of a weight matrix doing a dot product with the input.

**Layer** = all neurons firing simultaneously = one matrix multiplication + activation.

**Attention** = each word looking at every other word to update its numbers with context.

**Training** = starting with random matrices and nudging them billions of times until predictions are accurate.

**Gradient** = a number per weight saying how much it contributed to the error.

**Backpropagation** = computing all gradients by working backwards through the network.

**Loss** = one number measuring how wrong the current prediction is.

**Transformer** = stack of attention + feedforward layers that turns words into contextual vectors and predicts the next word.

---

*Notes compiled from a full conversation covering linear algebra foundations through to transformer architecture and ML training.*

