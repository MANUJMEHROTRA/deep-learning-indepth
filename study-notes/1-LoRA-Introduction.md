


# Comprehensive Guide: Matrix Rank and LoRA in Fully Connected Layers

This document compiles a series of questions and detailed answers ranging from the concept of the rank of a matrix to an in-depth discussion on using LoRA (Low-Rank Adaptation) in a 4-layer fully connected dense network. It also includes comparisons, code examples, and diagrammatic explanations.



## 1. What is the Rank of a Matrix and How is it Calculated?

### **Overview of Matrix Rank**
The **rank** of a matrix is the maximum number of **linearly independent** rows or columns in the matrix. It represents the dimension of the **column space** (or **row space**) of the matrix.

Mathematically, the rank of a matrix \( A \) is denoted as:
\[
\text{rank}(A)
\]
and it satisfies:
\[
\text{rank}(A) \leq \min(m, n)
\]
for an \( m \times n \) matrix.

![image](/images/matrixFactor.png)

### **How to Calculate the Rank of a Matrix**

#### **1. Using Row Echelon Form (REF) or Reduced Row Echelon Form (RREF)**
- **Process**: Convert the matrix into **REF** or **RREF** using Gaussian or Gauss-Jordan elimination.
- **Interpretation**: The rank is the **number of nonzero rows** in the echelon form.

#### **2. Using Determinants (for Square Matrices)**
- **Process**: Check the determinant of square submatrices.
- **Interpretation**: If an \( n \times n \) submatrix has a nonzero determinant, then it is full-rank for that size.

#### **3. Using Singular Value Decomposition (SVD)**
- **Process**: Compute the singular values of the matrix.
- **Interpretation**: The rank equals the number of nonzero singular values.

### **Example Calculation Using Row Echelon Form**
Given the matrix:
\[
A = \begin{bmatrix} 
1 & 2 & 3 \\ 
4 & 5 & 6 \\ 
7 & 8 & 9 
\end{bmatrix}
\]
1. **Row Operations**:
   - Subtract multiples of the first row:
     \[
     \begin{bmatrix} 
     1 & 2 & 3 \\ 
     0 & -3 & -6 \\ 
     0 & -6 & -12 
     \end{bmatrix}
     \]
   - Normalize the second row:
     \[
     \begin{bmatrix} 
     1 & 2 & 3 \\ 
     0 & 1 & 2 \\ 
     0 & -6 & -12 
     \end{bmatrix}
     \]
   - Eliminate the last row:
     \[
     \begin{bmatrix} 
     1 & 2 & 3 \\ 
     0 & 1 & 2 \\ 
     0 & 0 & 0 
     \end{bmatrix}
     \]
2. **Conclusion**: Two nonzero rows imply a rank of **2**.

### **Python Code to Compute Rank**

```python
import numpy as np

A = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9]])

rank = np.linalg.matrix_rank(A)
print("Rank of the matrix:", rank)
```

*Output:*
```
Rank of the matrix: 2
```

![image](/images/withLoRA.jpeg)

![image](/images/withoutLoRA.jpeg)


## 2. How is LoRA Used in a 4-Layer Fully Connected Dense Layer?

### **Overview of LoRA (Low-Rank Adaptation)**
LoRA is a **parameter-efficient fine-tuning** method designed for adapting large models. Instead of updating the full weight matrix, LoRA **adds low-rank matrices** \( A \) and \( B \) to the weight update:
\[
W' = W + AB
\]
- **\( W \)**: The original weight matrix (frozen during fine-tuning).
- **\( A \) and \( B \)**: Trainable low-rank matrices.

### **Step 1: Baseline Model Architecture (Without LoRA)**
A simple 4-layer fully connected network is defined as follows:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_baseline_model(input_dim):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),  # Layer 1
        layers.Dense(64, activation='relu'),  # Layer 2
        layers.Dense(32, activation='relu'),  # Layer 3
        layers.Dense(10, activation='softmax')  # Output layer (10 classes)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

input_dim = 100  # Example input dimension
baseline_model = create_baseline_model(input_dim)
baseline_model.summary()
```

*Note: This model trains all weights, which may lead to high memory usage and slower training.*

### **Step 2: Integrating LoRA into Dense Layers**
For each dense layer, instead of training the entire weight matrix \( W \), we freeze \( W \) and add trainable matrices \( A \) and \( B \).

![image](/images/1_dn7Oz9OaVAUkRJDW6PHPYA.webp)


### **LoRA Layer Implementation in Keras**

```python
class LoRALayer(layers.Layer):
    def __init__(self, units, rank=4, **kwargs):
        super(LoRALayer, self).__init__(**kwargs)
        self.units = units
        self.rank = rank  # Low-rank dimension
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Frozen weight matrix W (Not trainable)
        self.W = self.add_weight(shape=(input_dim, self.units),
                                 initializer="glorot_uniform",
                                 trainable=False)
        
        # Trainable low-rank matrices A and B
        self.A = self.add_weight(shape=(input_dim, self.rank),
                                 initializer="random_normal",
                                 trainable=True)
        self.B = self.add_weight(shape=(self.rank, self.units),
                                 initializer="random_normal",
                                 trainable=True)

    def call(self, inputs):
        # Compute W' = W + AB
        return tf.matmul(inputs, self.W + tf.matmul(self.A, self.B))

def create_lora_model(input_dim, rank=4):
    inputs = keras.Input(shape=(input_dim,))
    x = LoRALayer(128, rank=rank)(inputs)  # Layer 1 with LoRA
    x = layers.ReLU()(x)
    x = LoRALayer(64, rank=rank)(x)  # Layer 2 with LoRA
    x = layers.ReLU()(x)
    x = LoRALayer(32, rank=rank)(x)  # Layer 3 with LoRA
    x = layers.ReLU()(x)
    outputs = LoRALayer(10, rank=rank)(x)  # Output layer with LoRA
    outputs = layers.Softmax()(outputs)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

lora_model = create_lora_model(input_dim, rank=4)
lora_model.summary()
```

### **Step 3: Comparison: Training Efficiency & Performance**

| Model Type                | Trainable Parameters       | Memory Usage | Training Speed | Performance (Accuracy) |
|---------------------------|----------------------------|--------------|----------------|------------------------|
| **Baseline (Full Fine-tuning)** | All layers trainable        | High         | Slower         | High                   |
| **LoRA-based Model**      | Only matrices \( A \) & \( B \) are trainable | Low          | Faster         | Comparable (Slight Trade-off) |

*LoRA reduces the number of trainable parameters dramatically while retaining comparable performance.*

---

## 3. Diagrammatic Comparison of the Two Models

### **Baseline Model (Without LoRA)**
Each layer uses a fully trainable weight matrix \( W \).

```
Input  →  Dense (Trainable W)  →  ReLU  
       →  Dense (Trainable W)  →  ReLU  
       →  Dense (Trainable W)  →  ReLU  
       →  Dense (Trainable W)  →  Softmax  
```
- **All weights updated** → High memory usage, slower training.

### **LoRA-based Model (With LoRA)**
Each layer uses a frozen weight matrix \( W \) along with trainable matrices \( A \) and \( B \):

```
Input  →  Dense (Frozen W + Trainable A & B)  →  ReLU  
       →  Dense (Frozen W + Trainable A & B)  →  ReLU  
       →  Dense (Frozen W + Trainable A & B)  →  ReLU  
       →  Dense (Frozen W + Trainable A & B)  →  Softmax  
```
- **Only A & B are updated** → Lower memory usage, faster fine-tuning.

### **Visual Representation**

#### **Without LoRA (Standard Dense Layer)**
```
         Input
           ↓
    ┌───────────────┐
    │ W (Trainable) │
    └───────────────┘
           ↓
         Output
```

#### **With LoRA (Modified Dense Layer)**
```
         Input
           ↓
    ┌─────────────────────────┐
    │  W (Frozen)             │
    │  A (Trainable)          │
    │  B (Trainable)          │
    └─────────────────────────┘
           ↓
         Output
```

---

## 4. Proving with an Example that LoRA Trains Fewer Weights

### **Standard Dense Layer Calculation (Non-LoRA)**
For a layer with:
- **Input dimension** \( d \)
- **Output neurons** \( k \)

The total trainable parameters (weights + biases) are:
\[
\text{Parameters} = (d \times k) + k
\]

#### **Example:**
Let \( d = 512 \) and \( k = 256 \):
\[
\text{Trainable Parameters} = (512 \times 256) + 256 = 131,328
\]

For **4 layers** of the same size:
\[
\text{Total} = 4 \times 131,328 = 525,312
\]

### **LoRA Layer Calculation**
Using LoRA with rank \( r \):
- **\( A \in \mathbb{R}^{d \times r} \)**
- **\( B \in \mathbb{R}^{r \times k} \)**
- \( W \) is frozen.

Trainable parameters per layer become:
\[
\text{Parameters} = (d \times r) + (r \times k) + k
\]

#### **Example:**
For \( d = 512 \), \( k = 256 \), and \( r = 8 \):
\[
\text{Trainable Parameters} = (512 \times 8) + (8 \times 256) + 256 = 4,096 + 2,048 + 256 = 6,400
\]

For **4 layers**:
\[
\text{Total} = 4 \times 6,400 = 25,600
\]

### **Python Code to Verify Calculations**

```python
# Define dimensions
d = 512  # Input dimension
k = 256  # Output neurons
r = 8    # LoRA rank

# Standard Dense Layer parameters
dense_params = (d * k) + k

# LoRA Layer parameters
lora_params = (d * r) + (r * k) + k

# Compute for 4 layers
total_dense = 4 * dense_params
total_lora = 4 * lora_params

print(f"Trainable Parameters (Standard Dense): {total_dense}")
print(f"Trainable Parameters (LoRA, rank={r}): {total_lora}")
print(f"Reduction: {100 * (1 - total_lora / total_dense):.2f}%")
```

*Expected Output:*
```
Trainable Parameters (Standard Dense): 525312
Trainable Parameters (LoRA, rank=8): 25600
Reduction: 95.13%
```

---

## 5. Detailed Comparison of Trainable Weights per Layer: Non-LoRA vs. LoRA

### **Non-LoRA (Standard Dense Layer)**
For each dense layer:
- **Weights \( W \)**: \( d \times k \) (trainable)
- **Biases \( b \)**: \( k \) (trainable)

**Example Layers:**

| Layer   | Weight Dimensions      | Trainable Weights Count | Biases Count | Total per Layer  |
|---------|------------------------|-------------------------|--------------|------------------|
| Layer 1 | \( 512 \times 256 \)   | 131,072                 | 256          | 131,328          |
| Layer 2 | \( 256 \times 128 \)   | 32,768                  | 128          | 32,896           |
| Layer 3 | \( 128 \times 64 \)    | 8,192                   | 64           | 8,256            |
| Layer 4 | \( 64 \times 10 \)     | 640                     | 10           | 650              |

**Total Trainable Parameters (Standard Model):**
\[
131,328 + 32,896 + 8,256 + 650 = 173,130
\]

### **LoRA-Based Model**
For each layer using LoRA:
- **Frozen Weight \( W \)**: Not trainable.
- **Trainable Matrices**:
  - \( A \) of dimensions \( d \times r \)
  - \( B \) of dimensions \( r \times k \)
- **Biases \( b \)**: \( k \) (trainable)

**Example Layers with \( r = 8 \):**

| Layer   | \( A \) Dimensions        | \( B \) Dimensions        | Biases | Total Trainable per Layer |
|---------|---------------------------|---------------------------|--------|---------------------------|
| Layer 1 | \( 512 \times 8 = 4,096 \)  | \( 8 \times 256 = 2,048 \)  | 256    | 6,400                     |
| Layer 2 | \( 256 \times 8 = 2,048 \)  | \( 8 \times 128 = 1,024 \)  | 128    | 3,200                     |
| Layer 3 | \( 128 \times 8 = 1,024 \)  | \( 8 \times 64 = 512 \)     | 64     | 1,600                     |
| Layer 4 | \( 64 \times 8 = 512 \)     | \( 8 \times 10 = 80 \)      | 10     | 602                       |

**Total Trainable Parameters (LoRA Model):**
\[
6,400 + 3,200 + 1,600 + 602 = 11,802
\]

### **Summary Comparison Table**

| Model Type         | Total Trainable Parameters (4 Layers) |
|--------------------|---------------------------------------|
| **Standard Dense** | 173,130                               |
| **LoRA (Rank = 8)**| 11,802                                |

*LoRA reduces trainable parameters by over 93% compared to the standard dense layer approach.*

---

## Conclusion

- **Matrix Rank**: We discussed the concept and various methods for computing the rank of a matrix.
- **LoRA in Dense Layers**: We explained how LoRA works by freezing the main weight matrix and training two low-rank matrices instead.
- **Comparisons & Examples**: Detailed numerical examples and code demonstrate the significant reduction in trainable parameters when using LoRA, both at a layer level and overall.
- **Visual and Diagrammatic Explanations**: Provided to clarify the structural differences between standard dense layers and LoRA-enhanced layers.

This markdown document should serve as a comprehensive guide from the basic concept of matrix rank to advanced model adaptation techniques using LoRA.

---

*End of Document*
```

---

This markdown file is fully compatible and includes all the discussions, explanations, diagrams, and code examples from our conversation.