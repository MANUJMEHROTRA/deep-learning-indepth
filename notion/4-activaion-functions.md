# **Activation Functions in Neural Networks**  
Activation functions play a crucial role in neural networks by introducing **non-linearity**, helping the model learn complex patterns. The choice of activation function impacts the network's performance, convergence, and ability to handle vanishing/exploding gradients.

---

## **1. Linear Activation Function**
### 🔹 Definition:  
\[
f(x) = ax
\]
### 🔹 Properties:  
- Outputs are proportional to inputs.
- No non-linearity, so it acts like a simple regression model.
- Used in the **output layer** of regression models.

### 🔹 When to Use?  
✅ If the task is **linear regression**.  
🚫 Not used in hidden layers (no ability to capture non-linear patterns).  

---

## **2. Step Function (Threshold Activation)**
### 🔹 Definition:  
\[
f(x) = 
\begin{cases} 
1, & x > 0 \\ 
0, & x \leq 0
\end{cases}
\]
### 🔹 Properties:
- Converts inputs into binary values.
- Was used in **early perceptrons**.
- Cannot handle complex patterns.

### 🔹 When to Use?  
✅ Simple binary classification tasks (NOT recommended in deep learning).  
🚫 Not differentiable (can't be used in backpropagation).  

---

## **3. Sigmoid Activation Function**
### 🔹 Definition:  
\[
f(x) = \frac{1}{1 + e^{-x}}
\]
### 🔹 Properties:
- Output range: **(0,1)**
- Used for **probability-based outputs**.
- **Smooth & differentiable**.
- **Suffers from vanishing gradient** in deep networks.

### 🔹 When to Use?  
✅ **Binary classification tasks** (output layer).  
✅ If probabilities are needed.  
🚫 Not recommended in hidden layers (slow learning, vanishing gradient).

### 🔹 Example in PyTorch:
```python
import torch
import torch.nn.functional as F

x = torch.tensor([-1.0, 0.0, 1.0])
sigmoid_output = torch.sigmoid(x)
print(sigmoid_output)
```

---

## **4. Tanh (Hyperbolic Tangent) Activation Function**
### 🔹 Definition:  
\[
f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
\]
### 🔹 Properties:
- Output range: **(-1,1)**.
- Zero-centered (better than Sigmoid).
- **Still suffers from vanishing gradient**.

### 🔹 When to Use?  
✅ If **negative values are meaningful** in the hidden layers.  
✅ Good for **centered data**.  
🚫 Avoid in deep networks due to vanishing gradient.

### 🔹 Example in PyTorch:
```python
tanh_output = torch.tanh(x)
print(tanh_output)
```

---

## **5. ReLU (Rectified Linear Unit)**
### 🔹 Definition:  
\[
f(x) = \max(0, x)
\]
### 🔹 Properties:
- Output range: **(0, ∞)**.
- Prevents vanishing gradients.
- Computationally efficient.
- **Issue:** Can cause **"Dead Neurons"** (neurons that never activate).

### 🔹 When to Use?  
✅ **Hidden layers** of deep neural networks.  
🚫 Avoid in cases with lots of negative values (use Leaky ReLU instead).

### 🔹 Example in PyTorch:
```python
relu_output = F.relu(x)
print(relu_output)
```

---

## **6. Leaky ReLU**
### 🔹 Definition:  
\[
f(x) = 
\begin{cases} 
x, & x > 0 \\ 
0.01x, & x \leq 0
\end{cases}
\]
### 🔹 Properties:
- **Fixes ReLU’s “dead neuron” problem**.
- Allows small negative values instead of zero.
- Helps with **gradient flow**.

### 🔹 When to Use?  
✅ **Deep networks to prevent dead neurons**.  
✅ **Good alternative to ReLU**.  

### 🔹 Example in PyTorch:
```python
leaky_relu_output = F.leaky_relu(x, negative_slope=0.01)
print(leaky_relu_output)
```

---

## **7. Parametric ReLU (PReLU)**
### 🔹 Definition:  
\[
f(x) = 
\begin{cases} 
x, & x > 0 \\ 
\alpha x, & x \leq 0
\end{cases}
\]
- Similar to Leaky ReLU but learns **alpha** dynamically.
- Used in deep learning architectures.

### 🔹 When to Use?  
✅ If you need an **adaptive** activation function.  
✅ Used in **CNNs and deep networks**.

### 🔹 Example in PyTorch:
```python
prelu = nn.PReLU()
prelu_output = prelu(x)
print(prelu_output)
```

---

## **8. ELU (Exponential Linear Unit)**
### 🔹 Definition:  
\[
f(x) = 
\begin{cases} 
x, & x > 0 \\ 
\alpha (e^x - 1), & x \leq 0
\end{cases}
\]
### 🔹 Properties:
- Allows **negative values**.
- Smoother gradient flow than ReLU.
- **Better than Leaky ReLU** for some deep networks.

### 🔹 When to Use?  
✅ **Deep networks with small datasets**.  
✅ If smooth gradients are required.  

### 🔹 Example in PyTorch:
```python
elu_output = F.elu(x, alpha=1.0)
print(elu_output)
```

---

## **9. SELU (Scaled Exponential Linear Unit)**
### 🔹 Definition:  
\[
f(x) = 
\begin{cases} 
\lambda x, & x > 0 \\ 
\lambda \alpha (e^x - 1), & x \leq 0
\end{cases}
\]
Where:
- \(\lambda \approx 1.0507\),  
- \(\alpha \approx 1.6733\).

### 🔹 Properties:
- **Self-normalizing** (helps stabilize gradients).
- Works best with **Lecun initialization**.
- Used in **deep self-normalizing networks**.

### 🔹 When to Use?  
✅ If **self-normalization** is needed.  
✅ Used in **high-performance deep networks**.  

### 🔹 Example in PyTorch:
```python
selu_output = F.selu(x)
print(selu_output)
```

---

## **10. Softmax (for Multi-Class Classification)**
### 🔹 Definition:  
\[
f(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
\]
### 🔹 Properties:
- Converts logits into **probabilities**.
- Used in **multi-class classification**.

### 🔹 When to Use?  
✅ **Multi-class classification (output layer).**  

### 🔹 Example in PyTorch:
```python
softmax_output = F.softmax(x, dim=0)
print(softmax_output)
```

---

## **Choosing the Right Activation Function**
| Activation | Type | Range | Used in |
|------------|------|-------|---------|
| **Linear** | Linear | (-∞, ∞) | Regression output layer |
| **Step** | Binary | {0,1} | Old Perceptron models (Not used now) |
| **Sigmoid** | Non-linear | (0,1) | Binary classification output layer |
| **Tanh** | Non-linear | (-1,1) | Hidden layers, centered data |
| **ReLU** | Non-linear | (0,∞) | Deep networks, CNNs, hidden layers |
| **Leaky ReLU** | Non-linear | (-∞,∞) | Deep networks (better than ReLU) |
| **PReLU** | Non-linear | (-∞,∞) | Adaptive activation for deep networks |
| **ELU** | Non-linear | (-∞,∞) | Deep networks, smooth training |
| **SELU** | Non-linear | (-∞,∞) | Self-normalizing deep networks |
| **Softmax** | Non-linear | (0,1) | Multi-class classification output |

---

# **Why Use Activation Functions in Neural Networks?**  

Activation functions are crucial in neural networks because they introduce **non-linearity**, allowing the model to learn complex patterns. Without an activation function, the network behaves like a **linear model**, which cannot capture intricate relationships in data.

### **Key Reasons for Using Activation Functions**
1. **Non-Linearity:**  
   - Real-world data is non-linear, and activation functions allow the network to model complex functions.
   
2. **Differentiability:**  
   - Activation functions must be differentiable so that backpropagation (gradient descent) can update the weights.

3. **Prevention of Vanishing/Exploding Gradients:**  
   - Certain functions (e.g., **ReLU, Leaky ReLU**) help mitigate vanishing/exploding gradients.

4. **Probabilistic Interpretation:**  
   - Activation functions like **Sigmoid and Softmax** provide probabilities, useful in classification tasks.

---

## **Which Activation Function to Use and When?**  

### **1. Linear Activation Function**
✅ **Use When:**  
- In the **output layer** of regression models where predictions need to be continuous values.  
🚫 **Avoid In:**  
- Hidden layers (since it doesn’t introduce non-linearity).  

---

### **2. Sigmoid Activation Function**
✅ **Use When:**  
- **Binary classification output layer** (probability scores between 0 and 1).  
🚫 **Avoid In:**  
- Deep hidden layers (**vanishing gradient problem**).  

---

### **3. Tanh Activation Function**
✅ **Use When:**  
- **Hidden layers** when data is centered around zero (better than Sigmoid).  
🚫 **Avoid In:**  
- Deep networks due to **vanishing gradients**.  

---

### **4. ReLU (Rectified Linear Unit)**
✅ **Use When:**  
- **Hidden layers of deep networks (CNNs, RNNs, Transformers)**.  
- Computationally efficient, avoids vanishing gradients.  
🚫 **Avoid In:**  
- Networks with too many negative values (**"Dead Neurons"** problem).  

---

### **5. Leaky ReLU**
✅ **Use When:**  
- **Hidden layers** in deep networks to avoid dead neurons.  
🚫 **Avoid In:**  
- Cases where **pure ReLU suffices**.  

---

### **6. Parametric ReLU (PReLU)**
✅ **Use When:**  
- Adaptive version of Leaky ReLU for **deep networks requiring more flexibility**.  

---

### **7. ELU (Exponential Linear Unit)**
✅ **Use When:**  
- **Deep networks with small datasets**, since it provides smoother gradients than ReLU.  
🚫 **Avoid In:**  
- Large datasets where **ReLU works fine**.  

---

### **8. SELU (Scaled Exponential Linear Unit)**
✅ **Use When:**  
- **Self-normalizing deep networks**, works best with Lecun initialization.  
🚫 **Avoid In:**  
- Networks without proper initialization.  

---

### **9. Softmax**
✅ **Use When:**  
- **Multi-class classification output layer** (converts logits to probability distributions).  
🚫 **Avoid In:**  
- Hidden layers (use ReLU/Tanh instead).  

---

## **Necessary and Sufficient Conditions for Choosing an Activation Function**
A good activation function should satisfy the following conditions:

### **Necessary Conditions (Must-Have)**
1. **Non-Linearity**  
   - If the activation function is linear, the network behaves like a single-layer model, limiting its capacity.

2. **Differentiability**  
   - Required for backpropagation (except in some special cases like binary threshold functions).

3. **Computational Efficiency**  
   - The function should be easy to compute to make training faster (e.g., ReLU is faster than Sigmoid/Tanh).

---

### **Sufficient Conditions (Good-to-Have)**
1. **Prevent Vanishing/Exploding Gradients**  
   - Activation functions like **ReLU and Leaky ReLU** help prevent the vanishing gradient problem.

2. **Zero-Centered Output**  
   - Helps in optimizing convergence (**Tanh is zero-centered, but Sigmoid is not**).

3. **Sparsity & Efficient Neuron Activation**  
   - Functions like **ReLU** promote sparsity by setting some activations to zero.

---

## **Summary Table**
| Activation | Type | Used in | Pros | Cons |
|------------|------|---------|------|------|
| **Linear** | Linear | Regression Output | Simple, interpretable | Cannot model complex patterns |
| **Sigmoid** | Non-linear | Binary classification output | Probabilistic interpretation | Vanishing gradient |
| **Tanh** | Non-linear | Hidden layers | Zero-centered, better than Sigmoid | Vanishing gradient |
| **ReLU** | Non-linear | Hidden layers in deep networks | Efficient, avoids vanishing gradient | Dead neurons |
| **Leaky ReLU** | Non-linear | Deep networks | Fixes dead neuron problem | Still non-adaptive |
| **PReLU** | Non-linear | Deep networks (CNNs) | Adaptive ReLU | More parameters |
| **ELU** | Non-linear | Deep networks (small datasets) | Smooth activation | Slower than ReLU |
| **SELU** | Non-linear | Self-normalizing deep networks | Ensures stable gradients | Requires Lecun initialization |
| **Softmax** | Non-linear | Multi-class classification output | Converts logits into probabilities | Not for hidden layers |

---

### **Final Guidelines**
- **Use Sigmoid/Softmax in the output layer** for binary/multi-class classification.  
- **Use ReLU (or variants like Leaky ReLU) in hidden layers** of deep networks.  
- **For self-normalizing networks, use SELU**.  
- **Avoid Sigmoid/Tanh in deep networks due to vanishing gradients**.  


### **Is ReLU Differentiable? Let's Prove It!**  
ReLU (Rectified Linear Unit) is defined as:  

\[
f(x) = \max(0, x)
\]

This function is **piecewise-defined** as:

\[
f(x) =
\begin{cases} 
x, & x > 0 \\ 
0, & x \leq 0
\end{cases}
\]

To check differentiability, we first compute its **derivative**.

#### **Step 1: Compute Left and Right Derivatives**  
The derivative of ReLU is:

\[
f'(x) =
\begin{cases} 
1, & x > 0 \\ 
0, & x < 0
\end{cases}
\]

At **\( x = 0 \)**, we check the left-hand and right-hand derivatives.

- **Left-hand derivative (\( x \to 0^- \)):**  
  \[
  \lim_{h \to 0^-} \frac{f(h) - f(0)}{h} = \lim_{h \to 0^-} \frac{0 - 0}{h} = 0
  \]

- **Right-hand derivative (\( x \to 0^+ \)):**  
  \[
  \lim_{h \to 0^+} \frac{f(h) - f(0)}{h} = \lim_{h \to 0^+} \frac{h - 0}{h} = 1
  \]

Since the left and right derivatives at \( x = 0 \) are **not equal** (\( 0 \neq 1 \)), **ReLU is not differentiable at \( x = 0 \)**.

---

### **Then Why is ReLU Used as an Activation Function?**  
Even though ReLU is **not differentiable at \( x = 0 \)**, it is still widely used because:

1. **It is differentiable everywhere except at a single point**  
   - Gradient-based optimization (e.g., backpropagation) only requires differentiability **almost everywhere**, and **in practice, gradient descent works fine** with ReLU.
   - **Workaround:** In most deep learning libraries, the derivative at \( x = 0 \) is approximated as either **0 or 1**.

2. **Solves the Vanishing Gradient Problem**  
   - Unlike **Sigmoid and Tanh**, whose gradients shrink towards zero in deep networks, ReLU maintains a **constant gradient (1 for \( x > 0 \))**, ensuring efficient learning.

3. **Computational Efficiency**  
   - ReLU involves **simple thresholding**, making it faster than exponentiation-based activations (e.g., Sigmoid, Tanh).

4. **Sparsity and Efficient Neuron Activation**  
   - ReLU sets negative values to **zero**, meaning some neurons do not activate, making the network more **efficient and sparse**.

---

### **Conclusion**  
ReLU is **not differentiable at \( x = 0 \)** but is still used because:  
✅ **It is differentiable almost everywhere** (except one point).  
✅ **It avoids the vanishing gradient problem.**  
✅ **It is computationally efficient.**  
✅ **It promotes sparsity, improving efficiency.**  



