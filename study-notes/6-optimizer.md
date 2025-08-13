Optimizers are essential components of training neural networks, as they adjust model parameters to minimize the loss function. Let’s break down the most common optimizers, their mathematical foundations, and Python implementations.

---

## **1. Gradient Descent (GD)**
Gradient Descent updates parameters in the direction of the negative gradient of the loss function.

### **Mathematical Formula**
\[
\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t)
\]
Where:
- \(\theta_t\) = parameter at step \(t\)
- \(\alpha\) = learning rate
- \(J(\theta_t)\) = loss function
- \(\nabla J(\theta_t)\) = gradient of the loss w.r.t. \(\theta_t\)

### **Python Code**
```python
import numpy as np

def gradient_descent(X, y, theta, lr=0.01, epochs=100):
    m = len(y)
    for _ in range(epochs):
        gradient = (1/m) * X.T @ (X @ theta - y)
        theta -= lr * gradient
    return theta
```
👉 **Problem**: Standard GD computes gradients over the entire dataset, making it slow for large datasets.

---

## **2. Stochastic Gradient Descent (SGD)**
SGD updates parameters using a single data point at a time, reducing computation.

### **Mathematical Formula**
\[
\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t; x_i, y_i)
\]
where \(x_i, y_i\) are a single training example.

### **Python Code**
```python
def stochastic_gradient_descent(X, y, theta, lr=0.01, epochs=100):
    m = len(y)
    for _ in range(epochs):
        for i in range(m):
            gradient = X[i].T * (X[i] @ theta - y[i])
            theta -= lr * gradient
    return theta
```
👉 **Pros**: Faster updates, useful for large datasets.  
👉 **Cons**: High variance in updates, causing instability.

---

## **3. Mini-Batch Gradient Descent**
A compromise between GD and SGD, using small batches of data.

### **Mathematical Formula**
\[
\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t; X_{batch}, y_{batch})
\]

### **Python Code**
```python
def mini_batch_gradient_descent(X, y, theta, lr=0.01, epochs=100, batch_size=32):
    m = len(y)
    for _ in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled, y_shuffled = X[indices], y[indices]
        for i in range(0, m, batch_size):
            X_batch, y_batch = X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]
            gradient = (1/len(y_batch)) * X_batch.T @ (X_batch @ theta - y_batch)
            theta -= lr * gradient
    return theta
```
👉 **Pros**: Balances speed and accuracy.  
👉 **Cons**: Requires tuning batch size.

---

## **4. Momentum-based Gradient Descent**
Momentum accelerates gradient descent by adding a velocity term.

### **Mathematical Formula**
\[
v_t = \beta v_{t-1} + (1 - \beta) \nabla J(\theta_t)
\]
\[
\theta_{t+1} = \theta_t - \alpha v_t
\]
Where:
- \( v_t \) is the velocity term.
- \( \beta \) (typically 0.9) controls past gradient influence.

### **Python Code**
```python
def momentum_gd(X, y, theta, lr=0.01, epochs=100, beta=0.9):
    m = len(y)
    v = np.zeros_like(theta)
    for _ in range(epochs):
        gradient = (1/m) * X.T @ (X @ theta - y)
        v = beta * v + (1 - beta) * gradient
        theta -= lr * v
    return theta
```
👉 **Pros**: Reduces oscillations, speeds up convergence.  
👉 **Cons**: Needs careful tuning of \( \beta \).

---

## **5. RMSprop (Root Mean Square Propagation)**
RMSprop scales the learning rate based on past gradients.

### **Mathematical Formula**
\[
s_t = \beta s_{t-1} + (1 - \beta) \nabla J(\theta_t)^2
\]
\[
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{s_t} + \epsilon} \nabla J(\theta_t)
\]
Where:
- \( s_t \) is an exponentially weighted moving average of squared gradients.
- \( \epsilon \) is a small constant to avoid division by zero.

### **Python Code**
```python
def rmsprop(X, y, theta, lr=0.01, epochs=100, beta=0.9, epsilon=1e-8):
    m = len(y)
    s = np.zeros_like(theta)
    for _ in range(epochs):
        gradient = (1/m) * X.T @ (X @ theta - y)
        s = beta * s + (1 - beta) * (gradient**2)
        theta -= lr * gradient / (np.sqrt(s) + epsilon)
    return theta
```
👉 **Pros**: Works well for non-stationary objectives.  
👉 **Cons**: Requires careful tuning of \( \beta \).

---

## **6. Adam (Adaptive Moment Estimation)**
Adam combines Momentum and RMSprop.

### **Mathematical Formula**
\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla J(\theta_t)
\]
\[
s_t = \beta_2 s_{t-1} + (1 - \beta_2) \nabla J(\theta_t)^2
\]
\[
\hat{m_t} = \frac{m_t}{1 - \beta_1^t}, \quad \hat{s_t} = \frac{s_t}{1 - \beta_2^t}
\]
\[
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{s_t}} + \epsilon} \hat{m_t}
\]
Where:
- \( m_t \) is the first moment estimate (mean of gradients).
- \( s_t \) is the second moment estimate (variance of gradients).
- \( \hat{m_t} \), \( \hat{s_t} \) are bias-corrected values.

### **Python Code**
```python
def adam(X, y, theta, lr=0.01, epochs=100, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = len(y)
    m_t, v_t = np.zeros_like(theta), np.zeros_like(theta)
    for t in range(1, epochs+1):
        gradient = (1/m) * X.T @ (X @ theta - y)
        m_t = beta1 * m_t + (1 - beta1) * gradient
        v_t = beta2 * v_t + (1 - beta2) * (gradient**2)
        m_t_hat = m_t / (1 - beta1**t)
        v_t_hat = v_t / (1 - beta2**t)
        theta -= lr * m_t_hat / (np.sqrt(v_t_hat) + epsilon)
    return theta
```
👉 **Pros**: Works well in most cases, adapts learning rates.  
👉 **Cons**: Can be computationally expensive.

---

## **Choosing the Right Optimizer**
| Optimizer | Pros | Cons |
|-----------|------|------|
| GD | Stable convergence | Slow on large datasets |
| SGD | Faster updates | High variance in updates |
| Mini-Batch GD | Efficient balance | Requires batch size tuning |
| Momentum | Reduces oscillations | Needs \( \beta \) tuning |
| RMSprop | Handles non-stationary objectives | Needs \( \beta \) tuning |
| Adam | Adaptive, best default choice | Computationally expensive |

Would you like a benchmarking comparison of these optimizers on real data? 🚀