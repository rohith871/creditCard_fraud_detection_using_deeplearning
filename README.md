# GAN-Based Synthetic Fraudulent Transaction Generation

A deep learning project that uses a **Generative Adversarial Network (GAN)** to generate synthetic fraudulent transaction data.

The project demonstrates how a Generator can learn the underlying distribution of real fraudulent transactions and produce new synthetic samples that resemble the original fraud data.

## Overview

Credit card fraud datasets are typically highly imbalanced, with fraudulent transactions representing only a small portion of all transactions. This makes it difficult for machine learning models to learn meaningful patterns from the minority fraud class.

This project explores the use of a **Generative Adversarial Network (GAN)** to generate additional synthetic fraudulent transaction samples.

The GAN consists of two neural networks:

* **Generator (G)** — generates synthetic fraudulent transactions from random noise.
* **Discriminator (D)** — distinguishes between real fraudulent transactions and generated synthetic transactions.

The two networks are trained adversarially:

```text
                 Random Noise
                      │
                      ▼
                ┌───────────┐
                │ Generator │
                └─────┬─────┘
                      │
                Synthetic Data
                      │
                      ▼
                ┌──────────────┐
                │Discriminator │
                └──────┬───────┘
                       │
                  Real / Fake
```

As training progresses, the Generator attempts to produce increasingly realistic fraudulent transactions while the Discriminator becomes better at identifying synthetic samples.

---

## How the GAN Is Trained

Training is performed in alternating steps rather than training both networks simultaneously.

### 1. Train the Discriminator

A batch of real fraudulent transactions is sampled from the dataset.

At the same time, the Generator creates a batch of synthetic transactions from randomly generated noise.

The Discriminator is trained using:

```text
Real transactions → label 1
Synthetic transactions → label 0
```

The Discriminator updates its weights to improve its ability to distinguish real fraud from generated fraud.

### 2. Train the Generator

A new batch of random noise is generated and passed through the Generator.

The resulting synthetic transactions are passed through the Discriminator.

During this stage, the Discriminator is frozen.

The Generator is trained with the target:

```text
Synthetic transactions → label 1
```

This tells the Generator:

> Produce synthetic transactions that the Discriminator classifies as real.

The training cycle therefore becomes:

```text
Train Discriminator
        ↓
Train Generator
        ↓
Train Discriminator
        ↓
Train Generator
        ↓
        ...
```

---

## Random Noise

The Generator does not directly receive real transaction data as its input.

Instead, it receives randomly generated noise:

```python
noise = np.random.normal(
    0,
    1,
    (num_samples, generator.input_shape[1])
)
```

The values are sampled from a normal distribution with:

* Mean = `0`
* Standard deviation = `1`

For example, if the Generator accepts 29-dimensional noise:

```text
[ 0.21, -1.13, 0.54, 0.72, ..., -0.31 ]
```

The Generator transforms this random vector into a synthetic transaction.

New noise is generated during each call, allowing the Generator to produce different synthetic samples.

---

## Generator Architecture

The Generator consists of several fully connected layers.

A simplified structure is:

```text
Random Noise
     │
     ▼
 Dense Layer
     │
    ReLU
     │
     ▼
 Dense Layer
     │
    ReLU
     │
     ▼
 Dense Layer
     │
    ReLU
     │
     ▼
 Output Layer
     │
   Linear
     │
     ▼
Synthetic Transaction
```

### Why ReLU in the Hidden Layers?

ReLU introduces non-linearity into the network:

[
ReLU(x)=max(0,x)
]

This allows the Generator to learn complex relationships between the input noise and the transaction features.

Without non-linear activation functions, stacking multiple linear layers would still result in an overall linear transformation.

### Why Linear in the Output Layer?

The final layer uses a linear activation so that the Generator can produce unrestricted continuous values.

ReLU would restrict the output to non-negative values:

```text
ReLU:

-2 → 0
-1 → 0
 0 → 0
 1 → 1
 2 → 2
```

A linear output allows:

```text
-2 → -2
-1 → -1
 0 →  0
 1 →  1
 2 →  2
```

This is useful when the numerical features in the dataset can contain both positive and negative values, particularly after feature scaling.

---

## Discriminator

The Discriminator is a binary classification network.

Its job is to determine whether a transaction is:

```text
1 → Real
0 → Fake
```

A simplified architecture is:

```text
Transaction Features
        │
        ▼
    Dense Layer
        │
       ReLU
        │
        ▼
    Dense Layer
        │
       ReLU
        │
        ▼
    Output Layer
        │
     Sigmoid
        │
        ▼
   Real / Fake
```

The final Sigmoid activation produces a value between 0 and 1.

For example:

```text
0.95 → likely real
0.08 → likely fake
```

---

## Batch Normalization

Batch Normalization is used inside the neural networks to normalize intermediate activations during training.

Conceptually:

```text
Layer
  ↓
Batch Normalization
  ↓
Activation
  ↓
Next Layer
```

It helps keep the values flowing through the network in a more controlled range and can make neural network training more stable.

Batch Normalization is different from `batch_size`.

### Batch Size

The batch size determines how many samples are processed during a training update.

For example:

```python
batch_size = 64
```

means 64 samples/noise vectors are used for the corresponding training update.

---

## Training Configuration

The training loop uses:

```python
num_epochs = 1000
batch_size = 64
half_batch = 32
```

For each iteration:

```text
32 real fraudulent transactions
+
32 generated fraudulent transactions
```

are used to train the Discriminator.

Then:

```text
64 new random noise vectors
```

are used to train the Generator through the combined GAN.

---

## Training Flow

The complete training process can be summarized as:

```text
                 START ITERATION
                       │
                       ▼
              Generate random noise
                       │
                       ▼
                  Generator
                       │
                       ▼
              Generate fake samples
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
       Sample real data      Fake data
              │                 │
              ▼                 ▼
       ┌─────────────────────────────┐
       │       Discriminator         │
       │                             │
       │ Real → 1    Fake → 0        │
       └──────────────┬──────────────┘
                      │
                      ▼
                Update D
                      │
                      ▼
             Generate NEW noise
                      │
                      ▼
                  Generator
                      │
                      ▼
                Fake samples
                      │
                      ▼
               Discriminator
                 (Frozen)
                      │
                      ▼
                Target = 1
                      │
                      ▼
                Update G
                      │
                      ▼
                Next iteration
```

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Generative Adversarial Networks (GANs)
* Deep Learning

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-folder>
```

Install the dependencies:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

---

## Running the Project

Open the notebook or Python script containing the GAN implementation and run the cells in order.

The main training process follows:

```python
generator = build_generator()
discriminator = build_discriminator()

gan = build_gan(generator, discriminator)

gan.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)
```

Then the GAN is trained using alternating Discriminator and Generator updates.

---

## Project Objective

The main objective of this project is to understand and implement the fundamentals of **Generative Adversarial Networks** for generating synthetic fraudulent transaction data.

The project covers:

* GAN architecture
* Generator and Discriminator networks
* Random noise generation
* ReLU and linear activation functions
* Batch Normalization
* Mini-batch training
* Alternating Generator and Discriminator training
* Adversarial learning
* Synthetic fraud data generation
