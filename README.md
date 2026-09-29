# CS5720-Home-Assignment-2

**Course:** CS5720 Neural Networks and Deep Learning, Fall 2026
**University:** University of Central Missouri
**Student Name:** Anuradha Kurupathi
**Student ID:** 700781806

## Overview

This repository contains my solutions for Home Assignment 2, covering RNN/LSTM models for text generation and sentiment classification (Q1–Q2), convolution and CNN feature extraction operations (Q3–Q4), and CNN architecture implementation (Q5). All code was written and executed in Google Colab.

## Repository Contents

* `Home_Assignment_2_anuradhaipynb.ipynb` — Colab notebook with all 5 questions, code, and outputs
* `README.md` — this file

## Q1: Implementing an RNN for Text Generation

* Loaded the Shakespeare text dataset (first 300,000 characters, 99,967 training sequences)
* Converted characters to integer ids (62 unique characters) and used an `Embedding` layer
* Built the model: `Embedding(62, 64) → LSTM(256) → Dense(62, Softmax)` (348,606 parameters)
* Trained for 20 epochs; loss dropped from 2.5264 to 0.9667
* Generated text one character at a time by sampling at temperatures 0.2, 0.5, 1.0, and 1.5
* Output: Training log and generated text samples for each temperature

Temperature scaling:

* Temperature rescales the predicted probabilities before sampling: `p' = exp(log(p)/T) / sum(exp(log(p)/T))`
* Low temperature (0.2): the distribution becomes sharper, so the model picks the most likely character. Text is safer but repetitive (my output repeated phrases like "the people was the people").
* Temperature 1.0: uses the model's original distribution.
* High temperature (1.5): the distribution flattens, so unlikely characters get picked more often. Text is more random and has more made-up words (my output had many non-words).

## Q2: Sentiment Classification Using RNN

* Loaded the IMDB dataset (25,000 train / 25,000 test reviews), keeping the top 10,000 words
* Padded all reviews to 200 tokens using `pad_sequences`
* Built the model: `Embedding(10000, 128) → LSTM(64) → Dense(1, Sigmoid)` (1,329,473 parameters)
* Trained for 3 epochs with a 20% validation split
* Output: Test accuracy **83.9%**, confusion matrix, classification report, and confusion matrix plot

Results:

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Negative | 0.93 | 0.73 | 0.82 |
| Positive | 0.78 | 0.95 | 0.85 |

Confusion matrix: `[[9144, 3356], [670, 11830]]`

* Patterns observed: The model leans toward predicting "positive" (recall 0.95 but precision 0.78). Validation accuracy peaked at epoch 2 (87.2%) and dropped at epoch 3 (84.0%) while training accuracy kept rising, which indicates overfitting.
* Why precision-recall tradeoff matters: Precision measures how many predicted positives are truly positive, and recall measures how many true positives were found. Changing the threshold improves one at the cost of the other, and the right balance depends on the application (precision when false positives are costly, recall when missing cases is costly). Accuracy alone can hide this, as my 84% accuracy hides the uneven behavior across the two classes.

## Q3: Convolution with Different Stride and Padding

* Defined a 5×5 input matrix (values 1–25) and a 3×3 kernel `[[0,1,0],[1,-4,1],[0,1,0]]`
* Applied convolution with `tf.nn.conv2d` for four settings
* Output: Feature maps for each case

| Setting | Output shape | Result |
|---|---|---|
| Stride 1, VALID | 3×3 | all zeros |
| Stride 1, SAME | 5×5 | non-zero values along the border |
| Stride 2, VALID | 2×2 | all zeros |
| Stride 2, SAME | 3×3 | non-zero values along the border |

* Observations: The kernel is a Laplacian filter and the input increases linearly, so interior outputs are 0. With SAME padding, zero-padding at the borders creates artificial edges, so non-zero values appear there. A larger stride gives a smaller output feature map.

## Q4: CNN Feature Extraction with Filters and Pooling

Task 1: Sobel Edge Detection

* Loaded a 512×512 grayscale sample image (`skimage.data.camera()`)
* Applied the given Sobel-X and Sobel-Y filters using `cv2.filter2D`
* Output: Displayed the original image, Sobel-X (vertical edges), and Sobel-Y (horizontal edges)

Task 2: Max Pooling and Average Pooling

* Created a random 4×4 matrix (seed 42)
* Applied 2×2 Max Pooling and 2×2 Average Pooling using Keras layers
* Output: Original matrix, max-pooled matrix `[[9, 7], [7, 7]]`, and average-pooled matrix `[[6, 4.75], [5, 4.75]]`

## Q5: Implementing and Comparing CNN Architectures

Task 1: AlexNet

* Built a simplified AlexNet with the required Conv2D, MaxPooling, Flatten, Dense, and Dropout(0.5) layers and a 10-neuron Softmax output (227×227×3 input)
* Output: Model summary with **58,322,314** total parameters

Task 2: Residual Block and ResNet

* Implemented `residual_block(input_tensor, filters)` with two Conv2D layers (3×3, 64 filters) and a skip connection added before the final ReLU
* Built a ResNet-like model: Conv2D (64, 7×7, stride 2) → 2 residual blocks → Flatten → Dense(128) → Softmax output (64×64×3 input)
* Output: Model summary with **8,547,210** total parameters
* Observation: Skip connections give gradients a direct path backward, which helps train deeper networks. AlexNet has far more parameters than the ResNet-like model, mostly because of its two 4096-unit Dense layers.

## How to Run

1. Open the `.ipynb` file in [Google Colab](https://colab.research.google.com/).
2. Switch to a GPU (Runtime → Change runtime type → T4 GPU).
3. Run all cells in order (Runtime → Run all).
4. Q1 and Q2 take a few minutes since they train the LSTM models.
