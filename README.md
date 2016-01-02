# Machine Learning fundamentals

This repository contains the code I produced in the context of the Machine Learning online course taught by *Andrew Ng*.
Each programming exercise involved me to fill in the core functions of each algorithm, the rest of the code belongs to (Andrew Ng - Coursera - Stanford University).

Typical examples of Supervised and Unsupervised learning algorithms have been implemented (see *Summary*). Each implementation was verified in the context of the course. This repo is therefore a good resource to review the basics of some state of the art machine learning techniques.

## Summary

1. **Linear regression**: predicting best city is best for restaurant-chain expansion
2. **Logistic regression**: predicting whether a student gets admitted into a university
3. **Multi-class Classification using a Regularized Logistic regression**: Hand-written digits images and **Neural Network**(just the forward-propagation)
4. **Neural Network Training and the Backpropagation algorithm**: Hand-written digits images
5. **Machine Learning Problem Solving**: handling high bias(underfitting) and high variance(overfitting) problems. (Using Linear and Polynomial Regression)
6. **Support Vector Machines**: Spam classification (Gaussian kernel) (IMPLEMENTED USING A LIBRARY)
7. **K-means** (Clustering) and **Principal Component Analysis** (Dimensionality Reduction)
8. **Anomaly Detection**: detecting anomalous behaviours in server computers (Gaussian Distribution) and **Collaborative filtering**: Movie ratings

As suggested by the lecturer, a singular attention was made towards using vectorized implementations instead of traditional for-loops in order to make good use of Linear Algebra library and have models that run faster.

### Fitting the parameters of the model

In each exercise, the **Batch Gradient Descent** algorithm was used to fit the parameters of the model. No Stochastic Gradient Descent(Updating the model's parameters on each example) nor Mini-Batch Gradient descent(Updating the model's parameters on a specified number of examples b) was implemented in this repository.

#### Memo: 
1. Compute the **regularization-term** for both cost and gradient(to prevent overfitting)
2. Compute the **Cost** for a given value of Theta
3. Compute the **Gradient** for a given value of Theta


## Running a single application

To run ex1, simply execute ex1 in Octave/MATLAB.

For more details about an exercise (Implementation, Data-set or Context) see the corresponding .pdf.