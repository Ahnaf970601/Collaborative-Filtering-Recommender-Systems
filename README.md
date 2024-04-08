# Collaborative Filtering Recommender Systems
 
Here I am going to make a Rating and Recommender System with Collaborative Filtering

Here is the Movie ratings Dataset-
The original dataset has 9000 movies rated by 600 users. The dataset has been reduced in size to focus on movies from the years since 2000. This dataset consists of ratings on a scale of 0.5 to 5 in 0.5 step increments.

Y: The ratings matrix of dimensions nm (number of movies) by nu (number of users) where Y(i, j) is the rating given by user j to movie i.
R: The binary indicator matrix of dimensions nm by nu where R(i, j) is 1 if user j gave a rating to movie i and 0 otherwise.
X: The feature matrix for movies of dimensions nm by n where the i-th row corresponds to the feature vector for the i-th movie.
W: The parameter matrix for users of dimensions nu by n where the j-th row corresponds to the parameter vector for the j-th user.
b: The bias vector for users.
The feature and parameter vectors both have n elements, and in this particular exercise, n is set to 10, so both X and W are matrices with 10 columns. These matrices will be used throughout the exercise to develop the collaborative filtering cost model.



2. Collaborative filtering learning algorithm-

The cost function for collaborative filtering typically measures the accuracy of predictions by summing up the squared differences between observed ratings and predictions for all user-item pairs for which ratings are available. The regularization terms are usually added to this cost function to avoid overfitting by penalizing the magnitudes of the parameters.


Here is my code-



As you can see I got the expected result of the collaborative cost function

3. Vectorized Implementation-

For a collaborative filtering algorithm, especially when dealing with large datasets, a vectorized implementation is crucial for computational efficiency. This typically means avoiding explicit loops in favor of matrix operations that can be efficiently computed by libraries such as NumPy in Python.

The vectorized cost function for collaborative filtering with matrix factorization, including regularization, can be written as:


Here is my code for Vectorized Implementation-


4. Movie Recommendations-

Now I will start training my algorithm to make a movie recommendations-



Let’s now train the collaborative filtering model. This will learn the parameters 𝐗 , 𝐖 , and 𝐛 .

The operations involved in learning 𝑤 , 𝑏 , and 𝑥  simultaneously do not fall into the typical ‘layers’ offered in the TensorFlow neural network package. Consequently, the flow used in Course 2: Model, Compile(), Fit(), Predict(), are not directly applicable. Instead, we can use a custom training loop.

Recall from earlier labs the steps of gradient descent.

repeat until convergence:
compute forward pass
compute the derivatives of the loss relative to parameters
update the parameters using the learning rate and the computed derivatives
TensorFlow has the marvelous capability of calculating the derivatives for you. This is shown below. Within the tf.GradientTape() section, operations on Tensorflow Variables are tracked. When tape.gradient() is later called, it will return the gradient of the loss relative to the tracked variables. The gradients can then be applied to the parameters using an optimizer. This is a very brief introduction to a useful feature of TensorFlow and other machine learning frameworks. Further information can be found by investigating "custom training loops" within the framework of interest.


5. Recommendations-

Now I will compute the ratings for all the movies and display the movies that are recommended.


