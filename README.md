# Bayesian-Classifier-for-Continuous-Variables

This is the Bayesian Learning machine I designed for my final project.

For my system, I decided to use a Naive Bayes Classifier. The reason I choose a Naive Bayes Classifier is because Naive Bayes does a really good job of handling large amounts of data, data with large amounts of attributes, and, most importantly, handles missing attributes very well.

For continuous attributes, I calculated the conditional class prior simply by using arithmetic, but because most of the attributes were continuous variables, to calculate the P(X=x ┤|  C ) I had to take a different approach. I decided to use the Gaussian Probability Density Function to calculate the conditional class prior. 

P(X=x ┤|  C )=1/(√2πσ^2 ) e^((x-μ)^2/(2σ^2 ))  

Where μ is the sample mean and σ^2 is the sample variance. The downside to this approach is that the Gaussian PDF assumes the sample distribution is gaussian. If the sample is not gaussian distributed it may lead to classification errors.

To prevent counting variables multiple times, I calculated the correlation between all the variables, generating a correlation array

r=(∑_(i=1)^N〖(x_i-x ̅)(y_i-y ̅)〗)/(√∑_(i=1)^N(x_i-x ̅ )^2  ∑_(i=1)^N(y_i-y ̅ )^2 )

I squared this value, the coefficient of determination, and set a correlation threshold, the variable LIMIT, to weed out any attributes the were too highly correlated. High correlation means that there is a high percentage that these two attributes were dependent on each other. The algorithm chose to ignore the y values in the attribute tuples. In testing I found ignoring attributes with a coefficient of determination higher than .7125 gave the best results

Improvements I plan to make are coding all variables as random variables and using a Radon-Nikodyn Density function to calculate the probabilities, and to use some form of linear regression to calculate the opitmal r^2 value.
