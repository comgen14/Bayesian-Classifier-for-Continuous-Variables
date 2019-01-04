# Bayesian-Classifier-for-Continuous-Variables

This is the Bayesian Learning machine I designed for my final project.

For my system, I decided to use a Naive Bayes Classifier. The reason I choose a Naive Bayes Classifier is because Naive Bayes does a really good job of handling large amounts of data, data with large amounts of attributes, and, most importantly, handles missing attributes very well.

For continuous attributes, I calculated the conditional class prior simply by using arithmetic, but because most of the attributes were continuous variables, to calculate the P(X=x ┤|  C ) I had to take a different approach. I decided to use the Gaussian Probability Density Function to calculate the conditional class prior. 
P(X=x ┤|  C )=1/(√2πσ^2 ) e^((x-μ)^2/(2σ^2 ))  

Where μ is the sample mean and σ^2 is the sample variance. The downside to this approach is that the Gaussian PDF assumes the sample distribution is gaussian. If the sample is not gaussian distributed it may lead to classification errors.
