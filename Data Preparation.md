# Data Clean

## Missing values

### Methods

#### Deletion

* **Listwise**

    Listwise deletion (complete-case analysis) removes all data for an observation that has one or more missing values. Particularly if the missing data is limited to a small number of observations, you may just opt to eliminate those cases from the analysis.
    
    
* **Pairwise**

    Pairwise deletion analyses all cases in which the variables of interest are present and thus maximizes all data available by an analysis basis. A strength to this technique is that it increases power in your analysis.
    
    
* **Dropping Variables**

    Sometimes you can drop variables if the data is missing for more than 60% observations but only if that variable is insignificant.
    
#### Imputation

* **Mean, median, mode**

    * **Mean:** We might choose to use the mean, for example, if the variable is otherwise generally normally distributed (and in particular does not have any skewness).
    
    * **Median:** If the data does exhibit some skewness though (e.g., there are a small number of very large values) then the median might be a better choice.
    
    * **Mode:** For categoric variables we might choose to use the mode as the default to fill in for the otherwise missing values.
    
    
* **Predictive models**

    * **Linear Regression:** Cases with complete data for the predictor variables are used to generate the regression equation; the equation is then used to predict missing values for incomplete cases.
    
        However, there are several disadvantages of this model which tend to outweigh the advantages. First, because the replaced values were predicted from other variables they tend to fit together “too well” and so standard error is deflated. One must also assume that there is a linear relationship between the variables used in the regression equation when there may not be one.  
        
    * **KNN(K Nearest Neighbors):** In this method, k neighbors are chosen based on some distance measure and their average is used as an imputation estimate. The method requires the selection of the number of nearest neighbors, and a distance metric. KNN can predict both discrete attributes (the most frequent value among the k nearest neighbors) and continuous attributes (the mean among the k nearest neighbors)
        
        **Continuous Data:** The commonly used distance metrics for continuous data are Euclidean, Manhattan and Cosine
        
        **Categorical Data:** Hamming distance is generally used in this case. It takes all the categorical attributes and for each, count one if the value is not the same between two points. The Hamming distance is then equal to the number of attributes for which the value was different.

## Outliers

### Methods

* **Standard Deviation**

    In statistics, If a data distribution is approximately normal then about 68% of the data values lie within one standard deviation of the mean and about 95% are within two standard deviations, and about 99.7% lie within three standard deviations.
    
    Therefore, if you have any data point that is more than 3 times the standard deviation, then those points are very likely to be anomalous or outliers.
    
    
* **Boxplots**

    Box plots are a graphical depiction of numerical data through their quantiles. It is a very simple but effective way to visualize outliers. Think about the lower and upper whiskers as the boundaries of the data distribution. Any data points that show above or below the whiskers, can be considered outliers or anomalous.
    
    ![Boxplots.png](attachment:Boxplots.png)
    
    
* **DBScan Clustering**

    DBScan is a clustering algorithm that’s used cluster data into groups. It is also used as a density-based anomaly detection method with either single or multi-dimensional data. Other clustering algorithms such as k-means and hierarchal clustering can also be used to detect outliers.
    
    * **Core Points**: In order to understand the concept of the core points, we need to visit some of the hyperparameters used to define DBScan job. First hyperparameter (HP) is **min_samples**. This is simply the minimum number of core points needed in order to form a cluster. second important HP is **eps**. **eps** is the maximum distance between two samples for them to be considered as in the same cluster.
    
    * **Border Points**: Points are in the same cluster as core points but much further away from the centre of the cluster.
    
    * **Noise Points**: Those are data points that do not belong to any cluster. They can be anomalous or non-anomalous and they need further investigation.
    
    ![DBScan.png](attachment:DBScan.png)
    
    
* **Isolation Forest**

    Isolation Forest is an unsupervised learning algorithm that belongs to the ensemble decision trees family. This approach is different from all previous methods. All the previous ones were trying to find the normal region of the data then identifies anything outside of this defined region to be an outlier or anomalous.
    
    Isolation forest’s basic principle is that outliers are few and far from the rest of the observations. To build a tree (training), the algorithm randomly picks a feature from the feature space and a random split value ranging between the maximums and minimums. This is made for all the observations in the training set. To build the forest a tree ensemble is made averaging all the trees in the forest.
    
    Then for prediction, it compares an observation against that splitting value in a “node”, that node will have two node children on which another random comparisons will be made. The number of “splittings” made by the algorithm for an instance is named: “path length”. As expected, outliers will have shorter path lengths than the rest of the observations.
    
    An outlier score can computed for each observation:
    
    $$s(x,n)=2^{-\frac{E(h(x))}{c(n)}}$$
    
    Where $h(x)$ is the path length of the sample x, and $c(n)$ is the ‘unsuccessful length search’ of a binary tree (the maximum path length of a binary tree from root to external node). n is the number of external nodes. After giving each observation a score ranging from 0 to 1: 1 meaning more outlyingness and 0 meaning more normality. A threshold can be specified (ie. 0.55 or 0.60)
    
    ![Isolation%20Forest.png](attachment:Isolation%20Forest.png)

## Unbalanced data

* **What is unbalanced data?**

    Unbalanced data refers to classification problems where we have unequal instances for different classes.
    
    
* **Why is unbalanced data a problem in machine learning?**

    Most machine learning classification algorithms are sensitive to unbalance in the predictor classes. Let’s consider an even more extreme example than our breast cancer dataset: assume we had 10 malignant vs 90 benign samples. A machine learning model that has been trained and tested on such a dataset could now predict “benign” for all samples and still gain a very high accuracy. An unbalanced dataset will bias the prediction model towards the more common class!
    
### Methods

* **Under-sampling**

    With under-sampling, we randomly select a subset of samples from the class with more instances to match the number of samples coming from each class. In our example, we would randomly pick 241 out of the 458 benign cases. The main disadvantage of under-sampling is that we lose potentially relevant information from the left-out samples.
    
    1. **Near miss**
    
        The general idea behind near miss is to only the sample the points from the majority class necessary to distinguish between other classes.
        
        **NearMiss-1** select samples from the majority class for which the average distance of the N closest samples of a minority class is smallest.
        
        ![Nearmiss_1.png](attachment:Nearmiss_1.png)
        
        **NearMiss-2** select samples from the majority class for which the average distance of the N farthest samples of a minority class is smallest.
        
        ![Nearmiss_2.png](attachment:Nearmiss_2.png)
        
    2. **Tomeks links**
    
        A Tomek’s link exists if two observations of different classes are the nearest neighbors of each other. In the figure below, a Tomek’s link is illustrated by highlighting the samples of interest in green.
        
        ![Tomik.png](attachment:Tomik.png)
    
        For this undersampling strategy, we'll remove any observations from the majority class for which a Tomek's link is identified. Depending on the dataset, this technique won't actually achieve a balance among the classes - it will simply "clean" the dataset by removing some noisy observations, which may result in an easier classification problem. As I discussed earlier, most classifiers will still perform adequately for imbalanced datasets as long as there's a clear separation between the classifiers. Thus, by focusing on removing noisy examples of the majority class, we can improve the performance of our classifier even if we don't necessarily balance the classes.
    
    
* **Over-sampling**

    With oversampling, we randomly duplicate samples from the class with fewer instances or we generate additional instances based on the data that we have, so as to match the number of samples in each class. While we avoid losing information with this approach, we also run the risk of overfitting our model as we are more likely to get the same samples in the training and in the test data, i.e. the test data is no longer independent from training data. This would lead to an overestimation of our model’s performance and generalizability.
    
    1. **SMOTE**
        
        Synthetic Minority Over-sampling Technique (SMOTE) is a technique that generates new observations by interpolating between observations in the original dataset.
        
        For a given observation $x_{i}$, a new (synthetic) observation is generated by interpolating between one of the k-nearest neighbors, $x_{zi}$.
        
        $$x_{new}=x_i+\lambda(x_{zi}-x_i)$$
        
        where $\lambda$ is a random number in the range [0, 1]. This interpolation will create a sample on the line between $x_i$ and $x_{zi}$.
           
    2. **ADASYN**
    
        Adaptive Synthetic (ADASYN) sampling works in a similar manner as SMOTE, however, the number of samples generated for a given $x_i$ is proportional to the number of nearby samples which do not belong to the same class as $x_i$. Thus, ADASYN tends to focus solely on outliers when generating new synthetic training examples.

    
* **Class weight**

    One of the simplest ways to address the class imbalance is to simply provide a weight for each class which places more emphasis on the minority classes such that the end result is a classifier which can learn equally from all classes.

    In a tree-based model where you're determining the optimal split according to some measure such as decreased entropy, you can simply scale the entropy component of each class by the corresponding weight such that you place more emphasis on the minority classes.
    
    In a gradient-based model, you can scale the calculated loss for each observation by the appropriate class weight such that you place more significance on the losses associated with minority classes.

# Feature Engineering

## Feature scaling

### Motivation

1. The range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.

2. Another reason why feature scaling is applied is that gradient descent converges much faster with feature scaling than without it.

3. It's also important to apply feature scaling if regularization is used as part of the loss function (so that coefficients are penalized appropriately).

### Methods

1. **Rescaling (min-max normalization)**

    Rescaling is the simplest method and consists in rescaling the range of features to scale the range in [0, 1] or [−1, 1].

$$x^{'}=\frac{x-min(x)}{max(x)-min(x)}$$

2. **Mean normalization**

$$x^{'}=\frac{x-avg(x)}{max(x)-min(x)}$$

3. **Standardization (Z-score Normalization)**

$$x^{'}=\frac{x-\bar{x}}{\sigma}$$

4. **Scaling to unit length**

$$x^{'}=\frac{x}{\lVert x \rVert}$$

## Categorical feature encoding

### Methods

1. **Ordinal Encoding**

    We use this technique when the categorical feature is ordinal. In this case, retaining the orders is important.
    
    
2. **One-hot Encoding**

    We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category. These newly created binary features are known as **Dummy variables**. 
    
    **Disadvantages**
    
    * In some caese, One-hot encoding introduce sparsity in the dataset. In other words, it creates multiple dummy features in the dataset without adding much information.
    
    * It might lead to a Dummy variable trap. It is a phenomenon where features are highly correlated. That means using the other variables, we can easily predict the value of a variable.


3. **Binary Encoding**

    In this encoding scheme, the categorical feature is first converted into numerical using an ordinal encoder. Then the numbers are transformed in the binary number. After that binary value is split into different columns.
    
    **Advantages**
    
    * Binary encoding is a memory-efficient encoding scheme as it uses fewer features than one-hot encoding.
    
    * It reduces the curse of dimensionality for data with high cardinality.


```python

```
