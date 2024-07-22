##########################################################################
# Business Goal: Predict whether a person's income is above $50K       ###
# Team One: Omogbolahan Alli, Tinashe Kaseke, Michael Avosa            ###
##########################################################################

# Given the business goal, we will use the Adult dataset to predict whether a person's income is above $50K. #nolint

# Install required packages if not already installed
install.packages(c("ggplot2","dplyr", "corrplot", "gridExtra", "DataExplorer", "xgboost", "data.table", "caret", "Matrix")) #nolint
############### Load libraries ######################################
lapply(c("ggplot2", "dplyr", "DataExplorer", "corrplot", "gridExtra", "xgboost", "data.table", "caret", "Matrix"), library, character.only = TRUE) #nolint
# Set Working Directory
setwd("/Users/gbolahanalli/Library/Mobile Documents/com~apple~CloudDocs/Projects/R/ML_With_R/data") # Set working directory #nolint

################# Load the Dataset ###################################
train_data <- read.csv("adult.data", header = FALSE, sep = ",") # nolint 
test_data <- read.csv("adult.test", header = FALSE, sep = ",") # nolint

# This dataset contains information about individuals and their income.
# This seems like a binary classification problem where we need to predict
# whether a person's income is above or below $50K. #nolint

#############################
# Structure of the Dataset ##
#############################

# Initial Exploration
head(train_data)  # Dimensions
head(test_data)   # First six rows

################# OBSERVATIONS ###################################
# We see that the training and test datasets have no column names.
# We will add the column names to the datasets.
# We also see that the test dataset has a missing first row. We will remove this row. #nolint
##################################################################

# Column names. These are the labels for our dataset.
column_names <- c("age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"); #nolint

# Add column names to the training and test datasets
colnames(train_data) <- column_names
colnames(test_data) <- column_names

# Remove the first row from the test dataset
test_data <- test_data[-1, ]

# Let's explore the datasets again
dim(train_data)  # Dimensions
dim(test_data)   # Dimensions

# Given we are trying to predict income, let's take a look at the label
unique(test_data$income) # Check unique values in the income column
unique(train_data$income) # Check unique values in the income column

# Percentage of test to total data
nrow(test_data) / (nrow(train_data) + nrow(test_data)) * 100

# Percentage of train to total data
nrow(train_data) / (nrow(train_data) + nrow(test_data)) * 100

################### INITIAL OBSERVATIONS ##########################
# We see a 33% test and 67% train split which is relatively good
# for our testing and training datasets. We do however need to check
# if there are no data quality issues in the datasets.
############### SUPERVISED OR UNSUPERVISED #########################
# Given the large example size and a significant number of labelled features,
# we can use supervised learning algorithms to predict the income.
# We can either use a Random Forest or XGBoost model to predict the
# income of individuals as they both handle missing values pretty well.
################### RANDOM FOREST #################################
# Here are some of the reasons why using Random Forest might be beneficial:
# 1. Random Forest is a powerful ensemble learning method that can be used for both classification and regression tasks. #nolint
# 2. It can handle large datasets with high dimensionality.
# 3. It is robust to overfitting and can handle noisy data.
# 4. It provides feature importance, which can help in understanding the most important features for prediction. #nolint

################### XGBOOST #######################################
# Here are some of the reasons why using XGBoost might be beneficial:
# 1. XGBoost is an optimized distributed gradient boosting library designed for speed and performance. #nolint
# 2. It is highly efficient and scalable, making it suitable for large datasets especially imbalanced ones. #nolint
# 3. It provides regularization techniques to prevent overfitting.
# 4. It has built-in cross-validation and early stopping to optimize model performance. #nolint

################### INITIAL OBSERVATIONS ###########################
# We observe that the dataset is largely imbalanced.
# Therefore the XGBoost model might be a better choice for this dataset.
# We also oberve that the dataset contains both categorical and numerical features. #nolint
# We will need to encode the categorical features before training the model. #nolint
####################################################################


##################################################################
# Let's do some data cleaning and Data Preprocessing #############
##################################################################

plot_missing(train_data) # Plot missing values in the training dataset
########## Visualize missing values ###############################
# This plot indicates that there are no missing values in any of
# the features of the training dataset. This simplifies the data
# cleaning process since no imputation is necessary.
##################################################################

plot_histogram(train_data) # Plot histograms for all numeric/continous columns
######### Histograms of Numerical Variables #######################
# 1. Age: The age distribution is right-skewed, with most
# individuals between 20 and 50 years old.
# 2. Capital Gain and Capital Loss: These variables are highly skewed with most
# values at 0, indicating that very few individuals have significant capital
# gains or losses.
# 3. Education Num: This variable appears to be categorical despite being
# numeric, with distinct peaks corresponding to different education levels.
# 4. Fnlwgt (Final Weight): This variable is also right-skewed, suggesting that
# most individuals have lower final weights, but there are some high-weight outliers. #nolint
# 5. Hours Per Week: This variable shows a normal distribution with a
# significant peak at around 40 hours per week, indicating that many
# individuals work full-time hours.
##################################################################


######################
# CORRELATION MATRIX #
######################
## Let's analyze the correlation matrix to identify highly correlated features #nolint
## which might indicate multicollinearity issues. #nolint

## Compute correlation matrix
cor_matrix <- cor(train_data %>% select_if(is.numeric))

## Visualize correlation matrix
corrplot(cor_matrix, method="color", tl.cex=0.8, tl.col="black", title="Correlation Matrix") #nolint

##################################################################
# We observe that there is no strong correlation between the
# numerical features in the dataset. This is good as it indicates.
# that there is no multicollinearity issue in the dataset
# Therefore we retain all the numerical features for training the model.
##################################################################

##################################################################
# 1. Age:Age has low to moderate correlations with other features,
# indicating it captures unique information.
# 2. Fnlwgt (Final Weight): Fnlwgt shows a moderate correlation with
# education_num. While this isn't a perfect correlation, it's something to note.
# 3. Education Num:Education Num has moderate correlations with fnlwgt,
# suggesting a potential redundancy. However, since this feature represents
# education level numerically, it likely holds distinct information valuable for the model. #nolint
# 4. Capital Gain and Capital Loss: Both capital_gain and capital_loss show
# very low correlations with other features. This indicates these features
# are not linearly related to other variables and might provide unique predictive information. #nolint
# 5. Hours Per Week:This feature also shows low correlations with other features, #nolint
# indicating it adds unique information to the dataset.
##################################################################

# Let see if there are outliers in the age feature
outliers <- boxplot.stats(train_data$age)$out 
table(outliers) # Frequency of outliers

# Bar plot for categorical variables - workclass
plot_workclass <- ggplot(train_data, aes(x=workclass)) + 
  geom_bar(fill="lightgreen") +
  theme_minimal() +
  labs(title="Distribution of Workclass", x="Workclass", y="Count") +
  theme(axis.text.x = element_text(angle=45, hjust=1))

# Bar plot for categorical variables - education
plot_education <- ggplot(train_data, aes(x=education)) + 
  geom_bar(fill="lightblue") +
  theme_minimal() +
  labs(title="Distribution of Education", x="Education", y="Count") +
  theme(axis.text.x = element_text(angle=45, hjust=1))

# Bar plot for categorical variables - marital_status
plot_marital_status <- ggplot(train_data, aes(x=marital_status)) + 
  geom_bar(fill="lightcoral") +
  theme_minimal() +
  labs(title="Distribution of Marital Status", x="Marital Status", y="Count") +
  theme(axis.text.x = element_text(angle=45, hjust=1))

# Bar plot for categorical variables - occupation
plot_occupation <- ggplot(train_data, aes(x=occupation)) + 
  geom_bar(fill="lightpink") +
  theme_minimal() +
  labs(title="Distribution of Occupation", x="Occupation", y="Count") +
  theme(axis.text.x = element_text(angle=45, hjust=1))

# Bar plot for categorical variables - relationship
plot_relationship <- ggplot(train_data, aes(x=relationship)) + 
  geom_bar(fill="#e0e0ff") +
  theme_minimal() +
  labs(title="Distribution of Relationship", x="Relationship", y="Count") +
  theme(axis.text.x = element_text(angle=45, hjust=1))

# Arrange the plots in a grid
grid.arrange(plot_workclass, plot_education, plot_marital_status, plot_occupation, plot_relationship, ncol = 2) #nolint

################ OBSERVATION ONE ##################################
# The "?" values in the "Workclass" and "Occupation" categories indicate
# missing data that should be handled.

# Replace all occurrences of "?" with NA across all features
# Assuming 'train_data' and 'test_data' are your datasets
train_data[train_data == "?"] <- NA
test_data[test_data == "?"] <- NA

#################  OBSERVATION TWO ################################
# The "Education" category has multiple levels that could be grouped
# The "Occupation" category has multiple levels that could be grouped
##################################################################

# Handling missing values (if any)
# For simplicity, let's drop rows with any missing values
train_data <- na.omit(train_data)
test_data <- na.omit(test_data)

# Convert factors to numeric and handle missing values
train_data <- train_data %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.factor), as.numeric))

# Similar for test data
test_data <- test_data %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(across(where(is.factor), as.numeric))

# Let's view the entire training dataset in a more readable format
View(train_data) # Entire training dataset
View(test_data) # Entire test dataset

# Adjusting the income variable from 1,2 to 0,1 for binary classification purposes #nolint
train_data$income <- as.numeric(train_data$income) - 1
test_data$income <- as.numeric(test_data$income) - 1

# Ensure 'income' is a factor with two levels (0 and 1)
train_data$income <- factor(train_data$income, levels = c(0, 1))
test_data$income <- factor(test_data$income, levels = c(0, 1))

View(train_data) # Entire training dataset
View(test_data) # Entire test dataset

# Summary of the test and training dataset
summary(train_data) # Summary of the training dataset
summary(test_data) # Summary of the test dataset

# Structure of the test and training dataset
str(train_data) # Structure of the training dataset
str(test_data) # Structure of the test dataset

set.seed(1) # Set seed for reproducibility
# Define the tuning grid
tune_grid <- expand.grid(nrounds = c(50, 100, 150), max_depth = c(3, 6, 9), eta = 0.1, gamma = 0, colsample_bytree = 0.8, min_child_weight = 1, subsample = 0.8 )

# Define cross-validation method
# This is a form of k-fold cross-validation, a standard method
# to estimate the skill of the model on new data.
# It involves splitting the training dataset into k groups,
# training the model on k-1 groups, and evaluating it on the remaining group.
train_control <- trainControl( method = "cv", number = 5, verboseIter = TRUE, savePredictions = "final") #nolint


# Train the XGBoost mod
xgb_model <- train(income ~ ., data = train_data, method = "xgbTree", tuneGrid = tune_grid, trControl = train_control, metric = "Accuracy", maximize = TRUE) #nolint

# Summary of the trained model
print(xgb_model)

# Predict on the test data
predictions <- predict(xgb_model, newdata = test_data)
confusionMatrix(predictions, as.factor(test_data$income))

# Check variable importance
importance <- varImp(xgb_model, scale = FALSE)
print(importance)
plot(importance)