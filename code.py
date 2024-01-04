# NAME: CHICHARI ANUSHA
# ROLL NO: 21CH10020

# In[1]:


# importing all the libraries needed
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# In[2]:


# Loading the dataset
data = pd.read_csv('car_evaluation.csv')


# In[3]:


# Labelling the columns based on the provided information
column_names = ['Price Buying', 'Price Maintenance', 'Doors', 'Persons', 'Lug_boot', 'Safety', 'Acceptability']
data.columns = column_names

# Encoding categorical features using LabelEncoder
label_encoders = {}
categorical_columns = ['Price Buying', 'Price Maintenance', 'Doors', 'Persons', 'Lug_boot', 'Safety', 'Acceptability']

for column in categorical_columns:
    label_encoder = LabelEncoder()
    data[column] = label_encoder.fit_transform(data[column])
    label_encoders[column] = label_encoder


# In[4]:


# Splitting the data into features (X) and target variable (y)
X = data.drop('Acceptability', axis=1)
y = data['Acceptability']

# Splitting the dataset into Train (60%), Validation (20%), and Test (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# In[5]:


# Convert Pandas DataFrames to NumPy arrays
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_val = X_val.to_numpy()
y_val = y_val.to_numpy()


# In[6]:


# Implementation: 
class TreeNode:
    def __init__(self, depth, max_depth, entropy_threshold):
        self.depth = depth  # Track the depth of this node
        self.max_depth = max_depth
        self.entropy_threshold = entropy_threshold
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None  # For leaf nodes

    def entropy(self, y):
        # Calculate entropy
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def best_split(self, X, y):
        num_samples, num_features = X.shape
        if num_samples <= 1:
            return None

        parent_entropy = self.entropy(y)
        best_info_gain = -1
        best_split = None

        for feature_index in range(num_features):
            unique_values = np.unique(X[:, feature_index])
            for threshold in unique_values:
                left_indices = X[:, feature_index] < threshold
                right_indices = ~left_indices
                if sum(left_indices) < 2 or sum(right_indices) < 2:
                    continue

                left_entropy = self.entropy(y[left_indices])
                right_entropy = self.entropy(y[right_indices])
                info_gain = parent_entropy - (left_entropy + right_entropy)

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_split = (feature_index, threshold)

        return best_split

    def build_tree(self, X, y):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        if (self.max_depth is not None and self.depth >= self.max_depth) or \
                (num_samples < 2) or \
                (num_classes == 1) or \
                (self.entropy(y) < self.entropy_threshold):
            self.value = np.argmax(np.bincount(y))
            return

        best_split = self.best_split(X, y)

        if best_split is not None:
            feature_index, threshold = best_split
            self.feature_index = feature_index
            self.threshold = threshold
            left_indices = X[:, feature_index] < threshold
            right_indices = ~left_indices
            self.left = TreeNode(self.depth + 1, self.max_depth, self.entropy_threshold)
            self.right = TreeNode(self.depth + 1, self.max_depth, self.entropy_threshold)
            self.left.build_tree(X[left_indices], y[left_indices])
            self.right.build_tree(X[right_indices], y[right_indices])
        else:
            self.value = np.argmax(np.bincount(y))

    def predict(self, X):
        if self.value is not None:
            return self.value
        if X[self.feature_index] < self.threshold:
            return self.left.predict(X)
        else:
            return self.right.predict(X)


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, entropy_threshold=0.05):
        self.max_depth = max_depth
        self.entropy_threshold = entropy_threshold
        self.root = None

    def fit(self, X, y):
        self.root = TreeNode(depth=0, max_depth=self.max_depth, entropy_threshold=self.entropy_threshold)
        self.root.build_tree(X, y)

    def predict(self, X):
        predictions = [self.root.predict(x) for x in X]
        return np.array(predictions)


# In[7]:


# Experiment 1: Varying Entropy Threshold
thresholds = [0, 0.25, 0.5, 0.75, 1]
train_accuracy = []
val_accuracy = []

for threshold in thresholds:
    # Create a Decision Tree classifier with the current threshold
    clf = DecisionTreeClassifier(max_depth=None, entropy_threshold=threshold)

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Make predictions on training and validation data
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)

    # Calculate accuracy on training and validation data
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    train_accuracy.append(train_acc)
    val_accuracy.append(val_acc)

# Plot Percentage Accuracy vs. Threshold
plt.figure(figsize=(10, 6))
plt.plot(thresholds, train_accuracy, label='Training Accuracy', marker='o')
plt.plot(thresholds, val_accuracy, label='Validation Accuracy', marker='o')
plt.xlabel('Entropy Threshold')
plt.ylabel('Percentage Accuracy')
plt.title('Effect of Entropy Threshold on Accuracy')
plt.legend()
plt.grid()
plt.show()


# In[8]:


# Plotting Percentage Accuracy vs. Threshold as a bar chart
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(thresholds))

plt.bar(index, train_accuracy, bar_width, label='Training Accuracy', alpha=0.7)
plt.bar(index + bar_width, val_accuracy, bar_width, label='Validation Accuracy', alpha=0.7)

plt.xlabel('Entropy Threshold')
plt.ylabel('Percentage Accuracy')
plt.title('Effect of Entropy Threshold on Accuracy')
plt.xticks(index + bar_width / 2, thresholds)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adding data labels
for i in range(len(thresholds)):
    plt.text(index[i] - 0.02, train_accuracy[i] + 0.01, f'{train_accuracy[i]:.2f}', fontsize=10, ha='center')
    plt.text(index[i] + bar_width - 0.02, val_accuracy[i] + 0.01, f'{val_accuracy[i]:.2f}', fontsize=10, ha='center')

plt.show()


# In[9]:


# Plotting Size vs. Threshold
tree_sizes = []  # To store the size (depth) of the decision tree for each threshold

for threshold in thresholds:
    # Creating a Decision Tree classifier with the current threshold
    clf = DecisionTreeClassifier(max_depth=None, entropy_threshold=threshold)

    # Training the classifier
    clf.fit(X_train, y_train)

    # Getting the size (depth) of the decision tree
    tree_size = clf.root.depth  # Assuming root node stores depth
    tree_sizes.append(tree_size)

# Plotting Size vs. Threshold
plt.figure(figsize=(10, 6))
plt.plot(thresholds, tree_sizes, marker='o')
plt.xlabel('Entropy Threshold')
plt.ylabel('Size of Decision Tree (Depth)')
plt.title('Size of Decision Tree vs. Entropy Threshold')
plt.grid()
plt.show()


# In[10]:


# Finding the optimal threshold based on validation accuracy
optimal_threshold = thresholds[np.argmax(val_accuracy)]
print(f'Optimal Entropy Threshold based on Validation Accuracy: {optimal_threshold}')


# In[11]:


# Convert Pandas DataFrames to NumPy arrays for the test set
X_test = X_test.to_numpy()


# In[12]:


# Experiment2:
# Experiment 2(a): Calculate overall training and testing accuracy with optimal hyperparameter

# Create a Decision Tree classifier with the optimal threshold
clf_optimal = DecisionTreeClassifier(max_depth=None, entropy_threshold=optimal_threshold)

# Train the classifier
clf_optimal.fit(X_train, y_train)

# Make predictions on training and testing data
y_train_pred_optimal = clf_optimal.predict(X_train)
y_test_pred_optimal = clf_optimal.predict(X_test)

# Calculate accuracy on training and testing data
train_acc_optimal = accuracy_score(y_train, y_train_pred_optimal)
test_acc_optimal = accuracy_score(y_test, y_test_pred_optimal)

print(f'Overall Training Accuracy with Early Stopping: {train_acc_optimal:.2f}')
print(f'Overall Testing Accuracy with Early Stopping: {test_acc_optimal:.2f}')


# In[13]:


# Experiment 2(b): Plot percentage accuracy on training and validation data after each branch formation
current_depth = 0
max_depth = 20  # I took the range up to a maximum depth of 20
train_accuracy_per_depth = []
val_accuracy_per_depth = []

while current_depth <= max_depth:
    # Make predictions on training and validation data
    y_train_pred_optimal = clf_optimal.predict(X_train)
    y_val_pred_optimal = clf_optimal.predict(X_val)

    # Calculate accuracy on training and validation data
    train_acc_optimal = accuracy_score(y_train, y_train_pred_optimal)
    val_acc_optimal = accuracy_score(y_val, y_val_pred_optimal)

    train_accuracy_per_depth.append(train_acc_optimal)
    val_accuracy_per_depth.append(val_acc_optimal)

    # Build the tree up to the next depth
    clf_optimal.root.build_tree(X_train, y_train)

    current_depth += 1

# Plotting Accuracy vs Depth
plt.figure(figsize=(10, 6))
depths = range(1, len(train_accuracy_per_depth) + 1)
plt.plot(depths, train_accuracy_per_depth, label='Training Accuracy', marker='o')
plt.plot(depths, val_accuracy_per_depth, label='Validation Accuracy', marker='o')
plt.xlabel('Depth of Decision Tree')
plt.ylabel('Percentage Accuracy')
plt.title('Accuracy vs Depth of Decision Tree (Early Stopping)')
plt.legend()
plt.grid()
plt.show()


# In[14]:


# Create a DecisionTreeClassifier instance
clf_exp2c = DecisionTreeClassifier(entropy_threshold=optimal_threshold)

# Experiment 2(c): Stop when validation accuracy decreases and analyze accuracy on training and testing data
current_depth = 0
train_accuracy_per_depth_exp2c = []
val_accuracy_per_depth_exp2c = []

while True:
    # Build the tree up to the current depth
    clf_exp2c.root = TreeNode(depth=0, max_depth=current_depth, entropy_threshold=optimal_threshold)
    clf_exp2c.root.build_tree(X_train, y_train)

    # Make predictions on training and validation data
    y_train_pred_exp2c = clf_exp2c.predict(X_train)
    y_val_pred_exp2c = clf_exp2c.predict(X_val)

    # Calculate accuracy on training and validation data
    train_acc_exp2c = accuracy_score(y_train, y_train_pred_exp2c)
    val_acc_exp2c = accuracy_score(y_val, y_val_pred_exp2c)

    train_accuracy_per_depth_exp2c.append(train_acc_exp2c)
    val_accuracy_per_depth_exp2c.append(val_acc_exp2c)

    # Check if validation accuracy starts decreasing
    if current_depth > 0 and val_accuracy_per_depth_exp2c[current_depth] < val_accuracy_per_depth_exp2c[current_depth - 1]:
        break

    current_depth += 1

# Find the total number of nodes at the point when validation accuracy starts to decrease
total_nodes_at_stop_exp2c = np.sum([2 ** i for i in range(current_depth + 1)])

# Plotting Accuracy vs Depth
plt.figure(figsize=(10, 6))
depths_exp2c = range(1, len(train_accuracy_per_depth_exp2c) + 1)
plt.plot(depths_exp2c, train_accuracy_per_depth_exp2c, label='Training Accuracy', marker='o')
plt.plot(depths_exp2c, val_accuracy_per_depth_exp2c, label='Validation Accuracy', marker='o')
plt.xlabel('Depth of Decision Tree')
plt.ylabel('Percentage Accuracy')
plt.title('Accuracy vs Depth of Decision Tree (Experiment 2(c))')
plt.legend()
plt.grid()
plt.show()

print(f'Total number of nodes at stopping point in Experiment 2(c): {total_nodes_at_stop_exp2c}')
print(f'Validation Accuracy in Experiment 2(c): {val_accuracy_per_depth_exp2c[-1]:.2f}')
print(f'Training Accuracy in Experiment 2(c): {train_accuracy_per_depth_exp2c[-1]:.2f}')


# In[15]:


# Experiment 3: 
# Function to print rules for classification
def print_rules(node, antecedent="IF"):
    if node is None:
        return

    # If it's a leaf node, print the classification result
    if node.value is not None:
        class_label = label_encoders['Acceptability'].classes_[node.value]
        print(f"{antecedent} => THEN (Class {class_label})")

    # If it's not a leaf node, recursively generate rules for children
    if node.feature_index is not None:
        left_rule = f"{antecedent} AND ({column_names[node.feature_index]} == {node.threshold})"
        right_rule = f"{antecedent} AND NOT ({column_names[node.feature_index]} == {node.threshold})"
        print_rules(node.left, left_rule)
        print_rules(node.right, right_rule)

# Printing rules for Experiment 1 (Optimal Threshold)
print("Rules for Experiment 1 (Optimal Threshold):")
print_rules(clf_optimal.root)

# Printing rules for Experiment 2 (Early Stopping)
print("\nRules for Experiment 2 (Early Stopping):")
print_rules(clf_optimal.root)







