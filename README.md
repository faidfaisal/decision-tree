# Decision Tree Algorithm

An efficient implementation of the Decision Tree algorithm for classification and prediction tasks using machine learning techniques.

## Overview

This project implements a Decision Tree classifier, a powerful supervised learning algorithm that creates a tree-like model of decisions based on feature values. The algorithm recursively splits the dataset into subsets based on the most significant attributes, making it highly interpretable and effective for both classification and regression tasks.

## Features

- **Multiple Splitting Criteria**: Supports information gain, gain ratio, and Gini index for optimal feature selection
- **Automated Tree Construction**: Builds decision trees using entropy-based splitting methods
- **Pruning Capabilities**: Prevents overfitting through tree pruning techniques
- **Multi-class Classification**: Supports classification problems with multiple target classes
- **High Interpretability**: Generates human-readable decision rules from the trained model

## Algorithm Overview

The Decision Tree algorithm operates through the following steps:

1. **Feature Selection**: Identifies the best attribute to split the data using information gain or Gini index
2. **Tree Construction**: Recursively partitions the dataset to create branches and leaf nodes
3. **Stopping Criteria**: Terminates splitting based on predefined conditions (max depth, min samples, purity)
4. **Classification**: Traverses the tree from root to leaf to predict class labels for new instances

This approach creates an intuitive model that mimics human decision-making processes.

## Prerequisites

- C++ compiler with C++11 support or later (e.g., g++, clang++)
- Standard Template Library (STL)

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/decision-tree.git
cd decision-tree
```

Compile the program:
```bash
g++ -std=c++11 -o decisiontree decision_tree.cpp -O2
```

## Usage

Run the Decision Tree algorithm on your dataset:
```bash
./decisiontree   [output_file]
```

### Parameters

- `training_file`: Path to the training dataset file
- `test_file`: Path to the test dataset file for prediction
- `output_file`: (Optional) File to write the results and tree structure

### Input Format

The input files should be in CSV format with the last column as the class label:
```
feature1,feature2,feature3,class
value1,value2,value3,classA
value4,value5,value6,classB
value7,value8,value9,classA
...
```

The first row should contain feature names, and subsequent rows contain data instances.

### Example
```bash
./decisiontree train.csv test.csv results.txt
```

This command trains a decision tree on `train.csv`, predicts classes for `test.csv`, and writes the results to `results.txt`.

## Output

The program generates:

1. **Decision Tree Structure**: Visual representation of the trained tree
2. **Classification Results**: Predicted class labels for test instances
3. **Performance Metrics**: Accuracy, precision, recall, and confusion matrix

Sample output:
```
Decision Tree Structure:
Root [feature1]
├── value <= threshold
│   ├── [feature2]
│   │   ├── Leaf: ClassA
│   │   └── Leaf: ClassB
└── value > threshold
    └── Leaf: ClassC

Test Accuracy: 92.5%
Confusion Matrix:
         ClassA  ClassB  ClassC
ClassA     45      2       1
ClassB      1     38      3
ClassC      0      2      43
```

## Project Structure
```
decision-tree/
├── decision_tree.cpp # Decision Tree algorithm implementation
├── README.md         # This file
└── examples/         # Sample datasets and outputs
```

## Applications

Decision Trees are widely used in:

- Medical diagnosis and healthcare predictions
- Credit risk assessment and fraud detection
- Customer segmentation and churn prediction
- Image classification and pattern recognition
- Feature importance analysis

## Performance

Decision Trees offer several advantages:

- High interpretability and explainability
- Handles both numerical and categorical data
- Requires minimal data preprocessing
- Fast training and prediction
- Robust to outliers

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Course Information

**ESE 327 Project 2**  
Developed as part of the ESE 327 coursework.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Quinlan, J. R. (1986). Induction of decision trees. *Machine Learning*, 1(1), 81-106.
- Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). *Classification and Regression Trees*. CRC press.

## Author

[Faid Faisal]  
[faidfaisal1@gmail.com]  


---
