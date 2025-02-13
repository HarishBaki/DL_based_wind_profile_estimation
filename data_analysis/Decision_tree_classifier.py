import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature        # Feature index to split on
        self.threshold = threshold    # Threshold value for splitting
        self.left = left              # Left child (Node)
        self.right = right            # Right child (Node)
        self.value = value            # Class label if the node is a leaf

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth    # Maximum depth of the tree
        self.tree = None              # Root of the decision tree

    def fit(self, X, y):
        # Recursively build the decision tree
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # Stopping criteria: If the node is pure or max depth is reached
        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth == self.max_depth):
            return Node(value=np.bincount(y).argmax())

        # Find the best split
        feature, threshold = self._find_best_split(X, y)

        # Split the data
        X_left, y_left = X[X[:, feature] <= threshold], y[X[:, feature] <= threshold]
        X_right, y_right = X[X[:, feature] > threshold], y[X[:, feature] > threshold]

        # Recursively build the left and right subtrees
        left = self._build_tree(X_left, y_left, depth + 1)
        right = self._build_tree(X_right, y_right, depth + 1)

        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def _find_best_split(self, X, y):
        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                y_left = y[X[:, feature] <= threshold]
                y_right = y[X[:, feature] > threshold]

                gini = (len(y_left) / len(y)) * self._gini_impurity(y_left) + \
                       (len(y_right) / len(y)) * self._gini_impurity(y_right)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini_impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p ** 2)

    def predict(self, X):
        return np.array([self._predict_instance(x, self.tree) for x in X])

    def _predict_instance(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_instance(x, node.left)
        else:
            return self._predict_instance(x, node.right)
