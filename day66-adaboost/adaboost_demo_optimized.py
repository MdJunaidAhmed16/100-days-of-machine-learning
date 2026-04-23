"""
Optimized AdaBoost Implementation Demo
This script demonstrates AdaBoost algorithm with refactored, DRY code
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from mlxtend.plotting import plot_decision_regions


class AdaBoostDemo:
    """AdaBoost demonstration with clean, optimized code"""
    
    def __init__(self, X1, X2, labels):
        """Initialize dataset"""
        self.df = pd.DataFrame({
            'X1': X1,
            'X2': X2,
            'label': labels
        })
        self.X = self.df[['X1', 'X2']].values
        self.y = self.df['label'].values
        self.ensemble = []
        self.alphas = []
        
    def visualize_data(self, title="Dataset Visualization"):
        """Visualize the dataset"""
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.df['X1'], y=self.df['X2'], hue=self.df['label'], 
                       s=100, palette='Set1')
        plt.title(title)
        plt.show()
        
    def initialize_weights(self):
        """Initialize uniform weights for all samples"""
        self.df['weights'] = 1 / len(self.df)
        return self.df['weights'].values
        
    def train_weak_learner(self, max_depth=1):
        """Train a weak learner (decision tree stump)"""
        model = DecisionTreeClassifier(max_depth=max_depth)
        model.fit(self.X, self.y, sample_weight=self.df['weights'].values)
        return model
        
    def calculate_model_weight(self, error):
        """Calculate model weight (alpha) based on classification error"""
        if error == 0:
            return 10  # Large weight for perfect classifier
        if error >= 1:
            return 0.001  # Small weight for worse than random
        return 0.5 * np.log((1 - error) / error)
        
    def calculate_error(self, predictions):
        """Calculate weighted classification error"""
        misclassified = (predictions != self.y).astype(int)
        return np.sum(self.df['weights'].values * misclassified)
        
    def update_weights(self, predictions, alpha):
        """Update sample weights based on classifier performance"""
        errors = (predictions != self.y).astype(int)
        self.df['weights'] *= np.exp(alpha * errors)
        self.df['weights'] /= self.df['weights'].sum()  # Normalize
        
    def train(self, n_estimators=3, max_depth=1, visualize=False):
        """Train AdaBoost ensemble"""
        self.initialize_weights()
        
        for iteration in range(n_estimators):
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Train weak learner
            model = self.train_weak_learner(max_depth=max_depth)
            predictions = model.predict(self.X)
            
            # Calculate error and model weight
            error = self.calculate_error(predictions)
            alpha = self.calculate_model_weight(error)
            
            print(f"Error: {error:.4f}")
            print(f"Model Weight (alpha): {alpha:.4f}")
            
            # Store model and weight
            self.ensemble.append(model)
            self.alphas.append(alpha)
            
            # Visualize this iteration
            if visualize:
                self.visualize_iteration(iteration + 1, model, alpha, predictions)
            
            # Update weights for next iteration
            self.update_weights(predictions, alpha)
            print(f"Updated weights: {self.df['weights'].values}")
            
    def visualize_iteration(self, iteration, model, alpha, predictions):
        """Visualize decision boundary for current iteration"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot tree structure
        plt.sca(ax1)
        plot_tree(model, ax=ax1, filled=True)
        ax1.set_title(f"Iteration {iteration}: Decision Tree (alpha={alpha:.4f})")
        
        # Plot decision regions
        plt.sca(ax2)
        plot_decision_regions(self.X, self.y, clf=model, ax=ax2, legend=2)
        ax2.set_title(f"Iteration {iteration}: Decision Boundary")
        
        plt.tight_layout()
        plt.show()
        
    def predict(self, X=None):
        """Make predictions using the ensemble"""
        if X is None:
            X = self.X
            
        # Get predictions from all weak learners
        predictions = np.array([model.predict(X) for model in self.ensemble])
        
        # Weighted voting
        weighted_predictions = self.alphas * (2 * predictions - 1)
        return np.sign(np.sum(weighted_predictions, axis=0)).astype(int)
        
    def evaluate(self):
        """Evaluate ensemble accuracy"""
        predictions = self.predict()
        accuracy = np.mean(predictions == self.y)
        print(f"\nFinal Ensemble Accuracy: {accuracy:.4f}")
        return accuracy
        
    def show_summary(self):
        """Display summary of training"""
        summary_df = pd.DataFrame({
            'Weak Learner': [f"Tree {i+1}" for i in range(len(self.ensemble))],
            'Error': [self.calculate_error(model.predict(self.X)) 
                     for model in self.ensemble],
            'Alpha': self.alphas
        })
        print("\n" + "="*50)
        print("ADABOOST ENSEMBLE SUMMARY")
        print("="*50)
        print(summary_df)
        return summary_df


# ============== MAIN EXECUTION ==============
if __name__ == "__main__":
    # Initialize data
    X1 = [1, 2, 3, 4, 5, 6, 6, 7, 9, 9]
    X2 = [5, 3, 6, 8, 1, 9, 5, 8, 9, 2]
    labels = [1, 1, 0, 1, 0, 1, 0, 1, 0, 0]
    
    # Create AdaBoost demo instance
    adaboost = AdaBoostDemo(X1, X2, labels)
    
    # Visualize original data
    adaboost.visualize_data("Original Dataset")
    
    # Train ensemble
    print("Training AdaBoost Ensemble...")
    adaboost.train(n_estimators=3, max_depth=1, visualize=True)
    
    # Evaluate and show results
    adaboost.evaluate()
    adaboost.show_summary()