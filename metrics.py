import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Your actual predictions
y_true = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]  # 0: real, 1: fake
y_pred = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1]  # model predictions

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    """
    Calculate and display various performance metrics for the deepfake detection model.
    
    Args:
        y_true: Ground truth labels (0 for real, 1 for fake)
        y_pred: Predicted labels (0 for real, 1 for fake)
    
    Returns:
        dict: Dictionary containing accuracy, f1 score, and confusion matrix
    """
    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = conf_matrix.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Generate detailed classification report
    class_report = classification_report(y_true, y_pred, target_names=['Real', 'Fake'])
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save the plot
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'precision': precision,
        'recall': recall,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    }
    
    # Print detailed results
    print("\n=== Deepfake Detection Model Performance ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nDetailed Classification Report:")
    print(class_report)
    
    return metrics

def plot_metrics_over_time(metrics_history):
    """
    Plot metrics over time to show model performance trends.
    
    Args:
        metrics_history: List of metric dictionaries over time
    """
    timestamps = range(len(metrics_history))
    accuracies = [m['accuracy'] for m in metrics_history]
    f1_scores = [m['f1_score'] for m in metrics_history]
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, accuracies, label='Accuracy', marker='o')
    plt.plot(timestamps, f1_scores, label='F1 Score', marker='s')
    plt.title('Model Performance Over Time')
    plt.xlabel('Evaluation Point')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig('performance_trend.png')
    plt.close()

def save_metrics_to_file(metrics, filename='model_metrics.txt'):
    """
    Save metrics to a text file for future reference.
    
    Args:
        metrics: Dictionary containing model metrics
        filename: Name of the file to save metrics to
    """
    with open(filename, 'w') as f:
        f.write("=== Deepfake Detection Model Metrics ===\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write("True Negatives: {}\n".format(metrics['true_negatives']))
        f.write("False Positives: {}\n".format(metrics['false_positives']))
        f.write("False Negatives: {}\n".format(metrics['false_negatives']))
        f.write("True Positives: {}\n".format(metrics['true_positives']))

# Example usage:
if __name__ == "__main__":
    # Example data
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])  # 0: real, 1: fake
    y_pred = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1])  # model predictions
    
    # Calculate and display metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Save metrics to file
    save_metrics_to_file(metrics)
    
    # Example of tracking metrics over time
    metrics_history = [
        {'accuracy': 0.8, 'f1_score': 0.75},
        {'accuracy': 0.85, 'f1_score': 0.82},
        {'accuracy': 0.87, 'f1_score': 0.84},
        {'accuracy': 0.90, 'f1_score': 0.88}
    ]
    plot_metrics_over_time(metrics_history)
