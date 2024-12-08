import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCrossEntropyWithSelectivePenalty(nn.Module):
    def __init__(self, alpha=0.5, class_mapping=None):
        """
        Custom loss function combining CrossEntropyLoss and a distance-based penalty, 
        with specific cases excluded from penalties.
        :param alpha: Weight for the penalty term (between 0 and 1).
        :param class_mapping: List of class values corresponding to the class indices 
                              (e.g., [90, 91, ..., 100, "below 90", "above 100"]).
        """
        super(CustomCrossEntropyWithSelectivePenalty, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.class_mapping = class_mapping if class_mapping else list(range(90, 101)) + ["below 90", "above 100"]

    def forward(self, logits, targets):
        """
        Compute the custom loss.
        :param logits: Model output logits (before softmax) of shape [batch_size, num_classes].
        :param targets: Ground truth class indices of shape [batch_size].
        :return: Combined loss value.
        """
        # CrossEntropyLoss
        ce_loss = self.cross_entropy(logits, targets)

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)

        # Compute distance penalty
        batch_size, num_classes = logits.shape
        penalty = 0.0

        for i in range(batch_size):
            true_class = targets[i].item()  # Ground truth class index
            true_value = self.class_mapping[true_class]  # Ground truth SpO2 value

            for pred_class in range(num_classes):
                pred_value = self.class_mapping[pred_class]  # Predicted SpO2 value
                prob = probs[i, pred_class].item()  # Probability of the predicted class

                # Skip penalties for:
                # 1. Correct predictions
                # 2. Both predicted and true classes in "below 90" or "above 100" range
                if pred_class == true_class or (
                    (true_value == "below 90" and pred_value == "below 90") or
                    (true_value == "above 100" and pred_value == "above 100")
                ):
                    continue

                # Compute distance penalty (absolute difference between predicted and true SpO2 values)
                distance = abs(true_value - pred_value) if isinstance(true_value, int) and isinstance(pred_value, int) else 0
                penalty += prob * distance

        penalty = penalty / batch_size  # Average penalty over the batch

        # Combine CrossEntropyLoss and penalty
        combined_loss = ce_loss + self.alpha * penalty
        return combined_loss
