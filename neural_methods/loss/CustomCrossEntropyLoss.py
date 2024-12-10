import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCrossEntropyWithSelectivePenalty(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Custom loss function combining CrossEntropyLoss and a distance-based penalty,
        with specific cases excluded from penalties.
        :param alpha: Weight for the penalty term (between 0 and 1).
        """
        super(CustomCrossEntropyWithSelectivePenalty, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.alpha = alpha

        # Updated class mapping: [below 90, 90, ..., 100, above 100]
        self.class_mapping = ["below 90"] + list(range(90, 101)) + ["above 100"]

        # Indices for special ranges
        self.below_90_index = 0
        self.above_100_index = len(self.class_mapping) - 1

    def forward(self, logits, targets):
        """
        Compute the custom loss.
        :param logits: Model output logits (before softmax) of shape [batch_size, num_classes].
        :param targets: Ground truth class indices of shape [batch_size].
        :return: Combined loss value.
        """
        # Compute CrossEntropyLoss
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
                    (true_class == self.below_90_index and pred_class == self.below_90_index) or
                    (true_class == self.above_100_index and pred_class == self.above_100_index)
                ):
                    continue

                # Compute distance penalty for numeric classes only
                if isinstance(true_value, int) and isinstance(pred_value, int):
                    distance = abs(true_value - pred_value)  # Absolute difference between true and predicted values
                    # penalty += prob * distance
                    alpha=0.1
                    penalty += alpha * distance

        penalty = penalty / batch_size  # Average penalty over the batch

        # Combine CrossEntropyLoss and penalty
        combined_loss = ce_loss + self.alpha * penalty
        return combined_loss
