import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCrossEntropyWithSelectivePenalty(nn.Module):
    def __init__(self, alpha=0.8):
        """
        Custom loss function combining CrossEntropyLoss and a distance-based penalty,
        with specific cases excluded from penalties.
        :param alpha: Weight for the penalty term (between 0 and 1).
        """
        super(CustomCrossEntropyWithSelectivePenalty, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')  # No reduction for single sample
        self.alpha = alpha

        # Updated class mapping: [below 90, 90, ..., 100, above 100]
        self.class_mapping = ["below 90"] + list(range(90, 101)) + ["above 100"]

        # Indices for special ranges
        self.below_90_index = 0
        self.above_100_index = len(self.class_mapping) - 1

    def forward(self, logits, target):
        """
        Compute the custom loss for a single sample.
        :param logits: Model output logits (before softmax) of shape [num_classes].
        :param target: Ground truth class index (scalar).
        :return: Combined loss value for the sample.
        """
        # Compute CrossEntropyLoss for the single sample
        ce_loss = self.cross_entropy(logits, target)

        # Compute softmax probabilities and remove batch dimension
        probs = F.softmax(logits.squeeze(0), dim=0)  # Ensure logits have shape [num_classes]

        # Get true class and SpO₂ value
        true_class = target.item()  # Ground truth class index
        true_value = self.class_mapping[true_class]  # Ground truth SpO₂ value

        # Initialize penalty
        penalty = 0.0

        # Compute penalty
        for pred_class in range(len(self.class_mapping)):
            pred_value = self.class_mapping[pred_class]  # Predicted SpO₂ value
            prob = probs[pred_class].item()  # Probability of the predicted class
            # print("Logits shape (squeezed):", logits.squeeze(0).shape)
            # print("Probs shape (after squeeze):", probs.shape)
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
                penalty += prob * distance

        # Combine CrossEntropyLoss and penalty
        combined_loss = ce_loss + self.alpha * penalty
        return combined_loss

