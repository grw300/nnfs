import numpy as np

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss

class CategoricalCrossEntropyLoss(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred*y_true,
                axis=1
            )

        correct_confidences_clipped = np.clip(correct_confidences, 1e-7, 1 - 1e-7)

        negative_log_likelihoods = -np.log(correct_confidences_clipped)

        return negative_log_likelihoods