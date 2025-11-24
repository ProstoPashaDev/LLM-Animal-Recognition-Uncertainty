import numpy as np


class ECE:
    def compute_ece(self, probs, labels, num_bins=5):
        """
        Compute Expected Calibration Error (ECE).

        Parameters:
            probs (list or np.array): model-predicted probabilities (0 to 1)
            labels (list or np.array): 1 if prediction was correct, else 0
            num_bins (int): number of bins for calibration

        Returns:
            float: ECE score
        """
        probs = np.array(probs)
        labels = np.array(labels)

        bin_edges = np.linspace(0, 1, num_bins + 1)
        ece = 0.0

        for i in range(num_bins):
            start, end = bin_edges[i], bin_edges[i + 1]

            # indices of predictions in the bin
            in_bin = (probs > start) & (probs <= end)

            if np.sum(in_bin) == 0:
                continue

            # mean predicted confidence and accuracy in bin
            conf_avg = np.mean(probs[in_bin])
            acc_avg = np.mean(labels[in_bin])

            bin_prob = np.sum(in_bin) / len(probs)

            ece += bin_prob * abs(conf_avg - acc_avg)

        return ece
