import numpy as np


class NLL:

    def __init__(self):
        self.nlls = []

    def compute_nll(self, p_correct):
        """
        Compute Negative Log-Likelihood (NLL) for one prompt and remembers answer
        for futher NLL averaging

        Parameters:
            p_correct: list or np.array
                Probabilities assigned by the model to the correct answer.
                Each value must be in [0,1].

        Returns:
            float: average NLL
        """
        p_correct = np.array(p_correct)

        # Clip probabilities to avoid log(0)
        eps = 1e-12
        p_correct = np.clip(p_correct, eps, 1.0)

        nll = -np.mean(np.log(p_correct))
        self.nlls.append(nll)
        return nll

    def get_avg_nll(self):
        """
        :return: Mean NLL
        """
        return np.mean(self.nlls)
