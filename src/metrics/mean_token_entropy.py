import math


class MeanTokenEntropy:

    def token_entropy(self, token_top_logprobs):
        """
        Compute entropy for one token from its probability distribution.
        token_top_logprobs: list of {token, logprob} dicts for that position
        """
        entropy = 0.0
        for item in token_top_logprobs:
            p = math.exp(item["logprob"])  # convert logprob → probability
            entropy += p * (-item["logprob"])  # p * log(1/p) = -p log p
        return entropy

    def mean_token_entropy_openai(self, response):
        """
        Receives a GPT API response with logprobs and returns mean entropy.
        """
        entropies = []
        tokens = response.choices[0].logprobs

        # tokens are in tokens["top_logprobs"]
        for top_probs in tokens["top_logprobs"]:
            if top_probs is None:
                continue  # safety (first token may have None)
            H = self.token_entropy(top_probs)
            entropies.append(H)

        if len(entropies) == 0:
            return None

        return sum(entropies) / len(entropies)
