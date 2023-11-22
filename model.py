import pandas as pd


class Pursuit:
    def __init__(self, gamma: float, threshold: float):
        self.gamma = gamma
        self.threshold = threshold
        self.lexicon = {}

    def load_data(self, path: str):
        utterances = []
        meanings = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip() == "":
                    continue

                if line.isupper():
                    meanings.append(line.split())
                else:
                    utterances.append(line.split())

        self.data = pd.DataFrame(
            {'utterances': utterances, 'meanings': meanings})

        names = list(set(self.data.utterances.explode()))
        goldens = list(set(self.data.meanings.explode()))

        self.matrix = pd.DataFrame(index=names, columns=goldens).fillna(0.0)

    def initialize(self, utterance: str, meanings: list[str]):
        hypothesis = self.matrix[meanings].max(axis=0).idxmin()
        self.matrix.at[utterance, hypothesis] = self.gamma

        return hypothesis

    def reward(self, utterance: str, meaning: str):
        value = self.matrix.at[utterance, meaning]
        self.matrix.at[utterance, meaning] += (1 - self.gamma * value)

    def punish(self, utterance: str, meaning: str):
        self.matrix.at[utterance, meaning] *= (1 - self.gamma)

    def train(self):
        for _, row in self.data.iterrows():
            utterances = row['utterances']
            meanings = row['meanings']

            for word in utterances:
                if word in self.lexicon:
                    continue

                if self.matrix.loc[word].eq(0).all():
                    hypothesis = self.initialize(word, meanings)

                if hypothesis in meanings:
                    self.reward(word, hypothesis)
                else:
                    self.punish(word, hypothesis)
                    hypothesis = self.initialize(word, meanings)
                    self.reward(word, hypothesis)

                if self.matrix.at[word, hypothesis] > self.threshold:
                    self.lexicon[word] = hypothesis


if __name__ == "__main__":
    pursuit = Pursuit(.02, .7)
    pursuit.load_data('./Data/Train/train.txt')
    pursuit.train()

    print(pursuit.lexicon)
