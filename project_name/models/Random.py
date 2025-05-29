from sklearn.metrics import classification_report


class RandomModel:
    def __init__(self, data, mlb):
        self.mlb = mlb
        self.predictions = self.makePredictions(data, self.mlb)
        self.test(data)

    @staticmethod
    def makePredictions(data, mlb):
        guesses = mlb.transform(['alternative rock',
                                 'indie rock',
                                 'pop rock'])[0]
        return [guesses] * len(data)

    def test(self, y_test_binary):
        print(classification_report(y_test_binary,
                                    self.predictions,
                                    target_names=self.mlb.classes_))
