from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

# https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

# turn string into token list
def sanitize(str):
    return str.split()


class BleuScorer:
    #ref is the testing set, cand is the generated set
    def __init__(self, ref, cand):
        self.ref = ref
        self.cand = cand

    def compute_bleu1(self, ref, cand):
        # references = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
        # candidates = ['this', 'is', 'a', 'test']
        print(len(ref))
        print(len(cand))
        score = corpus_bleu(ref, cand, weights=(1,0,0,0) )
        return score

    def compute_bleu4(self, ref, cand):
        score = corpus_bleu(ref, cand, weights=(0.25,0.25,0.25,0.25) )
        return score

    # return bleu1 and bleu4 score
    def compute_score(self):
        bleu1 = self.compute_bleu1(self.ref, self.cand)
        bleu4 = self.compute_bleu4(self.ref, self.cand)
        return bleu1, bleu4
        


# Example usage

# # correct captions
# references = ['this is a test', 'a man walking two dogs']
# # generated captions
# candidates = ['this is test', 'man walking dog']
# scorer = BleuScorer(references, candidates)
# bleu1, bleu4 = scorer.compute_score()
# print("bleu1: %f, bleu4: %f" % (bleu1, bleu4))
