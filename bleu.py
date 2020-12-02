from nltk.translate.bleu_score import sentence_bleu

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
        score = sentence_bleu([ref], cand, weights=(1,0,0,0) )
        return score

    def compute_bleu4(self, ref, cand):
        score = sentence_bleu([ref], cand, weights=(0,0,0,1) )
        return score

    # return bleu1 and bleu4 score
    def compute_score(self):
        bleu1 = 0
        bleu4 = 0
        for i in range(len(self.ref)):
            san_ref = sanitize(self.ref[i])
            san_cand = sanitize(self.cand[i])
            bleu1 += self.compute_bleu1(san_ref, san_cand)
            bleu4 += self.compute_bleu4(san_ref, san_cand)

        return bleu1/len(self.ref), bleu4/len(self.ref)
        


# Example usage

# # correct captions
# references = ['this is a test', 'a man walking two dogs']
# # generated captions
# candidates = ['this is test', 'man walking dog']
# scorer = BleuScorer(references, candidates)
# bleu1, bleu4 = scorer.compute_score()
# print("bleu1: %f, bleu4: %f" % (bleu1, bleu4))
