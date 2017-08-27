import csv, collections, os
import numpy as np
import nltk


class load_sent_word_net(object):
    def __init__(self):
        sent_scores = collections.defaultdict(list)

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SentiWordNet_3.0.0_20130122.txt'),
                  'r') as csvfile:

            reader = csv.reader(csvfile, delimiter='\t', quotechar='"')

            for line in reader:
                if line[0].startswith('#'):
                    continue
                if len(line) == 1:
                    continue

                POS, ID, PosScore, NegScore, SynsetTerms, Gloss = line

                if len(POS) == 0 or len(ID) == 0:
                    continue

                for term in SynsetTerms.split(" "):
                    term = term.split("#")[0]
                    term = term.replace("-", " ").replace("_", " ")
                    key = "%s/%s" % (POS, term.split("#")[0])
                    sent_scores[key].append((float(PosScore), float(NegScore)))
                    # this is where magic happens. You classify all words by + or -.

        for key, value in sent_scores.items():
            sent_scores[key] = np.mean(value, axis=0)

        self.sent_scores = sent_scores

    def score_word(self, word):
        pos = nltk.pos_tag([word])[0][1]
        return self.score(word, pos)

    def score_sentence(self, sentence):
        pos = nltk.pos_tag(sentence)
        mean_score = np.array([0.0, 0.0])
        for j in range(len(pos)):
            mean_score += self.score(pos[j][0], pos[j][1])

        return mean_score

    def score(self, word, pos):
        if pos[0:2] == 'NN':
            pos_type = 'n'
        elif pos[0:2] == 'JJ':
            pos_type = 'a'
        elif pos[0:2] == 'VB':
            pos_type = 'v'
        elif pos[0:2] == 'RB':  # adverb
            pos_type = 'r'
        else:
            pos_type = 0

        if pos_type != 0:
            dic_loc = pos_type + '/' + word
            pos_neg_scores = self.sent_scores[dic_loc]  # dictionary location
            if len(pos_neg_scores) == 2:
                return pos_neg_scores
            else:
                return np.array([0.0, 0.0])
        else:
            return np.array([0.0, 0.0])

    def posvector(self, sentence):  # position vector

        pos_vector = nltk.pos_tag(sentence)
        vector = np.zeros(4)

        for j in range(len(sentence)):
            pos = pos_vector[j][1]
            if pos[0:2] == 'NN':
                vector[0] += 1
            elif pos[0:2] == 'JJ':
                vector[1] += 1
            elif pos[0:2] == 'VB':
                vector[2] += 1
            elif pos[0:2] == 'RB':
                vector[3] += 1

        return vector
