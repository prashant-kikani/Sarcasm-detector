import numpy as np
import pickle
import os
from . import feature_extract
from . import topic

# pickle files....
obj1 = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vecdict.p'), 'r')
obj2 = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'classif.p'), 'r')

# vector of features of all the data.......
vec = pickle.load(obj1)
# SVC classifier that is already trained in traintest.py file......
classifier = pickle.load(obj2)

obj1.close()
obj2.close()

# this is required because of various OS have various slashes in paths...
topic_mod = topic.topic(model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'topics.tp'),
                        dicttp=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'topics_dict.tp'))


def tweetscore(sentence):
    features = feature_extract.dialogue_act_features(sentence, topic_mod)
    # classifier can only get data in numerical from so convert in vector form.
    features_vec = vec.transform(features)
    score = classifier.decision_function(features_vec)[0]
    # sigmoid and other manipulations........
    per = int(round(2.0 * (1.0 / (1.0 + np.exp(-score)) - 0.5) * 1000.0))

    return per
