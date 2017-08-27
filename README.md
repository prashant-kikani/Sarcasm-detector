# Sarcasm-detector
Sarcastic sentence detector. Trained by sarcastic tweets.<br />
I will surely recommend to read research paper of stanford of sarcasm-detector. That peper made my 70% task easy.<br />
Here is link : http://cs229.stanford.edu/proj2015/044_report.pdf<br />
Other important links...<br />
http://www.cs.utah.edu/~huangrh/official-sarcasm-cameraReady-v2.pdf<br />
http://www.aclweb.org/anthology/W13-1605<br />
Classification of positive & negative words is done by using SentiWordnet 3.0 Link : http://sentiwordnet.isti.cnr.it/
<br /><br />
Detecting sarcasm is a natural language processing task.<br />
I made this this project by taking reference of many sites and papers like http://cs229.stanford.edu/proj2015/044_report.pdf<br />

I get data from twitter api by tracking the word "#sarcasm" in all tweets.<br />
Twitter provides an api : https://dev.twitter.com/ via which anyone can get live data of all the tweets.<br />
I used : Python with Numpy, Scipy, Scikit-learn, NLTK, gensim, Textblob and tweepy.<br />

<b><u>Features</u></b>......<br />

n-grams: More precisely, unigrams and bigrams. These are just collections of one word (example: really, great, awesome, etc.) and two words (example: really great, super awesome, very weird, etc.). To extract those, each tweet was tokenized, stemmed, uncapitalized and then each n-gram was added to a binary feature dictionary.<br />

Sentiments: My hypothesis here is that sarcastic tweets might be more negative than non-sarcastic tweets or the other way around. Moreover, there is often a big contrast of sentiments in sarcastic tweets. What I mean by this is that tweets often start with a very positive sentiment and end with a very negative sentiment (example: I love being cheated on #sarcasm). Sentiment analysis of tweets is a subject on its own so the idea here is to have something simple that can test my hypothesis. To this end I first split each tweet in one, two and three parts, and then do a sentiment analysis on all parts of the three splittings. I used two distinct sentiment analyzers. The first one is my own quick and dirty implementation which uses the SentiWordNet dictionary. This dictionary gives a positive and a negative sentiment score to each word of the English language. By looking up words in this dictionary, we can give a sentiment score to each part of the tweets. The other implementation of the sentiment analysis used the python library TextBlob which has a built-in sentiment score function.<br />

Topics: There are words that are often grouped together in the same tweets (example: saturday, party, night, friends, etc.). We call these groups of words topics. If we first learn the topics, then the classifier will just have to learn which topics are more associated with sarcasm and that will make the supervised learning easier and more accurate. To learn the topics, I used the python library gensim which implements topic modeling using latent Dirichlet allocation (LDA). We first feed all the tweets to the topic modeler which learns the topics. Then each tweet can be decomposed as a sum of topics, which we use as features.<br />

Classifier is support vectore machine...  
