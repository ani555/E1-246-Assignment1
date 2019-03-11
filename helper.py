import re
import os
import csv
import pickle
import math
import random
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter


def subsample(train_ids, threshold=1e-5, remove_stopwords=False):

	dropped_words=[]
	if remove_stopwords:
		dropped_words = stopwords.words('english')
	else:
		words = get_words(train_ids)
		words = dict(Counter(words))
		total_words = sum(words.values())
		word_norm_freq  = {word: freq/total_words for word, freq in words.items()}
		word_drop_prob = {word: 1-math.sqrt(norm_freq) for word, norm_freq in word_norm_freq.items()}
		dropped_words = [word for word, drop_prob in word_drop_prob.items() if random.random() >= drop_prob]
	
	return dropped_words



def get_sentences(train_ids):

	text = reuters.raw(train_ids).lower()
	paras = text.split('\n\n')

	paras = [para.replace('\n','.\n',1) for para in paras] # process the header in paras
	paras = [para.replace('&lt','',1) for para in paras] # remove this weird symbol present in the text
	sentences = []
	
	for para in paras:
		sentences.extend(sent_tokenize(para))

	sentences = [re.findall('([a-z]\.[a-z]\.|[a-z]+)',sentence) for sentence in sentences]
	sentences = [sentence for sentence in sentences if len(sentence)!=0]

	return sentences




def get_words(train_ids):

	sentences = get_sentences(train_ids)
	words = []
	for sentence in sentences:
		words.extend(sentence)

	return words

def get_vocab(train_ids, dropped_words=[], max_vocab_size=50000, threshold=2):

	
	train_words = get_words(train_ids)
	train_words = [word for word in train_words if len(word)>1 or word=='a']
	vocab = Counter(train_words).most_common(max_vocab_size)
	vocab = [(key, val) for key, val in vocab if key not in dropped_words]

	for i in range(len(vocab)-1,-1,-1):
		if vocab[i][1] < threshold:
			vocab.pop(i)
		else:
			break;
	vocab = dict(vocab)

	return vocab

def get_mapping_dicts(vocab):
	# returns {word: idx} and {idx: word} dicts
	word2idx = {word:idx for idx,word in enumerate(vocab)}
	idx2word = {idx:word for idx, word in enumerate(vocab)}
	return word2idx, idx2word

def get_unigram_probs(vocab):

	counts =  list(vocab.values())
	total_count = sum(counts)
	unigram_probs = [count/total_count for count in counts]
	return unigram_probs


def preprocess(file_ids, vocab, dropped_words=[]):
	sentences = get_sentences(file_ids)
	sents = []
	for sentence in sentences:
		proc_sent = []

		for word in sentence:
			word = word.lower()

			
			if word in vocab:
				proc_sent.append(word)

		sents.append(proc_sent)
	return sents

def get_word_pairs(sentences, word2idx, win_radius):
	X = []
	Y = []

	for sentence in sentences:
		for i,word in enumerate(sentence):
			for context_word in sentence[max(0, i-win_radius):min(len(sentence), i+win_radius+1)]:
				if(context_word!=word):
					X.append(word2idx[word])
					Y.append([word2idx[context_word]])

	return X, Y

def get_fileids(val_split=0.1):

#	doc_filter = ['acq', 'money-fx']
	train_ids = list(filter(lambda f: 'training' in f, reuters.fileids()))
	test_ids = list(filter(lambda f: 'test' in f, reuters.fileids()))

#	random.shuffle(train_ids)
	num_val_samples = int(val_split * len(train_ids))
	val_ids = train_ids[-num_val_samples:]
	train_ids = train_ids[:-num_val_samples]

	return train_ids, val_ids, test_ids


def get_train_val_test_data(train_ids, val_ids, test_ids, word2idx, vocab, dropped_words=[], win_radius=2):

	sent_train = preprocess(train_ids, vocab, dropped_words)
	sent_val = preprocess(val_ids, vocab)
	sent_test = preprocess(test_ids, vocab)

	X_train, Y_train = get_word_pairs(sent_train, word2idx, win_radius)
	X_val, Y_val = get_word_pairs(sent_val, word2idx, win_radius)
	X_test, Y_test = get_word_pairs(sent_test, word2idx, win_radius)

	return X_train, Y_train, X_val, Y_val, X_test, Y_test



def get_test_data(test_ids, word2idx, vocab):

	sent_test = preprocess(test_ids, vocab)
	X = []
	for sentence in sent_test:
		X.extend([word2idx[word] for word in sentence])
	return X
	
def generate_batches(X, Y, batch_size=128):

	num_samples = len(X)

	for batch_i in range(0, num_samples//batch_size):
		start = batch_i*batch_size
		X_batch = X[start: start+batch_size]
		Y_batch = Y[start: start+batch_size]
		yield X_batch, Y_batch



def load_simlex(file_path):

	# loads a dictionary from SimLex-999.txt with word pairs as key and SimLex-999 scores as values
	file = os.path.join(file_path, 'SimLex-999.txt')
	with open(file, 'r') as f:
		data = csv.reader(f, delimiter='\t')
		header = next(data)
		simlex_dict = {(row[0],row[1]):(float(row[3]), row[2]) for row in data}

	return simlex_dict

def calculate_corr(similarity, X, idx2word, word2idx, vocab, simlex_dict, threshold=30):

	words_list = [idx2word[word_id] for word_id in X]
	pred_dict = {}
	true_dict = {}
	count=0
	for (word1, word2), val in simlex_dict.items():
		if word1 in words_list and word2 in vocab:
			if vocab[word1]>=threshold and vocab[word2]>=threshold:
				pred_dict[(word1, word2)] = similarity[word2idx[word2], X.index(word2idx[word1])]
				true_dict[(word1, word2)] = simlex_dict[(word1, word2)]
				count+=1
		elif word2 in words_list and word1 in vocab:
			if vocab[word1]>=threshold and vocab[word2]>=threshold:
				pred_dict[(word1, word2)] = similarity[word2idx[word1], X.index(word2idx[word2])]
				true_dict[(word1, word2)] = simlex_dict[(word1, word2)]	
				count+=1
	#print('match word pairs {}'.format(count))
	

	corr = 0.0

	if true_dict:
		true_mean = 1.0*sum([val for key, val in true_dict.items()])/len(true_dict)
		pred_mean = 1.0*sum([val for key, val in pred_dict.items()])/len(pred_dict)
		num = 0.0
		den1 = 0.0
		den2 = 0.0
		for val1, val2 in zip(true_dict.values(), pred_dict.values()):
			num += (val1-true_mean)*(val2-pred_mean)
			den1 += (val1-true_mean)**2
			den2 += (val2-pred_mean)**2
		corr = num/math.sqrt(den1*den2)
	return corr, count

def save_obj(embeddings, file_path, file_name):

	if not os.path.exists(file_path):
		os.makedirs(file_path)

	file = os.path.join(file_path, file_name)
	with open(file,'wb') as f:
		pickle.dump(embeddings, f)

def load_obj(file_path, file_name):

	file = os.path.join(file_path, file_name)
	with open(file, 'rb') as f:
		obj = pickle.load(f)
	return obj

def load_analogy_questions(file_path, vocab, k):

	k_most_common = Counter(vocab).most_common(k)
	k_most_common = dict(k_most_common)
	file = os.path.join(file_path, 'questions-words.txt')
	questions = []
	with open(file, 'r') as f:
		data = csv.reader(f, delimiter=' ')
		for row in data:
			if row[0] == ':':
				pass
			elif row[0] in k_most_common and row[1] in k_most_common and row[2] in k_most_common and row[3] in k_most_common:
				questions.append([row[0], row[1], row[2], row[3]])
	return questions

def visualize_words(embeddings, labels):

	tsne = TSNE(n_components=2, n_iter=2000, init='pca')
	emb_2d = tsne.fit_transform(embeddings)

	plt.figure(figsize=(16, 16))

	for i in range(labels.shape[0]):

		plt.scatter(emb_2d[i,0], emb_2d[i,1])
		plt.annotate(labels[i], xy=(emb_2d[i,0], emb_2d[i,1]), xytext=(5,2), textcoords='offset points', ha='right', va='bottom')
	plt.show()

