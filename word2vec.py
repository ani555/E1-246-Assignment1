import nltk
import numpy as np
import tensorflow as tf
import math
import os
import helper
import pickle
import argparse
import json
import pprint
import time

nltk.download('reuters')
nltk.download('punkt')
nltk.download('stopwords')


parser = argparse.ArgumentParser(description='Word2Vec')
parser.add_argument('--mode', dest='mode', default='train', help='train or test or simlex or analogy or visualize', required=True)
parser.add_argument('--config_file', dest='config_file', default='config.json', help='config file name with path')
parser.add_argument('--save_path', dest='save_path', default='ckpt/', help='Path where model will be saved')
parser.add_argument('--load_path', dest='load_path', default='ckpt/', help='Path from where model will be loaded')
parser.add_argument('--data_path', dest='data_path', default='eval_data', help='Path where evaluation data files are present')
parser.add_argument('--test_words', nargs='+', help='Optionally pass some words while testing if not passed the default test set from reuters corpus will be used')

args = parser.parse_args()



def negative_sampling_loss(labels, weights, bias, inputs, neg_samples, num_classes, unigram_probs):

	neg_sample_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
		true_classes = tf.cast(labels,dtype=tf.int64),
		num_true = 1,
		num_sampled = neg_samples,
		distortion = 0.75,
		range_max = num_classes,
		unique = True,
		unigrams = unigram_probs)
	


	unigram_probs = tf.convert_to_tensor(unigram_probs, tf.float64)
	# labels is a matrix convert to a vector
	labels_flat = tf.reshape(labels, [-1])



	# embeddings for the labels (batch_size x d), bias for the labels (batch_size x 1)
	tgt_embeddings = tf.nn.embedding_lookup(weights, labels_flat)
	tgt_biases = tf.nn.embedding_lookup(bias, labels_flat)

	# embeddings for each negative sample (neg_samples x d), bias for each negative sample (neg_samples x 1) and unigram_probs (neg_samples x 1)
	neg_sample_embeddings = tf.nn.embedding_lookup(weights, neg_sample_ids)
	neg_sample_biases = tf.nn.embedding_lookup(bias, neg_sample_ids)
	neg_unigram_probs = tf.nn.embedding_lookup(unigram_probs, neg_sample_ids)

	# one logit (similarity between target word and current center word) (batch_size x 1)
	pos_logits = tf.reduce_sum(tf.multiply(inputs, tgt_embeddings), axis=1) + tgt_biases

	# neg_samples logits (similarity between neg words and center word) (batch_size x neg_samples)
	neg_logits = tf.matmul(inputs, neg_sample_embeddings, transpose_b=True)
	neg_unigram_probs = neg_unigram_probs[tf.newaxis, :]
	cost = tf.reduce_mean(-tf.log(tf.sigmoid(pos_logits)) - tf.reduce_sum(tf.multiply(tf.log(1-tf.sigmoid(neg_logits)),neg_unigram_probs), axis=1),axis=0)
	
	return cost 



def build_model(vocab_size, embed_size, unigram_probs, neg_samples=15):

	#define inputs
	x = tf.placeholder(tf.int32, shape=[None,], name='input')
	y = tf.placeholder(tf.int32, shape=[None,1], name='target')
	lr = tf.placeholder(tf.float64, name='learning_rate')

	#define the network
	embeddings = tf.Variable(tf.random_uniform([vocab_size, embed_size],-1.0,1.0, dtype=tf.float64), name='embeddings')
	Vc = tf.nn.embedding_lookup(embeddings, x) #embedding of the center word
	context_weights = tf.Variable(tf.truncated_normal([vocab_size, embed_size],stddev=1.0/math.sqrt(embed_size),dtype=tf.float64), name='context_weights')
	context_bias = tf.Variable(tf.zeros(vocab_size,dtype=tf.float64), name='context_bias')

	#define loss function and the optimizer
	loss = negative_sampling_loss(labels=y, weights=context_weights, bias=context_bias, inputs=Vc, neg_samples=neg_samples, num_classes=vocab_size, unigram_probs=unigram_probs)
	optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
	
	#add some extra nodes to the graph for evaluation purposes
	embeddings_norm = embeddings / tf.sqrt(tf.reduce_mean(tf.square(embeddings),1,keepdims=True)) #normalized embeddings
	Vc_norm = Vc / tf.sqrt(tf.reduce_mean(tf.square(Vc))) #normalized center word vector
	cosine_similarity = tf.matmul(embeddings_norm, Vc_norm, transpose_b=True, name='similarity') #cosine similarity

	model = {
			'input':x, 
			'target':y, 
			'learning_rate':lr, 
			'loss':loss, 
			'optimizer':optimizer, 
			'similarity': cosine_similarity, 
			'embeddings':embeddings_norm
			}

	return model


def calculate_loss(sess, model, X, Y, batch_size=128):

	num_batches = len(X)//batch_size
	avg_loss = 0.0 

	for batch_i, (input_batch, target_batch) in enumerate(helper.generate_batches(X, Y, batch_size)):

		loss = sess.run(model['loss'], feed_dict = {model['input']: input_batch, model['target']: target_batch})
		avg_loss += loss

	avg_loss /= num_batches

	return avg_loss


def train(sess, model, X_train, Y_train, X_val, Y_val, X_test, Y_test, idx2word, word2idx, vocab, save_path=None, batch_size=128, loss_batch=128, learning_rate=0.1, epochs=10, train_loss_freq=5000, val_loss_freq=100000):
	
	sess.run(tf.global_variables_initializer())
	num_batches = len(X_train) // batch_size
	train_loss_list = []
	val_loss_list = []
	val_word_ids = list(set(X_val))
	for epoch_i in range(epochs):
		avg_train_loss = 0.0 
		save_train_loss = 0.0 

		for batch_i, (input_batch, target_batch) in enumerate(helper.generate_batches(X_train, Y_train, batch_size)):
			_, train_loss = sess.run([model['optimizer'], model['loss']] , 
				feed_dict={
				model['input']:input_batch, 
				model['target']:target_batch, 
				model['learning_rate']:learning_rate
				})
			avg_train_loss += train_loss
			save_train_loss += train_loss

			if batch_i%train_loss_freq==0 and batch_i!=0:
				avg_train_loss /= train_loss_freq
				print('Epoch {:>3} Batch {:>4}/{} Avg Train Loss ({} batches):{:.6f}'.format(epoch_i,batch_i,num_batches, train_loss_freq, avg_train_loss))
				avg_train_loss = 0.0



	if save_path is not None:
		saver = tf.train.Saver()
		saver.save(sess, save_path)
		print('Model Saved')

	print('----------Training Complete----------')		
		
	# calculate final train, val and test losses on the trained model
	final_train_loss = calculate_loss(sess, model, X_train, Y_train, loss_batch)
	final_val_loss = calculate_loss(sess, model, X_val, Y_val, loss_batch)
	final_test_loss = calculate_loss(sess, model, X_test, Y_test, loss_batch)

	print('Final Train Loss: {:.6f}, Val Loss: {:.6f}, Test Loss: {:.6f}'.format(final_train_loss, final_val_loss, final_test_loss))

				
	
	helper.save_obj(train_loss_list,save_path,'train_loss.pkl')
	helper.save_obj(val_loss_list,save_path,'val_loss.pkl')
		
	trained_embeddings = model['embeddings'].eval()

	return trained_embeddings


def test(sess, test_graph, X_test, load_path, idx2word, top_k=10):


	loader = tf.train.import_meta_graph(load_path+'.meta')
	loader.restore(sess, load_path)
	
	inputs = test_graph.get_tensor_by_name('input:0')
	similarity = test_graph.get_tensor_by_name('similarity:0')
	X_test = list(set(X_test))
	
	sim_scores = sess.run(similarity, feed_dict = {inputs: X_test})
	sim_scores = sim_scores.T
	for i in range(len(X_test)):
		closest_words = ''
		print('Words closest to {}'.format(idx2word[X_test[i]]))
		closest_words_idx = (-sim_scores[i,:]).argsort()[1:top_k+1].tolist()
		for idx in  closest_words_idx:
			closest_words = closest_words + idx2word[idx] + ' '
		print(closest_words)


def build_analogy_graph(load_path, top_k=15):

	embeddings_norm = tf.convert_to_tensor(helper.load_obj(load_path,'embeddings.pkl'), dtype=tf.float32)

	input_a = tf.placeholder(dtype=tf.int32, shape=[None,])		
	input_b = tf.placeholder(dtype=tf.int32, shape=[None,])
	input_c = tf.placeholder(dtype=tf.int32, shape=[None,])

	a_embed = tf.gather(embeddings_norm, input_a)
	b_embed = tf.gather(embeddings_norm, input_b)
	c_embed = tf.gather(embeddings_norm, input_c)

	# remove component of a from b and add component of c
	# eg: Man:King :: Woman: ?
	d_embed = (b_embed - a_embed) + c_embed

	# find all possible d's by calculating the similarity with the word in vocab
	# d_embed: (batch_size x d) and embeddings_norm: (W x d) similarity: (batch_size x W)
	similarity = tf.matmul(d_embed, embeddings_norm, transpose_b=True) 

	# find the top k words closest to the given question

	_, result = tf.math.top_k(similarity, k=top_k)

	model={
	'input_a':input_a,
	'input_b':input_b,
	'input_c':input_c,
	'output':result
	}

	return model



def eval_analogy_task(sess, questions, model, word2idx, idx2word, num_print=5, print_top_k=15):

	questions = [q for q in questions if q[0] in word2idx and q[1] in word2idx and q[2] in word2idx and q[3] in word2idx]
	w0 = [word2idx[q[0]] for q in questions]
	w1 = [word2idx[q[1]] for q in questions]
	w2 = [word2idx[q[2]] for q in questions]
	w3 = [q[3] for q in questions]

	result = sess.run(model['output'], feed_dict={model['input_a']:w0,model['input_b']:w1,model['input_c']:w2})
	k = result.shape[1]
	m = len(questions)
	num_print = min(num_print, m)
	accuracy = 0.0
	for i in range(m):
		if word2idx[w3[i]] in result[i,:]:
			accuracy+=1
	accuracy/=m

	for i in range(num_print):
		print('{}:{}::{}:{}'.format(idx2word[w0[i]],idx2word[w1[i]],idx2word[w2[i]],w3[i]))
		print('Top {} predictions:'.format(print_top_k))
		top_k_words=''
		for j in range(print_top_k): 
			top_k_words = top_k_words + idx2word[result[i,j]] + ' '
		print(top_k_words)
	print('Total questions: {}'.format(m))
	print('Accuracy for the analogy task: {:.2f}'.format(accuracy))



def calculate_simlex_scores(sess, simlex_graph, vocab, simlex_dict, idx2word, word2idx, load_path, threshold=100):

	loader = tf.train.import_meta_graph(load_path+'.meta')
	loader.restore(sess, load_path)
	
	inputs = simlex_graph.get_tensor_by_name('input:0')
	similarity = simlex_graph.get_tensor_by_name('similarity:0')

	vocab_word_indices = [word2idx[word2] for (word1, word2), (_,_) in simlex_dict.items() if word1 in vocab and word2 in vocab]
	sim_scores = sess.run(similarity, feed_dict = {inputs: vocab_word_indices})

	simlex_n = {(word1, word2): val1 for (word1, word2), (val1, val2) in simlex_dict.items() if val2=='N'}
	simlex_a = {(word1, word2): val1 for (word1, word2), (val1, val2) in simlex_dict.items() if val2=='A'}
	simlex_v = {(word1, word2): val1 for (word1, word2), (val1, val2) in simlex_dict.items() if val2=='V'}
	overall = {(word1, word2): val1 for (word1, word2), (val1, val2) in simlex_dict.items()}
	simlex_n_score, num_n = helper.calculate_corr(sim_scores, vocab_word_indices, idx2word, word2idx, vocab, simlex_n, threshold)
	simlex_a_score, num_a = helper.calculate_corr(sim_scores, vocab_word_indices, idx2word, word2idx, vocab, simlex_a, threshold)
	simlex_v_score, num_v = helper.calculate_corr(sim_scores, vocab_word_indices, idx2word, word2idx, vocab, simlex_v, threshold)
	overall_score, num = helper.calculate_corr(sim_scores, vocab_word_indices, idx2word, word2idx, vocab, overall, threshold)

	print('Simlex scores for {} noun pairs: {:.4f}'.format(num_n, simlex_n_score))
	print('Simlex scores for {} adjective pairs: {:.4f}'.format(num_a, simlex_a_score))
	print('Simlex scores for {} verb pairs: {:.4f}'.format(num_v, simlex_v_score))
	print('Overall Simlex score for {} pairs: {:.4f}'.format(num, overall_score))




def main():

	mode = args.mode
	data_path = args.data_path
	save_path = args.save_path
	load_path = args.load_path
	config_file = args.config_file
	test_words = args.test_words

	with open(args.config_file, 'r') as f:
		config = json.load(f)

	pprint.pprint(config)
	embed_size = config['embed_dims']
	neg_samples = config['neg_samples']
	batch_size = config['batch_size']
	win_radius = config['win_radius']
	epochs = config['epochs']
	learning_rate = config['learning_rate']
	subsample = config['subsample']
	subsample_threshold = config['subsample_threshold']
	remove_stopwords = config['remove_stopwords']
	low_freq_threshold = config['low_freq_threshold']
	simlex_threshold = config['simlex_threshold']
	loss_batch = config['final_loss_batch_size']
	train_loss_freq = config['train_loss_disp_freq']
	val_loss_freq = config['val_loss_disp_freq']
	k_most_common = config['k_most_common']
	print_top_k = config['print_top_k']
	analogy_top_k = config['analogy_top_k']
	dropped_words = []

	
	
	with tf.device('/device:GPU:0'):
	
		if mode == 'train':

			train_ids, val_ids, test_ids = helper.get_fileids(val_split=0.2)
	
			if subsample:
				dropped_words = helper.subsample(train_ids, threshold = subsample_threshold, remove_stopwords=remove_stopwords)
				
			vocab = helper.get_vocab(train_ids, dropped_words, threshold = low_freq_threshold)
			vocab_size = len(vocab)
			word2idx, idx2word = helper.get_mapping_dicts(vocab)
			unigram_probs = helper.get_unigram_probs(vocab)

			X_train, Y_train, X_val, Y_val, X_test, Y_test = helper.get_train_val_test_data(train_ids, val_ids, test_ids, word2idx, vocab, dropped_words, win_radius=win_radius)
			print('Length of train set {} val set {} test set {}'.format(len(X_train),len(X_val), len(X_test)))	
			train_graph = tf.Graph()
			with tf.Session(graph = train_graph) as sess:
				model = build_model(vocab_size, embed_size, unigram_probs, neg_samples)
				start = time.time()
				trained_embeddings = train(sess, model, X_train, Y_train, X_val, Y_val, X_test, Y_test, idx2word, word2idx, vocab, learning_rate=learning_rate, save_path=save_path, batch_size=batch_size, loss_batch=loss_batch, epochs=epochs, train_loss_freq=train_loss_freq, val_loss_freq = val_loss_freq)		
				end = time.time()
				print('Training took {:.4f} mins'.format((end-start)/60))
			
			helper.save_obj(trained_embeddings,save_path,'embeddings.pkl')
			helper.save_obj(vocab, save_path, 'vocab.pkl')
			helper.save_obj((word2idx, idx2word), save_path, 'mappings.pkl')

		if mode != 'train':
			vocab = helper.load_obj(load_path, 'vocab.pkl')
			word2idx, idx2word = helper.load_obj(load_path, 'mappings.pkl')	

		if mode == 'test':

			if test_words is None:
				X_test = helper.get_test_data(test_ids, word2idx, vocab)
			else:
				X_test = [word2idx[word] for word in test_words if word in vocab]
	
			print('Length of test set {}'.format(len(X_test)))
	
			test_graph = tf.Graph()
			with tf.Session(graph = test_graph) as sess:
				test(sess, test_graph, X_test, load_path, idx2word, top_k=print_top_k)
	
		if mode == 'analogy':
			analogy_graph = tf.Graph()
			with tf.Session(graph=analogy_graph) as sess:
				analogy_model = build_analogy_graph(load_path, top_k=analogy_top_k)
				questions = helper.load_analogy_questions(data_path, vocab, k=k_most_common)
				eval_analogy_task(sess, questions, analogy_model, word2idx, idx2word, num_print=100, print_top_k=print_top_k)	
	
		if mode == 'visualize':
	
			indices = np.random.choice(vocab_size, 500)
			labels = np.array(list(vocab.keys()))[indices]
			embeddings = helper.load_obj(load_path, 'embeddings.pkl')
			helper.visualize_words(embeddings[indices], labels)	

		if mode == 'simlex':
			simlex_dict = helper.load_simlex(data_path)
			simlex_graph = tf.Graph()

			with tf.Session(graph=simlex_graph) as sess:
				calculate_simlex_scores(sess, simlex_graph, vocab, simlex_dict, idx2word, word2idx, load_path, simlex_threshold)




if __name__ == '__main__':

	main()


	

