import os


def mean_average_precision(similarities, rank=None):
	total_avg_precision = 0
	for id_i, similarities_i in similarities:
		number_of_examples = 0
		number_of_relevant_examples = 0
		total_precision = 0
		for similarity, id_j in similarities_i:
			number_of_examples += 1
			if rank is not None and number_of_examples > rank:
				break
			if id_i == id_j:
				number_of_relevant_examples += 1
				total_precision += number_of_relevant_examples / number_of_examples
		if number_of_relevant_examples > 1:
			raise NameError('MAP calculation is wrong!')
		
		if number_of_relevant_examples == 1:
			avg_precision = total_precision/number_of_relevant_examples
			total_avg_precision += avg_precision
	return total_avg_precision/len(similarities)
	

def rank_k_accuracy(similarities, k):
	number_of_hits = 0
	for id_i, similarities_i in similarities:
		number_of_examples = 0
		for similarity, id_j in similarities_i:
			number_of_examples += 1
			if number_of_examples > k:
				break
			if id_i == id_j:
				number_of_hits += 1
				break
	return number_of_hits/len(similarities) 
	
	
def comprehension_results(comprehension_results_dir):
	ranks = [1, 2, 3]
	total_rank = {x:0 for x in ranks}
	total_mAP = 0
	
	target_paths = sorted(list(os.listdir(comprehension_results_dir)))
	
	for filename in target_paths:
		with open(os.path.join(comprehension_results_dir, filename), 'r') as f:
			results = eval(f.read())
			sequence_set = set([])
			for _, key in results[0][1]:
				sequence_set.add(key)
			print(filename, len(sequence_set))
			mAP = mean_average_precision(results)
			total_mAP += mAP
			print('\tmAP:', mAP)
			for rank in ranks:
				acc = rank_k_accuracy(results, rank)
				print('\trank{:d} accuracy:'.format(rank), acc)
				total_rank[rank] += acc

	print('Average Results')
	print('Average mAP:', total_mAP/len(target_paths))
	for rank in total_rank:
		print('Average rank{:d} accuracy:'.format(rank), total_rank[rank]/len(target_paths))

