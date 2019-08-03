import os
import math
import statistics
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

def calculate_bleu(results):
	bleu_scores = []
	for key in results:
		references = results[key][1]
		hypothesis = results[key][0]
		bleu_score = sentence_bleu(references, hypothesis)
		bleu_scores.append(bleu_score)
	return statistics.mean(bleu_scores), statistics.stdev(bleu_scores)


def calculate_meteor(results):
	meteor_scores = []
	for key in results:
		references = results[key][1]
		hypothesis = results[key][0]
		score = meteor_score([' '.join(reference) for reference in references], ' '.join(hypothesis))
		meteor_scores.append(score)
	return statistics.mean(meteor_scores), statistics.stdev(meteor_scores)
	
def generation_results(generation_results_path):
	with open(generation_results_path, 'r') as f:
		results = eval(f.read())
	nltk.download('wordnet')
	print("bleu score (mean, stdev):", calculate_bleu(results))
	print("meteor score (mean, stdev):", calculate_meteor(results))
