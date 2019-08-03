import os
import argparse
from viref_wo_a.comprehension_test import comprehension_test as comprehension_test_viref_wo_a
from viref_wo_a.generation_test import generation_test as generation_test_viref_wo_a
from viref_wo_e.comprehension_test import comprehension_test as comprehension_test_viref_wo_e
from viref_wo_e.generation_test import generation_test as generation_test_viref_wo_e
from viref.comprehension_test import comprehension_test as comprehension_test_viref
from viref.generation_test import generation_test as generation_test_viref

from comprehension_results import comprehension_results
from generation_results import generation_results



def main():
	parser = argparse.ArgumentParser(description='Test(comprehension or generation) one of the models: viref, viref-a or viref-e.')
	parser.add_argument('--model', choices=['viref', 'viref-a', 'viref-e'], default='viref')
	parser.add_argument('--cuda', action="store_true")
	parser.add_argument('--max-refexp-len', type=int, default=25)
	parser.add_argument('--batch-size', type=int, default=10)
	parser.add_argument('--hidden-size', type=int, default=256)
	parser.add_argument('--num-layers', type=int, default=6)
	parser.add_argument('--dropout', type=float, default=0.2)
	parser.add_argument('--features-dir',  type=str, default='features')
	parser.add_argument('--dataset-split-dir', type=str, default='dataset_split')
	parser.add_argument('--data-dir', type=str, default='data')
	parser.add_argument('--type', choices=['generation', 'comprehension'], default='generation')
	args = parser.parse_args()

	print('args:', args)

	if args.model == 'viref':
		if args.type == 'generation':
			generation_test_viref(args, os.path.join('viref', 'model_save'))
			generation_results(os.path.join('viref', 'generation_results.txt'))
		elif args.type == 'comprehension':
			comprehension_test_viref(args, os.path.join('viref', 'model_save'))
			comprehension_results(os.path.join('viref', 'comprehension_results'))
	elif args.model == 'viref-a':
		if args.type == 'generation':
			generation_test_viref_wo_a(args, os.path.join('viref_wo_a', 'model_save'))
			generation_results(os.path.join('viref_wo_a', 'generation_results.txt'))
		elif args.type == 'comprehension':
			comprehension_test_viref_wo_a(args, os.path.join('viref_wo_a', 'model_save'))
			comprehension_results(os.path.join('viref_wo_a', 'comprehension_results'))
	elif args.model == 'viref-e':
		if args.type == 'generation':
			generation_test_viref_wo_e(args, os.path.join('viref_wo_e', 'model_save'))
			generation_results(os.path.join('viref_wo_e', 'generation_results.txt'))
		elif args.type == 'comprehension':
			comprehension_test_viref_wo_e(args, os.path.join('viref_wo_e', 'model_save'))
			comprehension_results(os.path.join('viref_wo_e', 'comprehension_results'))

if __name__ == '__main__':
	main()
