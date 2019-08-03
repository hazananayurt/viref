import os
import argparse
from viref_wo_a.train import train as train_viref_wo_a
from viref_wo_e.train import train as train_viref_wo_e
from viref.train import train as train_viref

def main():
	parser = argparse.ArgumentParser(description='Train one of the models: viref, viref-a or viref-e.')
	parser.add_argument('--model', choices=['viref', 'viref-a', 'viref-e'], default='viref')
	parser.add_argument('--cuda', action="store_true")
	parser.add_argument('--max-refexp-len', type=int, default=25)
	parser.add_argument('--batch-size', type=int, default=10)
	parser.add_argument('--hidden-size', type=int, default=256)
	parser.add_argument('--num-layers', type=int, default=6)
	parser.add_argument('--dropout', type=float, default=0.2)
	parser.add_argument('--learning-rate', type=float, default=1e-4)
	parser.add_argument('--epoch', type=int, default=300)
	parser.add_argument('--features-dir',  type=str, default='features')
	parser.add_argument('--dataset-split-dir', type=str, default='dataset_split')
	parser.add_argument('--data-dir', type=str, default='data')
	args = parser.parse_args()
	print('args:', args)
	
	if args.model == 'viref':
		train_viref(args, save_path=os.path.join('viref', 'model_save'))
	elif args.model == 'viref-a':
		train_viref_wo_a(args, save_path=os.path.join('viref_wo_a', 'model_save'))
	elif args.model == 'viref-e':
		train_viref_wo_e(args, save_path=os.path.join('viref_wo_e', 'model_save'))

if __name__ == '__main__':
	main()
