## Script to pre-process ASAP dataset (training_set_rel3.tsv) based on the essay IDs

import argparse
import codecs
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-file', dest='input_file', required=True, help='Input TSV file')
args = parser.parse_args()

def extract_based_on_ids(dataset, id_file):
	counter_done = 0
	counter_notdone = 0
	lines = []
	with open(id_file) as f:
		for line in f:
			id = line.strip()
			try:
				lines.append(dataset[id])
				print("done")
				counter_done = counter_done +1
			except:
				print('ERROR: Invalid ID %s in %s' % (id, id_file), file=sys.stederr)
				counter_notdone = counter_notdone +1
	print("Done, notDone", counter_done, counter_notdone)
	return lines

def create_dataset(lines, output_fname):
	f_write = open(output_fname, 'w', encoding = 'utf-8')
	# f_write.write(dataset['header'])
	for line in lines:
		f_write.write(line)

def collect_dataset(input_file):
	dataset = dict()
	lcount = 0
	# with open(input_file) as f:
	with codecs.open(input_file, "r", encoding = 'utf-8', errors='ignore') as f:
		print(f)
		for line in f:
			# print(line)
			lcount += 1
			if lcount == 1:
				dataset['header'] = line
				continue
			parts = line.split('\t')
			assert len(parts) >= 6, 'ERROR: ' + line
			dataset[parts[0]] = line
	return dataset

dataset = collect_dataset(args.input_file)
for fold_idx in range(0, 5):
	for dataset_type in ['dev', 'test', 'train']:
		lines = extract_based_on_ids(dataset, 'fold_%d/%s_ids.txt' % (fold_idx, dataset_type))
		create_dataset(lines, 'fold_%d/%s.tsv' % (fold_idx, dataset_type))