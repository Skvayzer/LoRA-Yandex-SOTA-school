__author__='kilimanj4r0'

"""
Author: Vladimir Makharev
Date: 14/11/2022
Description:
    This script aims to evaluate the output of data-to-text NLG models by 
    computing popular automatic metrics such as BLEU, METEOR, TER.
    
    ARGS:
        usage: eval.py [-h] -R REFERENCE -H HYPOTHESIS [-nr NUM_REFS]
               [-m METRICS]
        
        required arguments:
          -R REFERENCE, --reference REFERENCE
                                reference translation
          -H HYPOTHESIS, --hypothesis HYPOTHESIS
                                hypothesis translation

        optional arguments:
          -h, --help            show this help message and exit
          -nr NUM_REFS, --num_refs NUM_REFS
                                number of references
          -m METRICS, --metrics METRICS
                                evaluation metrics to be computed

    EXAMPLE:
        python3 eval.py -R data/en/references/reference -H data/en/hypothesis -nr 6 -m bleu,meteor,ter
"""

import argparse
import codecs
from tabulate import tabulate
import evaluate

# Required: pip install evaluate nltk sacrebleu -U

def parse(refs_path, hyps_path, num_refs):
    print('Starting to parse inputs...')

    references = []
    for i in range(num_refs):
        fname = refs_path + str(i) if num_refs > 1 else refs_path
        with codecs.open(fname, 'r', 'utf-8') as f:
            texts = f.read().split('\n')
            for j, text in enumerate(texts):
                if len(references) <= j:
                    references.append([text])
                else:
                    references[j].append(text)
    with codecs.open(hyps_path, 'r', 'utf-8') as f:
        hypothesis = f.read().split('\n')

    print('Finishing to parse inputs')
    return references, hypothesis


def compute_bleu(references, hypothesis):
    print('Computing BLEU...')
    bleu = evaluate.load('bleu')
    result =  bleu.compute(predictions=hypothesis, references=references)
    print(f'BLEU computed: {result["bleu"] * 100}')
    return result['bleu'] * 100

def compute_meteor(references, hypothesis):
    print('Computing METEOR...')
    meteor = evaluate.load('meteor')
    result = meteor.compute(predictions=hypothesis, references=references)
    print(f'METEOR computed: {result["meteor"] * 100}')
    return result['meteor'] * 100

def compute_ter(references, hypothesis):
    print('Computing TER...')
    ter = evaluate.load('ter')
    result = ter.compute(predictions=hypothesis, references=references)
    print(f'TER computed: {result["score"]}')
    return result['score']

def run(refs_path, hyps_path, num_refs, metrics='bleu,meteor,ter'):
    metrics = args.metrics.lower().split(',')
    references, hypothesis = parse(refs_path, hyps_path, num_refs)
    
    result = {}
    
    print('Evaluation started...')
    if 'bleu' in metrics:
        result['bleu'] = compute_bleu(references, hypothesis)
    if 'meteor' in metrics:
        result['meteor'] = compute_meteor(references, hypothesis)
    if 'ter' in metrics:
        result['ter'] = compute_ter(references, hypothesis)
    print('Evaluation finished...')    
    return result


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-R", "--reference", help="reference translation", required=True)
    arg_parser.add_argument("-H", "--hypothesis", help="hypothesis translation", required=True)
    arg_parser.add_argument("-nr", "--num_refs", help="number of references", type=int, default=4)
    arg_parser.add_argument("-m", "--metrics", help="evaluation metrics to be computed", default='bleu,meteor,ter')

    args = arg_parser.parse_args()

    print('Reading input...')
    refs_path = args.reference
    hyps_path = args.hypothesis
    num_refs = args.num_refs
    metrics = args.metrics
    print('Read input finished...')

    result = run(refs_path=refs_path, hyps_path=hyps_path, num_refs=num_refs, metrics=metrics)
    
    headers, values = [], []
    for metric, value in result.items():
        headers.append(metric.upper())
        values.append(round(value, 2))
    print(tabulate([values], headers=headers, tablefmt='orgtbl'))
