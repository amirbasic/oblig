from conllu import parse
from ner_eval import Evaluator
from smart_open import open
from collections import namedtuple
from copy import deepcopy


def f1(precision, recall):
    if precision == 0 and recall == 0:
        score = 0
    else:
        score = 2 * (precision * recall) / (precision + recall)
    return score


# wrong file format compared to the teachers eval_on_test.py
def real_evaluation(path_gold, pred_conllu):

    gold = parse(open(path_gold, "r", encoding='utf8').read())
    predictions = parse(open(pred_conllu, "r", encoding='utf8').read())
    
    gold_labels = []
    for sentence in gold:
        sentence = [token["misc"]["name"] for token in sentence]
        gold_labels.append(sentence)

    predicted_labels = []
    for sentence in predictions:
        sentence = [token["misc"]["name"] for token in sentence]
        predicted_labels.append(sentence)

    entities = ["PER", "ORG", "LOC", "GPE_LOC", "GPE_ORG", "PROD", "EVT", "DRV"]

    evaluator = Evaluator(gold_labels, predicted_labels, entities)
    results, results_agg = evaluator.evaluate()

    print("F1 score:")
    for entity in results_agg:
        prec = results_agg[entity]["strict"]["precision"]
        rec = results_agg[entity]["strict"]["recall"]
        #print(f"{entity}:\t{f1(prec, rec):.4f}")
    prec = results["strict"]["precision"]
    rec = results["strict"]["recall"]
    print(f"Overall score: {f1(prec, rec):.4f}")