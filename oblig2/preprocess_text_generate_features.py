import pandas as pd 
import torch 
from sklearn.model_selection import train_test_split
from typing import List, Tuple

class CollateFunctor:
    def __init__(self, padding_index: int, max_length: int):
        self.padding_index = padding_index
        self.max_length = max_length

    def __call__(self, samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids_premise = [p for p, h, y in samples]
        input_ids_hypothesis = [h for p, h, y in samples]
        labels = [y for p, h, y in samples]

        input_ids_padded_premise = torch.nn.utils.rnn.pad_sequence(
                                    input_ids_premise,
                                    batch_first = True,
                                    padding_value = self.padding_index
                                )
        
        input_ids_padded_hypothesis = torch.nn.utils.rnn.pad_sequence(
                                        input_ids_hypothesis,
                                        batch_first = True,
                                        padding_value = self.padding_index
                                    )
        
        input_ids_padded_premise = input_ids_padded_premise[:, :self.max_length]
        input_ids_padded_hypothesis = input_ids_padded_hypothesis[:, :self.max_length]
        
        labels = torch.LongTensor(labels)
        
        return input_ids_padded_premise,input_ids_padded_hypothesis, labels
    