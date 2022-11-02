# -*- coding: utf-8 -*-
"""This is imported from End_to_end_GPU.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/namiyousef/argument-mining/blob/develop/experiments/changmao/End-to-end_GPU_v1.ipynb

# End-to-end

This notebook should form the core skeleton of the 'run' function

## Colab Set up
"""
# -- public imports

#    base
import os
import gc
import json
import time
import warnings

#    pytorch
import torch
from torch.utils.data import DataLoader

#    huggingface
from transformers import AutoTokenizer, AutoModelForTokenClassification

#    other libs
import pandas as pd
from pandas.testing import assert_frame_equal

import plac


# -- private imports

#    ml-utils
from mlutils.torchtools.metrics import FScore

#    colab-dev-tools
from colabtools.utils import move_to_device, get_gpu_utilization
from colabtools.config import DEVICE

#    argminer
from argminer.data import ArgumentMiningDataset, TUDarmstadtProcessor, PersuadeProcessor
from argminer.evaluation import inference
from argminer.utils import encode_model_name
from argminer.config import LABELS_MAP_DICT, MAX_NORM


# -- globals
TEST_SIZE = 0.3

# -- define plac inputs
@plac.annotations(
    dataset=("name of dataset to use", "positional", None, str),
    strategy=("name of training strategy to use in form {level}_{labellin scheme}", "positional", None, str),
    model_name=("name of HuggingFace model to use", "positional", None, str),
    max_length=("max number of tokens", "positional", None, int),
    test_size=("test size to use (as a percentage of entire dataset)", "option", None, float),
    batch_size=("Batch size for training", "option", "b", int),
    epochs=("Number of epochs to train for", "option", "e", int),
    save_freq=("How frequently to save model, in epochs", "option", None, int),
    verbose=("Set model verbosity", "option", None, int),
    run_inference=("Flag to run inference or not", "option", 'i', bool)
)
def main(dataset, strategy, model_name, max_length, test_size, batch_size, epochs, save_freq, verbose, run_inference):
    """
    Function to run models as a script. This function expects the following directory structure:
    - cwd (usually $TMPDIR on cluster)
        - df_{dataset}_postprocessed.json
        - train-test-split.csv (for TUDarmstadt)
    :param dataset:
    :param strategy:
    :param model_name:
    :param max_length:
    :param batch_size:
    :param epochs:
    :param save_freq:
    :param verbose:
    :return:
    """

    # configs
    strat_name, strat_label = strategy.split('_')
    df_label_map = LABELS_MAP_DICT[dataset][strat_label]
    num_labels = len(set(df_label_map.label))

    # get data processor and train_test_splits
    try:
        processor = TUDarmstadtProcessor() if dataset == 'TUDarmstadt' else PersuadeProcessor()
        processor = processor.from_json()
        if dataset == 'TUDarmstadt':
            if test_size == 0:
                warnings.warn(
                    f"Detected test_size={test_size}. "
                    f"No splits being made. "
                    f"Train and test data treated identicallu.",
                    UserWarning, stacklevel=2
                )
                df_total = processor.dataframe
                df_dict = {
                    'train': df_total,
                    'test': df_total
                }
            else:
                df_dict = processor.get_tts(test_size=test_size)
        else:
            df_dict = processor.get_tts(test_size=test_size)

        df_train = df_dict.get('train')[['text', 'labels']] # TODO standardise this, so that you don't need to apply this..
        df_test = df_dict.get('test')[['text', 'labels']]

    except Exception as e: # TODO granularise errors
        raise Exception(f'Error occurred during data processing and splitting. Full logs: {e}')

    # get tokenizer, model and optimizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        # TODO force_download
        # TODO add option for optimizer
        optimizer = torch.optim.Adam(params=model.parameters())
    except Exception as e:
        raise Exception(f'Error from tokenizer and model loading. Full logs {e}')

    try:
        train_set = ArgumentMiningDataset(df_label_map, df_train, tokenizer, max_length, strategy)
        test_set = ArgumentMiningDataset(df_label_map, df_test, tokenizer, max_length, strategy, is_train=False)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    except Exception as e:
        raise Exception(f'Error occured during dataset/loader creation. Full logs {e}')

    if not os.path.exists('models'):
        os.makedirs('models')
        print('models directory created!')


    metrics = [FScore(average='macro')]
    scores = {
        'scores': {
            metric.__class__.__name__: [] for metric in metrics
        },
        'epoch_scores': {
            metric.__class__.__name__: [] for metric in metrics
        },
        'epoch_batch_ids': {
            metric.__class__.__name__: [] for metric in metrics
        }
    }

    model.to(DEVICE)
    print(f'Model pushed to device: {DEVICE}')
    for epoch in range(epochs):
        model.train()
        start_epoch_message = f'EPOCH {epoch + 1} STARTED'
        print(start_epoch_message)
        print(f'{"-" * len(start_epoch_message)}')
        start_epoch = time.time()

        start_load = time.time()
        training_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            start_train = time.time()
            inputs = move_to_device(inputs, DEVICE)
            targets = move_to_device(targets, DEVICE)

            optimizer.zero_grad()

            loss, outputs = model(
                labels=targets,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_dict=False
            )
            gpu_util = 'NO GPU' if DEVICE == 'cpu' else get_gpu_utilization()
            training_loss += loss.item()

            # backward pass

            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=MAX_NORM
            )

            loss.backward()
            optimizer.step()

            # metrics
            for metric in metrics:
                score = metric(outputs, targets)
                scores['scores'][metric.__class__.__name__].append(score.item())

            del targets, inputs, loss, outputs
            gc.collect()
            torch.cuda.empty_cache()

            end_train = time.time()

            if verbose > 1:
                print(
                    f'Batch {i + 1} complete. '
                    f'Time taken: load({start_train - start_load:.3g}),'
                    f'train({end_train - start_train:.3g}),'
                    f'total({end_train - start_load:.3g}). '
                    f'GPU util. after train: {gpu_util}. '
                    f'Metrics: {" ".join([f"{metric_name}({score_list[-1]:.3g})" for metric_name, score_list in scores["scores"].items()])}'
                )
            start_load = time.time()

        for metric in metrics:
            score = scores['scores'][metric.__class__.__name__][:i+1]
            avg_score = sum(score)/len(score)
            scores['epoch_scores'][metric.__class__.__name__].append(avg_score)
            scores['epoch_batch_ids'][metric.__class__.__name__].append(i)

        print_message = f'Epoch {epoch + 1}/{epochs} complete. ' \
                        f'Time taken: {start_load - start_epoch:.3g}. ' \
                        f'Loss: {training_loss/(i+1): .3g}. ' \
                        f'Metrics: {" ".join([f"{metric_name}({score_list[-1]:.3g})" for metric_name, score_list in scores["epoch_scores"].items()])}'

        if verbose:
            print(f'{"-" * len(print_message)}')
            print(print_message)
            print(f'{"-" * len(print_message)}')


        if epoch % save_freq == 0:
            encoded_model_name = encode_model_name(model_name, epoch+1)
            save_path = f'models/{encoded_model_name}'
            model.save_pretrained(save_path)
            print(f'Model saved at epoch {epoch+1} at: {save_path}')

    encoded_model_name = encode_model_name(model_name, 'final')
    save_path = f'models/{encoded_model_name}'
    model.save_pretrained(save_path)
    print(f'Model saved at epoch {epoch + 1} at: {save_path}')

    with open('training_scores.json', 'w') as f:
        json.dump(scores, f)
        print('Saved scores.')

    # load trained model
    if run_inference:
        trained_model = AutoModelForTokenClassification.from_pretrained(save_path)
        df_metrics, df_scores = inference(trained_model, test_loader)
        df_scores.to_json('scores.json')



if __name__ == '__main__':
    plac.call(main)