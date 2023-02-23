import os
import sys
import argparse
from random import seed
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Set, Tuple, Union
import collections.abc

import torch
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForImageClassification, AutoModelForMaskedImageModeling, TrainingArguments, Trainer
from sklearn.metrics import *
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    ToTensor,
)

from datasets import load_dataset
from datasets import load_metric
from datasets import Dataset, Image

from Vision.load_data import get_data_split, get_all_data
from models.vit.modeling_vit import ViTForImageClassification, ViTForMaskedImageModeling
from models.swin.modeling_swin import SwinForImageClassification, SwinForMaskedImageModeling


def one_hot(y_):
    y_ = y_.reshape(len(y_))

    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

class MaskGenerator:
    """
    A class to generate boolean masks for the pretraining task.
    A mask is a 1D tensor of shape (model_patch_size**2,) where the value is either 0 or 1,
    where 1 indicates "masked".

    Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-pretraining/run_mim.py
    """

    def __init__(self, input_size=384, grid_layout=6, mask_patch_size=8, model_patch_size=4, mask_ratio=0.6, mask_method="vertical"):
        
        input_size = input_size if isinstance(input_size, collections.abc.Iterable) else (input_size, input_size)
        grid_layout = grid_layout if isinstance(grid_layout, collections.abc.Iterable) else (grid_layout, grid_layout)
        model_patch_size = model_patch_size if isinstance(model_patch_size, collections.abc.Iterable) else (model_patch_size, model_patch_size)
        mask_patch_size = mask_patch_size if isinstance(mask_patch_size, collections.abc.Iterable) else (mask_patch_size, mask_patch_size)
        
        self.input_size = input_size
        self.grid_layout = grid_layout
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        self.mask_method = mask_method

        # if self.input_size[0] % self.mask_patch_size[0] != 0 or self.input_size[1] % self.mask_patch_size[1] != 0:
        #     raise ValueError("Input size must be divisible by mask patch size")
        # if self.input_size[0] % self.grid_layout[0] != 0 or self.input_size[1] % self.grid_layout[1] != 0:
        #     raise ValueError("Input size must be divisible by graph size")
        # if self.mask_patch_size[0] % self.model_patch_size[0] != 0 or self.mask_patch_size[1] % self.model_patch_size[1] != 0:
        #     raise ValueError("Mask patch size must be divisible by model patch size")

        # mask a column in each line graph sub-image
        if self.mask_method == "vertical":
            self.mask_patch_size = (self.input_size[0] // self.grid_layout[0], self.mask_patch_size[1])
        elif self.mask_method == "random":
            pass
        
        self.rand_size = (self.input_size[0] // self.mask_patch_size[0], self.input_size[1] // self.mask_patch_size[1])
        self.scale = (self.mask_patch_size[0] // self.model_patch_size[0], self.mask_patch_size[1] // self.model_patch_size[1])

        self.token_count = self.rand_size[0]*self.rand_size[1]
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size[0], self.rand_size[1]))
        mask = mask.repeat(self.scale[0], axis=0).repeat(self.scale[1], axis=1)

        return torch.tensor(mask.flatten())
        # return torch.tensor(mask)


def fine_tune_hf(
    model_path,
    model_loader,
    output_dir,
    train_dataset,
    val_dataset,
    test_dataset,
    image_size,
    grid_layout,
    patch_size,
    mask_patch_size,
    mask_ratio,
    mask_method,
    epochs,
    train_batch_size,
    eval_batch_size,
    save_steps,
    logging_steps,
    learning_rate,
    seed,
    save_total_limit,
    do_train,
    finetune,
    train_from_scratch
    ):  

    # loading model and feature extractor
    if do_train and not finetune:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path, ignore_mismatched_sizes=True)
        config.update(
                {
                    "image_size": image_size,
                    "grid_layout": grid_layout,
                    "mask_patch_size": mask_patch_size
                }
            )
        if train_from_scratch:
            model = model_loader(config)
        else:
            model = model_loader.from_pretrained(model_path, config=config, ignore_mismatched_sizes=True)
    else:
        # if not train, load the fine-tuned model saved in output_dir
        if os.path.exists(output_dir):
            dir_list = os.listdir(output_dir) # find the latest checkpoint
            latest_checkpoint_idx = 0
            for d in dir_list:
                if "checkpoint" in d:
                    checkpoint_idx = int(d.split("-")[-1])
                    if checkpoint_idx > latest_checkpoint_idx:
                        latest_checkpoint_idx = checkpoint_idx
    
            if latest_checkpoint_idx > 0 and os.path.exists(os.path.join(output_dir, f"checkpoint-{latest_checkpoint_idx}")):
                ft_model_path = os.path.join(output_dir, f"checkpoint-{latest_checkpoint_idx}")
                feature_extractor = AutoFeatureExtractor.from_pretrained(ft_model_path)
                model = model_loader.from_pretrained(ft_model_path, ignore_mismatched_sizes=True, image_size=image_size, grid_layout=grid_layout)
                print("load from: ", ft_model_path)
            else: # don't have a fine-tuned model
                feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
                if train_from_scratch:
                    config = AutoConfig.from_pretrained(model_path, ignore_mismatched_sizes=True, image_size=image_size, grid_layout=grid_layout)
                    model = model_loader.from_config(config)
                else:
                    model = model_loader.from_pretrained(model_path, ignore_mismatched_sizes=True, image_size=image_size, grid_layout=grid_layout)
        else:
            feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
            if train_from_scratch:
                config = AutoConfig.from_pretrained(model_path, ignore_mismatched_sizes=True, image_size=image_size, grid_layout=grid_layout)
                model = model_loader.from_config(config)
            else:
                model = model_loader.from_pretrained(model_path, ignore_mismatched_sizes=True, image_size=image_size, grid_layout=grid_layout)

    # define image transformation function
    train_transforms = Compose(
            [   
                # Resize(feature_extractor.size),
                # RandomResizedCrop(feature_extractor.size),
                ToTensor(),
                # normalize,
            ]
        )
    val_transforms = Compose(
            [
                # Resize(feature_extractor.size),
                # CenterCrop(feature_extractor.size),
                ToTensor(),
                # normalize,
            ]
        )

    mask_generator = MaskGenerator(
        input_size=image_size,
        grid_layout=grid_layout,
        mask_patch_size=mask_patch_size,
        model_patch_size=patch_size,
        mask_ratio=mask_ratio,
        mask_method=mask_method
    )

    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
        example_batch["mask"] = [mask_generator() for image in example_batch["image"]]
        return example_batch

    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
        example_batch["mask"] = [mask_generator() for image in example_batch["image"]]
        return example_batch

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        mask = torch.stack([example["mask"] for example in examples])
        return {"pixel_values": pixel_values, "bool_masked_pos": mask}

    # transform the dataset
    train_dataset.set_transform(preprocess_train)
    val_dataset.set_transform(preprocess_val)
    test_dataset.set_transform(preprocess_val)

    # training arguments
    training_args = TrainingArguments(
    output_dir=output_dir,          # output directory
    num_train_epochs=epochs,              # total number of training epochs
    per_device_train_batch_size=train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=eval_batch_size,   # batch size for evaluation
    evaluation_strategy = "steps",
    save_strategy = "steps",
    learning_rate=learning_rate, # 2e-5
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    # fp16=True,
    # fp16_backend="amp",
    save_steps=save_steps,
    logging_steps=logging_steps,
    logging_dir=os.path.join(output_dir, "runs/"),
    save_total_limit=save_total_limit,
    seed=seed,
    load_best_model_at_end=True,
    remove_unused_columns=False
    )

    trainer = Trainer(
    model,
    training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=feature_extractor,
    data_collator=collate_fn,
    )

    # training the model with Huggingface 🤗 trainer
    if do_train:
        train_results = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()
    
    # evaluation results
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # arguments for dataset
    parser.add_argument('--dataset', type=str, default='P12') #
    parser.add_argument('--dataset_prefix', type=str, default='') #
    
    parser.add_argument('--withmissingratio', default=False, help='if True, missing ratio ranges from 0 to 0.5; if False, missing ratio =0') #
    parser.add_argument('--feature_removal_level', type=str, default='no_removal', choices=['no_removal', 'set', 'sample'],
                        help='use this only when splittype==random; otherwise, set as no_removal') #
    parser.add_argument('--predictive_label', type=str, default='mortality', choices=['mortality', 'LoS'],
                        help='use this only with P12 dataset (mortality or length of stay)')
    
    # arguments for huggingface training
    parser.add_argument('--model', type=str, default='vit', choices=['vit', 'swin', 'resnet', 'clip', 'convnext']) #
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1799)
    parser.add_argument('--save_total_limit', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--logging_steps', type=int, default=5)
    parser.add_argument('--save_steps', type=int, default=5)

    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--upsample', default=False)
    
    # argument for the images
    parser.add_argument('--grid_layout', default=None)
    parser.add_argument('--image_size', default=None)
    parser.add_argument('--mask_patch_size', type=int, default=None)
    parser.add_argument('--mask_ratio', type=float, default=None)
    parser.add_argument('--mask_method', type=str, default=None)

    # argument for ablation study
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--finetune', action='store_true', help='whether to load a fine-tuned model to continue')
    parser.add_argument('--train_from_scratch', action='store_true', help='whether to load a randomly initialized model')

    args = parser.parse_args()

    dataset = args.dataset
    dataset_prefix = args.dataset_prefix
    print(f'Dataset used: {dataset}, prefix: {dataset_prefix}.')

    epochs = args.epochs
    upsample = args.upsample
    image_size = grid_layout = None
    mask_patch_size = args.mask_patch_size
    mask_ratio = args.mask_ratio
    mask_method = args.mask_method

    if dataset == 'P12':
        base_path = '../../dataset/P12data'
        num_classes = 2
        image_size = 384
        grid_layout = 6
    elif dataset == 'P19':
        base_path = '../../dataset/P19data'
        num_classes = 2
        image_size = 384
        grid_layout = 6
    elif dataset == 'PAM':
        base_path = '../../dataset/PAMdata'
        num_classes = 8
        image_size = (256,320)
        grid_layout = (4,5)
        epochs = 30
    elif dataset == 'EthanolConcentration':
        base_path = '../../dataset/TSRAdata/Classification/EthanolConcentration'
        num_classes = 4
        image_size = 256
        grid_layout = 2
    elif dataset == 'Heartbeat':
        base_path = '../../dataset/TSRAdata/Classification/Heartbeat'
        num_classes = 2
        image_size = 384
        grid_layout = 3
    elif dataset == 'SpokenArabicDigits':
        base_path = '../../dataset/TSRAdata/Classification/SpokenArabicDigits'
        num_classes = 10
        image_size = 256
        grid_layout = 4
    elif dataset == 'SelfRegulationSCP1':
        base_path = '../../dataset/TSRAdata/Classification/SelfRegulationSCP1'
        num_classes = 2
        image_size = (256,384)
        grid_layout = (2,3)
    elif dataset == 'SelfRegulationSCP2':
        base_path = '../../dataset/TSRAdata/Classification/SelfRegulationSCP2'
        num_classes = 2
        image_size = 384
        grid_layout = 3
    elif dataset == 'JapaneseVowels':
        base_path = '../../dataset/TSRAdata/Classification/JapaneseVowels'
        num_classes = 9
        image_size = (288,384)
        grid_layout = (3,4)
    elif dataset == 'UWaveGestureLibrary':
        base_path = '../../dataset/TSRAdata/Classification/UWaveGestureLibrary'
        num_classes = 8
        image_size = 256
        grid_layout = 2
    
    """prepare the model for sequence classification"""
    model = args.model
    model_loader = AutoModelForMaskedImageModeling
    if model == "vit":
        model_path = "google/vit-base-patch16-224-in21k"
        model_loader = ViTForMaskedImageModeling
        patch_size = 16
    elif model == "swin":
        model_path = "microsoft/swin-base-patch4-window7-224-in22k"
        model_loader = SwinForMaskedImageModeling
        patch_size = 4

    feature_removal_level = args.feature_removal_level  # 'set' for fixed, 'sample' for random sample
    print(feature_removal_level)

    """While missing_ratio >0, feature_removal_level is automatically used"""
    if bool(args.withmissingratio) == True:
        missing_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    else:
        missing_ratios = [0]
    print('missing ratio list', missing_ratios)
    
    for missing_ratio in missing_ratios:
        
        """prepare for training"""
        n_runs = args.n_runs
        n_splits = args.n_splits
        subset = False

        for k in range(n_splits):

            split_idx = k + 1
            print('Split id: %d' % split_idx)
            if dataset == 'P12':
                if subset == True:
                    split_path = '/splits/phy12_split_subset' + str(split_idx) + '.npy'
                else:
                    split_path = '/splits/phy12_split' + str(split_idx) + '.npy'
            elif dataset == 'P19':
                split_path = '/splits/phy19_split' + str(split_idx) + '_new.npy'
            elif dataset == 'PAM':
                split_path = '/splits/PAM_split_' + str(split_idx) + '.npy'
            else:
                split_path = '/splits/split_' + str(split_idx) + '.npy'
            
            # the path to save models
            if args.output_dir is None:
                if args.train_from_scratch:
                    output_dir = f"../../ckpt/ImgMIM/{dataset_prefix}{dataset}_{model}_{mask_patch_size}_{mask_ratio}_from_scratch/split{split_idx}"
                else:
                    output_dir = f"../../ckpt/ImgMIM/{dataset_prefix}{dataset}_{model}_{mask_patch_size}_{mask_ratio}_{mask_method}/split{split_idx}"
            else:
                output_dir = args.output_dir

            # prepare the data:
            Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, split_path, dataset=dataset, prefix=dataset_prefix, upsample=upsample, missing_ratio=missing_ratio)
            print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))

            for m in range(n_runs):
                print('- - Run %d - -' % (m + 1))
                fine_tune_hf(
                model_path=model_path,
                model_loader=model_loader,
                output_dir=output_dir,
                train_dataset=Ptrain,
                val_dataset=Pval,
                test_dataset=Ptest,
                image_size=image_size,
                grid_layout=grid_layout,
                patch_size=patch_size,
                mask_patch_size=mask_patch_size,
                mask_ratio=mask_ratio,
                mask_method=mask_method,
                epochs=epochs,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                logging_steps=args.logging_steps,
                save_steps=args.save_steps,
                learning_rate=args.learning_rate,
                seed=args.seed,
                save_total_limit=args.save_total_limit,
                do_train=args.do_train,
                finetune=args.finetune,
                train_from_scratch=args.train_from_scratch
                )