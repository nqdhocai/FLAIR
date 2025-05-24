import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import os 

def build_dataset(gt: list, ques_df: pd.DataFrame) -> list:
    dataset = []
    qid2question = dict(zip(ques_df["qid"], ques_df["question"]))
    
    for item in gt:
        qid = int(item["qid"])
        question = qid2question.get(qid)
        if question is not None:
            dataset.append((question, item["cids"]))
    return dataset


def load_dataset(dataset_id: str = "ms-marco", data_dir="../data") -> tuple:
    """
    Load and prepare dataset for training, validation, and testing.
    
    Args:
        dataset_id (str): Name of the dataset directory. Defaults to "ms-marco".
        data_dir (str): Path to the data directory. Defaults to "../data".
    
    Returns:
        tuple: A tuple containing (train_dataset, valid_dataset, test_dataset, corpus).
    
    Raises:
        FileNotFoundError: If the dataset directory does not exist.
    """
    dataset_path = os.path.join(data_dir, dataset_id)

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory '{dataset_path}' does not exist.")

    corpus = pd.read_csv(f"{dataset_path}/corpus.csv")
    ques_train = pd.read_csv(f"{dataset_path}/question_train.csv")
    ques_test = pd.read_csv(f"{dataset_path}/question_test.csv")
    ques_valid = pd.read_csv(f"{dataset_path}/question_valid.csv")
    ques = pd.concat([ques_train, ques_valid, ques_test], ignore_index=True)

    with open(f"{dataset_path}/ground_truth.json", "r") as f:
        gt = json.load(f)
        new_gt = []
        for item in gt:
            for k, v in item.items():
                new_gt.append({
                    "qid": k,
                    "cids": v
                })
        gt = new_gt

    train_dataset = build_dataset(gt, ques_train)
    valid_dataset = build_dataset(gt, ques_valid)
    test_dataset  = build_dataset(gt, ques_test)

    # Print số lượng
    print(f"Corpus size       : {len(corpus):,}")
    print(f"Train questions   : {len(ques_train):,}")
    print(f"Valid questions   : {len(ques_valid):,}")
    print(f"Test questions    : {len(ques_test):,}")
    print(f"Total questions   : {len(ques):,}")
    print(f"Ground truth size : {len(gt):,}\n")

    return train_dataset, valid_dataset, test_dataset, corpus