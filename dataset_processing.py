import os
import pickle
from datasets import load_from_disk

def dataset_processing(args, tokenizer):
    # Check if tokenized dataset path exists, create directory if not
    tokenized_dataset_dir = f"datasets/{args.model_name}"
    tokenized_dataset_path = os.path.join(tokenized_dataset_dir, "tokenized_dataset.pkl")
    
    # 创建目录，如果它不存在
    os.makedirs(tokenized_dataset_dir, exist_ok=True)
    
    # 如果已经存在 tokenized 数据集，直接加载
    if os.path.exists(tokenized_dataset_path):
        with open(tokenized_dataset_path, 'rb') as f:
            train_dataset, valid_dataset = pickle.load(f)
        return train_dataset, valid_dataset

    # 加载原始数据集
    dataset = load_from_disk(args.dataset_path)

    def tokenize_function(examples):
        return tokenizer(examples["text"], max_length=args.max_length, truncation=True)

    tokenized_dataset = dataset["train"].map(
        tokenize_function,
        batched=True,
        batch_size=10,
        num_proc=20,
        remove_columns=dataset["train"].column_names
    )

    # 分割为训练集和验证集
    train_valid_split = tokenized_dataset.train_test_split(test_size=args.eval_dataset_ratio)
    train_dataset = train_valid_split['train']
    valid_dataset = train_valid_split['test']
    

    # 保存 tokenized 数据集
    with open(tokenized_dataset_path, 'wb') as f:
        pickle.dump((train_dataset, valid_dataset), f)

    return train_dataset, valid_dataset

def dataset_processing_llama13(args, tokenizer):
    # Check if tokenized dataset path exists, create directory if not
    tokenized_dataset_dir = f"datasets/{args.model_path}"
    tokenized_dataset_path = os.path.join(tokenized_dataset_dir, "tokenized_dataset.pkl")
    
    # 创建目录，如果它不存在
    os.makedirs(tokenized_dataset_dir, exist_ok=True)
    
    # 如果已经存在 tokenized 数据集，直接加载
    if os.path.exists(tokenized_dataset_path):
        with open(tokenized_dataset_path, 'rb') as f:
            train_dataset, valid_dataset = pickle.load(f)
        return train_dataset, valid_dataset

    # 加载原始数据集
    dataset = load_from_disk(args.dataset_path)

    # 定义一个分词函数，用于对样本中的"text"字段进行分词处理
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],             # 对每个样本的 "text" 字段进行分词
            max_length=args.max_length,   # 设置最大长度，超过的部分将被截断
            truncation=True,              # 如果文本超过最大长度则截断
            padding='max_length'          # 对所有样本填充至最大长度，保证输入一致性
        )

    # 使用 map 方法对训练数据集进行批量分词处理
    tokenized_dataset = dataset["train"].map(
        tokenize_function,                          # 使用上面定义的分词函数
        batched=True,                               # 批量处理样本（每次处理一个batch）
        batch_size=10,                              # 每个批次处理10个样本
        num_proc=20,                                # 使用20个进程并行加速处理
        remove_columns=dataset["train"].column_names  # 分词后移除原始列，仅保留分词结果
    )


    # 分割为训练集和验证集
    train_valid_split = tokenized_dataset.train_test_split(test_size=args.eval_dataset_ratio) # 按照划分比例进行划分
    train_dataset = train_valid_split['train']
    valid_dataset = train_valid_split['test']
    


    # 将处理后的训练集和验证集保存到本地文件，使用 pickle 序列化
    with open(tokenized_dataset_path, 'wb') as f:
        # 将 train_dataset 和 valid_dataset 作为元组打包，并写入到指定路径的文件中
        # 后续使用可以直接加载
        pickle.dump((train_dataset, valid_dataset), f)


    return train_dataset, valid_dataset
