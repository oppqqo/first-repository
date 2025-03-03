# 使用中文镜像源
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 使用dataset和dataloader
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# 导入语料库
import datasets
from datasets import load_dataset
from datasets import load_from_disk
from datasets import DatasetDict
# 导入分词器
from transformers import AutoTokenizer
# from transformers import BertTokenizer

# 选择语料库"llamafactory/alpaca_gpt4_zh"，如果没有就下载，如果有就加载
print('load dataset....')
model_name = 'alpaca_gpt4_zh' # 模型名称
dataset_path = 'd:/VSProject/data' # 语料库路径
if os.path.exists(dataset_path+'/'+model_name):
    dataset = load_from_disk(dataset_path)
else:
    dataset = load_dataset("llamafactory/alpaca_gpt4_zh",cache_dir=dataset_path)
    # 储存语料库到本地data文件夹
    dataset.save_to_disk(dataset_path) 

# # 查看语料库结构
# print(dataset)

# 导入分词器
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

# 建立词汇表
# 添加特殊标记
special_tokens_dict = {
    'bos_token': '[SOS]',  # 句子开始标记
    'eos_token': '[EOS]',  # 句子结束标记
    'pad_token': '[PAD]',  # 填充标记
    'unk_token': '[UNK]',  # 未知词标记
    'sep_token': '[SEP]',  # 分隔标记
    'cls_token': '[CLS]'   # 分类标记
}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
vocab = tokenizer.get_vocab()
# print(f"添加了 {num_added_tokens} 个特殊标记")

# # 测试词汇表信息
# print("词汇示例(word:index):",{word : vocab[word] for word in ['[SOS]','[EOS]','[PAD]','[UNK]','[SEP]','[CLS]','我','你','他']})

# 转换为PyTorch的Dataset
dataset = Dataset.from_dict(dataset['train'])
print(dataset)
