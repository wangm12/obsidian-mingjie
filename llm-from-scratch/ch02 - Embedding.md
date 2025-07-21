# embedding
![[embedding_model.png]]
**深度学习模型无法处理视频、音频和文本等原始数据格式。 因此，我们使用嵌入模型将这些原始数据转换为密集向量表示，这样深度学习架构就可以轻松理解和处理原始数据。 具体来说，此图展示了将原始数据转换为三维数值向量的过程。 需要注意的是，不同的数据格式需要不同的嵌入模型。 例如，为文本设计的嵌入模型不适用于嵌入音频或视频数据。**


目前已经开发了多种算法和框架来生成词嵌入。 
其中最早和最流行的例子之一是 Word2Vec 方法。 Word2Vec 训练神经网络架构是通过预测给定目标词的上下文或反之来生成词嵌入。 Word2Vec 架构的主要思想是，出现在相似上下文中的词往往具有相似的含义。 因此，当将词嵌入投影到二维空间中以便于可视化时，可以看到相似的术语聚集在一起
![[word2vec.png]] **如果词嵌入是二维的，我们可以将它们绘制在二维散点图中以便于可视化，如此图所示。在使用词嵌入技术，例如 Word2Vec 时，对应于相似概念的词在嵌入空间中通常彼此接近。例如，在嵌入空间中，不同类型的鸟类相对于国家和城市更为靠近。**

词嵌入可以有不同的维度，从一维到数千维不等。 如图2.3所示，我们可以选择二维词嵌入以便于可视化。 更高的维度可能会捕捉到词之间更多细微的关系，但作为代价，计算效率将会下降。

虽然我们可以使用预训练模型如 Word2Vec 来为机器学习模型生成嵌入，但大型语言模型（LLMs）通常会生成它们自己的嵌入，这些嵌入是输入层的一部分，并在训练期间更新。 将嵌入作为 LLM 训练的一部分进行优化，而不是使用 Word2Vec 的优势在于，嵌入被优化以适应手头的特定任务和数据。

然而，在处理大型语言模型（LLMs）时，我们通常使用的嵌入维度远高于图2.3中显示的维度。 对于GPT-2和GPT-3，嵌入大小（通常被称为模型隐藏状态的维度）根据具体的模型变种和大小而变化。 这是性能与效率之间的权衡。 
最小的GPT-2（1.17亿参数）和GPT-3（1.25亿参数）模型使用768维的嵌入大小来提供具体示例。最大的GPT-3模型（1750亿参数）使用的嵌入大小为12,288维。

# tokenizing text
![[tokenize.png]]

### 是否移除空格
在开发一个简单的分词器时，是否应该将空格编码为单独的字符或者直接移除它们，这取决于我们的应用及其需求。 移除空格可以减少内存和计算需求。然而，保留空格在我们训练对文本的精确结构敏感的模型时可能是有用的 （例如，Python代码对缩进和间距非常敏感）。 这里，我们为了简化和简洁化分词输出而移除空格。 随后，我们将进入到一个包括空格的分词方案。

# 将token转换为token ID
为了将先前生成的token映射到token ID，我们首先需要构建一个所谓的词汇表。 这个词汇表定义了我们如何将每个独特的词和特殊字符映射到一个独特的整数，如图 2.6 所示。

## token dic
**图 2.6 我们通过将训练数据集中的整个文本分割成单个token来构建词汇表。 这些单独的token随后按字母顺序进行排序，并移除重复的token。 然后，将这些独特的token聚集成一个词汇表，该词汇表定义了从每个独特token到一个独特整数值的映射。 所展示的词汇表为了说明目的故意保持词汇量较小，并且为了简化没有包含标点符号或特殊字符。**
![[token_id_dic.png]]

当我们想要将大型语言模型（LLM）的输出从数字转换回文本时，我们也需要一种方法将token ID 转换回文本。 为此，我们可以创建一个词汇表的逆向版本，将token ID 映射回对应的token标记。

让我们通过 Python 来实现一个完整的分词器类，其中包括一个 编码(encode) 方法，该方法将文本分割成token，并通过词汇表执行字符串到整数的映射以生成token ID。 此外，我们还实现一个 解码(decode) 方法，该方法执行整数到字符串的反向映射，将token ID 转换回文本。

## 分词器
**分词器实现共有两个常用方法：一个是编码方法，另一个是解码方法。 编码方法接收样本文本，将其分割为单独的标记，并通过词汇表将这些标记转换为标记 ID。 解码方法接收token ID，将它们转换回文本token，并将这些文本token连接成自然文本。**
![[分词器.png]]

# 添加特殊的上下文tokens

## <|UNK|> 
**我们在词汇表中添加特殊的标记来处理特定的上下文。
例如，我们添加<|UNK|> token表示新的和未知的单词，这些单词不是训练数据的一部分，因此也不是现有词汇表的一部分。
此外，我们还添加了一个<|endoftext|> token，我们可以使用它来分隔两个不相关的文本源。**
![[tokenize_unknow_endoftext.png]]

## <|endoftext|>
**当处理多个独立的文本源时，我们在这些文本间添加叫做<|endoftext|>的tokens。这些<|endoftext|>tokens作为标记，标志着一个特定段落的开始和结束，这使得LLM能更有效地处理和理解文本。**
![[endoftext_token.png]]

# Byte pair encoding 字节对编码（BPE）

## tiktoken
https://github.com/openai/tiktoken, which implements the BPE algorithm very efficiently based on source code in Rust.

```
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
  
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of some" integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)
# [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252]

strings = tokenizer.decode(integers)
print(strings)
# 'Hello, do you like tea? <|endoftext|> In the sunlit terraces of'
```

## summary

基于上述的token IDs以及解码的文本，我们可以做出2点有价值的观察。\
第一，**`<|endoftext|>`**词元（token）被赋值了一个很大的token ID，例如，50256。
事实上，被用于训练诸如GPT-2，GPT-3以及被ChatGPT使用的原始模型的BPE分词器，总计词汇的规模是50257，其中`<|endoftext|>`被指定为最大的token ID。

第二，上述的BPE分词器可以正确的解码和编码没有见过的词汇，
例如"someunknownPlace"。BPE解码器可以处理任何没有见过的词汇。那么，他是怎么无需使用`<|unk|>`词元就做到这个的呢？
- BPE使用的算法会将不在预定义词表里的单词分解为更小的子单词单元或者甚至是独立的字母，使BPE可以处理词表外的单词。所以，基于这种算法，如果分词器在分词时遇到了不熟悉的单词，他会使用一系列的子单词词元或者字母来替换它

![[BPE_unknown_words.png]]

将不认识的单词分解为更小的的词元或字母保证了分词器，以及后续被训练的大模型可以处理任意的文本，即使文本里包含了从来没在测试数据里出现过的单词。

# 使用滑动窗口进行数据采样

## data sampling
**给定一个文本样本，提取输入块作为 LLM 的输入子样本，LLM 在训练期间的任务是预测输入块之后的下一个单词。在训练过程中，我们会屏蔽掉目标词之后的所有单词。请注意，在 LLM 处理文本之前，该文本已经进行 token 化，为了便于说明，此图省略了 token 化步骤。**
![[data_sampling.png]]

## **高效的数据加载器**
![[高效的数据加载器.png]]**我们将所有输入存存储到一个名为 x 的张量中，其中每一行都代表一个输入上下文。同时，我们创建了另一个名为 y 的张量，用于存储对应的预测目标（即下一个单词），这些目标是通过将输入内容向右移动一个位置得到的。**

### (code) 一个用于批处理输入和目标的数据集 
```
import torch
from torch.utils.data import Dataset, DataLoader
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt) #A
        for i in range(0, len(token_ids) - max_length, stride): #B
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self): #C
        return len(self.input_ids)
    def __getitem__(self, idx): #D
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4,
        max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2") #A
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) #B
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last) #C
    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader) #A
    first_batch = next(data_iter)
    print(first_batch)
```
`first_batch`变量包含两个张量：第一个张量存储输入的 token ID，第二个张量存储目标的 token ID。由于`max_length`为4，因此这两个张量都只包含 4 个 toekn ID。需要注意的是，这里的输入大小 4 是相对较小的，仅用于演示。在实际训练语言模型时，输入大小通常至少为 256。

### 张量 - tensor
`torch.tensor` 是一个核心函数，用于从现有的数据（如 Python 列表或 NumPy 数组）创建**张量（Tensor）**。
张量是 PyTorch 中最基本的数据结构，类似于多维数组，是构建和训练神经网络的基础。

### Dataset
**负责封装和访问数据**：它像一个数据“仓库”，知道数据的总数 (`__len__`) 和如何获取单个样本 (`__getitem__`)

`Dataset` 是一个抽象类，它代表了你的整个数据集。当你要加载自定义数据（例如，一堆图片和它们对应的标签）时，你需要创建一个继承自 `Dataset` 的自定义类，并重写它的两个核心方法：

1. `__len__(self)`:
    
    - **作用**：返回数据集中样本的总数。
    - **为何需要**：`DataLoader` 需要知道总共有多少数据，以便确定迭代的次数和如何划分批次。
2. `__getitem__(self, idx)`:
    
    - **作用**：根据给定的索引 `idx`，加载并返回一个样本。
    - **为何需要**：这是获取单个数据点的关键。你可以在这里定义如何从磁盘读取一张图片、如何解析一个文本文件、如何进行数据预处理或数据增强等。`DataLoader` 会调用这个方法来收集数据。

### DataLoader
`DataLoader` **负责组织和批量加载数据**：它从 `Dataset` 中取出数据，然后打包成批次 (batch)，并提供诸如打乱数据 (shuffling)、并行加载等高级功能。

`DataLoader` 是一个迭代器，它包装了 `Dataset` 对象，使其能够轻松、高效地进行数据加载。它负责将 `Dataset` 中零散的样本自动组合成一个个批次 (batches)。
这对于模型训练至关重要，因为我们通常使用小批量梯度下降 (mini-batch gradient descent) 来更新模型权重。

`DataLoader` 提供了许多强大的功能，通过其参数进行配置：

- `dataset`: 你创建的 `Dataset` 对象。
- `batch_size` (int, optional): 每个批次加载的样本数（默认为 1）。这是最重要的参数之一。
- `shuffle` (bool, optional): 如果为 `True`，则在每个 epoch 开始时打乱数据顺序（默认为 `False`）。这有助于防止模型过拟合，提高泛化能力。
- `num_workers` (int, optional): 用于数据加载的子进程数（默认为 0）。当 `num_workers > 0` 时，PyTorch 会使用多个子进程并行地从 `Dataset` 中加载数据，这可以大大加快数据加载速度，尤其是在数据预处理比较耗时的情况下。
- `pin_memory` (bool, optional): 如果为 `True`，`DataLoader` 会将张量复制到 CUDA 固定内存中，这可以加速数据从 CPU 到 GPU 的传输。
- `drop_last` (bool, optional): 如果数据集的总样本数不能被 `batch_size` 整除，最后一个批次的样本数会小于 `batch_size`。如果设置为 `True`，则会丢弃这最后一个不完整的批次（默认为 `False`）。

# 构建词符嵌入
## code
```
import torch

input_ids = torch.tensor([2, 3, 5, 1])

# 为了简化问题，假设我们有一个只包含6个单词的小型词汇表，我们想要创建大小为3的嵌入
# `vocab_size = 6`: 定义了我们词汇表的大小。也就是说，我们的世界里总共只有 6 个独特的词（ID 从 0 到 5）。
# `output_dim = 3`: 定义了我们希望每个词的嵌入向量的维度（大小）。这里我们选择 3，意味着每个词都将被表示成一个 3 维的向量。在实际应用中，这个维度通常是 128, 256, 768 等更大的值。
vocab_size = 6
output_dim = 3

# 设置随机种子, `nn.Embedding` 在初始化时会随机生成权重，设置种子可以确保每次运行代码时，生成的随机数都是一样的，这样结果就可复现了。
torch.manual_seed(123)

# 我们创建了一个 `Embedding` 层的实例。它内部会自动创建一个形状为 `(vocab_size, output_dim)`，也就是 `(6, 3)` 的权重矩阵。这个矩阵就是我们前面说的“查找表”。
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

print(embedding_layer.weight)
# `embedding_layer(torch.tensor([3]))`: 这行代码演示了查找过程。我们将一个包含单个 ID `3` 的张量传给 `embedding_layer`。
# `embedding_layer` 接收到 ID `3`，就会去它的权重矩阵 `weight` 中**查找索引为 3 的那一行**（记住索引从0开始）。
# 从上面的权重矩阵中可以看到，第 4 行 (索引为 3) 正是 `[-0.4015, 0.9666, -1.1481]`。
# 输出结果和权重矩阵的第 4 行完全一致，验证了它就是一个查找操作。
tensor([[ 0.3374, -0.1778, -0.1690],
        [ 0.9178,  1.5810,  1.3010],
        [ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-1.1589,  0.3255, -0.6315],
        [-2.8400, -0.7849, -1.4096]], requires_grad=True)

# 要将ID为3的词符转换为一个3维向量，我们执行以下步骤：
print(embedding_layer(torch.tensor([3])))
# tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)

# 为了嵌入上面所有的四个`input_ids`值，我们执行以下操作：
print(embedding_layer(input_ids))
# tensor([[ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-2.8400, -0.7849, -1.4096],
        [ 0.9178,  1.5810,  1.3010]], grad_fn=<EmbeddingBackward0>)
```

* `torch.nn.Embedding` 层本质上就是一个**大型的查找表 (lookup table)**。你可以把它想象成一个矩阵，每一行都代表词汇表中一个词的嵌入向量。
* 

嵌入层本质上是一种查找操作：
![[embedding_layer.png]]

# 词位置编码 Encoding word positions

先前介绍的 embedding 层的生成方式中，相同的 token ID 总是被映射成相同的向量表示，不会在乎 token ID 在输入序列中的位置
![[embedding_word_token.png]]在原则上，具有确定性的、与位置无关的编码对于可重现性是有益的。然而，由于LLM的自注意力机制本身也不关注位置，因此将额外的位置信息注入到LLM中是有帮助的。

### 为什么需要位置嵌入？
像 `torch.nn.Embedding` 这样的层在查找词元（Token）的向量时，是独立进行的。它不知道一个词元出现在句子的哪个位置。对于 "人咬狗" 和 "狗咬人" 这两个句子，词元嵌入层会为 "人", "咬", "狗" 找出相同的向量，从而无法区分这两个句子的巨大语义差异。

Transformer 模型的核心机制（自注意力，Self-Attention）本身也不包含序列顺序的信息。为了解决这个问题，我们需要一种方法将词元的位置信息注入到模型中。**位置嵌入**就是解决方案。

其基本思想是：为句子中的每一个位置（例如，第0个位置，第1个位置，...）创建一个专属的嵌入向量。然后，将这个位置向量与该位置上的词元向量相加。这样，最终的输入向量就同时包含了**“这个词是什么”**和**“这个词在哪个位置”**两种信息。

为了实现这一点，有两种常用的位置编码方式：相对位置编码和绝对位置编码。

## 绝对位置编码
绝对位置编码与序列中的特定位置相关联。对于输入序列中的每个位置，都会添加一个唯一的位置编码到 token 中，来表示其确切位置。例如，第一个 token 将具有特定的位置编码，第二个 token 将具有另一个不同的位置编码，依此类推
![[绝对位置编码.png]]

## 相对位置编码
相对位置编码不是专注于 token 的绝对位置，而是侧重于 token 之间的相对位置或距离。这意味着模型学习的是 “彼此之间有多远” 而不是 “在哪个确切位置”的关系。这样做的好处是，即使模型在训练过程中没有看到这样的长度，它也可以更好地推广到不同长度的序列。

## code
```
vocab_size = 50257
output_dim = 256


token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# token ID 张量是 8x4 维的，这意味着数据批次由 8 个文本样本组成，每个样本有 4 个 token。
max_length = 4
dataloader = create_dataloader_v1( raw_text, batch_size=8, max_length=max_length, stride=max_len)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
# 这是一个 `8x4` 的张量。这代表一个数据批次，包含 **8 个文本样本**，每个样本由 **4 个词元（Token）** 组成。
# 每一行是一个独立的句子/文本片段。例如，第一行 `[ 40, 367, 2885, 1464]` 就是第一个文本样本。

print("Token IDs:\n", inputs)
Token IDs:
tensor([[ 40, 367, 2885, 1464],  
[ 1807, 3619, 402, 271],  
[10899, 2138, 257, 7026],  
[15632, 438, 2016, 257],  
[ 922, 5891, 1576, 438],  
[ 568, 340, 373, 645],  
[ 1049, 5975, 284, 502],  
[ 284, 3285, 326, 11]])

print("\nInputs shape:\n", inputs.shape)
# Inputs shape: torch.Size([8, 4])

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)
# torch.Size([8, 4, 256])
# 原始的 `8x4` 形状被扩展为 `8x4x256`。你可以将其理解为：我们有 8 个句子，每个句子有 4 个词，现在每个词都变成了一个 256 维的向量。

# 对于 GPT 模型的绝对嵌入方法，我们只需要创建另一个具有与 token_embedding_layer 相同维度的嵌入层：
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)
# torch.Size([4, 256])

input_embeddings = token_embeddings + pos_embeddings
# **广播机制 (Broadcasting)**: 这里 PyTorch 的广播机制发挥了作用。当你把一个 `8x4x256` 的张量和一个 `4x256` 的张量相加时，PyTorch 会自动将 `pos_embeddings` "扩展"或"复制" 8 次，使其形状变为 `8x4x256`，然后再进行逐元素的相加。
print(input_embeddings.shape)
# torch.Size([8, 4, 256])
```
![[位置编码实例.png]]