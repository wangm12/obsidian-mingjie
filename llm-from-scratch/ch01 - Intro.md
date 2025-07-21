# Encoder / Decoder simple structure 
![[encoder_decoder_structure.png]]
this figure shows the final stage of the translation process where the decoder has to generate only the final word ("Beispiel"), given the original input text ("This is an example") and a partially translated sentence ("Das ist ein"), to complete the translation.

## self-attention
A key component of transformers and LLMs is the self-attention mechanism (not shown), which allows the model to weigh the importance of different words or tokens in a sequence relative to each other.

# BERT / GPT structure
![[bert_gpt_structure.png]]

On the left, the encoder segment exemplifies BERT-like LLMs, which focus on **masked word** prediction and are primarily used for tasks like **text classification.** 
On the right, the decoder segment showcases GPT-like LLMs, designed for **generative tasks** and producing coherent text sequences.

# Zero-shot / Few-Shot structure
![[zeroshot_fewshot_strucutre.png]]

# GPT workflow
![[gpt_workflow.png]]

# Summary
- 大型语言模型（LLMs）已经彻底改变了自然语言处理领域，该领域之前主要依赖于明确的基于规则的系统和更简单的统计方法。LLMs的出现引入了新的深度学习驱动的方法，这些方法推动了对人类语言的理解、生成和翻译的进步。
- 现代LLMs主要通过两个步骤进行训练。
- 首先，它们在大量未标记的文本语料库上进行预训练，使用句子中下一个词的预测作为“标签”。
- 然后，它们在较小的、有标签的目标数据集上进行微调，以执行指令或进行分类任务。
- LLMs基于 transformer 架构。transformer 架构的关键思想是注意力机制，它在生成输出时，使LLM能够选择性地访问整个输入序列中的每一个词。
- 原始的 transformer 架构包括用于解析文本的编码器和用于生成文本的解码器。
- 用于生成文本和执行指令的LLMs，如GPT-3和ChatGPT，只实现了解码器模块，简化了架构。预训练LLMs需要数十亿词的大型数据集。在本书中，我们将在小型数据集上实现和训练LLMs，用于教育目的，但也将展示如何加载公开可用的模型权重。
- 尽管GPT类模型的一般预训练任务是预测句子中的下一个词，这些LLMs表现出了“突现”属性，如分类、翻译或总结文本的能力。
- 一旦LLM被预训练，得到的基础模型可以更有效地针对各种下游任务进行微调。
- 在定制数据集上微调的LLMs可以在特定任务上胜过通用的LLMs。