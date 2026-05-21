---
doc_type: weread-highlights-reviews
bookId: "CB_9Od9nP9nfDGY6wI6x75125pz"
title: "Build a Large Language Model (From Scratch)"
author: "Sebastian Raschka"
category: ""
publisher: "Manning Publications Co."
publishTime: ""
isbn: ""
cover: "https://res.weread.qq.com/wrepub/CB_9Od9nP9nfDGY6wI6x75125pz_parsecover"
wordCount: 0
newRating: 0
newRatingCount: 0
lastSync: "2026-05-21T11:50:27Z"
readingProgress: "91%"
readingTime: "0分钟"
finishedDate: "2025-06-12"
noteCount: 2
reviewCount: 2
bookmarkCount: 0
appLink: "weread://reading?bId=CB_9Od9nP9nfDGY6wI6x75125pz"
webLink: "https://weread.qq.com/web/reader/12342ea3643425f394f64396e50396e6644475936774936783735313235707a962"
---

# 元数据

![Build a Large Language Model (From Scratch)](https://res.weread.qq.com/wrepub/CB_9Od9nP9nfDGY6wI6x75125pz_parsecover)

| 项目 | 内容 |
|------|------|
| 书名 | [Build a Large Language Model (From Scratch)](https://weread.qq.com/web/reader/12342ea3643425f394f64396e50396e6644475936774936783735313235707a962) |
| 作者 | Sebastian Raschka |
| 出版社 | Manning Publications Co. |

---

# 2 Working with text data

> 📌 清单 2.4 处理未知单词的简单文本分词器
> ⏱ 2025-05-28 23:09:59

💭 你问到了一个非常关键的点！r'\1' 是正则表达式替换操作中的一个特殊语法，它与模式（pattern）中的**捕获组（capturing group）**紧密相关。

让我们再次回顾这行代码：

text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)    #2


模式（Pattern）: r'\s+([,.:;?!"()\'])'

\s+: 匹配一个或多个空白字符（空格、制表符等）。

([,.:;?!"()\']): 这是关键！括号 () 在正则表达式中创建了一个捕获组。

这个捕获组 ([,.:;?!"()\']) 会匹配并“捕获”（记住）列表 ',.:;?!"()\'' 中的任何一个标点符号。

因为这是模式中从左到右的第一个括号对 ()，所以它被称为捕获组 1。

替换字符串（Replacement String）: r'\1'

在 re.sub 的替换字符串中，\1 (或者在某些正则表达式引擎中是 $1) 是一个反向引用（backreference）。

它指的是在模式匹配过程中，第一个捕获组所捕获到的实际文本。

同理，\2 会引用第二个捕获组，\3 引用第三个，以此类推。\0 通常引用整个匹配到的字符串。

作用解释：

这行代码的目的是：

查找：文本中所有“一个或多个空格后面紧跟着一个特定标点符号”的组合。

例如，找到 " ," (空格后跟逗号) 或者 " !" (空格后跟感叹号) 或者 " ." (多个空格后跟句号)。

替换：将整个找到的组合（例如 " ,"）替换为仅仅那个标点符号本身（例如 ","）。

它是如何做到的？

当 re.sub 找到一个匹配 r'\s+([,.:;?!"()\'])' 的地方时：

\s+ 部分会匹配空格。

([,.:;?!"()\']) 部分会匹配紧随其后的标点符号，并且因为它是捕获组 1，这个标点符号会被“记住”。

然后，re.sub 会用 r'\1' 来替换整个匹配到的字符串。

\1 告诉 re.sub：“把我刚才在捕获组 1 中捕获到的那个标点符号放到这里。”

举例：

假设 text 是 "Hello , world !"

第一次匹配：

re.sub 找到 " ," (一个空格后跟一个逗号)。

\s+ 匹配了空格 。

([,.:;?!"()\']) 匹配并捕获了逗号 ,。所以，捕获组 1 的内容是 ,。

整个匹配是 " ,".

替换操作：用 \1 (也就是 ,) 来替换 " ,".

text 变成 "Hello, world !"

第二次匹配：

re.sub 在新的 text 中继续查找，找到 " !" (一个空格后跟一个感叹号)。

\s+ 匹配了空格 。

([,.:;?!"()\']) 匹配并捕获了感叹号 !。所以，捕获组 1 的内容是 !。

整个匹配是 " !".

替换操作：用 \1 (也就是 !) 来替换 " !".

text 变成 "Hello, world!"

最终结果是 "Hello, world!"，标点符号前的多余空格被清除了。

总结一下 r'\1' 的作用：
它告诉 re.sub 函数，在执行替换时，使用模式中第一个捕获组 (...) 所匹配到的具体内容作为替换结果的一部分（在这个例子中是全部）。这使得我们可以保留匹配到的模式中的一部分（这里是标点符号），而丢弃另一部分（这里是标点符号前的空格）。

r 前缀表示这是一个“原始字符串 (raw string)”，这在处理包含反斜杠 \ 的字符串（如正则表达式和文件路径）时很有用，可以防止反斜杠被解释为转义字符。对于 \1 这种简单的反向引用，r 不是绝对必需的，但对于包含其他可能被误解为转义序列的模式，使用 r 是个好习惯。
⏱ 2025-05-28 23:09:59

---

# Build a Large Language Model (From Scratch)_1

> 📌 keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)      ﻿ #4         values = values.view(b, num_tokens, self.num_heads, self.head_dim)           queries = queries.view(                                                          b, num_tokens, self.num_heads, self.head_dim                             )
> ⏱ 2025-06-03 18:16:40

---

# 4 Implementing a GPT model from scratch to generate text

> 📌 截至撰写本文时，GPT-3 的权重尚未公开。GPT-2 也是学习如何实现 LLM 的更好选择，因为它可以在单台笔记本电脑上运行，而 GPT-3 需要 GPU 集群进行训练和推理。根据 Lambda Labs（https://lambdalabs.com/），在单个 V100 数据中心 GPU 上训练 GPT-3 需要 355 年，在消费级 RTX 8000 GPU 上需要 665 年。
> ⏱ 2025-06-04 12:06:48

---

# appendix A Introduction to PyTorch

> 📌 清单 A.4 具有两个隐藏层的多层感知器
> ⏱ 2025-05-28 12:44:58

💭 这段代码定义了一个名为 NeuralNetwork 的神经网络模型类，它继承自 torch.nn.Module。下面是这段代码的详细解释：
这是一个 Python 类定义，用于创建一个神经网络模型。
class NeuralNetwork(torch.nn.Module):
class NeuralNetwork: 定义了一个名为 NeuralNetwork 的类。
(torch.nn.Module): 表示这个类继承自 torch.nn.Module。在 PyTorch 中，所有神经网络模块（层、模型等）都应该继承自 nn.Module。这样做的好处是可以使用 nn.Module 提供的许多便利功能，例如参数管理 (parameters())、自动求导 (backward()) 和模型保存/加载等。
def __init__(self, num_inputs, num_outputs):
这是一个特殊的方法，称为构造函数。当创建一个 NeuralNetwork 类的实例时，会自动调用这个方法。
self: 表示类的实例本身。
num_inputs: 这个参数表示网络的输入特征的数量。
num_outputs: 这个参数表示网络的输出特征（或类别）的数量。
super().__init__()
这行代码调用父类 torch.nn.Module 的构造函数。这是继承 nn.Module 时必须要做的事情，以确保父类得到正确的初始化。
self.layers = torch.nn.Sequential(...)
self.layers: 在类的实例中创建一个名为 layers 的属性。
torch.nn.Sequential(...): nn.Sequential 是一个容器模块，它可以按顺序地将多个层组合在一起。当输入数据通过 Sequential 容器时，它会依次通过容器中的每一个层。这使得构建和管理多个层的网络变得非常方便。
在 nn.Sequential 容器中定义了网络的各个层：
torch.nn.Linear(num_inputs, 30): 这是一个全连接（或称为线性）层。它执行的操作是线性的：output = input * weight + bias。
num_inputs: 指定输入特征的数量，与构造函数的参数一致。
30: 指定这一层的输出特征数量，也是下一个隐藏层的输入数量。这是第一个隐藏层。
torch.nn.ReLU(): 这是一个 ReLU (Rectified Linear Unit) 激活函数。它是一个非线性函数，对输入应用 max(0, x) 操作。在神经网络中引入非线性是至关重要的，因为它允许网络学习更复杂的模式。它通常跟在全连接层后面。
torch.nn.Linear(30, 20): 这是第二个全连接层。
30: 指定输入特征的数量，与前一个 ReLU 层的输出数量一致。
20: 指定这一层的输出特征数量，也是输出层的输入数量。这是第二个隐藏层。
torch.nn.ReLU(): 第二个 ReLU 激活函数，跟在第二个全连接层后面。
torch.nn.Linear(20, num_outputs): 这是网络的输出层。
20: 指定输入特征的数量，与前一个 ReLU 层的输出数量一致。
num_outputs: 指定输出特征的数量，与构造函数的参数一致。这取决于你的任务，例如对于分类任务，这可能是类别的数量；对于回归任务，这可能是预测值的数量。
def forward(self, x):
这是一个特殊的方法，定义了网络的前向传播过程。当你在一个 nn.Module 实例上调用它时（例如 model(input_data)），实际上就是调用了这个 forward 方法。
self: 表示类的实例本身。
x: 表示网络的输入数据，通常是一个 PyTorch 张量。
logits = self.layers(x)
这行代码将输入张量 x 传递给 self.layers（也就是我们用 nn.Sequential 定义的层序列）。
nn.Sequential 会按顺序执行其中的所有层，并将最终的输出赋值给 logits 变量。logits 通常是模型在应用最终激活函数（如果需要的话，例如 softmax 或 sigmoid）之前的原始输出值。
return logits
这行代码返回前向传播的最终结果 logits。
总结:
这个 NeuralNetwork 类定义了一个简单的全连接神经网络，包含两个隐藏层（每个隐藏层后都有一个 ReLU 激活函数）和一个输出层。网络的输入和输出维度由构造函数 __init__ 的参数 num_inputs 和 num_outputs 决定。forward 方法描述了数据如何通过网络从输入层流向输出层。
⏱ 2025-05-28 12:44:58

---
