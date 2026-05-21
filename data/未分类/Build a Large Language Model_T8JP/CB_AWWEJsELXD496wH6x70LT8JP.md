---
doc_type: weread-highlights-reviews
bookId: "CB_AWWEJsELXD496wH6x70LT8JP"
title: "Build a Large Language Model"
author: "Sebastian Raschka"
category: ""
publisher: "Manning Publications Co."
publishTime: ""
isbn: ""
cover: "https://res.weread.qq.com/wrepub/CB_AWWEJsELXD496wH6x70LT8JP_parsecover"
wordCount: 0
newRating: 0
newRatingCount: 0
lastSync: "2026-05-21T11:50:37Z"
readingProgress: "0%"
readingTime: "0分钟"
finishedDate: ""
noteCount: 0
reviewCount: 1
bookmarkCount: 0
appLink: "weread://reading?bId=CB_AWWEJsELXD496wH6x70LT8JP"
webLink: "https://weread.qq.com/web/reader/587425d3643425f415757454a73454c58443439367748367837304c54384a50868"
---

# 元数据

![Build a Large Language Model](https://res.weread.qq.com/wrepub/CB_AWWEJsELXD496wH6x70LT8JP_parsecover)

| 项目 | 内容 |
|------|------|
| 书名 | [Build a Large Language Model](https://weread.qq.com/web/reader/587425d3643425f415757454a73454c58443439367748367837304c54384a50868) |
| 作者 | Sebastian Raschka |
| 出版社 | Manning Publications Co. |

---

# appendix A Introduction to PyTorch

> 📌 清单 A.4 具有两个隐藏层的多层感知器
> ⏱ 2025-05-28 05:48:54

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
⏱ 2025-05-28 05:48:54

---
