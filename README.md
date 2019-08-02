# IDCNN-CRF-Pytorch
迭代膨胀卷积命名实体抽取
<br>
论文 [Fast and Accurate Entity Recognition with Iterated Dilated Convolutions](https://www.aclweb.org/anthology/D17-1283) pytorch实现
<br>
Git翻了一圈竟然没有pytorch的实现，于是撸了一个。
<br>
戳[这里](https://github.com/vdumoulin/conv_arithmetic)有动态原理图
<br>
除了没有Pooling操作和加了dilation，和普通卷积基本相同，并不神秘
<br>
### 几点总结
- 没有传说中的快，虽然每层是并行的，但是由于是多个CNN stacking结构（为了捕获长程信息），层与层之间还是串行的
- 对于文本较长的，意味着block层数(N)需要越多，按log(N)增长，容易出现梯度消失。这里我加了层归一化和relu缓解这个问题。
- 在长本文上（seq_length=256）试了一把，效果没有BILSTM好，谨慎使用
