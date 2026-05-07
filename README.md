这份学习笔记包括两个部分：
- 《Build a LLM from scratch》的学习笔记
-  KDA的学习笔记和对比实验

**chapter 1-7**是跟随书籍的学习笔记，包括代码和我的总结笔记。

**compare_labs**包括KDA的数学推导和对GPT、KDA和Hybrid三种架构的训练特性对比。

在对比试验中，发现了一些有趣的现象：KDA的验证loss对于训练loss，并且对序列长度有更好的扩展性。

<div align="center">
    <img src="./compare_labs/phs/compare.png" style="max-height:200px; width:auto;">
    <br><sub>KDA</sub>
</div>
<br>

KDA的 Delta Rule的隐式正则化是反常loss的重要原因，巧妙的状态更新/遗忘机制则让KDA对长度的敏感性更低。
