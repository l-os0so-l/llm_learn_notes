# RL: 从传统强化学习到LLM

## 数据集

我从Thyme-SFT数据集的三种类型数据single（单轮图片问答）、comp（无图片计算问答）和2round（2轮图片问答）中分别下载了334、333和333条数据，提取SFT格式数据中的**image、question和response中的answer部分**组成RL问答格式数据集。

原始数据中，图片以base64编码格式直接给出，同时在question中提供了图片的存储路径。在comp中，prompt为模型介绍可以使用代码对图片进行分析或者进行数学计算，还要求模型直接打印图片的保存路径（如果处理了图片）或者计算的数学结果。但略显奇怪的是，在真正需要处理图片的single和2round中没有上述prompt，尽管在sft格式数据中3类数据都会教导模型使用代码分析问题。

结合我的代码实现，我对原始数据做了以下修改：把base64编码格式的图片解码为jpg格式图片单独存储，并将question中的路径替换为正确的本地图片存储路径；将comp中对图片处理部分的prompt删除，并为single和2round数据加上正确的prompt。

```
### **Code Execution Guidelines:**

You may write Python code to process the image. The code will be executed in a secure sandbox, and its output will be provided back to you for further analysis.
* All Python code snippets **must** be wrapped as follows:
    <code>
    ```python
    # Load the image using the User Image Path provided above.
    ```
    </code>
* At the end of the code, save the processed image to `sandbox_temp_dir` and print its full file path. The system will automatically load it for the next round.
```

## PPO

### 概述

在LLM中，关键的组件在原本的**actor**、**critic**的基础上增加了**reference**和**RM**。

不同于传统强化学习中，通常由一个environment来对actor的action给出reward，这里我们需要对LLM的回答进行打分——这就是**RM**的角色，它可以是一个提取答案并评分的规则判决函数，也可以是一个更加强大的模型（比如这里，我们使用强大得多的qwen3.5 4B作为RM）。reference是一个经过预训练和监督微调的模型，它被用于确保actor不会偏离原始模型太远，比如学会“骗分”。

接下来，我会逐步介绍我此次实验中的LLM强化学习PPO实现。


### 初始化四大组件

本次实验中，以Qwen2.5-VL-3B-Instruct为基模，使用Lora分别配置Policy（Actor）和Critic。使用Lora进行微调，可以大幅度减少需要调整的参数，加快微调速度。

```
Lora中的r和lora_alpha这两个参数，作用是什么呢？

考虑原始模型的一个参数矩阵d*d，d可能是上千甚至更大。Lora会额外训练两个d*r维的矩阵A、B，结合lora_alpha，最终该层的输出为：

Y = Wx + (lora_alpha/r)*(A@B)x

相比起直接微调d*d的W，训练两个d*r的参数矩阵，计算量要小得多。同时通过 lora_alpha/r 控制Lora输出对整体输出的影响。

W′= W + α/r ⋅ B⋅A

A一般随机高斯初始化，B全0初始化。在训练初期输出大致正比于r，改变r，输出的幅度自然改变，更新幅度也自然改变。使用lora_alpha/r，避免r增大而不得不每次都必须调节学习率。

同时，使用lora_alpha/r来控制更新幅度，B和A负责学习方向。这样有利于训练，并且我们可以通过控制这个比值来控制lora输出对整体输出的影响。
```

**Policy**的初始化十分简单，直接使用peft库的get_peft_model函数将lora挂载到基模上，再把基模所有需要梯度的参数冻结即可。而对于**Critic**，我们在把Lora挂载到基模并冻结基模参数后，还需要为它加上一个value_head。在计算时，需要取基模最后一层transformer block的输出（hidden state）输入给value_head，计算出每一个token的value。

```
from_pretrained函数被用来加载预训练好的模型。

trust_remote_code=True：允许执行模型仓库中的自定义 Python 代码，我们当然需要执行里面定义的模型架构代码。

attn_implementation="sdpa"：pytorch官方提供的、根据硬件自动选择高效的attention实现方式。
```

**Reference**就是Qwen2.5-VL-3B-Instruct，它不会进行微调。**RM**使用Qwen3.5-4B，它只负责比较policy的给出的answer和gt给出评分，同样不进行训练。

Policy和Critic全程加载在GPU上，Reference和RM只在使用时搬运到GPU上，平时位于CPU。

### 数据准备

数据被加载为以下格式并被随机打乱顺序：
```
{
    "question": item["question"],
    "answer": item["answer"],
    "image": img,
    "type": data_type,
}
```
此时，img还只是图片存储路径（如果数据确实包含img）。

在训练时，每个step会从加载的1000条数据中随机采样一个batch，格式如下：
```
{
    "questions": [d["question"] for d in batch],
    "answers": [d["answer"] for d in batch],
    "images": [d["image"] for d in batch],
    "types": [d["type"] for d in batch],
}
```

大模型的输入需要特定的格式，具体到qwen2.5-vl-3B，我们为每一条数据构建初始message：
```
{
    "role": "user", 
    "content":
        [
            {"type": "image"}, # 如果确实存在img的话
            {"type": "text", "text": question},
        ] 
}
```

### 收集trajectory

#### _build_inputs_from_messages

在第一轮或者某轮已经有了message的情况下，我们还需要对message进一步处理得到inputs，这正是_build_inputs_from_messages函数做的事情。

输入为： <br>
messages_list: List[List[dict]]和images_list: Optional[List]=None

对于messages_list的每个样本的message，我们会需要检查是否存在content；如果存在，我们会检查是否存在image；如果存在我们会把iamge（图像本身）提取出来，然后组合成新的message（message本身不包括图像，如果存在图像传入模型，message只保留一个{"type":"image"}标识）。大致结构为：
```
[
    { # 一个样本
        "role": "user",
        "content": [
            {"type": "image"},           # ← 标记还在！apply_chat_template 能看到它
            {"type": "text", "text": "描述一下"},
            {"type": "image"},           # ← 第二个标记
        ]
    }
]
```
之后后送入**processor.apply_chat_template**生成带 prompt 的文本字符串。

images_list的{"type": "image"}默认已经在message里面了，这样后面处理图片时才对的上，这也是后面加入图像时需要把它放在最前面的原因，这些都是隐式约定的。

images_list是外部传入的图像（可能是图像本身也可能是路径），通常为题目的原始图像。我们之前保存提取的图像时，是每个样本单独保存。现在我们需要把images_list里的图像加到对应样本图像list的最前方。

接下来，我们使用**self.processor**生成inputs。pil_images是整个batch展平后顺序存储的图像，所有的图像会被处理为展平的图像token序列(n,)。在每个样本对应的input的image处，会生成<|vision_start|><|image_pad|><|vision_end|>这样的占位符，<|image_pad|>的数量和该image实际的patch数量相同。

inputs有一个image_grid_thw属性，形状为[total_images, 3]，对应每张图片的 patch 数 = T * H * W。这里image_grid_thw的顺序和pil_images相同，因此我们可以设法得到每一个样本的图片在总的image token序列中的范围。最终得到：
```
inputs['image_ranges'] = 
    [ 
        {
            'pixel_values': (patch_cumsum[i], patch_cumsum[i + 1]),
            'image_grid_thw': (image_cumsum[i], image_cumsum[i + 1]),
        }
    ]
```
pixel_values是每个样本的图片在总的image token序列中的范围，image_grid_thw是每个样本对应的pil_images中的图像范围。由于<|image_pad|>的数量和该image实际的patch数量相同，所以我们不需要知道每个pixel_values中的每一张图片具体的范围，顺序填充就行了。

在最后，把inputs中所有的tensor属性转移到GPU上。

#### 生成输出outputs

在得到inputs后，把'image_ranges'取出来，剩余的输入给模型的generate函数，返回outputs_id。由于每个样本的输入和输出长度都不尽相同，为了利用GPU的并行化计算能力，无论是processor处理得到的inputs还是generate返回的outputs_id，其中的每个样本都被padding到了同样的长度。为了避免在计算loss/logits/value的时候把padding错误的考虑其中，我们需要同步更新**attention_mask**来指明哪些部分是padding。

可以通过**processor.batch_decode**对生成的outputs_id进行解码，得到生成的文本内容。

#### 计算old_logprobs、ref_logprobs和value

在PPO中，每次收集到的trajectory会被多次用于更新。在重要性采样需要使用收集trajectory时的概率，也即是这里计算的**old_logprobs**。

其实计算过程很简单，把把模型forward需要的数据传给模型再计算一次，然后提取response部分每个id对应的概率。

函数还实现了熵的计算，这个其实可以用来作为正则项限制模型再多次更新时不要偏离收集trajectory时太远，但是我们使用clip来做到这一点，所以熵实际上没有被使用。

同样的做法，我们计算**ref_logprobs**，只不过这次进行forward计算的模型从Policy变成了Reference。在计算时，reference被移到GPU，之后马上移回CPU。

value的计算更加简单，直接使用critic进行forward计算就行。

nonzero(as_tuple=True)返回非0元素，as_tuple返回高级索引(tensor([x1,x2,....]),...),否则返回tensor([[x1,y1,..],[x2,y2,..]...])

得到response的mask，为1的位置时真实的生成。

#### 返回
responses, old_logprobs, ref_logprobs, values, sequences（response_ids）, prompt_len(inputs_length), inputs, response_mask

### 收集episodes

episodes中的每一个episode，其实就是一个样本的一条完整trajectory。

在每一轮结束后，收集inputs、old_logprobs等后续更新会使用到的一系列属性为一个turn，添加到episodes[i]。对于中间轮，如果模型写了格式正确的代码，sandbox会执行并返回结果。根据格式和代码执行情况给与模型少量的中间步骤奖励或者惩罚，这个奖励/惩罚就是模型中间步骤的反馈**r**。

在该轮的末尾，需要更新messages_list,模型的回复会被添加到message中去。
```
{
    "role": "assistant",
    "content": [{"type": "text", "text": response[idx]}]
}
```
sandbox返回的结果，也会被添加到message。
```
{
    "role": "user",
    "content":
    [
        {"type": "text", "text": sandbox_result["text"]},
        {"type": "image", "image": img} # 如果存在
    ]
}
```
即使没有执行代码，sandbox也会返回
```
{
    "text": (
        "<sandbox_output>\n"
        "(No code provided. Please continue reasoning and provide your final answer.)\n"
        "</sandbox_output>"
    ),
    "images": [],
}
```

出于显存考虑，每一轮模型只能看到初始message和上一轮的message——模型的回复和sandbox的返回。

持续进行上述过程，每一轮保留必要的数据、计算中间r并且更新messages_list，直到达到**最大交互轮数3**或者模型输出了由
```
<answer></answer>
```
包裹的答案（en...放松到仅输出
```
 <answer> 或 \boxed{ 
```
也行）。

如果模型超时，直接给r=timeout_penalty。如果有答案，使用RM进行评分作（RM输出0-10，映射到-1~1）。

最终，按照公式：
$$
G_t = r_t + \gamma * G_{t+1}
$$
计算每一步的**cumulated reward**。

### update

我们分交互轮次给整个batch的数据同时更新，这意味着，我们需要取出当前轮次还存在的所有样本的turn数据，然后把它们padding到相同的长度。不过说实话，无论是inputs、response还是logits和values，在generate函数里面就已经被padding到相同长度了，只要我们确保更新时使用的数据来自同一轮就没必要再进行padding。

#### 计算advantage和return

这里的计算要区分交互轮次和token两个层面。在传统的RL中，**cumulated reward**如此计算：
$$
G_t = r_t + \gamma*G_{t+1}
$$
在LLM的PPO实现中，t代表的第t轮。但是别忘了，每一轮还生成了若干token，并且每个token都是在此前的token基础上生成的。这意味着就像传统RL那样，我们也应该采样累计的方式来计算每个token的reward，而不是简单的把一个轮次的每个token等同。

我们把 $G_t$ 作为第t轮的最后一个token的reward，即：
$$
r_{last,t} = G_t
$$
对于第t轮次的第i个token，它的cumulated reward应该是：
$$
G_{i,t} = r_{i,t} + \gamma*G_{i+1,t}
$$
那么问题来了，每个token的即时反馈r来自于哪？

$$
r_{i,t} = -\beta_{KL}*[log\pi_θ(o_{i,t}|x_i,o_{i,<t}) - log\pi_{ref}(o_{i,t}|x_i,o_{i,<t})]
$$
当然，最后一个token的r会加上本轮的G。

怎么理解这个KL惩罚项的作用呢？
我们知道，KL惩罚的作用是使得Policy不会学习得偏离Reference太远。当r为正时，这意味着：
$$
log\pi_θ(o_{i,t}|x_i,o_{i,<t}) - log\pi_{ref}(o_{i,t}|x_i,o_{i,<t})<0 \\
\pi_θ(o_{i,t}|x_i,o_{i,<t}) < \pi_{ref}(o_{i,t}|x_i,o_{i,<t})
$$
但是正的r鼓励模型更新后增大 $\pi_θ(o_{i,t}|x_i,o_{i,<t})$ 的概率，使得二者差距更小。反之，减小 $\pi_θ(o_{i,t}|x_i,o_{i,<t})$ ,同样使得二者差距减小。

考虑**TD err**：
$$
\delta_{i,t} = r_{i,t}+ \gamma*V_{i+1,t} -  V_{i,t}
$$

**GAE**（Generalized Advantage Estimation）的计算公式如下：
$$
GAE_{i,t} = \delta_{i,t} + \lambda*\gamma*GAE_{i+1,t}\\
GAE_{last,t} = 0
$$

**Critic**的目标值是：
$$
Return_{i,t} = GAE_{i,t}+V_{i,t}
$$
实际上，如果令 $\lambda=1$ ，那么Return退化为简单的G，GAE退化为蒙特卡洛估计(A=G-V)；如果令$\lambda=0$，GAE退化为单步TD。

最后对A进行了归一化，消除不同样本间 advantage 的数值尺度差异，防止某些样本的梯度主导更新。

#### POlicy更新

Policy的loss分为两个部分：Policy_loss 和 entropy_loss。

对于之前收集到的response，我们使用Policy进行forward计算出new_logprobs和entropy。

Policy_loss的计算公式如下：
$$
loss_t = -\frac{1}{n}\sum_{i=1}^{n}min \{ration_{i,t}*A_{i,t}, clip(ration, 1-\epsilon,1+\epsilon)*A_{i,t}\}\\
ration_{i,t} = \exp ^{log\pi_{θnew}(o_{i,t}|x_i,o_{i,<t}) - log\pi_{θold}(o_{i,t}|x_i,o_{i,<t})} \\ =  \frac{\pi_{θnew}(o_{i,t}|x_i,o_{i,<t})}{\pi_{θold}(o_{i,t}|x_i,o_{i,<t})}
$$
其中，$o_{i,<t}$ 指的是$x_i$之前的模型能够看到的所有输出结果，也就是说在生成了这些内容的基础上下一个生成的token是$o_{i,t}$ 的概率，这里公式的表述可能有些不清楚。

entropy_loss计算公式如下：
$$
loss_t = -k*\frac{1}{n}\sum_{i=1}^{n}E_{i,t}
$$

最终的loss为二者之和。

怎么理解这两个loss呢？

我们知道通常DP中优化器是要让loss越低越好，这意味着我们要最大化那两项本身。如果$A_{i,t}>0$，我们自然希望增大 $\pi_{θnew}(o_{i,t}|x_i,o_{i,<t})$；如果$A_{i,t}<0$，我们自然希望减小 $\pi_{θnew}(o_{i,t}|x_i,o_{i,<t})$。这二者都与最小化loss的目标相吻合。clip则确保ratio被强制限制在 [1−ε,1+ε] 区间内。取得最小值，每次更新的幅度不会太大，导致ppo的off-policy规则不适用。

举个例子，A<0时，ration越小，ration * A 越大，loss越小。但是ration*A不会比 $(1-\epsilon)*A$ 更大，即ration不会比 $(1-\epsilon)$ 更小。

对于entropy_loss，当每个选项概率相等时，entropy最大。我们需要最大化这个loss，也就是鼓励各个选项的概率的分布更加均匀，这样可以增大模型的选择多样性。当然了，系数k确保模型不会因为太多样而无法收敛。

#### Critic更新

critic的loss计算十分简单：
$$
loss_t = \frac{1}{2n}*\sum_i^{n}(V_{i,t} - R_{i,t})^2
$$
就是简单的MSE。

#### PPO Update

对于一个trajectory的数据，我们会每一轮更新一次，然后更新PPO_epoch次数。之后重新收集trajectory，再更新ppo_epoch次数。上述过程不断重复....
```
for step in range(config.max_steps):       # ← iteration
    episodes = rollout(...)                # ← 收集 trajectories
    compute_rewards()
    
    for epoch in range(config.ppo_epochs): # ← ppo_epochs
        for round_idx in range(max_len):   # ← 按 round 分 minibatch
            trainer.ppo_step(...)          # ← 一次 optimizer step
```

### sandbox

sandbox的工作流程是：从模型的输出提取代码，检查没问题后进行执行，然后返回结果给模型。

#### 提取代码并检查

提取代码很简单，因为我们在prompt里面要求模型用```<code></code>```包裹代码，我们这里只执行了额外的可能出现的“\```python```”的过滤。

我们使用python的AST(抽象语法树)，检查包括：
- 是否存在缩进错误、括号不匹配等问题
- 检查导入的模块，必须在白名单的模块才允许使用
- 检查函数调用，黑名单里的函数调用一律禁止，规定路径外的读写一律禁止
- 禁止__import__，防止动态导入模块

最终，模型只允许在临时工作文件夹下读写，只允许在原始图片存储文件夹下进行读操作，其余路径不可读写。

#### 执行代码

我们规定模型在操作图片后，必修把图片保存到指定的文件夹下面并打印出文件保存地址。如果没有按照规定打印地址或者打印地址错误导致无法提取图片，模型会被判断为代码执行错误给与惩罚。

为了防止每次模型都保存图片为类似"figure1.png"这种名字导致覆盖问题，我为每个样本单独创建一个临时文件夹，并在代码执行前就注入这个临时变量。

#### 返回结果

我们首先尝试寻找打印的图片保存路径，如果找到了，就把它单独提取出来。借助这个路径我们会读取真正的图片，并且把路径和图片都加入到sandbox的返回中。

除了路径，模型可能在代码中打印其他内容，或者代码报错返回报错信息。为了避免序列太长，导致显存爆炸，我们会对文本内容进行截断。我们首先设定一个**EXEC_TRUNCATE（ET）**，对输出或者报错进行截断：输出保留头尾各1/2 ET长度的内容，报错则开始部分保留1/4 ET、末尾保留3/4 ET长度的内容。此后，我们还会对内容再次进行首尾截断，确保out/err各自总的序列长度不会超过400。


## GRPO

GRPO是在PPO的基础上进行了一些改动，最大的区别是GRPO没有critic，此外主要涉及episodes的收集、更新和KL散度计算。

GRPO对一个样本收集K次trajectory，这K条trajectory或者说k个episode为一组。

GRPO在计算advantage的时候，首先把中间轮和末尾轮的r加起来，然后再组内进行归一化，结果就是本episode的A。不同于PPO的token-level的A，GRPO一整个episode共享这一个A。

GRPO的loss由两部分组成。一个是新Policy和Reference的ration直接作为一个loss项，另一个则是新Policy和旧Policy的cliped loss。


```
# 1. 每个 token 都计算 ratio
ratio = torch.exp(new_logprobs - old_logprobs)   # [batch, response_len]

# 2. advantage 是 [batch]  scalar，广播到所有 token
advantages_expanded = advantages.unsqueeze(-1)    # [batch, 1]

# 3. 每个 token 单独算 clip loss
surr1 = ratio * advantages_masked                 # [batch, response_len]
surr2 = torch.clamp(ratio, 1-clip, 1+clip) * advantages_masked
policy_loss = -torch.min(surr1, surr2).sum() / mask_sum

# 4. 每个 token 单独算 KL，然后平均
log_ratio = ref_logprobs - new_logprobs  # log(pi_ref / pi_theta)
ratio_kl = torch.exp(log_ratio)          # pi_ref / pi_theta
kl_penalty = (ratio_kl - log_ratio - 1) * response_mask
kl_loss = kl_penalty.sum() / mask_sum
```

此外，无论是评分、sandbox还是数据处理，GRPO都与PPO无异。