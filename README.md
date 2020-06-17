# NLPCC2020-MAMS
NLPCC 2020 MAMS 多属性多情感分析任务 第一名解决方案

队伍名称：BaiDing  
队员：Fengqing Zhou, Jinhui Zhang, Tao Peng  
任务描述：http://tcci.ccf.org.cn/conference/2020/cfpt.php  
思路：使用多种预训练语言模型，采用加权投票的方式集成结果  

**输入**：按照常规思路将aspect term或者aspect category视为额外的句子，构造句对进行分类  
<p align="center" >
<strong> [CLS] context [SEP] aspect term/category [SEP]</strong> 
</p>

**模型**：主要选取5种预训练语言模型进行训练，包括BERT，ALBert，RoBERTa，XLNet和ERNIE。在五种与训练语言模型上对数据集进行微调，根据验证集的结果调节超参数。
BERT，ALBert，RoBERTa，XLNet模型构建是基于HuggingFace开源的transformers库构建。ERNIE模型是基于百度开源的paddlepaddle版本代码库构建。  

**加权投票**：由于ERNIE模型整体在验证集上的结果较好，在保存模型是save最好的两个中间状态，在加权投票时设置较大的权重。额外地，在测试阶段，我们合并训练集和测试集的数据，构造一个更大的训练集来训练ERNIE模型，也得到一组结果。最终的权重如下图所示。  

![](https://github.com/BaiDing213/NLPCC2020-MAMS/blob/master/weights.png)
