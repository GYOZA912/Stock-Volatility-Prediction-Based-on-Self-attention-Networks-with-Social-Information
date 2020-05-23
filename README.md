# Stock-Volatility-Prediction-Based-on-Self-attention-Networks-with-Social-Information

文件介绍：
utils.py 封装了一个记录日志的类，它既能将结果输出到控制台，也能写入txt中
hyperparams.py 封装的超参 （里面的v1，是版本1 的意思，如果修改了代码，这边改成v2就行了）

股票评论的文本分类：
（1）有无情绪分类
（2）情绪正负分类
共调用一个modules.py模型
sentiment_class.py 完成（1）的训练，测试，和预测 与（2）的训练 和 测试
（2）的预测:使用 generate_posneg.py

运行 generate_feature.py 生成股价预测需要的特征数据

运行 price_class.py (调用了price_modules.py) 选择 train = True or train = False 进行训练就可以

