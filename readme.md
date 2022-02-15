### 中文信息抽取论文复现与改良
参考：
https://github.com/zhengyanzhao1997/NLP-model

#### 运行
1. ner
   ```shell
   python train_ner.py
   ```
2. 三元组抽取
    ```shell
   python main_re_mutihead.py 
   ```
   

#### NER模型结果

| head                   | P      | R      | F1     |
| ---------------------- | ------ | ------ | ------ |
| GlobalPointer          | 0.8223 | 0.7868 | 0.8042 |
| Biaffine               | 0.8025 | 0.8012 | 0.8019 |
| Mutihead               | 0.8280 | 0.7743 | 0.8003 |
| EfficientGlobalPointer | 0.7980 | 0.7858 | 0.7918 |
| TxMutihead             | 0.8336 | 0.7680 | 0.7994 |

#### spo三元组抽取

|          | P      | R      | f1     |
| -------- | ------ | ------ | ------ |
| biaffine | 0.7228 | 0.7438 | 0.7331 |
|          |        |        |        |
|          |        |        |        |

