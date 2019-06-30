import torch
from tqdm import tqdm
from torchtext import data
from torchtext.vocab import Vectors

fix_length = 40
batch_size = 25         
label_batch_size = 8

my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

"""
torchText创建Field对象

中文介绍：
        https://state-of-art.top/2018/11/28/torchtext%E8%AF%BB%E5%8F%96%E6%96%87%E6%9C%AC%E6%95%B0%E6%8D%AE%E9%9B%86/
说明文档：
        https://torchtext.readthedocs.io/en/latest/data.html#fields
"""

# 按照空格分词
tokenize = lambda x: x.split()  
# 定义字段处理操作
TEXT = data.Field(sequential=True, tokenize=tokenize, 
                  use_vocab=True, batch_first=True,
                  lower=True, fix_length = fix_length,
                  include_lengths=True)
LABEL = data.Field(sequential=False, use_vocab=True,
                   batch_first=True)
                
class BatchWrapper(object):
        """
        包装batch，方便调用
        输入Iterator
        """
        def __init__(self, dl, x_var, y_vars):
                self.dl, self.x_var, self.y_vars = dl, x_var, y_vars

        def __iter__(self):
                for batch in self.dl:
                        x = getattr(batch, self.x_var)

                        if self.y_vars is not None:
                                temp = [getattr(batch, feat).unsqueeze(1) for feat in self.y_vars]
                                label = torch.cat(temp, dim=1).long()
                        else:
                                label = torch.zeros((1))
                        text = x[0]
                        length = x[1]
                        yield (text, label, length)

        def __len__(self):
                return len(self.dl)
        
        
        
class BatchIterator(object):
        """
        用于读取数据并得到迭代器的类
        输入三个文件路径，x_var和y_var这里设置的默认值可以根据需要修改
        """
        def __init__(self, train_path, val_path, test_path, x_var="phrase", y_var=["coarse"]):
                self.train_path = train_path
                self.val_path = val_path
                self.test_path = test_path
                self.x_var = x_var
                self.y_vars = y_var
                
        def get_dataset(self):
                """读取数据，返回Dataset"""
                
                #csv_data = pd.read_csv(train_path)
                
                # id数据对训练在训练过程中没用，使用None指定其对应的field 
                fields = [("id", None), ("phrase", TEXT), ("coarse", LABEL)]  
                #examples = []
                
                train, valid = data.TabularDataset.splits(
                                path="",        # 这里我把数据文件和.py文件放在同一个目录下，所以为空
                                train=self.train_path, validation=self.val_path,        # 其实是文件名
                                format='csv',
                                fields=fields)
                
                test = data.TabularDataset(
                                path=self.test_path,    # 这里也应该是文件相对路径
                                format='csv',
                                fields=fields)
                
                return train, valid, test
                
        
        
        def get_iter(self):
                """ 读csv文件并返回dataset & 包装后的迭代器 """
                
                train, valid, test = self.get_dataset()
                
                # 构建词向量Vector和词表Vocab（保存在Field的vocab属性）
                vectors = Vectors(name='myvector/glove/glove.6B.200d.txt')
                TEXT.build_vocab(train, vectors=vectors)
                LABEL.build_vocab(train)
                
                train_iter, val_iter, test_iter = data.BucketIterator.splits(
                                (train, valid, test), 
                                sort_key=lambda x: len(x.phrase),
                                batch_sizes=(batch_size, batch_size, label_batch_size), 
                                device=torch.device('cuda'))
                
                train_iter = BatchWrapper(train_iter, x_var=self.x_var, y_vars=self.y_vars)
                val_iter = BatchWrapper(val_iter, x_var=self.x_var, y_vars=self.y_vars)
                test_iter = BatchWrapper(test_iter, x_var=self.x_var, y_vars=None)
        
                return train_iter, val_iter, test_iter
                
  
                
                
if __name__ == "__main__":
        
        # 这里输入的三个csv文件是在斯坦福数据集的基础上处理过的，具体代码见read_file.py
        train_path = 'stanford-sentiment-treebank.train.csv'
        val_path = 'stanford-sentiment-treebank.dev.csv'
        test_path = 'stanford-sentiment-treebank.test.csv'
        
        # 获取dataset及iterator
        bi = BatchIterator(train_path, val_path, test_path)
        train, valid, test = bi.get_dataset()
        train_iter, valid_iter, test_iter = bi.get_iter()
        
        for test, label, length in tqdm(train_iter):
                #do something
                pass
