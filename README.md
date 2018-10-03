# PyTorch


### PyTorch构造卷积神经网络示意：

1. **Define a Convolution Neural Network**(设计网络结构：不包含优化部分)

```python
class Net(nn.Module)：
	def __init__(self):
        super(Net, self)__init__()    #此处的Net不知何意
        构造(卷积、池化)层、全连接层
        self.conv1 = nn.Conv2d(3, 6, 5) #[1]
    def forward(self, x):
        pass
        x = self.fc3(x)
        return x
```

``[1]``(3, 6, 5) = (n[l-1], n[l], filter_size)

2. **Define a Loss function and optimizer**（选择目标函数和优化方法）

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)  #net的参数被传入到optimizer中
```

3. **Train the network**（）

   ```python
   #循环中
   optimizer.zero_grad()             #清空缓存梯度
   outputs = net(inputs)             #预测
   loss = criterion(outputs, labels) #计算损失
   loss.backward()                   #根据损失计算梯度
   optimizer.step()                  #更新optimizer内的权重
   ```

4. **Predict**(预测)

```python
#通过optimizer更新的权重也自动更新了net内的权重，无需在替换
with torch.no_grad():
    outputs = net(images)
    #_, predicted = torch.max(outputs.data, 1)
```


### PyTorch Tutorial

1. [Tensors_Numpy_CUDA][1]
2. [Autograd_Tutorial][2]
3. 

[1]: https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py	"Tensor|Numpy|CUDA"

[2]: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html "Autograd"
