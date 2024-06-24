# 1-1 预测&回归模型基本概念和框架

##  1.1 线性实例

### 1.1.1 Step1 构建函数模型（Model or Net)

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426100456499.png" alt="image-20240426100456499" style="zoom:25%;" />

### 1.1.2 Step2  定义损失函数 （define Loss from Traning Data）

#### 均值误差MAE(L1 loss)

​                                                                                       $ l(y,\hat{y}) = |y-\hat{y}|$

#### 均方误差MSE (L2 loss)

​                                                                                      $l(y,\hat{y})=(y-\hat{y})^2$

#### 交叉熵损失Cross-Entropy

​                                                                                 $l(y,\hat{y})=-\sum^{}_{i} y_i\log{\hat{y}_i}$

#### Huber's Robust loss

​                                                                         $l(y,\hat{y})=\begin{cases} {|y-\hat{y}|-\frac{1}{2}} & {|y-\hat{y}|>1} \\ {(y-\hat{y})^2} & {otherwise} \\ \end{cases}$

<img src="C:\Users\10056\AppData\Local\Temp\WeChat Files\865510478668934090.jpg" alt="865510478668934090" style="zoom: 10%;" />

### 1.1.3 Step3  多轮训练计算参数 （Optimization）

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426104434134.png" alt="image-20240426104434134" style="zoom:25%;" />

先设置一个初始参数利用Gradient Descent计算参数

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426104608488.png" alt="image-20240426104608488" style="zoom:33%;" />

#### Two Hyperparameters

Batch size

Learn rate(步长 or 学习率)

## 1.2 非线性实例

怎么解决一个非线性的Regression问题？？

### 1.2.1 Step1 构建函数模型

#### ① 目标函数分解（常数＋若干非线性激活函数）



<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426105924104.png" alt="image-20240426105924104" style="zoom:33%;" />

利用函数叠加，red curve = a constant + sum of a set of unlineral function

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426110056682.png" alt="image-20240426110056682" style="zoom: 25%;" />

可以叠加为各种函数

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426110139192.png" alt="image-20240426110139192" style="zoom: 25%;" />

#### ② 非线性激活函数构造

此时分**Hard sigmoid**和**Soft sigmoid**

#### **③ Hard sigmoid**

$\left.hard\_sigmoid(x)=\left\{\begin{matrix}0&x<-2.5\\0.2x+0.5&-2.5<=x<=2.5\\1&x>2.5\end{matrix}\right.\right.$



<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426111405482.png" alt="image-20240426111405482" style="zoom:33%;" />







#### ④ Soft Sigmoid（_sigmoid_)

$y = c\frac{1}{1+e^{-(b+wx_1)}}=c*sigmoid(b+wx_1)$

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426111534890.png" alt="image-20240426111534890" style="zoom:25%;" />

#### ⑤ 对比解决Model bias

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426111822035.png" alt="image-20240426111822035" style="zoom:25%;" />

如何从最开始的线性层到后面的sigmoid？

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426112542342.png" alt="image-20240426112542342" style="zoom:25%;" />

对于sigmoid里面的参数，这是关于$w$和$b$的一个线性关系

$y = c\frac{1}{1+e^{-(b+wx_1)}}=c*sigmoid(b+wx_1)=c*sigmoid(r)$

$r = Wx+b$​

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426112831623.png" alt="image-20240426112831623" style="zoom:25%;" />

再利用$a_i=sigmoid(r_i)$​   ------->    $a_i=\sigma(r_i)$

通过运算后可以得到$y=b+\sum_ic_i\mathbf{a}$

即最终得到：$y=b+\sum_ic_isigmoid\left(b_i+\sum_jw_{ij}x_j\right)$

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426113709980.png" alt="image-20240426113709980" style="zoom:25%;" />

#### ⑥ 未知数统一化【终】



<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426113838923.png" alt="image-20240426113838923" style="zoom:25%;" />



将所有的未知数$W,\boldsymbol{b},\boldsymbol{c}^T,b$  统一看作一个未知向量$\boldsymbol{\theta}$

### 1.2.2 Step2  定义损失函数

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426114732253.png" alt="image-20240426114732253" style="zoom:25%;" />

这个地方损失函数的选取和之前的一样

### 1.2.3 Step3  多轮训练计算参数 （Optimization）

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426114949161.png" alt="image-20240426114949161" style="zoom:25%;" />

选一个初始的$\boldsymbol{\theta}^0$，计算梯度$g=\nabla L(\boldsymbol{\theta}^0)$

利用gradient descent计算合适的$\boldsymbol{\theta}$​



由于数据样本会比较大，因此通常随机选取**batch_size**个样本为一组，利用这一组进行参数的拟合，对每个**batch_size**都拟合后，称为一个**epoch**

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426162822359.png" alt="image-20240426162822359" style="zoom:25%;" />



### 1.2.4 Sigmoid ---> ReLU

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426165051582.png" alt="image-20240426165051582" style="zoom: 25%;" />

此时需要两个ReLU才能拟合一Sigmoid，

$y=b+\sum_ic_isigmoid\left(b_i+\sum_jw_{ij}x_j\right)$

$y=b+\sum_{2i}c_i\max\left(0,b_i+\sum_jw_{ij}x_j\right)$ 注意这里$\sum$下标为$2i$​



利用添加隐藏层进行模型叠加，把得到的$\boldsymbol{a}$当作$x$再进行一次拟合操作（相当于再来一层拟合）

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426165853855.png" alt="image-20240426165853855" style="zoom:25%;" />

把上面的那个激活函数叫做$Neuron$。整体叫**Neural Network**

深度叠加后就叫**Deep Learning**

为什么深度不能太深？**过拟合**

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426170630576.png" alt="image-20240426170630576" style="zoom: 25%;" />



## 1.3 模型补充



### Tips1 对于含条件判断的线性函数设计

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426175013331.png" alt="image-20240426175013331" style="zoom:25%;" />



### Tips2 Loss函数的正则化（Regularization）



​        Regularization：                                  **$L=\sum_{n}\left(\widehat{y}^{n}-\left(b+\sum w_{i}x_{i}\right)\right)^{2}+\lambda\sum(w_{i})^{2}$**



<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426174428715.png" alt="image-20240426174428715" style="zoom:25%;" />

通过调整$\lambda$的值，使得函数更加平滑

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240426174648193.png" alt="image-20240426174648193" style="zoom:25%;" />

但也不能太平滑，找到一个合适的$\lambda$可以大大提高精度









# 1-2  回归&分类模型基本概念和框架

相当于分类预测，比如现在有很多动物特征，如何根据各类动物特征将其分类，比如说现在给一个短四肢，四足爬行，等等性质，判断它是一个猫

How to do Classification?

Classification as Regression?

## 2.1 二元分类

### 2.1.1 二分类Binary Classification

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613110338174.png" alt="image-20240613110338174" style="zoom:33%;" />

现在假定一个分类目标和方法，有两个参数$x_1,x_2$进行分类，假如是利用线性分类：$y = b + w_1x_1+w_2x_2$，把该回归模型接近-1的记为class 1，接近1的记为class 2。

**Question：ＥＲＲＯＲ**

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613110524394.png" alt="image-20240613110524394" style="zoom:33%;" />

对于值距离分类线性太远，则会引起错误

对于理想分类器　Ideal　Alternatives

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613111020048.png" alt="image-20240613111020048" style="zoom:40%;" />

其中

​                                    $L(f)=\sum_n\delta(f(x^n)\neq\hat{y}^n)$

表示若

​                                                  $f(x^n)\neq\hat{y}^n$

则                                        $\delta(f(x^n)\neq\hat{y}^n)=1$​

从而可以控制损失函数，注意此时由于 是离散的情况**无法使用梯度下降方法**

### 2.1.2 概率模型分类

#### 2.1.2.1 模型建立步骤

##### step 1 建立概率模型

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613112111270.png" alt="image-20240613112111270" style="zoom: 33%;" />

抽到蓝色球，其中属于Box 1的概率

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613112220593.png" alt="image-20240613112220593" style="zoom:33%;" />



对于x属性，其中它属于$C_1$的概率

*Generative Model*                           $P(x)={P(x|C_1)P(C_1)+P(x|C_2)P(C_2)}$

先算  $P(C_1)$   and   $P(C_2)$​ 这是先验概率。

##### step 2 计算先验概率

假如说现在预测宝可梦，蓝色是水系，绿色是草系。$P(C_i)$​ 可以根据总数和个体关系进行直接计算。

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613113401824.png" alt="image-20240613113401824" style="zoom: 50%;" />

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613113549968.png" alt="image-20240613113549968" style="zoom:33%;" />

##### step 3 计算条件概率（利用高斯分布和极大似然估计）



每一个水系宝可梦都有一个vector系数作为feature

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613113654109.png" alt="image-20240613113654109" style="zoom:33%;" />

所以现在如何计算$P(x|C_1)$？，即如何计算海龟在水系杰尼龟中出现的概率？

**采用高斯分布来计算**

可以看到上图中间线性区域整体比较集中

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613114107648.png" alt="image-20240613114107648" style="zoom:33%;" />

其中$u$为均值，$\sum $为协方差矩阵 covariance matrix

*这里类似扔飞镖的集中圆心的概率，即使圆心的面积为0，但是集中圆心的概率不为0*

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613114421977.png" alt="image-20240613114421977" style="zoom:33%;" />

类似于把中心聚集点看作一个高斯分布的对称点，然后利用这个高斯分布计算概率。

对于$u$和$\sum $的计算，**Maximum  Likelihood极大似然估计**

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613114730462.png" alt="image-20240613114730462" style="zoom:33%;" />

$\begin{aligned}&L(\mu,\Sigma)=f_{\mu,\Sigma}(x^{1})f_{\mu,\Sigma}(x^{2})f_{\mu,\Sigma}(x^{3})\ldots\ldots f_{\mu,\Sigma}(x^{79})\\&f_{\mu,\Sigma}(x)=\frac{1}{(2\pi)^{D/2}}\frac{1}{|\Sigma|^{1/2}}\exp\left\{-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)\right\}\\&\mu^{*},\Sigma^{*}=arg\max_{\mu,\Sigma}L(\mu,\Sigma)\\&\mu^{*}=\frac{1}{79}\sum_{n=1}^{79}x^{n}\quad\Sigma^{*}=\frac{1}{79}\sum_{n=1}^{79}(x^{n}-\mu^{*})(x^{n}-\mu^{*})^{T}\end{aligned}$​

注意，此时在计算$\mu^*,\Sigma^*=arg\max_{\mu,\Sigma}L(\mu,\Sigma)$​时采用的穷举法

计算可以得到：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613115252575.png" alt="image-20240613115252575" style="zoom:33%;" />

最开始的$P(C_{1}|x)=\frac{P(x|C_{1})P(C_{1})}{P(x|C_{1})P(C_{1})+P(x|C_{2})P(C_{2})}$

##### step 4 设定判断条件

可以假定如果$P(C_{1}|x)>0.5------>$**x belongs to class1 (water)**

对于所有参数：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613115627262.png" alt="image-20240613115627262" style="zoom:50%;" />

##### step 5 结果分析（二元参数）

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613120206078.png" alt="image-20240613120206078" style="zoom:40%;" />

##### step 6 模型调整



<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613165017673.png" alt="image-20240613165017673" style="zoom:33%;" />

对于class 1 和 class 2 使用同一个$\sum$,此时的损失函数

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613165143996.png" alt="image-20240613165143996" style="zoom: 50%;" />

如果 $u^1=u^2$则可以直接得到一个$\sum$的线性关系$\Sigma=\frac{79}{140}\Sigma^{1}+\frac{61}{140}\Sigma^{2}$​

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613165510560.png" alt="image-20240613165510560" style="zoom:50%;" />

此时利用相同的协方差矩阵，则变为一个线性的回归模型

#### 2.1.2.2 概率分布的选择

**Naive Bayes Classifier**

朴素贝叶斯分布：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613170429970.png" alt="image-20240613170429970" style="zoom:33%;" />

此时每一个参数都是独立的。

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613170650510.png" alt="image-20240613170650510" style="zoom:33%;" /><img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613171153754.png" alt="image-20240613171153754" style="zoom:33%;" />

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613171234144.png" alt="image-20240613171234144" style="zoom:33%;" />

此时假定$\sum^1=\sum^2$则可以进行消除

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613171442447.png" alt="image-20240613171442447" style="zoom:33%;" />

最后可以得到一个只与x有关的线性关系。

$z=w^Tx+b$

而$P(C_1|x)=\sigma(w\cdot x+b)=\sigma(z)=\frac{1}{1+exp(-z)}$

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613172104827.png" alt="image-20240613172104827" style="zoom:50%;" />

这就是为什么当$\sum^1=\sum^2$​的时候，会变成一个线性的关系

### 2.1.3 逻辑回归

#### step 1：构建函数

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613201300487.png" alt="image-20240613201300487" style="zoom:33%;" />

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613201405596.png" alt="image-20240613201405596" style="zoom:33%;" />

#### step 2: 评价函数好坏（建立损失函数）

现在有一部分训练数据

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613201659659.png" alt="image-20240613201659659" style="zoom:33%;" />

如何评价$f_{w,b}(x)=P_{w,b}(C_{1}|x)$的好坏？

即使得每一个$f_{w,b}(x)$的值都尽量大。因此可以设计：

​                                       $            L(w,b)=f_{w,b}(x^1)f_{w,b}(x^2)\left(1-f_{w,b}(x^3)\right)\cdotp\cdotp\cdotp f_{w,b}(x^N)$

注意此时都是预测属于$C_1$的概率，由于$x^3$是属于$C_2$的因此在计算时可以利用$1-f_{w,b}(x^3)$得到预期结果$w^*,b^*=arg\max_{w,b}L(w,b)$

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613202201933.png" alt="image-20240613202201933" style="zoom: 67%;" />

使得$L(w,b)$最大，也就是使得$-lnL(w,b)$最小，而此时$-lnL(w,b)$如下

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613202348255.png" alt="image-20240613202348255" style="zoom: 33%;" />

此时可以对目标值进行修改带入：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613202550497.png" alt="image-20240613202550497" style="zoom:33%;" />

这里应该为 1 1 0 

然后结果的$-lnL(w,b)$​可以变为

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613202646107.png" alt="image-20240613202646107" style="zoom:50%;" />

这一步相当于对所有的算式给定一个统一的计算方式。此时的$L(w,b)$​

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613202906341.png" alt="image-20240613202906341" style="zoom: 50%;" />

 此时可以得到一个交叉熵的计算公式

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613203047907.png" alt="image-20240613203047907" style="zoom:33%;" />

**逻辑回归和线性回归的比较：**

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613203155357.png" alt="image-20240613203155357" style="zoom: 50%;" />

#### step 3: 找最合适的函数参数

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613204632840.png" alt="image-20240613204632840" style="zoom:80%;" />

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613205046604.png" alt="image-20240613205043216" style="zoom:15%;" />

此时计算后得到：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613205131819.png" alt="image-20240613205131819" style="zoom: 50%;" />

可以看出：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613205256152.png" alt="image-20240613205256152" style="zoom:33%;" />

相当于一个是用交叉熵来计算损失函数，一个是利用均方误差来计算损失函数。

为什么Logistics回归不能用均方误差？

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613205748485.png" alt="image-20240613205748485" style="zoom:33%;" />

可以看到，当$\hat{y}^n=0$时，即目标值为0，但预测值$f_{w,b}(x^n)=1$此时对于$w_i$的偏导数为0。

同理 当预测值$f_{w,b}(x^n)=0$时，偏导数也是为0

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613210133373.png" alt="image-20240613210133373" style="zoom:33%;" />

**梯度消失**，当举例目标很近和很远的时候，其偏导数都为0。导致梯度不知道是不是到达了最优。

#### Discriminative判别模型 v.s. Generative生成模型

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613210717101.png" alt="image-20240613210717101" style="zoom: 50%;" />

此时都是一样的计算方式，

逻辑回归属于Discriminative。利用计算$w,b$

概率模型分类则是采用计算概率的形式计算$\mu_1,\mu_2,\sum$

但是最终都会回到$P(C,|x)=\sigma(w\cdot x+h)$的形式

即一个类线性模型。可以得到各自的图像如下：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613211222790.png" alt="image-20240613211222790" style="zoom:50%;" />



一般来说Discriminative 模型效果会好于Generative模型，但也有例外。

**当模型数据比较少的时候Generalize模型可能会比Discriminative模型要好**

Example:

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613211622172.png" alt="image-20240613211622172" style="zoom:50%;" />

如果是朴素贝叶斯：

$x_i$独立作用：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613211704773.png" alt="image-20240613211704773" style="zoom:33%;" />

（下面那一行都是$C_2$）

可以计算得到：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613211759499.png" alt="image-20240613211759499" style="zoom:33%;" />

此时$P(C_1|x)<0.5$

此时主要原因是样本数目不均衡。

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613212007974.png" alt="image-20240613212007974" style="zoom:33%;" />

**主要是Generative模型会进行脑补，在样本数据比较少的时候可能会展现作用。**



## 2.2 多分类问题

### 2.2.1 Softmax

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613212903080.png" alt="image-20240613212903080" style="zoom: 25%;" />

利用softmax进行多分类：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613213134535.png" alt="image-20240613213134535" style="zoom: 50%;" />

<img src="C:\Users\10056\Desktop\微信图片_20240613213452.png" alt="微信图片_20240613213452" style="zoom:15%;" />



<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613213543033.png" alt="image-20240613213543033" style="zoom: 50%;" />

**Regression and Classification**

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616181416749.png" alt="image-20240616181416749" style="zoom: 33%;" />

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616181602347.png" alt="image-20240616181602347" style="zoom:33%;" />

***逻辑回归缺陷***

**XOR问题 线性不可分 **

解决方式一：Feature Transformation

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613213911461.png" alt="image-20240613213911461" style="zoom:33%;" />

把坐标映射为到$(0,0)$和$(1,1)$的距离，此时出要是把原来的$(x_1,x_2)$通过左边变换到$(\hat{x}_1,\hat{x}_2)$再进行分类，相当于一个先**编码Encode再分类**的过程

从而进行分类，但是这是人为定义的，机器无法实现。

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613214412894.png" alt="image-20240613214412894" style="zoom:50%;" />

这里对于Feature Teansformation

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240613214702541.png" alt="image-20240613214702541" style="zoom: 50%;" />

通过设定Logistics的直线参数$w,b$从而控制这四个点的$x_1,x_2$的输出，我们现在要把红色点给区分出来，可以设计参数，使得蓝色点的$x_1,x_2$输出后尽量保持一致，而红色点的输出有一定差距，从而使得更新后的结果如下：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240614094429301.png" alt="image-20240614094429301" style="zoom: 50%;" />

因此叠加后，可以获得

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240614094609407.png" alt="image-20240614094609407" style="zoom:50%;" />

这就是Deep Learning。



# 2. 深度学习基本思路

整体框架：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240614105649888.png" alt="image-20240614105649888" style="zoom:33%;" />

**Overfitting **过拟合就增大训练数据样本

## 2.1 局部最优和鞍点（凸优化）

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616115619053.png" alt="image-20240616115619053" style="zoom:33%;" />

对于**local minima**，很难解决

对于**sadle point**，利用泰勒多元微分展开

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616120037539.png" alt="image-20240616120037539" style="zoom:33%;" />

$H$为海塞矩阵，也就是二阶偏导

$L(\boldsymbol{\theta})\approx L(\boldsymbol{\theta}^{\prime})+\underbrace{(\boldsymbol{\theta}-\boldsymbol{\theta}^{\prime})^{T}g}+\underbrace{\frac{1}{2}(\boldsymbol{\theta}-\boldsymbol{\theta}^{\prime})^{T}H(\boldsymbol{\theta}-\boldsymbol{\theta}^{\prime})}$

此时处于鞍点，一阶导数为0，即第二项的$g$为0，因此只有$\frac12(\boldsymbol{\theta}-\boldsymbol{\theta}^{\prime})^TH(\boldsymbol{\theta}-\boldsymbol{\theta}^{\prime})$

通过判断这个的值，可以分析处于什么位置

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616120322562.png" alt="image-20240616120322562" style="zoom: 50%;" />

分析：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616120510571.png" alt="image-20240616120510571" style="zoom:33%;" />

此时就是判断$H$​的值就行了，正定？负定？

也可以判断二次型的特征值，判断特征值的正负情况

**举例**

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616120941817.png" alt="image-20240616120941817" style="zoom:33%;" />

现在有一个极简的模型$y=w_1w_2x$，要使输入1，尽可能得到输出也为1

此时损失函数：$L=(\hat{y}-w_1w_2x)^2=(1-w_1w_2)^2$

此时的导数：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616121235105.png" alt="image-20240616121235105" style="zoom: 50%;" />

可以得到：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616121301907.png" alt="image-20240616121301907" style="zoom:50%;" />

此时可以判断是否在一个**saddel point**

判断在鞍点后，可以利用海塞矩阵进行进一步求解最优

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616121530950.png" alt="image-20240616121530950" style="zoom: 50%;" />

<img src="C:\Users\10056\Documents\WeChat Files\wxid_tkf0bkkvtr2522\FileStorage\Temp\9b638df26af8c6f6a060ea1fa713e58.jpg" alt="9b638df26af8c6f6a060ea1fa713e58" style="zoom: 10%;" />

例如：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616122912927.png" alt="image-20240616122912927" style="zoom: 50%;" />

下面会介绍一些超参数，和解决自动调整lr的算法

<img src="https://img-blog.csdnimg.cn/fed6c89c69e94be3bc821b3d3b187a4f.gif" alt="img" style="zoom: 50%;" />

## 2.2 Optimization

### 2.2.1 Batch

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616123904844.png" alt="image-20240616123904844" style="zoom:50%;" />

Full batch理论上运算时间长，但是每一次更新参数的功能比较好，Small Batch，每次只拿一个来更新，速度会快，但是跳的较大。

现在采用GPU进行并行计算，对于Large Batch的运算时间大大降低

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616124149411.png" alt="image-20240616124149411" style="zoom:33%;" />

对比：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616125027857.png" alt="image-20240616125027857" style="zoom: 33%;" />



****

### 2.2.2 Momentum

传统梯度下降（一般）

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616125503339.png" alt="image-20240616125503339" style="zoom:33%;" />

也就是加梯度反方向

现在：

***Gradient Descent + Momentum***

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616125713382.png" alt="image-20240616125713382" style="zoom: 50%;" />

相当于每次找最优解的时候都会考虑到前一次的方向，利用前一次的方向和负梯度方向得到新一次的方向

*有点像惯性*

$m^i$是相当于之前的梯度$g^0,g^1,...,g^{i-1}$的一个权重和

$m^0=0$

$m^1=-\eta g^{0}$
$$m^2=-\lambda\eta g^0-\eta g^1$$

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616130247175.png" alt="image-20240616130247175" style="zoom:50%;" />

相当于在最低点（局部最优后还要考虑到之前的运动方向，此时即使梯度为0.但也会继续运动）

### 2.2.3 Learning Rate

学习率就是学习步长，在梯度下降是确定方向，Learning rate 是确定在该方向所走步长。

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616130935549.png" alt="image-20240616130935549" style="zoom: 67%;" />

可以从右下角看到，即使步长较小，但是也不能达到预期的x值

最开始的梯度下降，此时学习率lr是固定的,也就是$\eta$固定

​                                                                  $\theta_i^{t+1}\leftarrow\theta_i^t-\eta\boldsymbol{g}_i^t\\g_i^t=\frac{\partial L}{\partial\theta_i}|_{\theta=\theta^t}$

此时会出现上面讲到的问题，对于固定的lr不一定能到达最终的理想点

**此时思考能不能把学习率lr在梯度变化较小的地方大一点，在梯度变化较大的地方少一点？**

也就是能不能把$\eta$随着时间变化而改变：

​                                                                  $\theta_i^{t+1}\leftarrow\theta_i^t-\frac\eta{\sigma_i^t}g_i^t$

如何设置$\frac{n}{\sigma_{i}^{t}}$?

#### 2.2.3.1 Root Mean Square

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616131936737.png" alt="image-20240616131936737" style="zoom:33%;" />

对于：

​                                                       $\sigma_i^t=\sqrt{\frac{1}{t+1}\sum_{i=0}^t(g_i^t)^2}$

当最开始斜率小（梯度小）的时候（平缓一点的地方）$\sigma_i^t$就会很小，因为相邻的两个梯度都很小（变化小），此时$\sigma_i^t$就大，所以：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616132708090.png" alt="image-20240616132708090" style="zoom: 50%;" />

可以看到蓝色曲线的斜率很小，变化也小，所以其步长就会大一点。

**Adagrad算法**

[【快速理解Adagrad】通俗解释Adagrad梯度下降算法-CSDN博客](https://blog.csdn.net/qq_45193872/article/details/124153859)

```python
def sgd_adagrad(parameters, sqrs, lr):
	eps = 1e-10
	for param, sqr in zip(parameters, sqrs):
 	sqr[:] = sqr + param.grad.data ** 2
 	div = lr / torch.sqrt(sqr + eps) * param.grad.data
 	param.data = param.data - div
```

缺点：对于特定的情形：**新月形**无法较好的修改学习率

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616170535024.png" alt="image-20240616170535024" style="zoom: 50%;" />

#### 2.2.3.2 RMSProp

引入一个超参数$\alpha$

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616171855676.png" alt="image-20240616171855676" style="zoom:50%;" />

利用超参数$\alpha$调控现在的$g_i^t$的重要性

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616172122578.png" alt="image-20240616172122578" style="zoom:50%;" />

对于最开始的梯度不大，此时可以调整$\alpha$使得$\sigma_{i}^{t-1}$较大，对于中间地方可以调整$\alpha$使得踩一脚刹车，此时梯度较大，可以通过变大$\alpha$​使得速度降下来。

```python
def rmsprop_update(parameters, gradients, sq_grads, lr=0.01, beta=0.9, epsilon=1e-8):
    for param, grad in zip(parameters, gradients):
        sq_grads[param] = beta * sq_grads[param] 
                          + (1 - beta) * (grad ** 2)
        param_update = lr / (np.sqrt(sq_grads[param]) + epsilon) * grad
        param -= param_update

```

在这个函数中，`parameters` 是模型参数列表，`gradients` 是对应的梯度列表，`sq_grads` 是历史梯度平方的累积（需要初始化），`lr` 是学习率，`beta` 和 `epsilon` 是 RMSProp 算法的超参数。

#### 2.2.3.3 Adam

**Adam = RMSProp + Momentum**

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616173709769.png" alt="image-20240616173709769" style="zoom:50%;" />

Pytorch已经有内置的Adam套件

现在利用Adam后可以得到：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616173920278.png" alt="image-20240616173920278" style="zoom:50%;" />

中间红圈里面的突然向上或向下突变，主要是由于中间横向移动的时候，梯度一直很小，导致$\sigma_{i}^{t-1}$

积累到很小的程度，然后步长就突变，同理达到很大后，又会回来。

### 2.2.4 Learning Rate Scheduling

之前我们一直用的方法：

​                                                                  $\theta_i^{t+1}\leftarrow\theta_i^t-\frac\eta{\sigma_i^t}g_i^t$

此时我们都是控制$\eta$为一个定值，当我们采用效果还可以的Adam修改后，发现在$\sigma_{i}^{t-1}$积累到很小的程度的时候，会发生突变，现在可以使得$\eta\rightarrow\eta^t$,让这个值随着时间变化而变化，此时可以得到：

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616174633083.png" alt="image-20240616174633083" style="zoom:50%;" />

#### 2.2.4.1 Learing Rate Decay

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616174728101.png" alt="image-20240616174728101" style="zoom:33%;" />

强调随时间增长后单调递减

#### 2.2.4.2 Warm up

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616175413901.png" alt="image-20240616175413901" style="zoom:33%;" />

强调，先递增后递减，至于什么时候递增，什么时候递减要自行调配。目前主要用于训练***BERT***

### 2.2.5 Optimization 总结

<img src="C:\Users\10056\AppData\Roaming\Typora\typora-user-images\image-20240616180348913.png" alt="image-20240616180348913" style="zoom:50%;" />

