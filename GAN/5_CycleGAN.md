## 循环一致性GAN

- 循环一致性

	- 循环一致性想解决的问题

		由于 GAN 网络复杂时，生成的图像内容与输入图像无关，循环一致性想要解决 **如何让生成的图像内容与输入的图像内容相关联**
        
    - 循环一致性的核心思想

		原始输入 $A$ 通过生成器 $G_{ab}$ 获得图像 $B$ 后，可以获得与图像 $A$ 相同 domain 的图像 $C$ ，最终保证 $A$ 与 $C$ 一致


- **CycleGAN图像合成网络**

![](./img/8.png)

- 数据集准备

    两组数据集准备，实现一类数据向另一类数据进行转换

- 整体网络结构

    ![](./img/9.png)

    ![](./img/10.png)

    - 学习转换与还原的网络
    - 沿着channel方向进行归一化 
    - 生成器主要由ResNet + 反卷积 + 卷积 输出与输入大小相同的数据 
    - 判别器主要是 卷积 + norm + leakyrelu ，最终输出单层特征图

- 判别器网络

    PatchGAN

    - 输出一个$N \times N$的矩阵，基于感受野计算损失

- 损失函数设计以及数学原理（**体现循环一致性**）

	- 表示定义

		$G$ - $Domain$ $X$ 中的图像 转成  $Domain$ $Y$ 中的图像 生成器
        
        $F$ - $Domain$ $Y$ 中的图像 转成  $Domain$ $X$ 中的图像 生成器
        
        $D_{X}$ - 判断 图像 是否属于 $Domain$ $X$ 且真实图像 的 判别器
        
        $D_{Y}$ - 判断 图像 是否属于 $Domain$ $Y$ 且真实图像 的 判别器
        
    - 循环一致性损失

		从 $Domain$ $X$ 中的图像转为 $Domain$ $Y$ 的图像有：
        
        $$x => y = G(x) => F(G(x)) \approx x$$
        
        从 $Domain$ $Y$ 中的图像转为 $Domain$ $X$ 的图像有：
        
        $$y => x = F(y) => G(F(y)) \approx y$$
        
        则循环一致性损失函数为：
        
        $$L_{cyc}(G,F) = E_{x-P_{data}(x)}[||F(G(x)) - x||_{p}] + E_{y-P_{data}(y)}[||G(F(y)) - y||_{p}]$$
        
  - 总损失函数

	- 普通 GAN 损失函数：
    
        对抗训练 $G，D_Y$

        $$L_{GAN}(G,D_{Y}, X, Y) = E_{y-P_{data}(y)}[logD_Y(y)] + E_{x-P_{data}(x)}[log(1-D_Y(G(x)))]$$
    
    - 最终损失函数：
    
        $$ L(G,F,D_X,D_Y) = L_{GAN}(G,D_{Y}, X, Y) + L_{GAN}(G,D_{X}, Y, X) + \lambda L_{cyc}(G,F)$$
        
- CycleGAN 的创新点

	- 使用 $Instance Normalization$ 替代 $Batch Normalization$
	- 目标损失函数使用 $LSGAN$ 平方差损失替代传统的 $GAN$ 损失

		由于传统 GAN 损失生成图像质量不高并且在模型训练时不稳定，所以使用 **平方差** 损失替代      
        
        $$L_{LSGAN}(G,D_{Y}, X, Y) = E_{y-P_{data}(y)}[|D_Y(y)|^2] + E_{x-P_{data}(x)}[(1-D_Y(G(x)))^2]$$
        
    - 生成器使用残差网络，保存了图像的语义
    - 使用缓存历史图像训练生成器    