

# 基于MLSQL的AutoML算法做乳腺癌预测


## 背景



MLSQL 是由Kyligence的William Zhu主导并带领旗下的开源团队成员开发的一门面向大数据和 AI 的语言，也是一个真正整合数据管理，商业分析，机器学习的统一平台。与其他机器学习框架的相比（机器学习框架SparkML，深度学习框架Tensorflow，PaddlePaddle等），MLSQL更易于使用。其类SQL的语法，简单易懂，门槛更低，适用于BI，AI等场景。同时，底层支持分布式计算和分布式存储，支持海量数据的操作。在数据安全方面，MLSQL 语言内置数据安全并高度可扩展。此外，MLSQL 引擎支持 K8s 等各种云以及私有部署，切合未来云原生的主流趋势。MLSQL 是由Kyligence的William Zhu主导并带领旗下的开源团队成员开发的一门面向大数据和 AI 的语言，也是一个真正整合数据管理，商业分析，机器学习的统一平台。与其他机器学习框架的相比（机器学习框架SparkML，深度学习框架Tensorflow，PaddlePaddle等），MLSQL更易于使用。其类SQL的语法，简单易懂，门槛更低，适用于BI，AI等场景。同时，底层支持分布式计算和分布式存储，支持海量数据的操作。在数据安全方面，MLSQL 语言内置数据安全并高度可扩展。此外，MLSQL 引擎支持 K8s 等各种云以及私有部署，切合未来云原生的主流趋势。MLSQL 是由Kyligence的William Zhu主导并带领旗下的开源团队成员开发的一门面向大数据和 AI 的语言，也是一个真正整合数据管理，商业分析，机器学习的统一平台。与其他机器学习框架的相比（机器学习框架SparkML，深度学习框架Tensorflow，PaddlePaddle等），MLSQL更易于使用。其类SQL的语法，简单易懂，门槛更低，适用于BI，AI等场景。同时，底层支持分布式计算和分布式存储，支持海量数据的操作。在数据安全方面，MLSQL 语言内置数据安全并高度可扩展。此外，MLSQL 引擎支持 K8s 等各种云以及私有部署，切合未来云原生的主流趋势。​​

​​
因此，本文作者希望尝试利用MLSQL提供的能力，对全球医疗健康类的热门话题因此，本文作者希望尝试利用MLSQL提供的能力，对全球医疗健康类的热门话题因此，本文作者希望尝试利用MLSQL提供的能力，对全球医疗健康类的热门话题【良/恶性乳腺癌肿瘤预测】【良/恶性乳腺癌肿瘤预测】【良/恶性乳腺癌肿瘤预测】进行试验。进行试验。进行试验。​​

​​


## 实验准备​



* **操作系统：** MacOS/Linux ​​
* **JDK版本：** 安装 [JDK8-Mac](https://www.openlogic.com/openjdk-downloads?field_java_parent_version_target_id=416&field_operating_system_target_id=431&field_architecture_target_id=391&field_java_package_target_id=396)​​

* **VSCode：** 安装：[https://code.visualstudio.com](https://code.visualstudio.com)​​

* **MLSQL版本：** MLSQL 桌面版下载站点[http://download.mlsql.tech/](http://download.mlsql.tech/) 选择选择选择 [mlsql-mac-0.0.6.vsix](http://download.mlsql.tech/mlsql-mac-0.0.6.vsix) 版本​​


## 数据加载​

“良/恶性乳腺癌肿瘤预测”的问题属于二分类任务，待预测的类别分别是良性乳腺癌肿瘤和恶性乳腺癌肿瘤。本案例的数据来源于Kaggle的开源数据，具体数据的[下载地址](https://www.kaggle.com/tanaypatare/cancer-prediction-precision-99/data)

该数据来源于美国威斯康星州乳腺癌（诊断）数据集，记录了患者的医学影像数据。​​



下面，我们先用MLSQL的load语法，加载数据（MLSQL支持许多数据源的加载，具体可以参考下面，我们先用MLSQL的load语法，加载数据（MLSQL支持许多数据源的加载，具体可以参考下面，我们先用MLSQL的load语法，加载数据（MLSQL支持许多数据源的加载，具体可以参考 [https://mlsql-docs.kyligence.io/latest/zh-hans/lang/load.html](https://mlsql-docs.kyligence.io/latest/zh-hans/lang/load.html)）​​

​
```SQL
set dataPath='/tmp/data/data.csv';
load csv.`${dataPath}` where header='true' as data;
```

#### Step1 数据宏观观测
先用MLSQL的desc宏命令观察加载的表数据里头，原数据被识别成了string的类型，因此后期需要对数据类型做conver转化(【Friendly Notes.】可以在load的where加一句inferSchema="true"，这样就可以从csv里读取到原生数据类型，不再需要做数据类型转换) ​​

下面我们对原始数据的特征做大概的介绍：


| Metrics | Description  |
| --- | --- |
| diagnosis | 乳腺组织的诊断（良性/恶性） |
| radius_mean | 从中心到细胞周长上各点的平均距离 |
| texture_mean | 肿瘤细胞灰度值的标准偏差 |
| perimeter_mean | 核心肿瘤的平均大小 |
| area_mean | 核心肿瘤的平均面积大小 |
| smoothness_mean | 细胞半径长度的局部变化平均值 |
| compactness_mean | 计算方法如下公式 ![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/1.png?raw=true) |
| concavity_mean | 细胞轮廓凹面部分严重程度的平均值 |
| concave points_mean | 轮廓凹面部分数量的平均值 |
| symmetry_mean | 肿瘤细胞的对称性平均值 |
| fractal_dimension_mean | 肿瘤细胞分形维数的平均值 |
| radius_se| 从肿瘤细胞中心到周长上各点距离平均值的标准误差 |
| texture_se | 肿瘤细胞图像灰度值标准偏差的标准误差 |
| perimeter_se | 核心肿瘤的平均大小的标准误差 |
| area_se | 核心肿瘤的平均面积大小的标准误差 |
| smoothness_se | 肿瘤细胞半径长度局部变化的标准误差 |
| compactness_se | （肿瘤细胞紧凑程度）下面计算公式的标准差 ![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/2.png?raw=true)|
| concavity_se | 肿瘤细胞轮廓凹面部分严重程度的标准误差 |
| concave points_se | 肿瘤细胞轮廓凹面部分数量的标准误差 |
| symmetry_se | 肿瘤细胞的对称性标准误差 |
| fractal_dimension_se | 肿瘤细胞分形维数的标准误差 |
| radius_worst | 从肿瘤细胞中心到肿瘤细胞周界上各点的平均距离的“最差”或最大平均值 |
| texture_worst | 肿瘤细胞图像灰度值标准偏差的“最差”或最大平均值 |
| perimeter_worst | 核心肿瘤的平均大小的标准误差的“最差”或最大平均值 |
| area_worst | 核心肿瘤的平均面积的标准误差的“最差”或最大平均值 |
| smoothness_worst | 肿瘤细胞半径长度的局部变化标准误差的“最差”或最大平均值 |
| compactness_worst |  肿瘤细胞紧凑程度的“最差”或最大平均值 |
| concavity_worst |  细胞轮廓凹面部分严重程度的“最差”或最大平均值 |
| concave points_worst | 轮廓凹面部分数量的“最差”或最大平均值 |
| symmetry_worst | 肿瘤细胞的对称性的“最差”或最大平均值 |
| fractal_dimension_worst | 肿瘤细胞分形维数的“最差”或最大平均值 |


```SQL
!desc data;
```
![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/3.png?raw=true)

#### Step2 缺失值观测​

我们在进行模型训练时，不可避免的会遇到某些特征出现空值的情况。因此我们对于数据的空置统计，十分有必要，根据不同的场景我们需要对空置做不同的处理，比如drop或者填充。

统计缺失值， 从如下的统计结果来看，_c32的值存在很多的空置，显然，在这个案例上，我们做空置填充是没有意义的，因此我们将_c32这一列去除。


``` SQL
select SUM( case when `id` is null or `id`='' then 1 else 0 end ) as id,
SUM( case when `diagnosis` is null or `diagnosis`='' then 1 else 0 end ) as diagnosis,
SUM( case when `radius_mean` is null or `radius_mean`='' then 1 else 0 end ) as radius_mean,
SUM( case when `texture_mean` is null or `texture_mean`='' then 1 else 0 end ) as texture_mean,
SUM( case when `perimeter_mean` is null or `perimeter_mean`='' then 1 else 0 end ) as perimeter_mean,
SUM( case when `area_mean` is null or `area_mean`='' then 1 else 0 end ) as area_mean,
SUM( case when `smoothness_mean` is null or `smoothness_mean`='' then 1 else 0 end ) as smoothness_mean,
SUM( case when `compactness_mean` is null or `compactness_mean`='' then 1 else 0 end ) as compactness_mean,
SUM( case when `concavity_mean` is null or `concavity_mean`='' then 1 else 0 end ) as concavity_mean,
SUM( case when `concave points_mean` is null or `concave points_mean`='' then 1 else 0 end ) as concave_points_mean,
SUM( case when `symmetry_mean` is null or `symmetry_mean`='' then 1 else 0 end ) as symmetry_mean,
SUM( case when `fractal_dimension_mean` is null or `fractal_dimension_mean`='' then 1 else 0 end ) as fractal_dimension_mean,
SUM( case when `radius_se` is null or `radius_se`='' then 1 else 0 end ) as radius_se,
SUM( case when `texture_se` is null or `texture_se`='' then 1 else 0 end ) as texture_se,
SUM( case when `perimeter_se` is null or `perimeter_se`='' then 1 else 0 end ) as perimeter_se,
SUM( case when `area_se` is null or `area_se`='' then 1 else 0 end ) as area_se,
SUM( case when `smoothness_se` is null or `smoothness_se`='' then 1 else 0 end ) as smoothness_se,
SUM( case when `compactness_se` is null or `compactness_se`='' then 1 else 0 end ) as compactness_se,
SUM( case when `concavity_se` is null or `concavity_se`='' then 1 else 0 end ) as concavity_se,
SUM( case when `concave points_se` is null or `concave points_se`='' then 1 else 0 end ) as concave_points_se,
SUM( case when `symmetry_se` is null or `symmetry_se`='' then 1 else 0 end ) as symmetry_se,
SUM( case when `fractal_dimension_se` is null or `fractal_dimension_se`='' then 1 else 0 end ) as fractal_dimension_se,
SUM( case when `radius_worst` is null or `radius_worst`='' then 1 else 0 end ) as radius_worst,
SUM( case when `texture_worst` is null or `texture_worst`='' then 1 else 0 end ) as texture_worst,
SUM( case when `perimeter_worst` is null or `perimeter_worst`='' then 1 else 0 end ) as perimeter_worst,
SUM( case when `area_worst` is null or `area_worst`='' then 1 else 0 end ) as area_worst,
SUM( case when `smoothness_worst` is null or `smoothness_worst`='' then 1 else 0 end ) as smoothness_worst,
SUM( case when `compactness_worst` is null or `compactness_worst`='' then 1 else 0 end ) as compactness_worst,
SUM( case when `concavity_worst` is null or `concavity_worst`='' then 1 else 0 end ) as concavity_worst,
SUM( case when `concave points_worst` is null or `concave points_worst`='' then 1 else 0 end ) as concave_points_worst,
SUM( case when `symmetry_worst` is null or `symmetry_worst`='' then 1 else 0 end ) as symmetry_worst,
SUM( case when `fractal_dimension_worst` is null or `fractal_dimension_worst`='' then 1 else 0 end ) as fractal_dimension_worst,
SUM( case when `_c32` is null or `_c32`='' then 1 else 0 end ) as _c32
from data as data_id;
```

![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/4.png?raw=true)

![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/5.png?raw=true)

![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/6.png?raw=true)

![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/7.png?raw=true)

#### Step3 标签转换​

将预测标签列diagnosis进行数据转化，值为M的为得了乳腺癌的样本（1），否则就是没有得乳腺癌的样本（0），这样我们就完成了将预测列进行了标签化工作。​​

```SQL
select int(`id`), (case when `diagnosis` = 'M' then 1 else 0 end) as `diagnosis`,float(`radius_mean`),float(`texture_mean`),
float(`perimeter_mean`),float(`area_mean`),float(`smoothness_mean`),
float(`compactness_mean`),float(`concavity_mean`),float(`concave points_mean`),
float(`symmetry_mean`),float(`fractal_dimension_mean`),float(`radius_se`),float(`texture_se`),
float(`perimeter_se`),float(`area_se`),float(`smoothness_se`),float(`compactness_se`),float(`concavity_se`),
float(`concave points_se`),float(`symmetry_se`),float(`fractal_dimension_se`),float(`radius_worst`),float(`texture_worst`),
float(`perimeter_worst`),float(`area_worst`),float(`smoothness_worst`),float(`compactness_worst`),float(`concavity_worst`),
float(`concave points_worst`),float(`symmetry_worst`),float(`fractal_dimension_worst`)
from data as data1;

```

![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/8.png?raw=true)

**数据分析/可视化**
下面为了观测数据的分布，我们在下面做了数据分析和数据可视化，观测数据之间的关系，以及数据与目标列（diagnosis）的关系。


#### Step4 数据分布观测
```Python
#%env=source /usr/local/Caskroom/miniconda/base/bin/activate dev2
#%python
#%input=data1
#%schema=st(field(content,string),field(mime,string))
from pyjava.api.mlsql import RayContext,PythonContext
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyjava.api import Utils
# 这句是为了代码提示
context:PythonContext = context


ray_context = RayContext.connect(globals(),None)
data = ray_context.to_pandas()


plt.figure(figsize = (20, 15))
sns.set(style="darkgrid")
plotnumber = 1


for column in data:
    if plotnumber <= 30:
        ax = plt.subplot(5, 6, plotnumber)
        sns.histplot(data[column],kde=True)
        plt.xlabel(column)
        
    plotnumber += 1
# plt.show()

Utils.show_plt(plt,context)
```
![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/9.png?raw=true)

#### Step4 异常值观测
数据异常点对于相关系数的影响主要反映在异常点对数据离散程度的影响，对均值点的影响以及异常点偏离均值点的程度，因此我们用箱线图观测异常点数据。

```Python
#%env=source /usr/local/Caskroom/miniconda/base/bin/activate dev2
#%python
#%input=data1
#%schema=st(field(content,string),field(mime,string))
from pyjava.api.mlsql import RayContext,PythonContext
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyjava.api import Utils
# 这句是为了代码提示
context:PythonContext = context


ray_context = RayContext.connect(globals(),None)
data = ray_context.to_pandas()


plt.figure(figsize = (20, 15))
plotnumber = 1


for column in data:
    if plotnumber <= 30:
        ax = plt.subplot(5, 6, plotnumber)
        sns.boxplot(x=data[column])
        plt.xlabel(column)
       
    plotnumber += 1
plt.title("Distribution")


Utils.show_plt(plt,context)
```

![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/10.png?raw=true)

#### Step5 关系热力图

(关系型数据的可视化) 热力图体现了两个离散变量之间的组合关系。
可以通过热力图去观测，每个特征之间的关系程度，以及对n变量的影响程度，从而更好地选择特征列去训练模型。

``` Python
#%env=source /usr/local/Caskroom/miniconda/base/bin/activate dev2
#%python
#%input=data1
#%schema=st(field(content,string),field(mime,string))
from pyjava.api.mlsql import RayContext,PythonContext
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyjava.api import Utils
# 这句是为了代码提示
context:PythonContext = context


ray_context = RayContext.connect(globals(),None)
data = ray_context.to_pandas()


plt.figure(figsize = (30, 15))
sns.heatmap(data.corr(),annot=True)


Utils.show_plt(plt,context)
```
![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/11.png?raw=true)

#### Step6 特征关联关系分布图

```Python
#%env=source /usr/local/Caskroom/miniconda/base/bin/activate dev2
#%python
#%input=data1
#%schema=st(field(content,string),field(mime,string))
from pyjava.api.mlsql import RayContext,PythonContext
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyjava.api import Utils
# 这句是为了代码提示
context:PythonContext = context


ray_context = RayContext.connect(globals(),None)
data = ray_context.to_pandas()


sns.pairplot(data,
             x_vars=[
                          'area_worst',
                     'smoothness_worst',
                  'compactness_worst',
                     'concavity_worst',
                     'concave points_worst',
                        'symmetry_worst',
                      'fractal_dimension_worst'],
             y_vars=["diagnosis"])
Utils.show_plt(plt,context)
```

![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/12.png?raw=true)

```Python
#%env=source /usr/local/Caskroom/miniconda/base/bin/activate dev2
#%python
#%input=data1
#%schema=st(field(content,string),field(mime,string))
from pyjava.api.mlsql import RayContext,PythonContext
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyjava.api import Utils
# 这句是为了代码提示
context:PythonContext = context


ray_context = RayContext.connect(globals(),None)
data = ray_context.to_pandas()


sns.pairplot(data,
             x_vars=[  'concavity_se', 
                     'concave points_se',
                     'symmetry_se',
                     'fractal_dimension_se',
                     'radius_worst', 
                     'texture_worst',
                  'perimeter_worst'],
             y_vars=["diagnosis"])

Utils.show_plt(plt,context)
```

![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/13.png?raw=true)

```Python
#%env=source /usr/local/Caskroom/miniconda/base/bin/activate dev2
#%python
#%input=data1
#%schema=st(field(content,string),field(mime,string))
from pyjava.api.mlsql import RayContext,PythonContext
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyjava.api import Utils
# 这句是为了代码提示
context:PythonContext = context


ray_context = RayContext.connect(globals(),None)
data = ray_context.to_pandas()
sns.pairplot(data,
             x_vars=[
                     'fractal_dimension_mean',
                       'radius_se', 
                     'texture_se', 
                     'perimeter_se',
                     'area_se',
                     'smoothness_se',
                    'compactness_se'],
                y_vars=["diagnosis"])

Utils.show_plt(plt,context)
```

![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/14.png?raw=true)

```Python
#%env=source /usr/local/Caskroom/miniconda/base/bin/activate dev2
#%python
#%input=data1
#%schema=st(field(content,string),field(mime,string))
from pyjava.api.mlsql import RayContext,PythonContext
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyjava.api import Utils
# 这句是为了代码提示
context:PythonContext = context


ray_context = RayContext.connect(globals(),None)
data = ray_context.to_pandas()
sns.pairplot(data,
             x_vars=['radius_mean', 
                           'texture_mean', 
                              'area_mean', 
                     'smoothness_mean',
                     'compactness_mean', 
                     'concavity_mean',
                  'concave points_mean',
                     'symmetry_mean'],
                      y_vars=["diagnosis"])

Utils.show_plt(plt,context)
```

![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/15.png?raw=true)


#### 特征挑选
基于上述的特征与label的关系，我们挑选出 '***radius_mean***', '***perimeter_mean***','***area_mean***', '***smoothness_mean***', '***compactness_mean***', '***concavity_mean***','***concave points_mean***', '***symmetry_mean***', '***fractal_dimension_mean***',
 '***radius_se***', '***texture_se***', '***perimeter_se***', '***area_se***', '***smoothness_se***','***compactness_se***', '***concavity_se***', '***concave points_se***', '***symmetry_se***','***fractal_dimension_se***', '***radius_worst***', '***texture_worst***', '***perimeter_worst***', '***area_worst***', '***smoothness_worst***','***compactness_worst***', '***concavity_worst***', '***concave points_worst***', '***symmetry_worst***', '***fractal_dimension_worst***' 特征

```SQL
select `id` as id, `radius_mean`+`texture_mean`+`perimeter_mean`+`area_mean`+
`smoothness_mean`+`compactness_mean`+`concavity_mean`+`concave points_mean`+
`symmetry_mean`+`fractal_dimension_mean`+`radius_se`+`texture_se`+
`perimeter_se`+`area_se`+`smoothness_se`+`compactness_se`+`concavity_se`+
`concave points_se`+`fractal_dimension_se`+`symmetry_se`+`radius_worst`+
`texture_worst`+`perimeter_worst`+`area_worst`+`smoothness_worst`+
`compactness_worst`+`concavity_worst`+`concave points_worst`+`symmetry_worst`+
`fractal_dimension_worst` as Agg_of_all, `diagnosis` as diagnosis from data1 as data2;
select Min(`Agg_of_all`) as min_Agg_of_all, Max(`Agg_of_all`) as max_Agg_of_all from data2 as data3;
select (`Agg_of_all`-data3.`min_Agg_of_all`)/(data3.`max_Agg_of_all`-data3.`min_Agg_of_all`) as nor_Agg_of_all, `diagnosis` as label, `id` as id from data2, data3 as data4;
select * from data4 join data1 on data4.`id`=data1.`id` as data5;
```

![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/16.png?raw=true)

## 模型训练

#### Step 8 数据类型转换
```SQL
select array(`radius_mean`,`perimeter_mean`,`area_mean`,`smoothness_mean`,`compactness_mean`,`concavity_mean`,
`concave points_mean`,`symmetry_mean`,`fractal_dimension_mean`,`radius_se`,`texture_se`,`perimeter_se`,
`area_se`,`smoothness_se`,`compactness_se`,`concavity_se`,`concave points_se`,`symmetry_se`,`fractal_dimension_se`,
`radius_worst`,`texture_worst`,`perimeter_worst`,`area_worst`,`smoothness_worst`,`compactness_worst`,`concavity_worst`,
`concave points_worst`,`symmetry_worst`,`fractal_dimension_worst`) as features, `label` as label
from data5 as data6;
```

![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/17.png?raw=true)


#### Step9 数据切分
``` SQL
train data6 as RateSampler.`` 
where labelCol="label"
and sampleRate="0.7,0.3" as marked_dataset;
select * from marked_dataset where __split__=1
as testingTable;

select * from marked_dataset where __split__=0
as trainingTable;
```

#### Step10 数据训练

MLSQL支持了AutoML，内置了（随机森林，GBT算法，逻辑回归，线性回归等分类算法）
我们利用MLSQL的AutoML进行模型自动训练，并自动挑选出最好的模型作为目标模型【Note. 由于AutoML是一个最终分类器的sorting行为，所以evaluaTable是一个必填的字段】

```SQL
select vec_dense(features) as features ,label as label from trainingTable as trainData;

train trainData as AutoML.`/tmp/kaggle` where 
algos = "GBTs,LinearRegression,LogisticRegression,NaiveBayes,RandomForest"
and keepVersion="true" 
and evaluateTable="testData";
```

![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/18.png?raw=true)

从上图可以看到，RandomForest的模型表现最好，我们用RF的模型作为最终模型进行数据预测


## 模型预测

#### Step 11 数据预测

```SQL
select vec_dense(features) as features ,label as label from testingTable as testingData;
predict testingData as AutoML.`/tmp/kaggle` as res;
```

![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/19.png?raw=true)


我们需要对模型的预测结果进行评估，MLSQL支持了二分类算法预测的结果评估
下面是预测结果评估的代码,
从下面的evaluation metrics来看，模型的预测结果达到了recall接近1，同时precision和AUC直接逼近1，这个结果还是比较可观的。

```SQL
select replace(String(probability), "[","") as prob, label from res as res1;
select replace(prob, "]", "") as prob, label from res1 as res2;
select Double(split(prob, ",")[1]) as prob, label from res2 as res3;
run res3 as PredictionEva.`` where labelCol='label' and probCol='prob';
```

![avatar](https://github.com/ckeys/personalRepo/blob/master/docs/Kyligence/AutoML/demo/pics/20.png?raw=true)

## 模型注册

通过把模型注册成UDF的方式，用户可以快速部署模型，然后用func调用的方式预测数据

```SQL
register AutoML.`/tmp/kaggle` as two_category_predict;
select two_category_predict(features) as predicted,label from testData as output;
```
