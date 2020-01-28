# 环境
<a href="https://pypi.org/project/lightgbm" rel="nofollow"><img src="https://camo.githubusercontent.com/34244ae628b4cb096fa26305abc1304e5d1b5e33/68747470733a2f2f696d672e736869656c64732e696f2f707970692f707976657273696f6e732f6c6967687467626d2e7376673f6c6f676f3d707974686f6e266c6f676f436f6c6f723d7768697465" alt="Python Versions" data-canonical-src="https://img.shields.io/pypi/pyversions/lightgbm.svg?logo=python&amp;logoColor=white" style="max-width:100%;"></a>
[![image](https://img.shields.io/badge/PySpark-deepgreen.svg)](https://pypi.org/project/lightgbm/)
[![image](https://img.shields.io/badge/Lightgbm-deepgreen.svg)](https://pypi.org/project/lightgbm/)
[![image](https://img.shields.io/badge/conda-deepgreen.svg)](https://www.anaconda.com/)
[![image](https://img.shields.io/badge/PyCharm-deepgreen.svg)](https://www.jetbrains.com/pycharm/)
[![image](https://img.shields.io/badge/hadoop-deepgreen.svg)](https://hadoop.apache.org/)



## 详细说明：
<ul>
    <li>一、完成度
        <ul>
          <li>基本上按照IEEE CIS Fraud Detection top1的分享方案完成了数据预处理，特征工程，和模型训练等步骤。
          训练使用的mmlspark包，这个在github搜即可，流程我只跑通了程序中的debug模式，内存不够跑完。
</ul>   
   </li>
    <li>二、程序说明
        <ul>
          <li>程序只有PySpark_Model.py一个，运行的话，内存足够的情况下将debug设置为False，不够的话设置为True，我前面根目录写的配置只能跑debug模式。另外，前面建议先debug模式测试下环境和设置，然后再使用全部数据。运行的时候，先设置step=1，程序会跑完数据预处理和特征工程部分，然后将处理得到的csv文件保存在hdfs上面。然后step=2，程序开始训练模型，并输出auc值，这些可以自行修改。我测试了debug模式下auc在0.93左右，mmlspark的lightgbm参数基本上没有设置，设置好了应该还有提升，作者的cv大概在0.94，应该相差不大。
</ul>
   </li>
    </li>
    <li>三、总结
        <ul>
          <li>写完这个脚本，对Pyspark的api熟悉了很多，很多和pandas不一样的编写方式也适应了，不得不说，Pyspark处理数据真是比pandas难用了很多，而且很多坑，不写一遍跑出来结果，根本看不出问题，官方文档也不是特别详细，网络上的博客、csdn，基本上都是抄来抄去，这方面资料还是建议google+stackoverflow，偶尔能碰到些相似的问题解决方式，我基本上是靠自己试加猜来解决遇到的问题。另外，我电脑是单机版本的spark加hadoop，所以用起来总是感觉不太正常，内存方面也是一直炸，这个涉及到pyspark里面executor、driver内存的设置，遇到的坑我都在脚本里面详细注释了，可以仔细看看，还有一些其它的问题的解决办法我都写在里面了。最终，还是内存不足，pandas版本作者的脚本都能随便跑完，而且时间很短。以后还是不在单机弄了，感觉很难用，为了个内存问题，调了一两周，最后还是没通，感觉是哪里内存泄露还是内存没释放干净，对spark的机制不太了解，后面如果弄清了再修改下。mmlspark的资料网上也很少，找了很久还是没找到详细参数介绍，只能先用基本参数跑通了，而且有些参数按它的文档设置了也感觉没用，训练过程设置了参数也输出不了，只能在spark的web端看到一点点输出，和lightgbm的单机版差别还是有点，而且参数很少，又特别慢，等我有空再优化下。
    </li>
</ul>
   </li>
    </li>
    <li>四、参考
      <ul>
        <li>  top1 Notebook  https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600 

</ul>





