# 3d-model-classify-retrival

The project is only for discovering and can't really be used in real-world project

安装的python包见requirement.txt

##主目录
*requirement.txt 程序需要的依赖包可以通过pip install -r requirements.txt安装
*ModelNet2数据集文件夹（来自ModelNet40，选取两个）
*showPlot.py 展示图片的函数
*ModelGeneration.py 生成神经网络模型的文件
*model.h5 生成的模型的保存文件
*model.json 生成的模型的权重参数
*util.py 距离计算的公式的文件
*classifier.py 生成模型的分类和计算最相似的模型
*FeaGeneration.py 生成特征矩阵，被用于ModelGeneration和classifier
*Sph_harm_for2.ipynb 探索的jupyter文件

##运行步骤和命令行命令：
*首先安装需要的依赖
*首先通过cd进入主目录
*运行python ModelGeneration.py 得到model.h5和model.json，和模型输出效果分析
*运行python classifier.py ModelNet2/glass_box/test/glass_box_0246.off 即可对ModelNet2/glass_box/test/glass_box_0246.off这个文件输出原图和五个最相似的图片
