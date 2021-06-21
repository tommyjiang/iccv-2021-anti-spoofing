在`pkg`文件夹中有anaconda的安装包
# conda环境搭建

```shell
conda create -n anti python=3.7

conda activate anti

pip install -r requirements.txt
```

# 安装Ranger优化器

```shell
git clone https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
cd Ranger-Deep-Learning-Optimizer
pip install -e .
```

# 拉取数据

```shell
图片数据解压后放置于raw_data文件夹内
```
# 预处理

```shell
通过retinanet r50，完成人脸检测，检测结果是 extra_data/pts_v2.tar
基本思路为，设置[0.8 0.1 0.001]三个阈值档位，依次进行检测，在0.1及以上档位检出的标签给到2，在0.1-0.001之间输出的标签给到1，0.001仍为检到脸给0 通过该处理，完成色块图像，以及极端光照图像的简单划分，以及困难人脸的检出

```

# 首次训练
```shell
cd extra_data
tar -xvf pts_v2.tar
cd ../anti_code
python3 train_ccl.py --config_file="configs/ccl_mask.yml"
获取最佳valid的结果，以及模型文件model_1

使用该模型预测二阶段数据，获取初始伪标签
python3 test_TTA.py --config_file="configs/ccl_mask.yml"
得到标签为temp_res_test.txt

```
# 半监督方法增强结果
```shell
修改./anti_code/data/build.py 中第108行data_scores为此前生成的初始伪标签
进而调用半监督训练代码
python3 train_ccl_pesudo_update.py --config_file="configs/ccl_mask_pesudo_update.yml"  
取其第35epoch作为输出结果 ，第二次lr decay后5epoch，可以直接取过程中产出的结果文件 ../logs/xxxx/pesudo_ckpt/pesudo_scores_34.txt

```
# 后处理

```shell
调用后处理程序完成结果平滑
cd ./post_process
python3 post_process.py
运行前，需要更改代码中的读取路径以及结果生成路径。结果即为最终的推理结果。
考虑到valid和test上存在分布偏差，为EER求出的阈值过偏，导致结果偏移严重，因此直接采取0.5-0.7之间的数值作为阈值。根据此前train_ccl产生的最佳模型在validset上产生的结果及赛方提供的valid标签，计算最佳阈值点，并划分出真假结果。随后将neg设为thred-0.1，pos设为thred+0.1，作为前置结果合并提交

```
