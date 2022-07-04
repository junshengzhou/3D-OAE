## Dataset 
The overall directory structure should be:

```
│3D-OAE/
├──cfgs/
├──datasets/
├──data/
│   ├──ModelNet/
│   ├──ModelNetFewshot/
│   ├──ScanObjectNN/
│   ├──ShapeNet55-34/
│   ├──PCN/
├──.......
```
**ModelNet Dataset:** You can download the processed ModelNet data from [[Google Drive]](https://drive.google.com/drive/folders/1fAx8Jquh5ES92g1zm2WG6_ozgkwgHhUq?usp=sharing)[[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/d/4808a242b60c4c1f9bed/)[[BaiDuYun]](https://pan.baidu.com/s/18XL4_HWMlAS_5DUH-T6CjA )(code:4u1e) and save it in `data/ModelNet/modelnet40_normal_resampled/`. (You can download the offical ModelNet from [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip), and process it by yourself.) Finally, the directory structure should be:
```
│ModelNet/
├──modelnet40_normal_resampled/
│  ├── modelnet40_shape_names.txt
│  ├── modelnet40_train.txt
│  ├── modelnet40_test.txt
│  ├── modelnet40_train_8192pts_fps.dat
│  ├── modelnet40_test_8192pts_fps.dat
```

**ModelNet Few-shot Dataset:** We follow the previous work to split the original ModelNet40 into pairs of support set and query set. The split used in our experiments is public in [[Google Drive]](https://drive.google.com/drive/folders/1gqvidcQsvdxP_3MdUr424Vkyjb_gt7TW?usp=sharing)/[[Tsinghua Cloud]](https://cloud.tsinghua.edu.cn/d/d4aac5b8f02749e3bdaa/)/[[BaiDuYun]](https://pan.baidu.com/s/1s-Dn1s8cYpeaFVpd1jslzg)(code:bjbq). Download the split file and put it into `data/ModelNetFewshot`, then the structure should be:

```
│ModelNetFewshot/
├──5way10shot/
│  ├── 0.pkl
│  ├── ...
│  ├── 9.pkl
├──5way20shot/
│  ├── ...
├──10way10shot/
│  ├── ...
├──10way20shot/
│  ├── ...
```

**ShapeNet55/34 Dataset:** You can download the processed ShapeNet55/34 dataset at [[BaiduCloud](https://pan.baidu.com/s/16Q-GsEXEHkXRhmcSZTY86A)] (code:le04) or [[Google Drive](https://drive.google.com/file/d/1jUB5yD7DP97-EqqU2A9mmr61JpNwZBVK/view?usp=sharing)]. Unzip the file under `ShapeNet55-34/`. The directory structure should be

```
│ShapeNet55-34/
├──shapenet_pc/
│  ├── 02691156-1a04e3eab45ca15dd86060f189eb133.npy
│  ├── 02691156-1a6ad7a24bb89733f412783097373bdc.npy
│  ├── .......
├──ShapeNet-35/
│  ├── train.txt
│  └── test.txt
```


**PCN Dataset:** Download the offical data from [here](https://gateway.infinitescript.com/?fileName=ShapeNetCompletion) and unzip it into `data/PCN`. The directory structure should be:
```
│PCN/
├──train/
│  ├── complete
│  │   ├── 02691156
│  │   │   ├── 1a04e3eab45ca15dd86060f189eb133.pcd
│  │   │   ├── .......
│  │   ├── .......
│  ├── partial
│  │   ├── 02691156
│  │   │   ├── 1a04e3eab45ca15dd86060f189eb133
│  │   │   │   ├── 00.pcd
│  │   │   │   ├── 01.pcd
│  │   │   │   ├── .......
│  │   │   │   └── 07.pcd
│  │   │   ├── .......
│  │   ├── .......
├──test/
│  ├── complete
│  │   ├── 02691156
│  │   │   ├── 1d63eb2b1f78aa88acf77e718d93f3e1.pcd
│  │   │   ├── .......
│  │   ├── .......
│  ├── partial
│  │   ├── 02691156
│  │   │   ├── 1d63eb2b1f78aa88acf77e718d93f3e1
│  │   │   │   └── 00.pcd
│  │   │   ├── .......
│  │   ├── .......
├──val/
│  ├── complete
│  │   ├── 02691156
│  │   │   ├── 4bae467a3dad502b90b1d6deb98feec6.pcd
│  │   │   ├── .......
│  │   ├── .......
│  ├── partial
│  │   ├── 02691156
│  │   │   ├── 4bae467a3dad502b90b1d6deb98feec6
│  │   │   │   └── 00.pcd
│  │   │   ├── .......
│  │   ├── .......
├──PCN.json
└──category.txt
```

**ScanObjectNN Dataset:** Download the offical data from [here](http://103.24.77.34/scanobjectnn) and unzip it into `data/ScanObjectNN`. The directory structure should be:
```
│ScanObjectNN/
├──main_split/
│  ├── training_objectdataset_augmentedrot_scale75.h5
│  ├── test_objectdataset_augmentedrot_scale75.h5
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
├──main_split_nobg/
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
```