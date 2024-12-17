## WeatherDataHub
WeatherDataHub 是一个高效、全面的气象数据处理工具库，支持多种数据格式和来源，旨在帮助气象学家、数据科学家和研究人员轻松处理和分析各类气象数据。

### 支持的数据类型
1.模式数据
 - 支持全球/区域数值模式输出的处理和转换
 - 数据格式：GRIB、NetCDF (nc)、Zarr 等

2.雷达数据
 - 处理原始雷达观测数据及衍生产品
 - 数据格式：HDF、NetCDF

3.卫星数据
 - 支持多颗卫星观测数据的读取与处理
 - 数据格式：HDF、NetCDF、Zarr 等

4.站点数据
 - 处理地面观测站点数据，包括实测和统计信息
 - 数据格式：CSV、TXT 等

### 功能概述

 - 数据读取与格式转换
 - 支持多种气象数据格式的读取、转换和存储。
 - 空间与时间处理
 - 数据子区域提取、重网格化、时间序列分析等功能。
 - 批量处理与高性能计算
 - 基于并行处理和高性能工具加速大数据量的读取和计算。
 - 数据清洗与质量控制
 - 自动化的数据检查、异常值剔除和质量控制流程。
 - 数据可视化
 - 提供快速可视化功能，用于展示空间分布、时间序列和其他统计结果。
 - 数据探索性分析
 - 建模可行性分析

### 文件结构
```bash
WeatherDataHub/
│
├── data/                   # 示例数据文件夹
│   ├── radar/              # 雷达数据处理
│...
├── data_process/           # 数据处理脚本 normalization etc.
│   ├── model/              # 模式数据处理
│   ├── radar/              # 雷达数据处理
│   ├── satellite/          # 卫星数据处理
│   ├── station/            # 站点数据处理
│   ├── common/             # 通用处理脚本
│   ├── utils/              # 工具函数库
│   ├── io.py               # 数据读取与写入
│── data_eda/               # 数据探索
│   ├── data_cor/           # 相关性分析
│   ├── data_ml/            # machine learning方法建模分析
│   ├── data_dl/            # deep learning建模分析
│     ├── /scheme           # model{"name","parameter-quantity"}、data{"input","model-output","target"}、loss{"data-transform"}
│── data_visual/  
├── README.md               # 项目说明文档
├── requirements.txt        # 依赖库清单
```

[data_visual](https://github.com/zhengkai15/WeatherVis.git)

### 依赖库

本项目依赖以下 Python 库：
 - xarray
 - numpy
 - pandas
 - dask
 - netCDF4
 - h5py
 - cfgrib
 - zarr
 - matplotlib
 - cartopy
 - pyproj

### 许可证

本项目遵循 MIT License 开源许可协议。

### 致谢

感谢所有为气象数据处理和分析工具贡献力量的开发者和社区！

