# 基于向量数据库的内容查询
## 简介
基于Excel文件的其中某些项，建立向量数据库，通过向量相似度查询，返回列中其他项。
## 文件说明
1. build_xxxxxx.py 构建向量数据库
2. query_matching_xxxxxx.py 查询向量数据库
3. query_mix_matching_xxxx.py 混合查询向量数据库
4. commons.py 公共函数
5. fastapi-query.py fastapi接口服务程序


## 重新改写fastapi-query.py 
原老文件为bk可用于参考
### 改写思路和调用方法
1. 复用向量搜索和关键词搜索函数
2. 将keyword list 参数传入
3. 将vector db名 参数传入
4. 合并为两个接口
    * 纯向量匹配查询（0.5权重角度和0.5权重距离）
    * 混合向量匹配查询（0.2权重向量匹配和0.8权重关键字匹配）--权重支持参数可调
5. 调用方法见 xxx/docs 中详细介绍