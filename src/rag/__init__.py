"""知识库（RAG）子包。

模块划分
--------
* :mod:`src.rag.corpus` —— 语料抽取、清洗、分块（含字符偏移）
* ``src.rag.lexical`` —— 词法检索（TF-IDF 字符 n-gram），无网络时的兜底
* ``src.rag.dense`` —— 向量检索（embedding-3 + numpy 精确余弦）
* ``src.rag.retrieve`` —— 统一检索入口与结果融合

设计约束：不引入任何第三方向量库，索引全部是可删除、可重建的中间产物。
"""