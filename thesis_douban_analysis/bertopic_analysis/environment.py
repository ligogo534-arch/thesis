"""
BERTopic 实操环境检查脚本
用途:检查跑通 BERTopic pipeline 所需的核心依赖是否已安装、版本是否兼容
运行方式: python check_environment.py
"""

import sys
import importlib
from importlib.metadata import version, PackageNotFoundError


# 核心依赖清单: (import 名称, pip 包名称, 说明)
REQUIRED_PACKAGES = [
    ("pandas", "pandas", "数据处理"),
    ("numpy", "numpy", "数值计算"),
    ("jieba", "jieba", "中文分词"),
    ("sentence_transformers", "sentence-transformers", "生成句向量(用于加载 text2vec-base-chinese)"),
    ("umap", "umap-learn", "UMAP 降维"),
    ("hdbscan", "hdbscan", "HDBSCAN 聚类"),
    ("bertopic", "bertopic", "BERTopic 主体"),
    ("sklearn", "scikit-learn", "c-TF-IDF 等底层依赖"),
    ("torch", "torch", "sentence-transformers 底层依赖(embedding 模型推理)"),
]

# 可选但常用的依赖
OPTIONAL_PACKAGES = [
    ("plotly", "plotly", "BERTopic 可视化(visualize_topics 等)"),
    ("matplotlib", "matplotlib", "静态图表绘制"),
]


def check_package(import_name, pip_name, description, optional=False):
    try:
        importlib.import_module(import_name)
        try:
            v = version(pip_name)
        except PackageNotFoundError:
            v = "未知版本"
        status = "OK"
        print(f"[{status:>4}] {pip_name:<28} v{v:<12} - {description}")
        return True
    except ImportError:
        status = "缺失" if not optional else "可选-缺失"
        print(f"[{status:>4}] {pip_name:<28} {'':<13} - {description}")
        return False


def main():
    print(f"Python 版本: {sys.version}\n")

    print("=== 核心依赖 ===")
    missing_required = []
    for import_name, pip_name, desc in REQUIRED_PACKAGES:
        ok = check_package(import_name, pip_name, desc)
        if not ok:
            missing_required.append(pip_name)

    print("\n=== 可选依赖 ===")
    missing_optional = []
    for import_name, pip_name, desc in OPTIONAL_PACKAGES:
        ok = check_package(import_name, pip_name, desc, optional=True)
        if not ok:
            missing_optional.append(pip_name)

    print("\n" + "=" * 60)
    if missing_required:
        print("缺少以下核心依赖,请先安装:")
        print(f"  pip install {' '.join(missing_required)}")
    else:
        print("核心依赖已全部安装。")

    if missing_optional:
        print("\n(可选) 缺少以下依赖,不影响主流程但建议安装:")
        print(f"  pip install {' '.join(missing_optional)}")

    # 额外检查: 是否能实际加载 text2vec-base-chinese
    if "sentence-transformers" not in missing_required:
        print("\n=== 尝试加载 text2vec-base-chinese 模型 (需联网下载,首次会较慢) ===")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("shibing624/text2vec-base-chinese")
            test_vec = model.encode(["这是一个测试句子"])
            print(f"模型加载成功,句向量维度: {test_vec.shape}")
        except Exception as e:
            print(f"模型加载失败,错误信息: {e}")
            print("可能原因: 网络无法访问 huggingface.co, 或需要配置镜像源(如 hf-mirror.com)")


if __name__ == "__main__":
    main()