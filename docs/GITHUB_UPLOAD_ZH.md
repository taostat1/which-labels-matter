# 仓库维护说明

这个文件夹可以直接作为仓库根目录。它与 Overleaf 压缩包分开：这里主要放算法、演示、测试和复现入口，不放个人投稿信和大体积数据。

## 先看这几个文件

1. `README.md`：英文项目主页及安装说明。
2. `docs/ALGORITHM.md`：与论文 Algorithm 1 对应的公式和三阶段、十四步流程。
3. `src/whichlabels/api.py`：公共特征准备、查询、停止、回退及最终训练。
4. `src/whichlabels/labels.py`：真实标签接口与去重记账。
5. `docs/REPRODUCING.md`：模拟和真实数据实验入口。
6. `VALIDATION.json`：当前 CP 版本的软件检查及此前版本的历史记录。

## 本地运行

在仓库目录创建并激活 Python 虚拟环境，运行：

```text
python -m pip install -e ".[reproduce]"
python -m whichlabels.demo --output demo_output
python -m unittest discover -s tests -v
```

演示使用合成数据，不需要下载 GoEmotions、ADE20K 或模型权重。
输出包括候选、区间、比较标签数、最终训练新增标签数、总标签数、
回退情况及计算秒数。独立完整标签审计在两种方法结束之后进行。

## 作者信息与许可证

- 补真实作者与论文链接。`docs/CITATION.cff.example` 是待填写模板，
  填完后再复制为仓库根目录的 `CITATION.cff`。
- 选择代码许可证并添加 `LICENSE`。当前没有代作者指定开源许可证。
- 确认准备缓存和图像数据的公开方式，遵守原数据集及模型使用条款。
- `.gitignore` 已排除虚拟环境、特征缓存、标签数组和新运行结果。
- 不要把投稿信、作者私人联系记录或本地完整研究文件夹一起上传。

本地 0.2.1 版本使用 CP 精确二项区间，并补上弱正则下相同训练设计的精确相消。最新正文模拟使用 Simulation_Refinement_20261106.zip，真实实验与扩展使用 CP_Revision_20261106.zip；旧 EB 复现目录保留作历史记录。
仓库地址：https://github.com/taostat1/which-labels-matter 。后续修改可通过 Git 客户端提交和推送。

