# Repository Guidelines

## 项目结构与模块组织

`nanovllm/` 是主包。对外 API 位于 `llm.py` 和 `sampling_params.py`；运行时
调度、块管理、序列和模型执行位于 `engine/`。模型家族定义放在 `models/`，通用
算子和神经网络层放在 `layers/`，模型加载及上下文辅助逻辑放在 `utils/`。修改
特定模型时，应尽量将改动限制在对应模块中。`example.py` 是主要使用示例，
`bench.py` 用于本地吞吐基准测试，`docs/` 存放说明文档，`assets/` 存放图片资源。

## 构建、测试与开发命令

- `python -m pip install -e .`：以可编辑模式安装项目；需要 Python 3.10–3.12，
  并安装 `pyproject.toml` 中声明的依赖。
- `python example.py`：运行本地 Qwen3-0.6B 生成示例；模型未放在
  `~/huggingface/` 时，请先修改脚本内路径。
- `python bench.py`：预热引擎并输出本地生成吞吐量。
- `python -m compileall nanovllm`：快速检查包内 Python 文件的语法和编译情况。

当前未配置测试框架或 CI 命令。实现功能时请补充有针对性的测试，建议命名为
`test/test_<area>.py`，并直接运行，例如 `python test/test_scheduler.py`。若可用
小型 CPU 单元测试验证，请不要只依赖 GPU 环境测试。

## 代码风格与命名约定

遵循现有 Python 风格：使用四个空格缩进；导入顺序为标准库、第三方库、本地包。
函数、变量和模块采用 `snake_case`，类采用 `PascalCase`；名称应简洁并体现领域
概念，如 `Sequence`、`Scheduler`、`BlockManager`。修改既有代码时保留类型标注和
`@dataclass` 用法。项目未配置格式化或静态检查工具，请保持改动范围小，并匹配
周边代码的排版，而不要引入新的工具或风格。

## 提交与 Pull Request 规范

近期提交使用简短祈使式摘要，常带 conventional scope，例如：
`fix(scheduler): recalculate num_tokens after allocate`。修复或新功能应沿用此格式，
例如 `feat(engine): add request validation`，且每个提交只处理一项明确变更。PR 应
说明行为变化、受影响的模型或配置、已执行的验证；适用时关联 Issue。性能相关改动
应附上基准结果。不要提交模型权重、缓存或生成的性能分析产物。
