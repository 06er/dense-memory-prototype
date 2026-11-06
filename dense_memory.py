import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from typing import List, Optional

# ===================== 1. 定义记忆单元 Memory Cell（论文里的Dense Memory基本单元） =====================
@dataclass
class MemoryCell:
    """
    记忆单元：Dense Memory Bank 的基础存储对象
    对比RAG的chunk：RAG chunk是只读原文切片；MemoryCell可以被修改、合并、标记失效
    """
    content: str               # 文本事实
    embedding: np.ndarray      # 稠密向量
    is_valid: bool = True      # 是否有效（支持失效/删除，RAG一般不做）
    source: str = ""           # 来源标记
    timestamp: int = 0         # 时间戳，用于老化遗忘

class DenseMemoryBank:
    def __init__(self, embed_dim: int = 128, sim_threshold: float = 0.75):
        self.embed_dim = embed_dim
        self.sim_threshold = sim_threshold  # 相似度阈值，用于记忆合并
        self.memory_cells: List[MemoryCell] = []
        self._time_clock = 0

    def _mock_embedding(self, text: str) -> np.ndarray:
        """模拟embedding，真实场景替换成text-embedding / BGE / E5"""
        rng = np.random.default_rng(seed=hash(text) % 10000)
        vec = rng.normal(size=self.embed_dim)
        return vec / np.linalg.norm(vec)

    def ingest(self, text: str, source: str = "") -> MemoryCell:
        """
        【Ingest 写入模块】
        新信息流入，编码成记忆单元，先尝试和已有相似记忆合并，而不是无脑append
        这就是Dense的含义：不是简单堆块，持续维护记忆状态
        """
        emb = self._mock_embedding(text)
        new_cell = MemoryCell(
            content=text,
            embedding=emb,
            source=source,
            timestamp=self._time_clock
        )
        self._time_clock += 1

        # 找相似且有效的记忆单元，尝试合并
        if self.memory_cells:
            all_embs = np.stack([c.embedding for c in self.memory_cells if c.is_valid])
            sims = cosine_similarity(emb.reshape(1, -1), all_embs)[0]
            max_sim_idx = np.argmax(sims)
            max_sim = sims[max_sim_idx]

            valid_cells = [c for c in self.memory_cells if c.is_valid]
            if max_sim >= self.sim_threshold:
                # 找到相似记忆，触发合并（核心：修改旧记忆，不是新增）
                target_cell = valid_cells[max_sim_idx]
                # 这里简化：拼接内容；真实版本交给LLM做语义融合、冲突判断
                target_cell.content = f"[MERGED] {target_cell.content} | {text}"
                target_cell.embedding = (target_cell.embedding + emb) / 2
                target_cell.timestamp = self._time_clock
                return target_cell

        # 没有相似记忆，新增
        self.memory_cells.append(new_cell)
        return new_cell

    def retrieve(self, query: str, top_k: int = 3) -> List[MemoryCell]:
        """
        【Retrieve 稠密检索】
        和普通RAG一样是向量检索，但检索出来的是可编辑记忆单元，不是只读文档切片
        """
        if not self.memory_cells:
            return []
        q_emb = self._mock_embedding(query)
        valid_cells = [c for c in self.memory_cells if c.is_valid]
        if len(valid_cells) == 0:
            return []
        embs = np.stack([c.embedding for c in valid_cells])
        sims = cosine_similarity(q_emb.reshape(1, -1), embs)[0]
        # 按相似度排序取topk
        sorted_idx = np.argsort(-sims)[:top_k]
        return [valid_cells[i] for i in sorted_idx]

    def update_memory(self, old_cell: MemoryCell, new_content: str, mark_invalid: bool = False):
        """
        【Update 记忆回写模块，RAG没有这个！】
        推理之后发现矛盾/新结论，修改旧记忆，或者标记旧记忆失效
        这是Dense Memory最核心的差异化算子
        """
        if mark_invalid:
            old_cell.is_valid = False
            return
        old_cell.content = new_content
        old_cell.embedding = self._mock_embedding(new_content)
        old_cell.timestamp = self._time_clock
        self._time_clock += 1

    def forget_old(self, max_keep: int = 10):
        """【可选：遗忘机制，解决记忆膨胀】
        记忆太多的时候，淘汰最久没访问的冷记忆，解决论文提到的开销爆炸问题
        """
        valid = sorted([c for c in self.memory_cells if c.is_valid], key=lambda x: x.timestamp)
        if len(valid) > max_keep:
            need_drop = valid[: len(valid) - max_keep]
            for c in need_drop:
                c.is_valid = False

# ===================== 2. 推理控制器 Reason Controller（论文里的推理侧） =====================
class ReasonController:
    def __init__(self, memory_bank: DenseMemoryBank):
        self.memory = memory_bank

    def _mock_llm_reason(self, prompt: str) -> str:
        """模拟LLM推理，真实项目替换成 Ollama / OpenAI / transformers 调用"""
        return f"LLM_RESPONSE(基于记忆推理): {prompt[:120]}..."

    def reason_loop(self, user_query: str):
        """
        完整闭环：检索记忆 → 交给LLM推理 → LLM输出里判断是否要更新记忆
        """
        # step1：稠密检索相关记忆
        recalled_mem = self.memory.retrieve(user_query, top_k=3)
        mem_context = "\n".join([f"- {c.content}" for c in recalled_mem])

        # step2：基于记忆做推理
        prompt = f"""
        已知记忆：
        {mem_context}
        用户问题：{user_query}
        请回答，同时判断：是否和已有记忆冲突？是否需要修改记忆？
        """
        answer = self._mock_llm_reason(prompt)
        print("===== 推理回答 =====")
        print(answer)

        # step3：【关键】模拟LLM输出指令，触发记忆更新（真实场景：LLM输出结构化json：action=update/invalid/add）
        # 这里写一个简单规则演示：如果query包含“修正”，就更新第一条记忆
        if "修正" in user_query and len(recalled_mem) > 0:
            target = recalled_mem[0]
            new_fact = f"【修正后的事实】{user_query}"
            self.memory.update_memory(target, new_fact)
            print(f"\n===== 记忆已更新 =====")
            print(f"旧记忆：{target.content}")
            print(f"更新为：{new_fact}")

        return answer

# ===================== 3. 主程序演示闭环 =====================
if __name__ == "__main__":
    bank = DenseMemoryBank(embed_dim=128, sim_threshold=0.72)
    agent = ReasonController(bank)

    # 信息流持续流入（流式 ingest，不是一次性喂给LLM）
    print("=== 写入第一批记忆 ===")
    bank.ingest("实验A的准确率是0.72", source="实验日志1")
    bank.ingest("实验B使用学习率1e-4", source="实验日志2")
    bank.ingest("模型在短文本上效果更好", source="总结笔记")

    # 第一轮查询
    print("\n===== 第一轮查询：实验A准确率是多少 =====")
    agent.reason_loop("实验A准确率是多少？")

    # 第二轮：信息冲突，修正记忆（Dense Memory的核心能力）
    print("\n===== 第二轮查询：修正，实验A准确率实际是0.85 =====")
    agent.reason_loop("修正，实验A准确率实际是0.85")

    # 第三轮再查询，读到更新后的记忆
    print("\n===== 第三轮查询：再次问实验A准确率 =====")
    agent.reason_loop("实验A准确率现在是多少？")

    # 遗忘测试，控制记忆规模
    bank.forget_old(max_keep=5)
    print("\n=== 记忆老化遗忘完成 ===")
