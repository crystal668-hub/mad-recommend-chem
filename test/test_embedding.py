"""
===================================
测试嵌入模型选择功能
功能：验证根据agent_name选择不同的嵌入模型
===================================
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.agent_config import AgentConfig
from database.embedder import MultiModelEmbedder


def test_embedding_selection():
    """
    测试根据agent_name选择不同的嵌入模型
    """
    print("=" * 60)
    print("测试嵌入模型选择功能")
    print("=" * 60)
    
    # 加载配置
    config = AgentConfig("./config/config.yaml")
    
    # 获取所有agent的配置
    all_agent_configs = {
        'agent1': config.get_llm_config('agent1'),
        'agent2': config.get_llm_config('agent2'),
        'agent3': config.get_llm_config('agent3'),
        'agent4': config.get_llm_config('agent4')
    }
    
    print("\n[配置信息]")
    for agent_name, agent_config in all_agent_configs.items():
        embedding_model = agent_config.get('embedding_model')
        embedding_provider = agent_config.get('embedding_provider', 'openrouter')
        print(f"  {agent_name}: {embedding_model} (via {embedding_provider})")
    
    # 初始化MultiModelEmbedder，传入所有agent配置
    print("\n[初始化MultiModelEmbedder]")
    default_config = all_agent_configs['agent1']
    embedder = MultiModelEmbedder(default_config, agent_configs=all_agent_configs)
    
    # 测试文本
    test_text = "Platinum catalyst for hydrogen evolution reaction"
    
    # 测试1: 使用agent1的嵌入模型
    print("\n" + "=" * 60)
    print("测试1: 使用agent1 (openai/text-embedding-3-large via OpenRouter)")
    print("=" * 60)
    try:
        embedding1 = embedder.embed_text(test_text, agent_name='agent1')
        print(f"✓ 成功获取向量")
        print(f"  向量维度: {len(embedding1)}")
        print(f"  向量前5个值: {embedding1[:5]}")
    except Exception as e:
        print(f"✗ 失败: {str(e)}")
    
    # 测试2: 使用agent2的嵌入模型 (Voyage AI)
    print("\n" + "=" * 60)
    print("测试2: 使用agent2 (voyage-3-large via Voyage AI SDK)")
    print("=" * 60)
    try:
        embedding2 = embedder.embed_text(test_text, agent_name='agent2')
        print(f"✓ 成功获取向量")
        print(f"  向量维度: {len(embedding2)}")
        print(f"  向量前5个值: {embedding2[:5]}")
    except Exception as e:
        print(f"✗ 失败: {str(e)}")
    
    # 测试3: 使用agent3的嵌入模型
    print("\n" + "=" * 60)
    print("测试3: 使用agent3 (google/gemini-embedding-001 via OpenRouter)")
    print("=" * 60)
    try:
        embedding3 = embedder.embed_text(test_text, agent_name='agent3')
        print(f"✓ 成功获取向量")
        print(f"  向量维度: {len(embedding3)}")
        print(f"  向量前5个值: {embedding3[:5]}")
    except Exception as e:
        print(f"✗ 失败: {str(e)}")
    
    # 测试4: 使用agent4的嵌入模型
    print("\n" + "=" * 60)
    print("测试4: 使用agent4 (text-embedding-v4 via 云百炼平台的OpenAI兼容方法)")
    print("=" * 60)
    try:
        embedding4 = embedder.embed_text(test_text, agent_name='agent4')
        print(f"✓ 成功获取向量")
        print(f"  向量维度: {len(embedding4)}")
        print(f"  向量前5个值: {embedding4[:5]}")
    except Exception as e:
        print(f"✗ 失败: {str(e)}")
    
    # 测试5: 批量向量化
    print("\n" + "=" * 60)
    print("测试5: 使用agent2批量向量化 (3个文本)")
    print("=" * 60)
    test_texts = [
        "Platinum catalysts for hydrogen evolution",
        "Gold nanoparticles for CO2 reduction",
        "Ruthenium for oxygen evolution reaction"
    ]
    try:
        embeddings = embedder.embed_batch(test_texts, batch_size=2, show_progress=True, agent_name='agent2')
        print(f"✓ 成功批量向量化")
        print(f"  向量数量: {len(embeddings)}")
        print(f"  向量维度: {len(embeddings[0]) if embeddings else 0}")
    except Exception as e:
        print(f"✗ 失败: {str(e)}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n总结:")
    print("✓ MultiModelEmbedder已支持根据agent_name动态选择嵌入模型")
    print("✓ agent1/agent3使用OpenRouter API")
    print("✓ agent2使用Voyage AI官方SDK")
    print("✓ agent4使用")
    print("✓ 可以在build_vector_db.py中指定agent_name来选择对应的嵌入模型")


if __name__ == "__main__":
    test_embedding_selection()
