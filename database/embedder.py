"""
===================================
Embedder Module
Function: Use OpenAI-compatible SDK to call OpenRouter/Bailian API for text vectorization
===================================
"""

import os
import time
from typing import Dict, List, Optional

import voyageai
from openai import OpenAI
from tqdm import tqdm

from utils.logger import Logger

logger = Logger.create_module_logger("database.embedder")


class MultiModelEmbedder:
    """
    多模型文本向量化器
    支持根据agent配置动态切换嵌入模型
    支持使用OpenRouter API、Voyage Python Package
    支持OpenAI、Voyage AI、Google等多种嵌入模型
    """
    
    def __init__(self, model_config: Dict, agent_configs: Dict = None):
        """
        初始化多模型向量化器
        
        Args:
            model_config: 默认模型配置字典
            agent_configs: 所有agent的配置字典 
        """
        self.default_model = model_config.get('embedding_model', 'text-embedding-3-large')
        self.api_key = model_config.get('api_key')
        self.base_url = model_config.get('emb_url', 'https://openrouter.ai/api/v1/embeddings')
        
        # Save agent configurations for dynamic model selection
        self.agent_configs = agent_configs or {}
        
        # Lazy load Voyage AI clients and OpenAI clients
        self.voyage_clients: Dict[str, 'voyageai.Client'] = {}
        self.openai_clients: Dict[str, OpenAI] = {}
        
        # Get API Key from environment variable if needed
        if self.api_key and self.api_key.startswith('${') and self.api_key.endswith('}'):
            env_var = self.api_key[2:-1]
            self.api_key = os.environ.get(env_var)

        # Preload and normalize all agent embedding profiles
        self.agent_embedding_profiles = self._build_agent_embedding_profiles()
        
        logger.info("[OK] Initialized embedder")
        
        if self.agent_embedding_profiles:
            agent_models = {
                name: profile.get('embedding_model')
                for name, profile in self.agent_embedding_profiles.items()
            }
            agent_providers = {
                name: profile.get('embedding_provider')
                for name, profile in self.agent_embedding_profiles.items()
            }
            logger.info(f"Agent embedding model providers: {agent_providers}")

    def _resolve_env_var(self, value: Optional[str]) -> Optional[str]:
        if value and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            return os.environ.get(env_var)
        return value

    def _normalize_openai_base_url(self, emb_url: Optional[str]) -> Optional[str]:
        if not emb_url:
            return emb_url
        if emb_url.endswith('/embeddings'):
            return emb_url.rsplit('/embeddings', 1)[0]
        return emb_url

    def _infer_provider_by_agent(self, agent_name: str, agent_config: Dict) -> str:
        if agent_name in ['agent2']:
            return 'voyage'
        if agent_name in ['agent4']:
            return 'bailian'
        if agent_name in ['agent1', 'agent3']:
            return 'openrouter'

        embedding_provider = (agent_config or {}).get('embedding_provider', '')
        if embedding_provider:
            return embedding_provider.lower()
        return 'openrouter'

    def _build_agent_embedding_profiles(self) -> Dict[str, Dict]:
        profiles: Dict[str, Dict] = {}
        for agent_name, cfg in self.agent_configs.items():
            embedding_model = cfg.get('embedding_model', self.default_model)
            emb_url = cfg.get('emb_url', self.base_url)
            openai_base_url = self._normalize_openai_base_url(emb_url)
            api_key = self._resolve_env_var(cfg.get('api_key', self.api_key))
            voyage_api_key = self._resolve_env_var(cfg.get('voyage_api_key'))
            embedding_provider = self._infer_provider_by_agent(agent_name, cfg)

            profiles[agent_name] = {
                'embedding_model': embedding_model,
                'emb_url': emb_url,
                'openai_base_url': openai_base_url,
                'api_key': api_key,
                'voyage_api_key': voyage_api_key,
                'embedding_provider': embedding_provider
            }
        return profiles
    
    def get_model_for_agent(self, agent_name: str = None) -> str:
        """
        获取指定agent使用的向量模型
        
        Args:
            agent_name: agent name 
        
        Returns:
            str: embedding model name
        """
        if agent_name and agent_name in self.agent_configs:
            model = self.agent_configs[agent_name].get('embedding_model', self.default_model)
            return model
        return self.default_model
    
    def _get_voyage_client(self, agent_name: str) -> Optional['voyageai.Client']:
        """
        获取或创建Voyage AI客户端
        
        Args:
            agent_name: agent名称
            
        Returns:
            voyageai.Client或None
        """
        if agent_name not in self.voyage_clients:
            profile = self.agent_embedding_profiles.get(agent_name, {})
            voyage_api_key = profile.get('voyage_api_key')
            if voyage_api_key:
                self.voyage_clients[agent_name] = voyageai.Client(api_key=voyage_api_key)
                logger.info(f"Created Voyage AI client for {agent_name}")
            else:
                return None
                
        return self.voyage_clients.get(agent_name)

    def _get_openai_client(self, agent_name: str, api_key: str, base_url: str) -> OpenAI:
        if agent_name not in self.openai_clients:
            self.openai_clients[agent_name] = OpenAI(api_key=api_key, base_url=base_url)
        return self.openai_clients[agent_name]
        
    def _is_voyage_model(self, agent_name: str = None) -> bool:
        """
        判断指定agent是否使用Voyage向量模型
        
        Args:
            agent_name: agent name
            
        Returns:
            bool: whether using Voyage AI
        """
        if not agent_name or agent_name not in self.agent_embedding_profiles:
            return False

        embedding_provider = self.agent_embedding_profiles[agent_name].get('embedding_provider', '')
        return embedding_provider == 'voyage'

    def _is_bailian_model(self, agent_name: str = None) -> bool:
        """
        判断指定agent是否使用百炼向量模型
        
        Args:
            agent_name: agent name
            
        Returns:
            bool: whether using Aliyun Bailian
        """
        if not agent_name or agent_name not in self.agent_embedding_profiles:
            return False
        provider = self.agent_embedding_profiles[agent_name].get('embedding_provider', '')
        return provider in ['bailian', 'aliyun', 'qwen']

    def _get_bailian_base_url(self, agent_name: str) -> str:
        profile = self.agent_embedding_profiles.get(agent_name, {})
        return profile.get('emb_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings')

    def _get_agent_profile(self, agent_name: Optional[str]) -> Dict:
        if agent_name and agent_name in self.agent_embedding_profiles:
            return self.agent_embedding_profiles[agent_name]
        return {
            'embedding_model': self.default_model,
            'emb_url': self.base_url,
            'openai_base_url': self._normalize_openai_base_url(self.base_url),
            'api_key': self.api_key,
            'embedding_provider': 'openrouter'
        }

    def embed_text(self, text: str, retry: int = 3, agent_name: str = None) -> List[float]:
        """
        将单个文本转换为向量
        
        Args:
            text: input text
            retry: number of retries
            agent_name: agent name, used to select the corresponding embedding model
        
        Returns:
            List[float]: embedding vector
        """
        # Some providers return errors or empty responses for blank inputs; short-circuit to a
        # deterministic zero vector to keep pipelines robust.
        if text is None or not str(text).strip():
            model = self.get_model_for_agent(agent_name) if agent_name else self.default_model
            return [0.0] * self.get_embedding_dimension(model)

        # Determine which agent to use
        use_agent = agent_name
        
        # Check if using Bailian or Voyage AI
        if use_agent and self._is_bailian_model(use_agent):
            return self._embed_text_bailian(text, retry, use_agent)
        if use_agent and self._is_voyage_model(use_agent):
            return self._embed_text_voyage(text, retry, use_agent)
        return self._embed_text_openrouter(text, retry, use_agent)

    def embed_query(self, text: str, retry: int = 3, agent_name: str = None) -> List[float]:
        """
        Embed query text.

        Voyage models support different query/document embedding modes. For Voyage we use
        input_type="query"; for other providers we fall back to `embed_text`.
        """
        if text is None or not str(text).strip():
            model = self.get_model_for_agent(agent_name) if agent_name else self.default_model
            return [0.0] * self.get_embedding_dimension(model)

        use_agent = agent_name

        if use_agent and self._is_voyage_model(use_agent):
            voyage_client = self._get_voyage_client(use_agent)
            if not voyage_client:
                raise ValueError(
                    f"Failed to create Voyage AI client for {use_agent}, please check API key configuration"
                )

            model = self.get_model_for_agent(use_agent)
            for attempt in range(retry):
                try:
                    result = voyage_client.embed(
                        texts=[text],
                        model=model,
                        input_type="query",
                    )

                    if result and result.embeddings:
                        return result.embeddings[0]
                    raise Exception("Voyage AI returned empty result")

                except Exception as e:
                    logger.error(f"[ERROR] Query embedding failed (attempt {attempt + 1}/{retry}): {str(e)}")
                    if attempt < retry - 1:
                        time.sleep(2 ** attempt)

            raise Exception(f"Query embedding failed, retried {retry} times")

        return self.embed_text(text, retry=retry, agent_name=agent_name)
    
    def _embed_text_voyage(self, text: str, retry: int, agent_name: str) -> List[float]:
        """
        使用Voyage AI SDK进行文本向量化
        
        Args:
            text: input text
            retry: number of retries
            agent_name: agent name
            
        Returns:
            List[float]: embedding vector
        """
        voyage_client = self._get_voyage_client(agent_name)
        if not voyage_client:
            raise ValueError(f"Failed to create Voyage AI client for {agent_name}, please check API key configuration")
        
        model = self.get_model_for_agent(agent_name)
        
        for attempt in range(retry):
            try:
                # Use Voyage AI SDK
                result = voyage_client.embed(
                    texts=[text],
                    model=model,
                    input_type="document"  # Can be "document" or "query"
                )
                
                if result and result.embeddings:
                    return result.embeddings[0]
                else:
                    raise Exception("Voyage AI returned empty result")
                    
            except Exception as e:
                logger.error(f"[ERROR] Embedding failed (attempt {attempt + 1}/{retry}): {str(e)}")
                if attempt < retry - 1:
                    time.sleep(2 ** attempt)
        
        raise Exception(f"Embedding failed, retried {retry} times")

    def _embed_text_bailian(self, text: str, retry: int, agent_name: str) -> List[float]:
        """
        Use Aliyun Bailian OpenAI-compatible SDK for text embedding
        
        Args:
            text: input text
            retry: number of retries
            agent_name: agent name
            
        Returns:
            List[float]: embedding vector
        """
        profile = self._get_agent_profile(agent_name)
        api_key = profile.get('api_key')
        if not api_key:
            raise ValueError("Bailian API Key not configured")

        base_url = self._normalize_openai_base_url(self._get_bailian_base_url(agent_name))
        model = profile.get('embedding_model', self.default_model)
        client = self._get_openai_client(agent_name, api_key, base_url)

        for attempt in range(retry):
            try:
                result = client.embeddings.create(model=model, input=text)
                if result and result.data:
                    return result.data[0].embedding
                raise Exception("Bailian returned empty result")
            except Exception as e:
                logger.error(f"[ERROR] Embedding failed (attempt {attempt + 1}/{retry}): {str(e)}")
                if attempt < retry - 1:
                    time.sleep(2 ** attempt)

        raise Exception(f"Embedding failed, retried {retry} times")
    
    def _embed_text_openrouter(self, text: str, retry: int, agent_name: str = None) -> List[float]:
        """
        Use OpenRouter OpenAI-compatible SDK for text embedding
        
        Args:
            text: input text
            retry: number of retries
            agent_name: agent name
            
        Returns:
            List[float]: embedding vector
        """
        profile = self._get_agent_profile(agent_name)
        model = profile.get('embedding_model', self.default_model)
        base_url = profile.get('openai_base_url') or self._normalize_openai_base_url(profile.get('emb_url', self.base_url))
        api_key = profile.get('api_key')
        if not api_key:
            raise ValueError("OpenRouter API Key not configured")

        for attempt in range(retry):
            try:
                client = self._get_openai_client(agent_name or 'default', api_key, base_url)
                result = client.embeddings.create(model=model, input=text)
                if result and result.data:
                    return result.data[0].embedding
                raise Exception("OpenRouter returned empty result")
            except Exception as e:
                logger.error(f"[ERROR] Embedding failed (attempt {attempt + 1}/{retry}): {str(e)}")
                if attempt < retry - 1:
                    time.sleep(2 ** attempt)
        
        raise Exception(f"Embedding failed, retried {retry} times")
    
    def embed_batch(self, texts: List[str], batch_size: int = 10, show_progress: bool = True, agent_name: str = None) -> List[List[float]]:
        """
        Batch text embedding
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条
            agent_name: agent名称，用于选择对应的嵌入模型 
        
        Returns:
            List[List[float]]: list of embedding vectors
        """
        # Determine the model used for display and dimension retrieval
        if agent_name:
            model = self.get_model_for_agent(agent_name)
        else:
            model = self.default_model
        
        embeddings = []
        total_texts = len(texts)
        
        iterator = range(0, total_texts, batch_size)
        if show_progress:
            desc = f"Embedding progress [{model}]"
            iterator = tqdm(iterator, desc=desc, total=(total_texts + batch_size - 1) // batch_size)
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            
            for text in batch:
                try:
                    embedding = self.embed_text(text, agent_name=agent_name)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"\n[ERROR] Skipping text (index {len(embeddings)}): {str(e)}")
                    # Append zero vector when failure
                    embeddings.append([0.0] * self.get_embedding_dimension(model))
            
            # Avoid API rate limiting
            if i + batch_size < total_texts:
                time.sleep(0.5)
        
        return embeddings
    
    def get_embedding_dimension(self, model: str = None) -> int:
        """
        Get embedding dimension
        
        Args:
            model: model name (if None, use the default model)
        
        Returns:
            int: embedding dimension
        """
        use_model = model if model else self.default_model
        
        if 'voyage-3' in use_model.lower():
            return 1024
        elif 'large' in use_model.lower() or '3-large' in use_model.lower():
            return 3072
        elif 'small' in use_model.lower() or '3-small' in use_model.lower():
            return 1536
        elif 'gemini-embedding' in use_model.lower():
            return 3072
        elif 'embedding-v4' in use_model.lower():
            return 1024
        else:
            return 1536  


