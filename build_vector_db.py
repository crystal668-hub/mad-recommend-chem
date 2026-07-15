"""
======================================================
Vector Database Construction Script
Function: Parse Markdown files using LlamaIndex to build a Chroma vector database
======================================================
"""

import sys
from pathlib import Path
from datetime import datetime
from utils.environment import load_project_environment

load_project_environment()

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.agent_config import AgentConfig
from database.literature_types import LITERATURE_TYPE_CONFIGS
from database.vector_store import VectorStore
from database.text_processor import TextProcessor
from database.embedder import MultiModelEmbedder
from utils.logger import Logger, setup_logging

def build_vector_database(
    config_path: str = "./config/config.yaml",
    data_dir: str = "./data/raw",
    literature_type_configs: dict = None,
    agent_name: str = "agent2",
    chunk_size: int = 256,
    chunk_overlap: int = 50
):
    """
    Build vector database
    Parse Markdown with LlamaIndex, create Document objects, chunk and index
    Use Chromadb for persistent storage
    
    Args:
        config_path: 配置文件路径
        data_dir: 原始数据目录，包含各 Literature Type 的子目录
        literature_type_configs: Literature Type 目录和 CSV 元数据配置
        agent_name: 使用的Agent配置名称
        chunk_size: 分块大小
        chunk_overlap: 分块重叠大小
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Load config first so we can also use its logging section.
    config = AgentConfig(config_path)

    # Configure unified run logging (per-run logs under ./logs/runs/<run_id>/).
    setup_logging(config.config, run_id=f"build_vector_db_{timestamp}")
    logger = Logger.get_logger("MAD.build_vector_db")

    logger.info("Starting Chroma vector database build", extra={"event": "vector_db.build.start"})
    
    if literature_type_configs is None:
        literature_type_configs = LITERATURE_TYPE_CONFIGS

    #--------------------------------------------------
    # 1. Load configuration
    #--------------------------------------------------
    
    logger.info("[Step 1/5] Loading configuration...")
    
    # Get corresponding Agent configuration
    llm_config = config.get_llm_config(agent_name)
    vector_config = config.get_vector_store_config()
    rag_config = config.get_rag_config()
    
    # Use chunk parameters 
    chunk_size = rag_config.get('chunk_size', chunk_size)
    chunk_overlap = rag_config.get('chunk_overlap', chunk_overlap)
    
    # Get all agent configuration dictionaries
    all_agent_configs = {
        'agent1': config.get_llm_config('agent1'),
        'agent2': config.get_llm_config('agent2'),
        'agent3': config.get_llm_config('agent3'),
        'agent4': config.get_llm_config('agent4')
    }
    
    # Set different collection_name based on agent name
    base_collection_name = vector_config.get('collection_name', 'chemical_reactions_recommendation')
    collection_name = f"{base_collection_name}_{agent_name}"
    
    logger.info(f"✓ Agent_used: {agent_name}")
    logger.info(f"✓ base_model: {llm_config.get('model')}")
    logger.info(f"✓ embedding_model: {llm_config.get('embedding_model')}")
    logger.info(f"✓ collection_name: {collection_name}")
    logger.info(f"✓ chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")
    
    #--------------------------------------------------
    # 2. Use LlamaIndex to load documents
    #--------------------------------------------------

    logger.info("\n[Step 2/5] Loading literature data...")
    processor = TextProcessor(data_dir)
    
    # Check data directory structure
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"\n✗ Data directory does not exist: {data_dir}")
        logger.error("  Please ensure each Literature Type has Markdown and CSV metadata:")
        for _, cfg in literature_type_configs.items():
            logger.error(f"    {data_dir}/{cfg['path']}/*.md")
            logger.error(f"    {cfg['metadata_csv']}")
        return
    
    documents = processor.load_literature_type_documents(
        base_dir=data_dir,
        literature_type_configs=literature_type_configs,
    )  
    
    logger.info(f"\n✓ Loaded {len(documents)} Document objects")
    
    if len(documents) == 0:
        logger.error("\n✗ No documents found, please check the data directory")
        logger.error("  Supported file format: .md")
        logger.error(f" Data directory: {data_dir}")
        return
    
    #--------------------------------------------------
    # 3. Chunk documents
    #--------------------------------------------------

    logger.info("\n[Step 3/5] Chunking documents...")
    chunked_documents = processor.chunk_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    logger.info(f"✓ Number of chunked documents: {len(chunked_documents)}")
    
    #--------------------------------------------------
    # 4. Initialize embedder
    #--------------------------------------------------

    logger.info("\n[Step 4/5] Initializing embedder and generating embeddings...")
    try:
        embedder = MultiModelEmbedder(llm_config, agent_configs=all_agent_configs)
    except ValueError as e:
        logger.error(f"\n✗ Initialization failed: {str(e)}")
        return
    
    # Use doc attributes to get text and metadata
    texts = [doc.text for doc in chunked_documents]
    metadatas = [doc.metadata for doc in chunked_documents]
    
    logger.info(f"✓ Preparing to embed {len(texts)} texts")
    
    # Generate embeddings
    logger.info("\nStarting embedding...")
    try:
        embeddings = embedder.embed_batch(texts, batch_size=10, show_progress=True, agent_name=agent_name)
    except Exception as e:
        logger.error(f"\n✗ Embedding failed: {str(e)}")
        return
    
    logger.info(f"\n✓ Successfully generated {len(embeddings)} embeddings")
    logger.info(f"✓ Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
    
    #--------------------------------------------------
    # 5. Store to Chroma vector database
    #--------------------------------------------------

    logger.info("\n[Step 5/5] Storing to Chroma vector database...")
    
    vector_store = VectorStore(
        persist_directory=vector_config.get('persist_directory', './data/chroma_db'),
        collection_name=collection_name,  
        embedding_function=None  # Use precomputed embeddings 
    )
    
    # Clear existing data (optional)
    current_count = vector_store.get_collection_count()
    if current_count > 0:
        prompt = f"\nCollection already has {current_count} documents. Clear it? (y/n): "
        logger.info(prompt)
        user_input = input(prompt)
        if user_input.lower() == 'y':
            vector_store.reset_collection()
            logger.info("✓ Collection cleared")
    
    # Batch add documents and embeddings
    try:
        vector_store.add_documents(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
    except Exception as e:
        logger.error(f"\n✗ Failed to store: {str(e)}")
        return
    
    logger.info("✓ Successfully stored to Chroma database")
    logger.info(f"✓ Collection name: {collection_name}")
    logger.info(f"✓ Persist directory: {vector_config.get('persist_directory')}")
    logger.info(f"✓ Document count: {vector_store.get_collection_count()}\n")
    logger.info("=" * 60)
    logger.info("Chroma DB build complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    build_vector_database(
        config_path="./config/config.yaml",
        data_dir="./data/raw",              
        literature_type_configs=LITERATURE_TYPE_CONFIGS,
        agent_name="agent2",                
        chunk_size=256,                     
        chunk_overlap=50                    
    )
