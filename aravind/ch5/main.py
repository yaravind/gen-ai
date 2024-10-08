from langchain.chains.llm import LLMChain
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI

from aravind.ch5.embedder import get_embeddings
from aravind.ch5.indexer import create_milvus_client, create_collection, log_collection, create_index
from aravind.ch5.loader import get_docs_for_all_paths
from aravind.ch5.searcher import search_for, count_result_set, get_top_k
from genai_utils import log_elapsed_time, pretty_json
from loader import install_nltk
from chunker import get_chunks_for_all_docs
import time
import logging
from pymilvus import CollectionSchema, FieldSchema, DataType
from config import set_environment
import pymilvus

# Setup
set_environment()
install_nltk()
# Print the Milvus library version
print(f"Milvus library version: {pymilvus.__version__}")

# Define the logfmt format
FORMAT = '[%(asctime)s] %(levelname)s %(name)s - %(message)s'

# Configure logging
logging.basicConfig(filename='/Users/O60774/Downloads/logs/python-app.log', level=logging.ERROR,
                    format=FORMAT)
LOG = logging.getLogger('aravind.ch5')

logging.getLogger('pymilvus').setLevel(logging.ERROR)

logging.getLogger('aravind.ch5').setLevel(logging.DEBUG)
logging.getLogger('loader').setLevel(logging.ERROR)
logging.getLogger('embedder').setLevel(logging.ERROR)
logging.getLogger('chunker').setLevel(logging.ERROR)
logging.getLogger('indexer').setLevel(logging.DEBUG)
logging.getLogger('searcher').setLevel(logging.DEBUG)
logging.getLogger('genai_utils').setLevel(logging.DEBUG)

# Start process
git_repos = ["/Users/O60774/ws/learn-spark", "/Users/O60774/ws/old-gitbook-spark-notes",
             "/Users/O60774/ws/learn-functional-domain"]

start_time = time.time()
all_docs = get_docs_for_all_paths(git_repos)
LOG.debug(f"Total Markdown file count: {len(all_docs)}")
log_elapsed_time(start_time, "Load documents")

start_time = time.time()
all_chunks = get_chunks_for_all_docs(all_docs)
LOG.debug(f"Total chunk count: {len(all_chunks)}")
log_elapsed_time(start_time, "Create chunks")

start_time = time.time()
vectors = []
for chunk in all_chunks:
    vec = get_embeddings(chunk)
    vectors.append(vec)
LOG.debug(f"Vector length: {len(vectors)}")

log_elapsed_time(start_time, "Create vectors")

# LOG.debug(f"all chunk embedding count: {len(get_embeddings_for_all_chunks(all_chunks))}")
# er: EmbeddingResult = vectors[0]
# LOG.debug(type(er))

start_time = time.time()
URI = "/Users/O60774/Downloads/vec_stores/milvus-vector-store.db"
vec_store = create_milvus_client(URI, overwrite=True)
log_elapsed_time(start_time, "Create MilvusClient")

# STEP 3. CREATE A MILVUS COLLECTION AND DEFINE THE DATABASE INDEX.

# Create the schema
EMBEDDING_DIM = 768
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=10000)
]
schema = CollectionSchema(fields, description="Custom schema for EmbeddingResult")
start_time = time.time()
COLLECTION_NAME = "col_apache_spark"
INDEX_NAME = "idx_COSINE"
METRIC_TYPE = "COSINE"
create_collection(vec_store, name=COLLECTION_NAME, schema=schema, vector_dim=EMBEDDING_DIM, delete_if_exists=True,
                  auto_gen_id=True,
                  metric_type=METRIC_TYPE)
log_elapsed_time(start_time, "Create collection")

log_collection(vec_store, COLLECTION_NAME, INDEX_NAME)

# 4. Create index
start_time = time.time()
create_index(vec_store, COLLECTION_NAME, INDEX_NAME, METRIC_TYPE, vector_filed_name="vector", index_type="FLAT")
log_elapsed_time(start_time, "Create index")
log_collection(vec_store, COLLECTION_NAME, INDEX_NAME)

start_time = time.time()
data = [
    {
        # "id": i, Id is auto-created so no need to submit as part of the data
        "vector": vectors[i].vector,
        "source": vectors[i].source,
        "chunk": vectors[i].chunk
    }
    for i in range(len(vectors))
]
log_elapsed_time(start_time, "Convert EmbeddingResult to schema")

# Add the embedding vector to the Milvus vector store:
start_time = time.time()
vec_store.insert(COLLECTION_NAME, data=data, progress_bar=True)
log_elapsed_time(start_time, f"Insert {len(vectors)} vectors")

# Single-vector search
start_time = time.time()
user_question = "explain shuffle in spark"
res = search_for(user_question, vec_store, COLLECTION_NAME, METRIC_TYPE)
log_elapsed_time(start_time, f"Single-vector search")
LOG.debug(f"Search results count: {count_result_set(res)}")

# for re in get_top_k(res, 3):
#     print(pretty_json(re))

# Generate
prompt_text = """
System Instruction: You are a highly intelligent and responsive AI language model designed to assist with answering 
questions based on provided context. Here's the information you need:

### Retrieved Chunks:
{context}

### User Question:
{user_question}

### Response:
Using the context provided in the retrieved chunks, answer the user's question comprehensively and accurately.
"""
context_chunks = get_top_k(res, 2)

context = "\n---\n".join(
    [f"chunk {i}: {chunk_obj['chunk'].replace('\n\n', '\n').replace('\r\r', '\n')}" for i, chunk_obj in
     enumerate(context_chunks)])

# print("-" * 150)
# print(f"Context:\n {context}")

prompt_template = PromptTemplate.from_template(template=prompt_text)
#partial_prompt = prompt_template.partial(user_question=user_question)

# Interpolate each element of the list into the placeholders in the prompt
# for item in get_top_k(res, 2):
#     interpolated_text = prompt_template.format(context=context, user_question=user_question)
#     print(interpolated_text)
#     print("-" * 50)  # Separator for clarity

# Setup LLM and QA chain; set temperature low to keep hallucinations in check
LLM = ChatOpenAI(
    model_name="gpt-3.5-turbo", temperature=0, streaming=False
)

llm_chain = LLMChain(prompt=prompt_template, llm=LLM, verbose=False)
start_time = time.time()
answer = llm_chain.run({"context": context, "user_question": user_question})
log_elapsed_time(start_time, f"Generate response")

LOG.debug(f"Question: {user_question}")
LOG.debug(f"Answer  : {answer}")
print(answer)

# Shutdown
LOG.debug("Milvus client is connected. Close the connection.")
vec_store.close()
