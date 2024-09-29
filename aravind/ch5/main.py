from aravind.ch5.embedder import get_embeddings, get_embeddings_for_all_chunks, EmbeddingResult
from aravind.ch5.loader import get_docs_for_all_paths
from genai_utils import print_elapsed_time
from loader import install_nltk
from chunker import get_chunks_for_all_docs
from langchain_milvus import Milvus
from langchain.vectorstores import Milvus
from pymilvus import MilvusClient
import time
import logging
from pymilvus import CollectionSchema, FieldSchema, DataType

install_nltk()

# Configure logging
logging.basicConfig(filename='/Users/O60774/Downloads/logs/python-app.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s %(name)s - %(message)s')
LOG = logging.getLogger('aravind.ch5')

logging.getLogger('aravind.ch5').setLevel(logging.DEBUG)
logging.getLogger('embedder').setLevel(logging.ERROR)
logging.getLogger('loader').setLevel(logging.ERROR)
logging.getLogger('chunker').setLevel(logging.ERROR)
logging.getLogger('genai_utils').setLevel(logging.DEBUG)

# Start process
git_repos = ["/Users/O60774/ws/learn-spark", "/Users/O60774/ws/old-gitbook-spark-notes",
             "/Users/O60774/ws/learn-functional-domain"]

start_time = time.time()
all_docs = get_docs_for_all_paths(git_repos)
LOG.debug(f"Total Markdown file count: {len(all_docs)}")
print_elapsed_time(start_time, "Load documents")

start_time = time.time()
all_chunks = get_chunks_for_all_docs(all_docs)
LOG.debug(f"Total chunk count: {len(all_chunks)}")
print_elapsed_time(start_time, "Create chunks")

start_time = time.time()
vectors = []
for chunk in all_chunks:
    vec = get_embeddings(chunk)
    vectors.append(vec)
LOG.debug(f"Vector length: {len(vectors)}")

print_elapsed_time(start_time, "Create vectors")

# LOG.debug(f"all chunk embedding count: {len(get_embeddings_for_all_chunks(all_chunks))}")
# er: EmbeddingResult = vectors[0]
# LOG.debug(type(er))

start_time = time.time()
URI = "/Users/O60774/Downloads/milvus-vector-store.db"
EMBEDDING_DIM = 768
vec_store = MilvusClient(URI)
print_elapsed_time(start_time, "Create MilvusClient")

# Define the fields
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=10000)
]

# Create the schema
schema = CollectionSchema(fields, description="Custom schema for EmbeddingResult")

# STEP 3. CREATE A MILVUS COLLECTION AND DEFINE THE DATABASE INDEX.
# Uses Milvus AUTOINDEX, which defaults to HNSW.
start_time = time.time()
COLLECTION_NAME = "col_apache_spark"
vec_store.create_collection(collection_name=COLLECTION_NAME,
                            schema=schema,
                            dimension=EMBEDDING_DIM,
                            consistency_level="Eventually",
                            auto_id=True,
                            overwrite=True)
print_elapsed_time(start_time, "Create collection")
print(vec_store.get_collection_stats(COLLECTION_NAME))


def print_collection(collection_name: str):
    collection_info = vec_store.describe_collection(collection_name)

    print(collection_info)
    # Print the schema details
    print(f"Collection Name: {collection_info['collection_name']}")
    print(f"Description: {collection_info['description']}")
    print("Fields:")
    for field in collection_info['fields']:
        print(f"  - Name: {field['name']}, DataType: {field['type']}, Is Primary: {field.get('is_primary', False)}")


print_collection(COLLECTION_NAME)

start_time = time.time()
data = [
    {
        # "id": i,
        "vector": vectors[i].vector,
        "source": vectors[i].source,
        "chunk": vectors[i].chunk
    }
    for i in range(len(vectors))
]
print_elapsed_time(start_time, f"Convert EmbeddingResult to schema of the collection: {COLLECTION_NAME}")

# Add the embedding vector to the Milvus vector store:
start_time = time.time()
vec_store.insert(COLLECTION_NAME, data=data, progress_bar=True)
print_elapsed_time(start_time, f"Insert vectors{len(vectors)}")

print(vec_store.get_collection_stats(COLLECTION_NAME))

vec_store.close()
