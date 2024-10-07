import logging

from pymilvus import MilvusClient, CollectionSchema
from genai_utils import pretty_json

LOG = logging.getLogger('indexer')


def create_milvus_client(URI: str, overwrite: bool = False) -> MilvusClient:
    vec_store = MilvusClient(URI, overwrite=overwrite)
    return vec_store


def create_collection(vec_store: MilvusClient, name: str, schema: CollectionSchema,
                      vector_dim: int, delete_if_exists: bool = True,
                      auto_gen_id: bool = True,
                      consistency: str = "Strong", metric_type: str = "COSINE"):
    delete_collection_if_exists(vec_store, name) if delete_if_exists else None
    res = vec_store.create_collection(collection_name=name,
                                      schema=schema,
                                      dimension=vector_dim,
                                      consistency_level=consistency,
                                      auto_id=auto_gen_id,
                                      metric_type=metric_type)
    LOG.debug(res)


def log_collection(vec_store: MilvusClient, collection_name: str, index_name: str = None):
    collection_info = vec_store.describe_collection(collection_name)

    pretty_json(collection_info)
    # Print the schema details
    LOG.debug(f"Collection Name: {collection_info['collection_name']}")
    LOG.debug(f"Description: {collection_info['description']}")
    indexes = vec_store.list_indexes(collection_name)
    fields = collection_info['fields']
    LOG.debug(f"Field count:{len(fields)}")
    for field in fields:
        LOG.debug(f"  - Name: {field['name']}, DataType: {field['type']}, Is Primary: {field.get('is_primary', False)}")

    LOG.debug(f"Index(s) count: {len(indexes)}")
    for index in indexes:
        LOG.debug(f"Index: {index}")
        log_index(collection_name, index_name, vec_store)


def log_index(collection_name, index_name, vec_store):
    index_info = vec_store.describe_index(collection_name, index_name)
    LOG.debug(f"  - Index info: {pretty_json(index_info)}")


def delete_collection_if_exists(client: MilvusClient, collection_name: str):
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
        LOG.debug(f"Collection '{collection_name}' has been deleted.")
    else:
        LOG.debug(f"Collection '{collection_name}' does not exist.")


def create_index(vec_store: MilvusClient, coll_name: str, index_name: str, metric_type: str, vector_filed_name: str,
                 index_type: str,
                 sync: bool = True):
    # Set up the index parameters
    index_params = MilvusClient.prepare_index_params()
    # Add an index on the vector field.
    index_params.add_index(
        field_name=vector_filed_name,
        metric_type=metric_type,
        index_type=index_type,
        index_name=index_name
    )
    # 4.3. Create an index file
    vec_store.create_index(
        collection_name=coll_name,
        index_params=index_params,
        sync=sync  # Whether to wait for index creation to complete before returning. Defaults to True.
    )


