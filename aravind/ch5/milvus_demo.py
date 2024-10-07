import json
import random

from pymilvus import MilvusClient

client = MilvusClient("/Users/O60774/Downloads/vec_stores/milvus-demo.db", overwrite=True)

client.create_collection(
    collection_name="quick_setup",
    dimension=5,
    metric_type="IP"
)

colors = ["green", "blue", "yellow", "red", "black", "white", "purple", "pink", "orange", "brown", "grey"]
data = []

for i in range(1000):
    current_color = random.choice(colors)
    data.append({
        "id": i,
        "vector": [random.uniform(-1, 1) for _ in range(5)],
        "color": current_color,
        "color_tag": f"{current_color}_{str(random.randint(1000, 9999))}"
    })

res = client.insert(
    collection_name="quick_setup",
    data=data
)

print(res)

# client.create_partition(
#     collection_name="quick_setup",
#     partition_name="red"
# )
#
# client.create_partition(
#     collection_name="quick_setup",
#     partition_name="blue"
# )

red_data = [{"id": i, "vector": [random.uniform(-1, 1) for _ in range(5)], "color": "red",
             "color_tag": f"red_{str(random.randint(1000, 9999))}"} for i in range(500)]
blue_data = [{"id": i, "vector": [random.uniform(-1, 1) for _ in range(5)], "color": "blue",
              "color_tag": f"blue_{str(random.randint(1000, 9999))}"} for i in range(500)]

res = client.insert(
    collection_name="quick_setup",
    data=red_data
)

print(res)

res = client.insert(
    collection_name="quick_setup",
    data=blue_data
)

print(res)

# Single vector search
res = client.search(
    collection_name="quick_setup", # Replace with the actual name of your collection
    # Replace with your query vector
    data=[[0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592]],
    limit=5, # Max. number of search results to return
    search_params={"metric_type": "IP", "params": {}} # Search parameters
)

# Convert the output to a formatted JSON string
result = json.dumps(res, indent=4)
print(result)