from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import numpy as np
import ollama

#  1: Connect to Milvus Lite with a local db
connections.connect("default", uri="./milvus_demo.db")

# 2: Create collection in Milvus
dim = 768
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500)
]
schema = CollectionSchema(fields, "nomic-embed-text")
collection = Collection("nomic_embed_collection", schema)

# 3: Generate embeddings with Ollama and nomic-embed-text
def generate_embedding(text):
   response = ollama.embed(model="nomic-embed-text", input=text)
   print(text, " ", response, "\n")
   return response["embeddings"][0]

# Exaple texts
texts = [
  "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
  "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
  "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
  "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
  "Llamas are vegetarians and have very efficient digestive systems",
  "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
]

# Generate embeddings and stor in Milvus
embeddings = [generate_embedding(text) for text in texts]
entities = [
    embeddings,  # vectors
    texts  # texts
]
collection.insert(entities)

# 4: Create index for searching
index_params = {
    "index_type": "IVF_FLAT",  # Use IVF_FLAT
    "metric_type": "L2",  # Use L2 (Euclidean distance)
    "params": {"nlist": 128}  # clasters count
}
collection.create_index("embedding", index_params)

# 5: Load collection in memmory
collection.load()

# 6: Search by embeddings
query_text = "What animals are llamas related to?"
query_embedding = generate_embedding(query_text)

search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10}  # clasters count for searching
}

results = collection.search(
    data=[query_embedding],  # vector request
    anns_field="embedding",  # a field for searching
    param=search_params,
    limit=3,  # count of close results
    output_fields=["text"]  # original texts
)

# 7: Generate answers
def generate_response(results, distance_threshold=0.5):
    response = "Relevant answers:\n"
    for hits in results:
        for hit in hits:
            if hit.distance < distance_threshold:  # check a distance
                response += f"- {hit.entity.get('text')} (distance: {hit.distance:.2f})\n"
            else:
                response += "- No relevant answer.\n"
    return response

print(f"Question: {query_text}")
# Print results
print(generate_response(results))

# 8: Clear a collection
collection.drop()