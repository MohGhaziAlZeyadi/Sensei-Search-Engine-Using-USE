import os
import pickle
import tensorflow_hub as hub
import annoy

projected_dim = 64  #@param {type:"number"}
module_url = 'https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1' #@param {type:"string"}


embedding_dimension = projected_dim
index_filename = "index"
index = annoy.AnnoyIndex(embedding_dimension)
index.load(index_filename, prefault=True)
print('Annoy index is loaded.')
with open(index_filename + '.mapping', 'rb') as handle:
    mapping = pickle.load(handle)
print('Mapping file is loaded.')




def find_similar_items(embedding, num_matches=5):
    '''Finds similar items to a given embedding in the ANN index'''
    ids = index.get_nns_by_vector(
    embedding, num_matches, search_k=-1, include_distances=False)
    items = [mapping[i] for i in ids]
    return items


# Load the TF-Hub module
print("Loading the TF-Hub module...")
embed_fn = hub.load(module_url)
print("TF-Hub module is loaded.")

random_projection_matrix = None
if os.path.exists('random_projection_matrix'):
    print("Loading random projection matrix...")
    with open('random_projection_matrix', 'rb') as handle:
        random_projection_matrix = pickle.load(handle)
    print('random projection matrix is loaded.')

def extract_embeddings(query):
    '''Generates the embedding for the query'''
    query_embedding =  embed_fn([query])[0].numpy()
    if random_projection_matrix is not None:
        query_embedding = query_embedding.dot(random_projection_matrix)
    return query_embedding


test_embeddings = extract_embeddings("Hello Machine Learning!")
print(test_embeddings.shape)
print(test_embeddings[:10])



def similar_items(query, n = 5):
    """
    @param {type:"string"}
    """
    print(f"\nNEXT QUERY:\n{query}")
    query_embedding = extract_embeddings(query)
    items = find_similar_items(query_embedding, n)

    print("Results:")
    print("=========")
    for item in items:
        print(item)


query = ["LAND_ROVER NOISE FROM THE BACK OF THE CAR"]



for q in query:
     similar_items(q, n = 20)