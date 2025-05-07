# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Spacy
#     language: python
#     name: spacy
# ---

# +
import json
import spacy
import numpy as np
import glob
import os
from tqdm import tqdm
import scipy

# Load the spaCy model with word vectors
# You can use 'en_core_web_md' or 'en_core_web_lg' for better word vectors
# -

vectors = np.load("glove/word2vec-google-news-300.model.vectors.npy")

# +
# import gensim
# model = gensim.models.KeyedVectors.load_word2vec_format(
#     "glove/word2vec-google-news-300.model", binary=True
# )
# import pickle
# xx = pickle.load(open("glove/word2vec-google-news-300.model", "rb"))

# +

# nlp = spacy.load('en_core_web_md')
nlp_lg = spacy.load('en_core_web_lg')
nlp_tf = spacy.load('en_core_web_trf')

nlp("sonorism").vector
nlp_lg("sonorism").vector
# -

nlp = spacy.load("glove/300d/glove.840B.300d.forSpacy/")

sim = [
    ["chord", "chords"],
    ["dance music", "music"],
    ["rock music", "folk music"]
]
for (x,y) in sim:
    print(f"{x}, {y}: ", end="\t")
    print(np.log10(scipy.spatial.distance.cosine(
        nlp(x).vector,
        nlp(x).vector
    )))

# +
mainF = "/scratch/gpfs/jl8975/jlanglieb/tmp/graph/mto-project-admin/mto/data/"
# make sure ends in /


json_files = glob.glob(os.path.join(mainF, "*.json"))
for FN in tqdm(json_files):
    # FN = "mto.16.22.3.murphy.json"
    # Read the JSON file
    with open(mainF+os.path.basename(FN), 'r') as file:
        data = json.load(file)
    
    # Extract keywords
    keywords = data.get('keywords', [])
    
    # Create a dictionary to store keyword embeddings
    keyword_embeddings = {}
    
    # Process each keyword and get its vector representation
    for keyword in keywords:
        # Process the keyword with spaCy
        doc = nlp(keyword)

        
        # Get the vector for the entire keyword phrase
        # For multi-word keywords, this will be the average of the word vectors
        vector = doc.vector

        assert np.all(np.isclose(np.sum([v.vector for v in doc], axis=0)/len(doc), doc.vector))

        # Remove punctuation
        nonQuotVec = [v for v in doc if v.is_punct != True]
        vector = np.sum([v.vector for v in nonQuotVec], axis=0)/len(nonQuotVec)
        
        # if any of the tokens are 0, set whole thing to 0
        # Klumpenhouwer networks === networks otherwise!
        if np.any([np.sum(v.vector) == 0 for v in nonQuotVec]):
            vector = vector * 0
        
        # Store the vector in the dictionary
        keyword_embeddings[keyword] = vector.tolist()
    
    # Print the results
    # for keyword, embedding in keyword_embeddings.items():
    #     print(f"Keyword: {keyword}")
    #     print(f"Vector shape: {np.array(embedding).shape}")
    #     print(f"Vector: {np.array(embedding)[:5]}...")  # Show first 5 elements
    #     print("-" * 50)
    
    # Optionally save the embeddings to a new JSON file
    output_data = data.copy()
    output_data['keyword_embeddings'] = keyword_embeddings
    
    with open(mainF+"wEmbeddings/"+os.path.basename(FN)+".wEmbeddings.json", 'w') as file:
        json.dump(output_data, file, indent=4)
    
    print(f"Processed {len(keywords)} keywords and saved embeddings to out_with_embeddings.json")
# -


