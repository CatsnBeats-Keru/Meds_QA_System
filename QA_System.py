##### 0. Importing Libraries & API Key #####

import os
import pandas as pd
import numpy as np
import streamlit as st

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'



##### 1. Dataset Preprocessing #####

# Loading the dataset
df = pd.read_csv('drugsComTest_raw.csv')

# Pick top rated medicines & delete rows that are unused/null
indexInvalid = df[(df['rating'] < 7) | df['condition'].isnull()].index       # Grouping rows and columns that are going to be dropped into 'indexInvalid'
df.drop(indexInvalid, inplace=True)                                          # Dropping indexInvalid
df = df.drop(['review', 'uniqueID', 'date', 'usefulCount'], axis=1)          # Dropping 'review', 'uniqueID', 'date', and 'usefulCount' column
df.reset_index(drop=True, inplace=True)                                      # Resetting index

# Creating context column
new_row = []                                                                                                      # Define an empty array
for i in range(len(df)):                                                                                          # Insert context sentences to the array
  new_row.append(f'This medicine is called {df["drugName"][i]} and is used for treating {df["condition"][i]}')
df['Context'] = np.array(new_row)                                                                                 # Create new column for the array

# Saving the modified dataset
df.to_csv('drugstestnew.csv', index=False, encoding="utf-8")



##### 2. Dataset Embedding & Vectorising #####

# Splitting & embedding
loader = CSVLoader('drugstestnew.csv', encoding="utf-8", csv_args={'delimiter': ','})       # Load the edited dataset
rows = loader.load_and_split()                                                              # Split into rows
embeddings = HuggingFaceEmbeddings()                                                        # Apply the embedding

# Vectorising with ChromaDB
vec = Chroma.from_documents(rows, collection_name='drugstestnew')                           # Load dataset into ChromaDB (Vectorstore)



##### 3. Establishing the LLM & Interface #####

# Establishing the LLM
hub_llm = HuggingFaceHub(                                           
    repo_id='google/flan-t5-xl',
    model_kwargs={'temperature':0, "max_length":512},
    verbose=True
)

# Establishing QA chain
chain = load_qa_chain(hub_llm, chain_type="stuff")                  

# Establishing Streamlit UI
st.title("Drug Consultation Testing")                               # Title
prompt = st.text_input('Input your question here')                  # Textbox input for the user

if prompt:                                                          # If hits enter
    docs = vec.similarity_search(prompt)                            # Swap raw LLM with the dataset
    response = chain.run(input_documents=docs, question=prompt)
    st.write(response)                                              # python -m streamlit run QA_System.py