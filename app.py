import streamlit as st
import  time
import config
import json
import os
import sys
import uuid
import datetime
import glob
import time
import uuid
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
from dotenv import dotenv_values
from openai import AzureOpenAI
from azure.core.exceptions import AzureError
from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos import ThroughputProperties
from azure.cosmos import ThroughputProperties
from azure.cosmos import CosmosClient, PartitionKey
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import io


load_dotenv()





# specify the name of the .env file name 
env_name = "example.env" # following example.env template change to your own .env file name
config = dotenv_values(env_name)
# Azure Cosmos DB connection details
HOST = config['cosmos_host']
key = config['cosmos_key']



# Azure OpenAI connection details
openai_endpoint = config['openai_endpoint']
openai_key = config['openai_key']
openai_version = config['openai_version']
openai_embeddings_model = config['openai_embeddings_deployment']
openai_chat_model = config['AZURE_OPENAI_CHAT_MODEL']



dbsource = config['cosmosdbsourcedb'] 
cachecol = config['cosmosdbcachecol']

# Azure Storage Account settings
storage_account_name = config.get('storage_account_name', '')
storage_account_key = config.get('storage_account_key', '')
storage_connection_string = config.get('storage_connection_string', '')


# Create the OpenAI client
openai_client = AzureOpenAI(
  api_key = openai_key,  
  api_version = openai_version,  
  azure_endpoint =openai_endpoint 
)



ENDPOINT =  config['cosmos_host']
client = CosmosClient(ENDPOINT, key)



def get_completion(openai_client, model, prompt: str):    
   
    response = openai_client.chat.completions.create(
        model = model,
        messages =   prompt,
        temperature = 0.1
    )   
    return response.model_dump()


def chat_completion(user_message):
    # Dummy implementation of chat_completion
    # Replace this with the actual implementation
    response_payload = f"Response to: {user_message}"
    cached = False
    return response_payload, cached

def generate_embeddings(openai_client, text):
    """
    Generates embeddings for a given text using the OpenAI API v1.x
    """
    
    response = openai_client.embeddings.create(
        input = text,
        model= openai_embeddings_model
    
    )
    embeddings = response.data[0].embedding
    return embeddings

def get_similar_docs(openai_client, db, collection, query_text, limit, sim, typesearch):
    """
    Performs a search against the specified collection using different search methods.
    
    Args:
        openai_client: The OpenAI client for generating embeddings
        db: The database name
        collection: The collection name to search in
        query_text: The text to search for
        limit: Maximum number of results to return
        sim: Similarity threshold
        typesearch: Type of search to perform ('vector', 'full text', or 'hybrid')
        
    Returns:
        A list of documents that match the search criteria
    """
 
    mydbt = client.get_database_client(db)   
    cvector = mydbt.get_container_client(collection)
    
    if typesearch == "vector":
            query_vector = generate_embeddings(openai_client, query_text)
            query = f"""
                SELECT TOP @num_results  c.id,c.source, VectorDistance(c.embedding, @embedding) as SimilarityScore 
                FROM c
                WHERE VectorDistance(c.embedding,@embedding) > @similarity_score
                ORDER BY VectorDistance(c.embedding,@embedding)
            """
            results = cvector.query_items(
                query=query,
                parameters=[
                    {"name": "@embedding", "value": query_vector},
                    {"name": "@num_results", "value": limit},
                    {"name": "@similarity_score", "value": sim}
                ],
                enable_cross_partition_query=True, populate_query_metrics=True
            )   
          
         
    elif typesearch == "full text":
           
            query = f"""
                SELECT TOP @num_results  c.id,c.source 
                FROM c
                WHERE  FullTextContainsAll(c.text, @query_text)
                
            """
           
            results = cvector.query_items(
                query=query,
                parameters=[
                    {"name": "@query_text", "value": query_text},
                    {"name": "@num_results", "value": limit}
                ],
                enable_cross_partition_query=True, populate_query_metrics=True
            )
         
          
    elif typesearch == "hybrid":
        try:
            query_vector = generate_embeddings(openai_client, query_text)
            query_text_split = query_text.split()
          
            
            
            query = f"""
                SELECT TOP """+ str(limit)+"""  c.id,c.source,c.text 
                FROM c
                ORDER BY RANK RRF(VectorDistance(c.embedding,"""+ str(query_vector)+"""), FullTextScore(c.text, """+ str(query_text_split)+"""))
            """
           
            results = list(cvector.query_items(
                query=query,
                enable_cross_partition_query=True
            ))
       
        except Exception as e:
            print(f"Error during hybrid search query execution: {str(e)}")
            results = []
         
            
    products = []      
   
    if results and typesearch != "hybrid" : 
 
        for a in results:
           
            source = a['source']
            id = a['id']    
     
               
            mycolt = mydbt.get_container_client(source)  
            response = mycolt.read_item(item=id, partition_key=id)
            res = response.get('text')
            products.append({"text": res})
            
  
        
    else:
    
        
        web_tests_list = []
        # Iterate through the pages and append the items to the list
        for web_test in results:
            web_tests_list.append(web_test)
       
        
        for a in web_tests_list:
            text = a['text']
            products.append({"text": text})
            
     
        
        
        
         
    return products


def get_chat_history( username,completions=1):
    
   
    mydbt = client.get_database_client(dbsource)   
    container = mydbt.get_container_client(cachecol)
    
    results = container.query_items(
        query= '''
        SELECT TOP @completions *
        FROM c
        where c.name = @username
        ORDER BY c._ts DESC
        ''',
        parameters=[
            {"name": "@completions", "value": completions},
            {"name": "@cusername", "value": username},
        ], enable_cross_partition_query=True)
    results = list(results)
    return results


def list_collections(database_name=None):
    """
    Lists all collections/containers in the specified CosmosDB database.
    If no database name is provided, it uses the default database from config.
    Excludes the 'cache' collection from the results.
    
    Args:
        database_name (str, optional): Name of the database to list collections from. 
                                       Defaults to the dbsource from config.
    
    Returns:
        list: A list of collection names in the specified database (excluding 'cache').
    """
    try:
        # If no database name is provided, use the default database from config
        if database_name is None:
            database_name = dbsource
            
        # Get the database client
        db_client = client.get_database_client(database_name)
        
        # List all containers in the database
        containers = list(db_client.list_containers())
        
        # Extract the container IDs (names), excluding the 'cache' collection
        container_names = [container['id'] for container in containers if container['id'] != cachecol]
        
        return container_names
    
    except Exception as e:
        print(f"Error listing collections: {str(e)}")
        return []


# Azure Storage Functions
def get_blob_service_client(connection_string=None):
    """
    Get a Blob Service client using the storage connection string.
    
    Args:
        connection_string (str, optional): The storage account connection string.
            If not provided, uses the one from config.
            
    Returns:
        BlobServiceClient: The Blob Service client object or None if error occurs
    """
    try:
        conn_str = connection_string or storage_connection_string
        if not conn_str:
            return None
            
        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        return blob_service_client
    except Exception as e:
        print(f"Error creating blob service client: {str(e)}")
        return None

def list_containers():
    """
    List all containers in the Azure Storage account.
    
    Returns:
        list: A list of container names in the storage account or empty list if error
    """
    try:
        # Get a blob service client
        blob_service_client = get_blob_service_client()
        if not blob_service_client:
            return []
            
        # List containers
        containers = list(blob_service_client.list_containers())
        
        # Extract container names
        container_names = [container.name for container in containers]
        
        return container_names
    except Exception as e:
        print(f"Error listing containers: {str(e)}")
        return []

def list_blobs(container_name):
    """
    List all blobs in a specific container.
    
    Args:
        container_name (str): The name of the container to list blobs from
        
    Returns:
        list: A list of blob names in the container or empty list if error
    """
    try:
        # Get a blob service client
        blob_service_client = get_blob_service_client()
        if not blob_service_client:
            return []
            
        # Get a container client
        container_client = blob_service_client.get_container_client(container_name)
        
        # List blobs
        blobs = list(container_client.list_blobs())
        
        # Extract blob names and properties
        blob_info = [{
            'name': blob.name, 
            'size': blob.size, 
            'creation_time': blob.creation_time,
            'last_modified': blob.last_modified,
            'content_type': blob.content_settings.content_type if blob.content_settings else None
        } for blob in blobs]
        
        return blob_info
    except Exception as e:
        print(f"Error listing blobs in {container_name}: {str(e)}")
        return []

def upload_file_to_container(container_name, file_name, file_content):
    """
    Upload a file to a specific container in Azure Blob Storage.
    
    Args:
        container_name (str): The name of the container to upload the file to
        file_name (str): The name to give the file in the container
        file_content (bytes or BinaryIO): The content of the file to upload
        
    Returns:
        bool: True if upload successful, False otherwise
        str: URL of the uploaded blob or error message
    """
    try:
        # Get a blob service client
        blob_service_client = get_blob_service_client()
        if not blob_service_client:
            return False, "Could not create blob service client. Check your connection string."
            
        # Get a container client and create the container if it doesn't exist
        container_client = blob_service_client.get_container_client(container_name)
        if not container_client.exists():
            container_client.create_container()
            
        # Get a blob client
        blob_client = container_client.get_blob_client(file_name)
        
        # Upload the file
        blob_client.upload_blob(file_content, overwrite=True)
        
        # Get the URL of the blob
        blob_url = f"https://{storage_account_name}.blob.core.windows.net/{container_name}/{file_name}"
        
        return True, blob_url
    except Exception as e:
        error_message = f"Error uploading {file_name} to {container_name}: {str(e)}"
        print(error_message)
        return False, error_message

def create_container(container_name):
    """
    Create a new container in the Azure Storage account.
    
    Args:
        container_name (str): The name of the container to create
        
    Returns:
        bool: True if creation successful, False otherwise
        str: Success message or error message
    """
    try:
        # Get a blob service client
        blob_service_client = get_blob_service_client()
        if not blob_service_client:
            return False, "Could not create blob service client. Check your connection string."
            
        # Get a container client
        container_client = blob_service_client.get_container_client(container_name)
        
        # Create the container if it doesn't exist
        if not container_client.exists():
            container_client.create_container()
            return True, f"Container '{container_name}' created successfully"
        else:
            return True, f"Container '{container_name}' already exists"
    except Exception as e:
        error_message = f"Error creating container {container_name}: {str(e)}"
        print(error_message)
        return False, error_message

def cachesearch( vectors, username,similarity_score , num_results):
    # Execute the query
   
    mydbt = client.get_database_client(dbsource)   
    container = mydbt.get_container_client(cachecol)
    
    results = container.query_items(
        query= '''
        SELECT TOP @num_results  c.completion, VectorDistance(c.vector, @embedding) as SimilarityScore 
        FROM c
        WHERE VectorDistance(c.vector,@embedding) > @similarity_score and c.name = @usernames
        ORDER BY VectorDistance(c.vector,@embedding)
        ''',
        parameters=[
            {"name": "@embedding", "value": vectors},
            {"name": "@num_results", "value": num_results},
            {"name": "@usernames", "value": username},
            {"name": "@similarity_score", "value": similarity_score}
        ],
        enable_cross_partition_query=True, populate_query_metrics=True)
   
    formatted_results = []
    for result in results:
     
        formatted_results.append(result)

  
    return formatted_results


def cacheresponse(user_prompt, prompt_vectors, response, username):
    
   
    mydbt = client.get_database_client(dbsource)   
    container = mydbt.get_container_client(cachecol)
    
    docu = [user_prompt]
    
    # Create a dictionary representing the chat document
    chat_document = {
        'id':  str(uuid.uuid4()),  
        'prompt': user_prompt,
        'completion': response['choices'][0]['message']['content'],
        'completionTokens': str(response['usage']['completion_tokens']),
        'promptTokens': str(response['usage']['prompt_tokens']),
        'totalTokens': str(response['usage']['total_tokens']),
        'model': response['model'],
        'name': username,
        'vector': prompt_vectors
    }
    # Insert the chat document into the Cosmos DB container
    container.create_item(body=chat_document)
 
 
def createcachecollection():
    mydbt = client.get_database_client(dbsource)   
      
# Create the vector embedding policy to specify vector details
    vector_embedding_policy = {
    "vectorEmbeddings": [ 
        { 
            "path":"/vector" ,
             "dataType":"float32",
            "distanceFunction":"cosine",
            "dimensions":1536
        }, 
    ]
}

# Create the vector index policy to specify vector details
    indexing_policy = { 
    "vectorIndexes": [ 
        {
            "path": "/vector", 
            "type": "diskANN"
            
        }
    ]
    } 
   
  
# Create the cache collection with vector index
    try:
        mydbt.create_container_if_not_exists( id=cachecol, 
                                            partition_key=PartitionKey(path='/id'),
                                            offer_throughput=ThroughputProperties(auto_scale_max_throughput=1000, auto_scale_increment_percent=0), 
                                            indexing_policy=indexing_policy,
                                            vector_embedding_policy=vector_embedding_policy
                                            ) 
 

    except exceptions.CosmosHttpResponseError: 
        raise 
   
def clearcache ():
   
 
    mydbt = client.get_database_client(dbsource)   
  
    
      
# Create the vector embedding policy to specify vector details
    vector_embedding_policy = {
    "vectorEmbeddings": [ 
        { 
            "path":"/vector" ,
             "dataType":"float32",
            "distanceFunction":"cosine",
            "dimensions":1536
        }, 
    ]
}

# Create the vector index policy to specify vector details
    indexing_policy = { 
    "vectorIndexes": [ 
        {
            "path": "/vector", 
            "type": "diskANN"
            
        }
    ]
    } 
   
    mydbt.delete_container(cachecol)


# Create the cache collection with vector index
    try:
        mydbt.create_container_if_not_exists( id=cachecol, 
                                                  partition_key=PartitionKey(path='/id'), 
                                                  indexing_policy=indexing_policy,
                                                  vector_embedding_policy=vector_embedding_policy
                                                ) 
 

    except exceptions.CosmosHttpResponseError: 
        raise 
    return "Cache cleared."
     
def generatecompletionede(user_prompt, username, vector_search_results, chat_history):
    
    # Get the system prompt from session state if available, otherwise use the default
    if 'system_prompt' in st.session_state:
        system_prompt = st.session_state.system_prompt.replace('{username}', username)
    else:
        # Default system prompt
        system_prompt = '''
        You are an intelligent assistant for yourdata, translate the answer in the same langage use for the ask. You are designed to provide answers to user questions about user's data.
        You are friendly, helpful, and informative and can be lighthearted. Be concise in your responses.use the name of the file where the information is stored to provide the answer.
            - start with the hello ''' + username + '''
            - Only answer questions related to the information provided below. 
            '''

    # Create a list of messages as a payload to send to the OpenAI Completions API

    # system prompt
    
    messages = [{'role': 'system', 'content': system_prompt}]
    
    #chat history
    for chat in chat_history:
        messages.append({'role': 'user', 'content': chat['prompt'] + " " + chat['completion']})
    
    #user prompt
    messages.append({'role': 'user', 'content': user_prompt})

    #vector search results
    for result in vector_search_results:
        messages.append({'role': 'system', 'content': result['text']})

    
    # Create the completion
    response = get_completion(openai_client, openai_chat_model, messages)
  
    
    return response

def chat_completion(user_input, username, cachecoeficient, coefficient, maxresult, typesearch, collection):

    # Generate embeddings from the user input
    user_embeddings = generate_embeddings(openai_client, user_input)
    
    # Query the chat history cache first to see if this question has been asked before
    cache_results = cachesearch(user_embeddings, username, cachecoeficient, 1)

    if len(cache_results) > 0:
        return cache_results[0]['completion'], True
    else:
        # Use selected collection if provided and valid, otherwise use default
        
        # Perform vector search on the collection
        search_results = get_similar_docs(openai_client, dbsource, collection, user_input, maxresult, coefficient, typesearch)
        
        # Chat history
        chat_history = get_chat_history(username, 1)

        # Generate the completion
        completions_results = generatecompletionede(user_input, username, search_results, chat_history)

        # Cache the response
        cacheresponse(user_input, user_embeddings, completions_results, username)
        
        return completions_results['choices'][0]['message']['content'], False
    
    
# Application Streamlit
def main():
    st.title("Web application is a RAG application to ask question load using the template logic app wich load data in Azure Cosmos DB")

    # Coefficient used to set the similarity threshold for database searches
    coefficient = 0.7  # Default similarity coefficient for database searches
    # Coefficient used to determine similarity threshold for cache search
    cachecoeficient = 0.99
    maxresult = 5
    global chat_history
    chat_history = []
    selected_collection = None  # Initialize selected collection variable
    
    # Default username for the application
    username = "default_user"
    createcachecollection()
    # Initialize session state for login


    # Onglets
    tab1, tab2, tab3, tab4 = st.tabs(["Chat with your data", "Documents load", "Azure Storage", "Configurations"])

  
    with tab1:
            st.header("Chat")
            models = [
                "vector",
                "full text","hybrid"
                ]
            if st.button("clear the cache"):
              st.write("start clear the Cache")
              clearcache()
              st.write("Cache cleared.")
            
            typesearch = st.selectbox(
                'type search',
                    (models))
                
            st.markdown("---")
            st.subheader("Collection Selection")
              # Get collections for dropdown
            collections = list_collections()
            # Add placeholder for "Select a collection" at the beginning
            collections_with_placeholder = ["Select a collection..."] + collections
            selected_collection = st.selectbox(
                'Select collection to query',
                collections_with_placeholder
            )
            
            if selected_collection != "Select a collection...":
                st.write(f"Selected collection: {selected_collection}")
                # Show collection info
                try:
                    db_client = client.get_database_client(dbsource)
                    container_client = db_client.get_container_client(selected_collection)
                    container_props = container_client.read()
                    
                    # Display collection properties in an expandable section
                    with st.expander("Collection details"):
                        st.write(f"Collection ID: {container_props['id']}")
                      
                except Exception as e:
                    st.error(f"Error retrieving collection details: {str(e)}")
                
            st.write("Chatbot goes here")
            if "messages" not in st.session_state:
                st.session_state["messages"] = [
                {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
                ]
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])
           
                          
            if prompt := st.chat_input(placeholder="enter your question here ? "):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)   
                with st.chat_message("assistant"):
                    question = prompt
                    start_time = time.time()
                    response_payload, cached = chat_completion(question, username, cachecoeficient, coefficient, maxresult, typesearch, selected_collection)
                    end_time = time.time()
                    elapsed_time = round((end_time - start_time) * 1000, 2)
                    response = response_payload

                    details = f"\n (Time: {elapsed_time}ms)"
                    if cached:
                        details += " (Cached)"
                        chat_history.append([question, response + "for "+ username + details])
                    else:
                        chat_history.append([question, response + details])
        
                    st.session_state.messages.append({"role": "assistant", "content":chat_history})
                    st.write(chat_history)
            
   
       
    with tab2:
            
           
            
           
            collections2 = list_collections()
            # Add placeholder for "Select a collection" at the beginning
            collections_with_placeholder2 = ["Select a collection..."] + collections2
            selected_collection2 = st.selectbox(
                'Select collection',
                collections_with_placeholder2
            )
            
            if selected_collection2 != "Select a collection of db":
                st.write(f"Selected collection: {selected_collection2}")
                # Show collection info
              
            
          
                
                
                
                
             
            if st.button("show file loads  "):
                
                  
                st.write(f"LIST OF FILE LOAD ")  
                mydbt = client.get_database_client(dbsource)   
                container = mydbt.get_container_client(selected_collection2)
    
                results = list(container.query_items(
                query= '''
                SELECT distinct (c.documentName) FROM c
                ''',
                enable_cross_partition_query=True, populate_query_metrics=True))

                

                df = pd.DataFrame(results, columns=['documentName'])
                st.dataframe(df)
                
                st.write("made by emmanuel deletang in case of need contact him at edeletang@microsoft.com")
                
    with tab3:
            st.header("Azure Storage Manager")
            
            # Check if we have a connection to Azure Storage
            if not storage_connection_string or storage_connection_string == "DefaultEndpointsProtocol=https;AccountName=yourstorageaccount;AccountKey=yourstorageaccountkey;EndpointSuffix=core.windows.net":
                st.warning("⚠️ Storage account not configured. Please update your .env file with your Azure Storage connection details.")
                
                # Allow user to input connection string directly
                conn_string = st.text_input("Enter your Azure Storage connection string:", 
                                          value="DefaultEndpointsProtocol=https;AccountName=yourstorageaccount;AccountKey=yourstorageaccountkey;EndpointSuffix=core.windows.net",
                                          type="password")
                
                if st.button("Connect to Storage Account"):
                    if conn_string and conn_string != "DefaultEndpointsProtocol=https;AccountName=yourstorageaccount;AccountKey=yourstorageaccountkey;EndpointSuffix=core.windows.net":
                        # Try to connect with the provided connection string
                        blob_service_client = get_blob_service_client(conn_string)
                        if blob_service_client:
                            st.session_state['storage_connection_string'] = conn_string
                            st.success("✅ Successfully connected to Azure Storage!")
                            st.experimental_rerun()  # Rerun the app to refresh the UI
                        else:
                            st.error("❌ Failed to connect to Azure Storage. Please check your connection string.")
                    else:
                        st.error("❌ Please enter a valid connection string.")
            
            else:
                # Get the connection string from session state if available
                actual_conn_string = st.session_state.get('storage_connection_string', storage_connection_string)
                
                # Containers section
                st.subheader("Storage Containers")
                
                # Get list of containers
                containers = list_containers()
                
                if not containers:
                    st.warning("No containers found in the storage account or could not connect to the storage account.")
                else:
                    # Display containers in a dropdown
                    selected_container = st.selectbox("Select a container:", ["Select a container..."] + containers)
                    
                    if selected_container != "Select a container...":
                        # Blobs section
                        st.subheader(f"Blobs in {selected_container}")
                        
                        # Get list of blobs
                        blobs = list_blobs(selected_container)
                        
                        if not blobs:
                            st.info(f"No blobs found in container '{selected_container}'")
                        else:
                            # Create a DataFrame from the blobs info
                            blobs_df = pd.DataFrame(blobs)
                            # Convert size to more readable format
                            blobs_df['size_kb'] = blobs_df['size'].apply(lambda x: f"{x/1024:.2f} KB")
                            # Display the blobs in a table
                            st.dataframe(blobs_df[['name', 'size_kb', 'content_type', 'last_modified']])
                
              
                
                # Container selection for upload
                upload_container = st.selectbox("Select container for upload:", 
                                            ["Select a container..."] + containers,
                                            key="upload_container_select")
                
                # File uploader
                uploaded_file = st.file_uploader("Choose a file to upload", type=None)
                
                if uploaded_file is not None and upload_container != "Select a container...":
                    # Get file details
                    file_name = uploaded_file.name
                    file_content = uploaded_file.getvalue()
                    
                    # Show file details
                    st.write(f"File: {file_name}, Size: {len(file_content)/1024:.2f} KB")
                    
                    # Upload button
                    if st.button("Upload to Azure Storage"):
                        # Upload the file
                        success, message = upload_file_to_container(upload_container, file_name, file_content)
                        
                        if success:
                            st.success(f"✅ File uploaded successfully!")
                            st.markdown(f"**Blob URL:** [{message}]({message})")
                        else:
                            st.error(f"❌ Upload failed: {message}")
   
    with tab4:
            st.header("Configurations")
            
            # Initialize the system prompt in session state if not already present
            if 'system_prompt' not in st.session_state:
                st.session_state.system_prompt = '''
                You are an intelligent assistant for yourdata, translate the answer in the same langage use for the ask. You are designed to provide answers to user questions about user's data.
                You are friendly, helpful, and informative and can be lighthearted. Be concise in your responses.use the name of the file where the information is stored to provide the answer.
                    - start with the hello {username}
                    - Only answer questions related to the information provided below. 
                '''
            
            # System prompt configuration
            st.subheader("System Prompt Configuration")
            
            # Explain what the system prompt is
            st.info("""
            The system prompt is the initial instruction given to the AI model. 
            It guides how the AI responds to user queries. You can customize it to change the chatbot's behavior.
            
            Use {username} as a placeholder for the user's name in the prompt.
            """)
            
            # Text area for editing the system prompt
            new_system_prompt = st.text_area(
                "Edit System Prompt:",
                value=st.session_state.system_prompt,
                height=300
            )
            
            # Save button to update the system prompt
            if st.button("Save System Prompt"):
                st.session_state.system_prompt = new_system_prompt
                # Save the prompt to a JSON file
                if save_system_prompt(new_system_prompt):
                    st.success("✅ System prompt updated and saved successfully!")
                else:
                    st.error("❌ Failed to save system prompt.")
                
            # Reset button to restore the default system prompt
            if st.button("Reset to Default"):
                default_prompt = '''
                You are an intelligent assistant for yourdata, translate the answer in the same langage use for the ask. You are designed to provide answers to user questions about user's data.
                You are friendly, helpful, and informative and can be lighthearted. Be concise in your responses.use the name of the file where the information is stored to provide the answer.
                    - start with the hello {username}
                    - Only answer questions related to the information provided below. 
                '''
                st.session_state.system_prompt = default_prompt
                # Reset the prompt in the JSON file
                if save_system_prompt(default_prompt):
                    st.success("✅ System prompt reset to default and saved successfully!")
                else:
                    st.error("❌ Failed to reset system prompt.")

# New functions to save and load system prompt
def save_system_prompt(prompt_text):
    """
    Save the system prompt to a JSON file.
    
    Args:
        prompt_text (str): The system prompt text to save
        
    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        # Create a config directory if it doesn't exist
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
        os.makedirs(config_dir, exist_ok=True)
        
        # Save the prompt to a JSON file
        config_file = os.path.join(config_dir, 'system_prompt.json')
        with open(config_file, 'w') as f:
            json.dump({'system_prompt': prompt_text}, f)
            
        return True
    except Exception as e:
        print(f"Error saving system prompt: {str(e)}")
        return False

def load_system_prompt():
    """
    Load the system prompt from a JSON file.
    
    Returns:
        str: The system prompt text if loaded successfully, None otherwise
    """
    try:
        # Get the config file path
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
        config_file = os.path.join(config_dir, 'system_prompt.json')
        
        # Check if the file exists
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                return config_data.get('system_prompt')
        
        return None
    except Exception as e:
        print(f"Error loading system prompt: {str(e)}")
        return None

if __name__ == "__main__":
    main()
