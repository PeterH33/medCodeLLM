import os
import tee
import glob
import json
import requests
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from pydantic import BaseModel

# TODO I would like to not have to do the parse part over and over, some means of storing the vectors and ensuring that nothing changed in the context folder would be beneficial.

# TODO ZeroShot, Fewshot, RAG, Different models


# Load documents
def loadDocumentsFromFolder(folderPath):
    # NOTE Extend this later to cover other file types as needed, this should be a suitable starting point, but pdfs will likely require inclusion
    patterns = ["*.txt", "*.csv"]
    fileNames = []
    for p in patterns:
        fileNames.extend(glob.glob(os.path.join(folderPath, p)))
    documents = []
    for fileName in fileNames:
        try:
            with open(fileName, "r", encoding="utf-8") as f:
                content = f.read()
            doc = Document(page_content=content, metadata={'source_file': os.path.basename(fileName)})
            documents.append(doc)
        except Exception as e:
            print(f'Error in loadDocumentsFromFolder() reading {fileName}: {e}')
    # NOTE I believe that I will want to make the system allow for an empty file with a y/n check to continue if the user does not have documents loaded
    if not documents:
        raise RuntimeError(f'No documents in {folderPath}')
    return documents


def setupRAG():
    CONTEXT_DOCS_PATH = './docs'
    documents = loadDocumentsFromFolder(CONTEXT_DOCS_PATH)

    # Split docs into chunks
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=500,
        length_function=len,
    )
    splitTexts= textSplitter.split_documents(documents)

    # Create local embeddings
    try:
        embeddingModel = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    except Exception as e:
        print(f'Error creating local embeddings: {e}')

    # Store chunks in vector DB
    try:
        vectorStore = FAISS.from_documents(splitTexts, embeddingModel)
    except Exception as e:
        print(f'Error storing chunks in Vector DB: {e}')

    # Setup retriever
    retriever = vectorStore.as_retriever(search_kwargs={'k':3})
    return retriever
    # RAG Prepwork ends


# # NOTE Original RAG Query system, consider this structure in the future, perhaps this method works better with the retrievalQA chain
# def askRAGQuestion(question, model, retriever):
#     try:
#         # Define llm
#         # NOTE There are a ton of options on this method, investigate more later
#         llm = ChatOllama(model=model)

#         # Create retrievalQA chain
#         qaChain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type='stuff',
#             retriever=retriever,
#             return_source_documents=True,
#         )
#     except Exception as e:
#         print(f'Error in settingup llm and qaChain in askRAGQuestion(): {e}')
#     try:
#         result = qaChain.invoke({"query": question})
#         print(f'ans: {result["result"]}')
#         print('\nSource Documents: ')
#         for i, sourceDocChunk in enumerate(result['source_documents']):
#             print(f'ChunkNumber:{i+1}, File: {sourceDocChunk.metadata.get("source_file", "N/A")}')
#             print(f'Content Snippet: {sourceDocChunk.page_content[:350]}')
#     except Exception as e:
#         print(f'Error in askRAGQuestion(): {e}')

def askRAGQuestion(question, model, retriever, jsonSchema=None):
    try:
        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(question)
        
        # Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create the enhanced prompt with context, not sure if this is better than qaChain above, worth looking into
        enhanced_prompt = f"""Context information from relevant documents:
{context}

Question: {question}

Please answer the question based on the context provided above."""
        
        url = "http://localhost:11434/api/generate"
        prompt_data = {
            "model": model,
            "prompt": enhanced_prompt,
            "format": jsonSchema
        }
        
        response = requests.post(url, json=prompt_data, stream=True)
        
        collected = ""  
        
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                print(data["response"], end="", flush=True)
                collected += data["response"]
                
            if data.get("done"):
                break
        
        print()
        
        # Print source documents
        # print('\nSource Documents:')
        # for i, sourceDocChunk in enumerate(relevant_docs):
        #     print(f'ChunkNumber: {i+1}, File: {sourceDocChunk.metadata.get("source_file", "N/A")}')
        #     print(f'Content Snippet: {sourceDocChunk.page_content[:350]}')
        #     print()
        
        # # Return the collected response and sources
        # return {
        #     "result": collected,
        #     "source_documents": relevant_docs
        # }
        
    except Exception as e:
        print(f'Error in askRAGQuestion(): {e}')
        return None


def rawOllamaCall(question, model):
    url = "http://localhost:11434/api/generate"

    prompt = {
        "model": model,
        "prompt": question,
    }

    response = requests.post(url, json=prompt, stream=True)

    # Stream output clean
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                print(data["response"], end="", flush=True)
            if data.get("done") == 'true':
                break

def jsonSchemaOllamaCall(question, model, jsonSchema):
    url = "http://localhost:11434/api/generate"

    prompt = {
        "model": model,
        "prompt": question,
        "format": jsonSchema
    }

    response = requests.post(url, json=prompt, stream=True)

    collected = ""  # hold streamed JSON text

    for line in response.iter_lines():
        if not line:
            continue

        data = json.loads(line.decode("utf-8"))

        if "response" in data:
            # Print characters exactly as the model streams them
            print(data["response"], end="", flush=True)

        if data.get("done"):
            break



# --Test Queries--
print('\n ====== Program Start ======\n')


# Doctors Notes extraction
doctorsNotes = loadDocumentsFromFolder('./doctorNotes')

# Need to define the JSON format that I want information exported as. This initial structure is simply pulled from an interview with one of the contributing physicians. Keeping it in this format in code allows for editing in the future should it be needed.
outputStructureComplex = '''{
    "medical_record": {
        "original_document": "",
        "codes": {
        "diagnostic_codes": [],
        "procedure_codes": [],
        "billing_codes": []
        },
        "subjective": {
        "chief_complaint": "",
        "history_of_present_illness": "",
        "past_medical_history": "",
        "surgical_history": "",
        "pregnancy_history": "",
        "menstrual_history": "",
        "social_history": {
            "sexual_activity": "",
            "drug_use": "",
            "lifestyle": ""
        },
        "alcohol_use": "",
        "current_medications": [],
        "allergies": [],
        "review_of_systems": {
            "systems_reviewed": [
            {
                "system_name": "cardiovascular",
                "findings": ""
            },
            {
                "system_name": "respiratory",
                "findings": ""
            }
            ]
        }
        },
        "objective": {
        "vital_signs": {
            "temperature_celsius": null,
            "blood_pressure_mmHg": "",
            "heart_rate_bpm": null,
            "respiratory_rate_bpm": null,
            "oxygen_saturation_percent": null
        },
        "physical_exam": [],
        "lab_results": [],
        "imaging": [],
        "diagnostic_procedures": []
        },
        "assessment": {
        "summary": "",
        "differential_diagnosis": [],
        "working_diagnosis": ""
        },
        "plan": {
        "expected_follow_up": "",
        "management_plan": [
            {
            "organ_system": "cardiovascular",
            "actions": []
            }
        ]
        },
        "orders": {
        "medications_ordered": [],
        "referrals_made": [],
        "labs_ordered": [],
        "imaging_ordered": []
        }
    }
}'''

outputStructureSimple = '''{
    "medical_record": {
        "original_document": "",
        "codes": {
        "diagnostic_codes": [],
        },
        "subjective": {
        "chief_complaint": "",
        },
        "objective": {
        "vital_signs": {
            "temperature_celsius": "",
            "blood_pressure_mmHg": "",
            "heart_rate_bpm": "",
            "respiratory_rate_bpm": "",
            "oxygen_saturation_percent": ""
        },
        "physical_exam": "",
        "lab_results": [],
        "imaging": [],
        "diagnostic_procedures": [],
        },
        "assessment": {
        "summary": "",
        "differential_diagnosis": "",
        "working_diagnosis": "",
        },
        "plan": {
        "expected_follow_up": "",
        "management_plan": "",
        },
        "orders": {
        "referrals_made": [],
        }
    }
}'''

outputStructureTrivial = '''{
    "medical_record": {
        "original_document": "",
        "diagnostic_code": [],
        "diagnoses": [],
    }
}'''

class jsonStructureTrivial(BaseModel):
    original_document: str
    diagnostic_codes: list[str]
    diagnoses: list[str]


# Instructions to follow the json promt - refine as needed

with open('prompts/instructionPrompt.txt', 'r') as f:
    instructionPrompt = f.read()

with open('prompts/fewShotPrompt.txt', 'r') as f:
    fewShotPrompt = f.read()


# NOTE Specify results path for different tests, this can be cleaner.
RESULTS_PATH = './results/rag'

retriever = setupRAG()

tee.startTee(RESULTS_PATH)


models = ["deepseek-r1:8b", 'llama3.2:latest', 'gemma3:4b', 'mistral:7b', 'phi4:14b']

# NOTE EDIT BEFORE RUNS - Ouput document heading 
print('Running test using few-shot prompting, and trivial Json schema for Ollama Structured output. gpt-oss replaced with gemma3:270m')

print('Iniital prompt:')
print('Directions: ' + instructionPrompt + ' RAG Context ' + 'Doctors note: <Variable>')


for note in doctorsNotes:
    for model in models:
    
        question = 'Directions: ' + instructionPrompt + 'Desired output Json structure: ' + outputStructureTrivial + 'Doctors note:' + note.page_content
        fewShotQuestion = 'Directions: ' + instructionPrompt + fewShotPrompt + 'Doctors note:' + note.page_content

        print(f'\n\n==========================\nStarting query using model {model} please wait...')
        startTime = time.perf_counter()

        # askRAGQuestion(question, model)
        # rawOllamaCall(question, model)
        # jsonSchemaOllamaCall(question, model, jsonStructureTrivial.model_json_schema())
        # jsonSchemaOllamaCall(fewShotQuestion, model, jsonStructureTrivial.model_json_schema())
        askRAGQuestion(question, model, retriever, jsonStructureTrivial.model_json_schema())

        endTime = time.perf_counter()

        print('\nTime to completion')
        print(f'Time: {endTime-startTime:.6f} seconds')

tee.endTee()

# Interesting accidental result, not including the doctors notes it went ahead and halucinated some notes.
