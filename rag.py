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

# TODO I would like to not have to do the parse part over and over, some means of storing the vectors and ensuring that nothing changed in the context folder would be beneficial.

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

DOCS_PATH = './docs'
documents = loadDocumentsFromFolder(DOCS_PATH)

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


# RAG Prepwork ends


# Query system
def askRAGQuestion(question, model):
    try:
        # Define llm
        # NOTE There are a ton of options on this method, investigate more later
        llm = ChatOllama(model=model)

        # Create retrievalQA chain
        qaChain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever,
            return_source_documents=True,
        )
    except Exception as e:
        print(f'Error in settingup llm and qaChain in askRAGQuestion(): {e}')
    try:
        result = qaChain.invoke({"query": question})
        print(f'ans: {result["result"]}')
        print('\nSource Documents: ')
        for i, sourceDocChunk in enumerate(result['source_documents']):
            print(f'ChunkNumber:{i+1}, File: {sourceDocChunk.metadata.get("source_file", "N/A")}')
            print(f'Content Snippet: {sourceDocChunk.page_content[:350]}')
    except Exception as e:
        print(f'Error in askRAGQuestion(): {e}')


def rawOllamaCall(question, model):
    url = "http://localhost:11434/api/generate"

    prompt = {
        "model": model,
        "prompt": question
    }

    response = requests.post(url, json=prompt, stream=True)

    # Stream output clean
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                print(data["response"], end="", flush=True)
            if data.get("done", False):
                break

# --Test Queries--
print('\n Program Start')

# For each note and for each model that I want to inspect, we need to run a question.

# Doctors Notes extraction
doctorsNotes = loadDocumentsFromFolder('./doctorNotes')

# Need to define the JSON format that I want information exported as. This initial structure is simply pulled from an interview with one of the contributing physicians. Keeping it in this format in code allows for editing in the future should it be needed.
outputStructure = '''{
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

# TODO Need to cleanup output formatting and the terminal window size, perhaps outputting to a file instead of print line is the better way to go.
# TODO Include the prompt that was sent to the system before 
# Instructions to follow the json promt - refine as needed
directionsForJSON = 'Provide an output using only the provided json structure, do not deviate from it, not not create new fields, any time that there is an array multiple datapoints can be added to the array. Use the following JSON structure and the provided context to process and output the information contained in the doctors note. Include the full original doctors note verbatum in the section labeled original_document. '

RESULTS_PATH = './results'

tee.startTee('./results')

models = ["deepseek-r1:8b"]
for model in models:
    for note in doctorsNotes:
    
        question = 'Directions: ' + directionsForJSON + 'Desired output Json structure: ' + outputStructure + 'Doctors note:' + note.page_content

        print(f'Starting RAG query using model {model} please wait...')
        start = time.perf_counter()

        askRAGQuestion(question, model)
        ragend = time.perf_counter()

        rawOllamaCall(question, model)
        rawend = time.perf_counter()

        print('\nTime to completion')
        print(f'RAG Time: {ragend-start:.6f} seconds')
        print(f'Raw Time: {rawend-ragend:.6f} seconds')

tee.endTee()

# Interesting accidental result, not including the doctors notes it went ahead and halucinated some notes.
