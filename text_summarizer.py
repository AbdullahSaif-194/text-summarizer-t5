from fastapi import FastAPI,Request
from transformers import AutoTokenizer,T5ForConditionalGeneration
import re
import torch
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
class Dialogue_Class(BaseModel):
    dialogue :str
  
templates = Jinja2Templates(directory=".")    
model = T5ForConditionalGeneration.from_pretrained("./Text_Summarizer_Model")
tokenizer = AutoTokenizer.from_pretrained("./Text_Summarizer_Model")
def clean_data(dialogue):
    dialogue = re.sub(r"\r?\n|\r"," ",dialogue) # remove lines
    dialogue = re.sub(r"\s+"," ",dialogue) # remove extra spaces like tab
    dialogue = re.sub(r"<.*?>"," ",dialogue) # remove html tags
    dialogue = dialogue.strip().lower()
    return dialogue
def tokenize(data):
    inputs = tokenizer(data,padding="max_length",max_length=512,truncation=True,return_tensors="pt")
    return inputs

def summarizering(dialogue):
    dialogue = clean_data(dialogue)
    inputs =  tokenize(dialogue)
    targets = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length = 150,
        num_beams = 4,
        early_stopping = True,
    )
    summary = tokenizer.decode(targets[0],skip_special_tokens=True)
    return summary

@app.post("/summarize/") 
async def summarizer(dialogue_input : Dialogue_Class):
    summary = summarizering(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")
    