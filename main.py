from fastapi import FastAPI, UploadFile, File, Form, HTTPException # type: ignore
# add absolute path from src 
import sys
sys.path.append('src')
from search import run
from schema import get_schema
import arg_utils
import json
from dotenv import load_dotenv
import os
from starlette.concurrency import run_in_threadpool

load_dotenv()
keys_order = ["Name", "Dialect_Subsets", "HF_Link", "Link", "License", "Year", "Language", "Dialect", "Source", "Domain", "Form", "Annotation_Style", "Description", "Volume", "Unit", "Provider", "Derived_From", "Paper_Title", "Paper_Link", "Script", "Tokenized", "Host", "Access", "Cost", "Has_Splits", "Partial", "Tasks", "Venue_Title", "Venue_Type", "Venue_Name", "Authors", "Affiliations", "Abstract", "Added_By"]

app = FastAPI()

@app.post("/run")
async def run_extraction(link: str =  Form(''), schema_name: str = Form(''), file: UploadFile = File(None), model_name: str = Form('')):
    # Build a fresh args object per request so concurrent requests don't clobber
    # each other's settings on a shared global.
    _args = arg_utils.Args(**arg_utils.get_default_args())
    _args.model_name = model_name
    _args.schema_name = schema_name
    _args.format = 'pdf_plumber'
    _args.overwrite = True
    _args.log = True

    # `run` is a blocking, long-running function (PDF download/parse, LLM calls).
    # Running it directly in this async endpoint would block the event loop and
    # freeze every other request. Offload it to a worker thread instead.
    results = await run_in_threadpool(run, link, file, _args)

    if model_name not in results or "metadata" not in results.get(model_name, {}):
        raise HTTPException(
            status_code=422,
            detail="Failed to process paper. Check that the link is valid and the paper could be downloaded.",
        )

    metadata = results[model_name]['metadata']
    metadata['Added_By'] = model_name
    print(metadata)
    return {'model_name': model_name, 'metadata': metadata}

@app.post("/schema")
async def get_schema_endpoint(name: str =  Form('')):
    schema = get_schema(name)
    schema_dict = json.loads(schema.schema())
    for line in open('GUIDELINES.md', 'r').readlines():
        if '**' in line:
            key = line.strip().split('**')[1].replace(' ', '_')
            schema_dict[key]['description'] = line.strip().split('**')[-1].strip().capitalize()
    schema_dict ['Added_By'] = {
        "answer_type": "str",
        "answer_min": 1,
        "answer_max": 1,
        "description": "Your full name"
    }
    return {key: schema_dict[key] for key in keys_order}
