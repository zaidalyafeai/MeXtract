from search import get_metadata
from rich import print
from schema import TextSchema
from type_classes import *

def extract(text, model_name, schema_name = "ar", backend = "openrouter", max_model_len = 8192, max_output_len = 2084, schema = None):
    message, metadata, cost, error = get_metadata(
        text, model_name, schema_name=schema_name, schema = schema, backend = backend, log = False, max_model_len = max_model_len, max_output_len = max_output_len
    )
    return metadata

text = """
MeXtract: Light-Weight Metadata Extraction from Scientific Papers

Zaid Alyafeai, Maged AlShaibani, Bernard Ghanem

Introduction
MeXtract is a 3.0 Billion parameter chat model based on fine-tuning the chat version of Qwen2.5 3B.

Model Overview
MeXtract is evaluated on the MOLE+ benchmark, which is an extension of MOLE with model schema.

Language	Number of papers
Arabic	21
Russian	21
French	21
English	21
Japanese	21
Multilingual	21
Model	21

Development Context
We release the models freely, along with the code on GitHub (https://github.com/IVUL-KAUST/MeXtract
) and on Hugging Face through https://huggingface.co/collections/IVUL-KAUST/mextract-68e63fb946b06f5031d4e3ef
.
The models are released under the Apache-2.0 License.

Conclusion
MeXtract is created through instruction tuning and preference optimization.
The model achieves state-of-the-art results on the MOLE+ benchmark.

preprint arXiv:2402.03177 [cs.CL] 5 Feb 2025
"""

class ExampleSchema(TextSchema):
    Paper_Title: Field(Str, 1, 5)
    Authors: Field(List[Str], 1, 5)
    Name : Field(Str, 1, 5)
    Num_Parameters: Field(Float, 1, 999)
    Unit: Field(Str, 1, 5, ['Million', 'Billion', 'Trillion'])
    Type: Field(Str, 1, 1, ['Base', 'Chat', 'Code'])
    Benchmarks: Field(List[Str], 1, 5)
    Access: Field(Str, 1, 1, ['Free', 'Paid'])
    Host: Field(Str, 1, 5)
    Link: Field(URL, 1, 1)
    HF_Link: Field(URL, 1, 1)
    License: Field(Str, 1, 5)
    Paper_Link: Field(URL, 1, 1)
    

metadata = extract(
    text, "IVUL-KAUST/MeXtract-3B", schema=ExampleSchema, backend = "transformers"   
)
print(metadata)