from schema import TextSchema
from type_classes import *
from search import extract


class ExampleSchema(TextSchema):
    Name: Field(Str, 1, 5)
    Hobbies: Field(List[Str], 1, 1, ['Hiking', 'Swimming', 'Reading'])
    Age : Field(Int, 1, 100)
    Married: Field(Bool, 1, 1)

text = """
My name is Zaid. I am 25 years old. I like swimming and reading. I am is married. 
"""
metadata = extract(
    text, "IVUL-KAUST/MeXtract-3B", schema=ExampleSchema, backend = "transformers"   
)
print(metadata)

## {'Name': 'Zaid', 'Hobbies': ['Swimming'], 'Age': 25, 'Married': True}