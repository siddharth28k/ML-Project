# ML-Project
Creating a service on FastAPI for ML models using the HuggingFace transformers library and converting it into V2 inference protocol. 

First I import the specific data formats as metadata for the desired pipelines so that it can be used dynamically.
Then I have created a function to converts input data into v2_payload using the above mentioned metadata. It returns the converted payload.
Then I used transformers library(HuggingFace) to create a pipeline and define the input by loading the converted payload json file and set the inputs for the HF model. The pipeline is executed and the results of the model are returned
Then the output is simply added to the v2 response and returned(inputs and other details can be mentioned too, I have skipped that part).

This whole code is tested on Swagger UI (FastAPI) 

Sample input:

hf-pipeline: text-generation 
model-deployed-url: sshleifer/tiny-gpt2 (any model would work as long as the pipeline is same)

input for swagger UI: {"args": "Once upon a time "}

