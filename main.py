from fastapi import FastAPI
from transformers import pipeline
import uvicorn
import argparse
import json
import base64
from PIL import Image
from io import BytesIO

app = FastAPI()

METADATA = {
    "text-classification": {
        "inputs": [
            {"name": "args", "datatype": "str"}
        ],
        "outputs": []
    },
    "token-classification": {
        "inputs": [
            {"name": "args", "datatype": "str"}
        ],
        "outputs": []
    },
    "object-detection": {
        "inputs": [
            {"name": "inputs", "datatype": "str"},
            {"name": "inputs", "datatype": "base64"},
            {"name": "inputs", "datatype": "pillow_image"}
        ],
        "outputs": []
    },
    "text-generation": {
        "inputs": [
            {"name": "args", "datatype": "str"}
        ],
        "outputs": []
    }
}

def convert_to_v2_payload(pipeline_name: str, payload: dict):
    input_format = METADATA[pipeline_name]["inputs"][0]["datatype"]
    v2_payload = {
        "id": "string",
        "parameters": {
            "content_type": "application/json",
            "headers": {},
            "additionalProp1": {}
        },
        "inputs": [],
        "outputs": []
    }

    if input_format == "str":
        input_data = {
            "name": METADATA[pipeline_name]["inputs"][0]["name"],
            "datatype": "string",
            "shape": [],
            "parameters": {
                "content_type": "application/json",
                "headers": {},
                "additionalProp1": {}
            },
            "data": json.dumps(payload)
        }
        v2_payload['inputs'].append(input_data)
    elif input_format == "base64":
        image_bytes = base64.b64decode(payload["inputs"])
        image = Image.open(BytesIO(image_bytes))
        image_data = image.tobytes()

        input_data = {
            "name": METADATA[pipeline_name]["inputs"][1]["name"],
            "datatype": "bytes",
            "shape": [image.height, image.width, 3],  # Assuming RGB image
            "parameters": {
                "content_type": "application/octet-stream",
                "headers": {},
                "additionalProp1": {}
            },
            "data": base64.b64encode(image_data).decode("utf-8")
        }
        v2_payload['inputs'].append(input_data)
    elif input_format == "pillow_image":
        image_bytes = payload["inputs"].tobytes()
        image = Image.open(BytesIO(image_bytes))
        image_data = image.tobytes()

        input_data = {
            "name": METADATA[pipeline_name]["inputs"][2]["name"],
            "datatype": "bytes",
            "shape": [image.height, image.width, 3],  # Assuming RGB image
            "parameters": {
                "content_type": "application/octet-stream",
                "headers": {},
                "additionalProp1": {}
            },
            "data": base64.b64encode(image_data).decode("utf-8")
        }
        v2_payload['inputs'].append(input_data)
    
    return json.dumps(v2_payload)

def perform_inference(pipeline_name: str, model_deployed_url: str, converted_payload: dict):
    # Load the specified Hugging Face pipeline
    hf_pipeline = pipeline(pipeline_name, model=model_deployed_url)

    # Convert the converted_payload back to a dictionary
    converted_payload = json.loads(converted_payload)

    # Extract the inputs from the converted_payload
    inputs = converted_payload["inputs"]

    # Prepare the inputs for the Hugging Face pipeline
    hf_inputs = {}

    for input_data in inputs:
        input_name = input_data["name"]
        input_datatype = input_data["datatype"]
        input_shape = input_data["shape"]
        input_parameters = input_data["parameters"]
        input_value = input_data["data"]

        if input_datatype == "str":
            hf_inputs["text_inputs"] = input_value
        elif input_datatype == "base64":
            input_bytes = base64.b64decode(input_value)
            input_image = Image.open(BytesIO(input_bytes))
            hf_inputs[input_name] = input_image

    # Perform the inference using the loaded model and converted payload
    inference_result = hf_pipeline(**hf_inputs)  # Pass inputs as keyword arguments

    # Return the inference result
    return inference_result


def convert_to_v2_response(pipeline_name: str, inference_result):
    # Prepare the response in the V2 inference protocol format
    v2_response = {
        "id": "string",
        "parameters": {
            "content_type": "application/json",
            "headers": {},
            "additionalProp1": {}
        },
        "inputs": [],
        "outputs": []
    }

    # Add the outputs to the response
    v2_response["outputs"] = inference_result

    # Return the converted response in the V2 inference protocol format
    return json.dumps(v2_response)

@app.post("/inference")
async def inference(pipeline_name: str, model_deployed_url: str, payload: dict):
    converted_payload = convert_to_v2_payload(pipeline_name, payload)
    inference_result = perform_inference(pipeline_name, model_deployed_url, converted_payload)
    converted_response = convert_to_v2_response(pipeline_name, inference_result)
    return converted_response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_pipeline", type=str, help="Hugging Face pipeline name")
    parser.add_argument("--model_deployed_url", type=str, help="Deployed endpoint URL of the model")
    args = parser.parse_args()

    uvicorn.run("main:app", reload=True)
