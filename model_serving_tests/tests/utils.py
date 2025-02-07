import os

import pytest

from .conftest import client
import asyncio
import aiohttp
from typing import Any, Generator, Optional
import yaml
from jinja2 import BaseLoader, Environment
from abc import ABC, abstractmethod
from ocp_resources.pod import Pod
from kubernetes.dynamic.client import DynamicClient
import logging
from model_serving_tests.tests.constant import (
    INFERE_DIR, RUNTIME_DIR, STORAGE_DIR)

LOGGER = logging.getLogger(__name__)


class PodNotFoundError(Exception):
    """Exception raised when a pod is not found."""
    pass


class Jinja2Loader(ABC):
    """Abstract base class for Jinja2 template loaders."""

    @abstractmethod
    def from_jinja2(self, src: str, variables: dict) -> str:
        """Render a Jinja2 template with the provided variables.

        Args:
            src (str): The Jinja2 template as a string.
            variables (dict): A dictionary of variables to render the template.

        Returns:
            str: The rendered template.
        """
        pass


class TemplateLoader(Jinja2Loader):
    """Concrete implementation of Jinja2Loader using Jinja2's BaseLoader."""

    def from_jinja2(self, src: str, variables: dict) -> str:
        """Render the given Jinja2 template with the provided variables.

        Args:
            src (str): The Jinja2 template as a string.
            variables (dict): A dictionary of variables to render the template.

        Returns:
            str: The rendered template.
        """
        template = Environment(loader=BaseLoader()).from_string(src)
        return template.render(variables)


def parse_resource_template(yaml_file_path: Any,
                            context: Any,
                            output_file_path: Any) -> None:
    """Read a YAML file with placeholders, replace them, and write the modified YAML to a new file.

    Args:
        yaml_file_path (Any): Path to the input YAML file.
        context (Any): Context dictionary to render the YAML file.
        output_file_path (Any): Path to the output YAML file.
    """
    # Read the YAML file
    with open(yaml_file_path, 'r') as file:
        yaml_content = file.read()

    # Create TemplateLoader instance
    loader = TemplateLoader()

    # Render the YAML content with the given context
    rendered_content = loader.from_jinja2(yaml_content, context)

    # Convert the rendered content back to YAML
    try:
        # Load the rendered content to validate and format it as YAML
        yaml_data = yaml.safe_load(rendered_content)
        with open(output_file_path, 'w') as file:
            yaml.dump(yaml_data, file, default_flow_style=False)
        print(f"Processed YAML has been written to {output_file_path}")
    except yaml.YAMLError as e:
        print(f"Error processing YAML: {e}")


def get_predictor_pod(client: DynamicClient, namespace: str, is_name: str) -> Pod:
    """Retrieve a predictor pod from a Kubernetes namespace.

    Args:
        client (DynamicClient): The Kubernetes dynamic client.
        namespace (str): The namespace to search for the pod.
        is_name (str): The base name of the pod to search for.

    Returns:
        Pod: The found pod object.

    Raises:
        PodNotFoundError: If no predictor pod is found in the namespace.
    """
    for pod in Pod.get(dyn_client=client, namespace=namespace):
        if is_name + "-predictor" in pod.name:
            LOGGER.info(f"Pod name {pod.name}")
            return pod

    raise PodNotFoundError(f"No predictor pod found in namespace {namespace}")


async def _send_chat_completion(messages, url):
    """Send a chat completion request to a specified URL.

    Args:
        messages: The messages to send in the request.
        url: The URL to send the request to.

    Returns:
        dict: The JSON response from the server, or an error message.
    """
    # Make sure to define your URL here
    print('Starting OpenAI request')
    async with aiohttp.ClientSession() as session:
        async with session.post(url=url,
                                json={"messages": messages, "model": "gpt", "seed": 1350},
                                headers={"Content-Type": "application/json"}, ssl=False) as response:
            if response.status == 200:
                try:
                    json_response = await response.json()
                    return json_response
                except aiohttp.ContentTypeError:
                    print('Error: Response content is not JSON')
                    return {'error': 'Invalid response content'}
            else:
                print(f'Error: Received status code {response.status}')
                return {'error': f'Status code {response.status}'}


async def _send_async_requests(prompts_messages, url=None):
    """Send asynchronous chat completion requests and return the responses.

    Args:
        prompts_messages: A list of message lists to send in the requests.
        url: The URL to send the requests to.

    Returns:
        list: A list of responses from the server.
    """
    tasks = [_send_chat_completion(msgs, url) for msgs in prompts_messages]
    responses: tuple[Any] = await asyncio.gather(*tasks)
    print([resp for resp in responses])
    responses = [
        resp.get('choices', [{}])[0].get('message', {}).get('content', '').strip() if 'error' not in resp else resp[
            'error'] for resp in responses]
    return responses


def create_runtime_manifest_from_template(deployment_type: str, runtime_image: str,
                                          runtime_name: str,
                                          raw_port: int = 8033) -> None:
    """Create a runtime manifest from a template and save it to a file.

    Args:
        runtime_image(str): runtime image
        deployment_type (str): The type of deployment (e.g., 'rawdeployment').
        runtime_name (str, optional): The name of the runtime. Defaults to "serving_runtime".
        raw_port (int, optional): The raw port to use. Defaults to 8033.
    """
    data = {
        "entrypoint": "vllm_tgis_adapter",
        "runtime_image": runtime_image,
        'tgi_raw_port': {
            'port': raw_port,
            'name': 'h2c',
            'protocol': 'TCP'
        }
    }

    if deployment_type.lower() != "rawdeployment":
        del data["tgi_raw_port"]
        del data["entrypoint"]
        data["deployment_mode"] = deployment_type
    parse_resource_template(yaml_file_path=RUNTIME_DIR / 'vLLM' / f'{runtime_name}.yaml', context=data,
                            output_file_path=RUNTIME_DIR / 'vLLM' / f'{runtime_name}_updated.yaml')


def create_isvc_manifest_from_template(deployment_mode: Any,
                                       model_name: Any,
                                       accelerator_type: Any = None,
                                       model_format: Any = None,
                                       runtime_name: Any = None,
                                       storage_uri: Any = None,
                                       gpu_count: Any = None,
                                       new_args: Any = None,
                                       env_vars: Any = None) -> Any:
    """Create an ISVC manifest from a template and save it to a file.

    Args:
        accelerator_type(Any): Accelerator type on which test will run like Nvidia,AMD,Gaudi
        deployment_mode (Any): The deployment mode.
        model_name (Any): The name of the model.
        model_format (Any, optional): The format of the model. Defaults to None.
        runtime_name (Any, optional): The name of the runtime. Defaults to None.
        storage_uri (Any, optional): The URI for storage. Defaults to None.
        gpu_count (Any, optional): The number of GPUs to use. Defaults to None.
        new_args (Any, optional): Additional arguments. Must be a list if provided. Defaults to None.
        env_vars (Any, optional): Environment variables. Must be a list of dictionaries with 'name' and 'value' keys if provided. Defaults to None.

    Raises:
        ValueError: If new_args or env_vars are not of the expected types or formats.
    """
    data = {}

    if deployment_mode:
        data["deployment_mode"] = deployment_mode
    if model_name:
        data["model_name"] = model_name
    if accelerator_type is not None:
        accel_type = accelerator_type.lower()

        # Determine the GPU locator based on the accelerator type
        if accel_type == "nvidia":
            data["gpu_locator"] = "nvidia.com/gpu"
        elif accel_type == "amd":
            data["gpu_locator"] = "amd.com/gpu"
        elif accel_type in ["intel", "gaudi", "habana"]:
            data["gpu_locator"] = "habana.ai/gaudi"
        else:
            LOGGER.info("Using COU")
    if model_format is not None:
        data["model_format"] = model_format

    if runtime_name is not None:
        data["runtime_name"] = runtime_name

    if storage_uri is not None:
        data["storage_uri"] = storage_uri

    if gpu_count is not None:
        data["gpu_count"] = gpu_count

    # Validate and add new_args if it's a valid list or None
    # "new_args"= ["--new-arg1=value1", "--new-arg2=value2"],  # New args to add
    if new_args is not None:
        if not isinstance(new_args, list):
            raise ValueError("new_args must be a list or None")
        data["new_args"] = new_args

    # Validate and add env_vars if it's a valid list or None
    # "env_vars"= [{"name": "NEW_VAR1", "value": "value1"}, {"name": "NEW_VAR2", "value": "value2"}]
    if env_vars is not None:
        if not isinstance(env_vars, list):
            raise ValueError("env_vars must be a list or None")
        if not all(isinstance(item, dict) and 'name' in item and 'value' in item for item in env_vars):
            raise ValueError("Each item in env_vars must be a dictionary")
    parse_resource_template(yaml_file_path=INFERE_DIR / model_name / f'{model_name}.yaml', context=data,
                            output_file_path=INFERE_DIR / model_name / f'{model_name}_updated.yaml')


def create_s3_secret_manifest(name="s3_seceret"):
    data = {
        "aws_access_key_id": os.getenv('AWS_ACCESS_KEY_ID'),
        "aws_secret_access_key": os.getenv('AWS_SECRET_ACCESS_KEY')
    }
    parse_resource_template(yaml_file_path=STORAGE_DIR / f'{name}.yaml', context=data,
                            output_file_path=STORAGE_DIR / f'{name}.yaml')
