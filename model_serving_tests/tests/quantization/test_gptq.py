from typing import Any, Callable
import pytest
from kubernetes.dynamic.client import DynamicClient
from ocp_resources.resource import Resource
from model_serving_tests.endpoint_utility.openai_utility import OpenAIClient
from model_serving_tests.endpoint_utility.grpc_utility import TGISGRPCPlugin
from model_serving_tests.tests.utils import create_runtime_manifest_from_template, create_isvc_manifest_from_template, \
    get_predictor_pod, create_s3_secret_manifest
import logging
import time

LOGGER = logging.getLogger(__name__)

# The model is convertible to gptq_marlin during runtime based on HW
# more detail: https://docs.vllm.ai/en/latest/quantization/supported_hardware.html

MODEL_NAMES = ["llama-2-7b-chat-gptq"]
DEPLOYMENT_TYPES = ["RawDeployment", "Serverless"]

COMPLETION_QUERY = {
    "text": "Write a code to find the maximum value in a list of numbers.",
    "output_tokens": 1000
}

CHAT_QUERY = [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Do you have mayonnaise recipes?"
      }
    ]


@pytest.mark.smoke
@pytest.mark.parametrize("deployment_type", DEPLOYMENT_TYPES)
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_llama_2_7b_chat_gptq_simple(client: DynamicClient,
                                     run_static_command: Callable[[str], None],
                                     response_snapshot: Any,
                                     create_namespace: Callable[[str], Resource],
                                     create_secret_from_file: Callable[[str], Resource],
                                     create_service_account: Callable[[str], Resource],
                                     create_serving_runtime_from_file: Callable[[str, str], Resource],
                                     create_isvc_from_file: Callable[[str, str], Resource],
                                     model_name: str,
                                     deployment_type: str,
                                     runtime: str,
                                     runtime_image: str,
                                     accelerator_type: str,
                                     runtime_name: str) -> None:
    """
    Test function for validating the deployment and serving of a model in a Kubernetes environment.

    This function performs the following steps:
    1. Creates necessary Kubernetes resources (namespace, secret, service account, serving runtime, and inference service).
    2. Waits for the predictor pod to be in a "Running" and "Ready" state.
    3. Depending on the deployment type, performs port-forwarding or uses the provided URL to access the model.
    4. Sends requests to the model and compares responses with predefined snapshots.

    Args:
        client (DynamicClient): The client used to interact with the Kubernetes cluster.
        run_static_command (Callable[[str], None]): A function to execute static commands in the environment.
        response_snapshot (Any): A snapshot object for response comparison.
        create_namespace (Callable[[str], Resource]): A function to create a new Kubernetes namespace.
        create_secret_from_file (Callable[[str], Secret]): A function to create a Kubernetes secret from a file.
        create_service_account (Callable[[str], ServiceAccount]): A function to create a Kubernetes service account.
        create_serving_runtime_from_file (Callable[[str, str], ServingRuntime]): A function to create a serving runtime from a file.
        create_isvc_from_file (Callable[[str, str], InferenceService]): A function to create an inference service from a file.
        model_name (str): The name of the model to be deployed.
        deployment_type (str): The type of deployment (e.g., "rawdeployment" or "serverless").
        runtime (str, optional): The runtime environment. Defaults to "vLLM".
        runtime_name (str, optional): The name of the serving runtime. Defaults to "serving_runtime".
    """
    namespace_name = model_name.lower()
    create_runtime_manifest_from_template(deployment_type, runtime_image, runtime_name)
    create_isvc_manifest_from_template(deployment_type, model_name, accelerator_type=accelerator_type, gpu_count=1)
    create_s3_secret_manifest()
    namespace = create_namespace(namespace_name)
    secret = create_secret_from_file(namespace=namespace.name)
    service_account = create_service_account(namespace=namespace.name)
    serving_runtime = create_serving_runtime_from_file(namespace=namespace.name, path=runtime)
    inference_service = create_isvc_from_file(namespace=namespace.name, model_name=model_name)
    time.sleep(10)
    predictor_pod = get_predictor_pod(client, namespace=namespace.name, is_name=inference_service.name)
    predictor_pod.wait_for_status("Running", timeout=600)
    predictor_pod.wait_for_condition("Ready", "True", timeout=600)
    time.sleep(10)
    LOGGER.info(f"Model statuts: {inference_service.instance.status.modelStatus.states.activeModelState}")
    if inference_service.instance.status.modelStatus.states.activeModelState != "Loaded":
        pytest.fail("Model is not in Loaded state")
    if deployment_type.lower() == "rawdeployment":
        #grpc
        cmd = f"oc -n {namespace_name} port-forward pod/{predictor_pod.name} 8033:8033"
        run_static_command(cmd)
        url = "localhost:8033"
        tgis_client = TGISGRPCPlugin(host=url, model_name=model_name, streaming=True)
        all_token = tgis_client.make_grpc_request(COMPLETION_QUERY)
        LOGGER.info(all_token)
        model_info = tgis_client.get_model_info()
        LOGGER.info(model_info)
        stream = tgis_client.make_grpc_request_stream(COMPLETION_QUERY)
        LOGGER.info(stream)
        assert all_token == response_snapshot
        assert model_info == response_snapshot
        assert stream == response_snapshot
        # Forward port to access the service locally
        cmd = f"oc -n {namespace_name} port-forward pod/{predictor_pod.name} 8080:8080"
        run_static_command(cmd)
        url = "http://localhost:8080"

        openai_client = OpenAIClient(host=url, model_name=model_name)
        completion_response = openai_client.request_http(endpoint="/v1/completions", query=COMPLETION_QUERY)
        chat_response = openai_client.request_http(endpoint="/v1/chat/completions", query=CHAT_QUERY)

        assert completion_response == response_snapshot
        assert chat_response == response_snapshot

    elif deployment_type.lower() == "serverless":
        url = inference_service.instance.status.url
        LOGGER.info(url)

        openai_client = OpenAIClient(host=url + ":443", model_name=model_name)
        completion_response = openai_client.request_http(endpoint="/v1/completions", query=COMPLETION_QUERY)
        chat_response = openai_client.request_http(endpoint="/v1/chat/completions", query=CHAT_QUERY)

        assert completion_response == response_snapshot
        assert chat_response == response_snapshot
    else:
        LOGGER.warning("Deployment type is not provided correctly.")


MODEL_NAMES = ["mistral-7b-v01-gptq-marlin-4bit"]


@pytest.mark.smoke
@pytest.mark.parametrize("deployment_type", DEPLOYMENT_TYPES)
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_mistral_7b_v01_gptq_marlin_4bit_simple(client: DynamicClient,
                                                run_static_command: Callable[[str], None],
                                                response_snapshot: Any,
                                                create_namespace: Callable[[str], Resource],
                                                create_secret_from_file: Callable[[str], Resource],
                                                create_service_account: Callable[[str], Resource],
                                                create_serving_runtime_from_file: Callable[[str, str], Resource],
                                                create_isvc_from_file: Callable[[str, str], Resource],
                                                model_name: str,
                                                deployment_type: str,
                                                runtime: str,
                                                runtime_image: str,
                                                accelerator_type: str,
                                                runtime_name: str) -> None:
    """
    Test function for validating the deployment and serving of a model in a Kubernetes environment.

    This function performs the following steps:
    1. Creates necessary Kubernetes resources (namespace, secret, service account, serving runtime, and inference service).
    2. Waits for the predictor pod to be in a "Running" and "Ready" state.
    3. Depending on the deployment type, performs port-forwarding or uses the provided URL to access the model.
    4. Sends requests to the model and compares responses with predefined snapshots.

    Args:
        client (DynamicClient): The client used to interact with the Kubernetes cluster.
        run_static_command (Callable[[str], None]): A function to execute static commands in the environment.
        response_snapshot (Any): A snapshot object for response comparison.
        create_namespace (Callable[[str], Resource]): A function to create a new Kubernetes namespace.
        create_secret_from_file (Callable[[str], Secret]): A function to create a Kubernetes secret from a file.
        create_service_account (Callable[[str], ServiceAccount]): A function to create a Kubernetes service account.
        create_serving_runtime_from_file (Callable[[str, str], ServingRuntime]): A function to create a serving runtime from a file.
        create_isvc_from_file (Callable[[str, str], InferenceService]): A function to create an inference service from a file.
        model_name (str): The name of the model to be deployed.
        deployment_type (str): The type of deployment (e.g., "rawdeployment" or "serverless").
        runtime (str, optional): The runtime environment. Defaults to "vLLM".
        runtime_name (str, optional): The name of the serving runtime. Defaults to "serving_runtime".
    """
    namespace_name = "mistral-marlin-test"
    create_runtime_manifest_from_template(deployment_type, runtime_image, runtime_name)
    create_isvc_manifest_from_template(deployment_type, model_name, accelerator_type=accelerator_type, gpu_count=1)
    create_s3_secret_manifest()
    namespace = create_namespace(namespace_name)
    secret = create_secret_from_file(namespace=namespace.name)
    service_account = create_service_account(namespace=namespace.name)
    serving_runtime = create_serving_runtime_from_file(namespace=namespace.name, path=runtime)
    inference_service = create_isvc_from_file(namespace=namespace.name, model_name=model_name)
    time.sleep(10)
    predictor_pod = get_predictor_pod(client, namespace=namespace.name, is_name=inference_service.name)
    predictor_pod.wait_for_status("Running", timeout=600)
    predictor_pod.wait_for_condition("Ready", "True", timeout=600)
    time.sleep(10)
    LOGGER.info(f"Model statuts: {inference_service.instance.status.modelStatus.states.activeModelState}")
    if inference_service.instance.status.modelStatus.states.activeModelState != "Loaded":
        pytest.fail("Model is not in Loaded state")
    if deployment_type.lower() == "rawdeployment":
        #grpc
        cmd = f"oc -n {namespace_name} port-forward pod/{predictor_pod.name} 8033:8033"
        run_static_command(cmd)
        url = "localhost:8033"
        tgis_client = TGISGRPCPlugin(host=url, model_name=model_name, streaming=True)
        all_token = tgis_client.make_grpc_request(COMPLETION_QUERY)
        LOGGER.info(all_token)
        model_info = tgis_client.get_model_info()
        LOGGER.info(model_info)
        stream = tgis_client.make_grpc_request_stream(COMPLETION_QUERY)
        LOGGER.info(stream)
        assert all_token == response_snapshot
        assert model_info == response_snapshot
        assert stream == response_snapshot
        # Forward port to access the service locally
        cmd = f"oc -n {namespace_name} port-forward pod/{predictor_pod.name} 8080:8080"
        run_static_command(cmd)
        url = "http://localhost:8080"

        openai_client = OpenAIClient(host=url, model_name=model_name)
        completion_response = openai_client.request_http(endpoint="/v1/completions", query=COMPLETION_QUERY)
        chat_response = openai_client.request_http(endpoint="/v1/chat/completions", query=CHAT_QUERY)

        assert completion_response == response_snapshot
        assert chat_response == response_snapshot

    elif deployment_type.lower() == "serverless":
        url = inference_service.instance.status.url
        LOGGER.info(url)

        openai_client = OpenAIClient(host=url + ":443", model_name=model_name)
        completion_response = openai_client.request_http(endpoint="/v1/completions", query=COMPLETION_QUERY)
        chat_response = openai_client.request_http(endpoint="/v1/chat/completions", query=CHAT_QUERY)

        assert completion_response == response_snapshot
        assert chat_response == response_snapshot
    else:
        LOGGER.warning("Deployment type is not provided correctly.")