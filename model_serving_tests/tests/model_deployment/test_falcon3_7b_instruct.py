from typing import Any, Callable
import pytest
import re
import subprocess
import csv
import os
from kubernetes.dynamic.client import DynamicClient
from ocp_resources.resource import Resource
from model_serving_tests.endpoint_utility.openai_utility import OpenAIClient
from model_serving_tests.endpoint_utility.grpc_utility import TGISGRPCPlugin
from model_serving_tests.tests.utils import create_runtime_manifest_from_template, create_isvc_manifest_from_template, \
    get_predictor_pod, create_s3_secret_manifest
import logging
import time

LOGGER = logging.getLogger(__name__)

MODEL_NAMES =  ['falcon3-7b-instruct']
DEPLOYMENT_TYPES = ["RawDeployment"]

COMPLETION_QUERY = {
    "text": "List the top five breeds of dogs and their characteristics.",
}

CHAT_QUERY = [
    {
        "role": "user",
        "content": "Explain quantum computing in simple terms."
    }
]

def get_vllm_version(namespace, pod_name):
    cmd = f'oc exec -n {namespace} {pod_name} -- python -c "import vllm; print(vllm.__version__)"'
    result = subprocess.check_output(cmd, shell=True, text=True)
    return result.strip()

def get_vllm_throughput_logs(namespace, pod_name):
    cmd = f"oc logs -n {namespace} {pod_name} | grep 'Avg prompt throughput'"
    result = subprocess.getoutput(cmd)
    return result

def log_server_performance(model_name, version, logs):
    LOGGER.info(f"Model: {model_name}")
    LOGGER.info(f"VLLM Version: {version}")
    LOGGER.info("====SERVER LOGS====")
    LOGGER.info(logs)

def parse_vllm_logs(logs, start_time, used_entries):
    parsed = []
    for line in logs.split("\n"):
        time_match = re.search(r"(\d{2}:\d{2}:\d{2})", line)
        if not time_match:
            continue
        log_time = time_match.group(1)

        if log_time < start_time:
            continue
        if line in used_entries:
            continue

        match = re.search(r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s", line)
        if match:
            entry = ({
                "prompt_tokens_per_sec": float(match.group(1)),
                "generation_tokens_per_sec": float(match.group(2))
                })
            parsed.append(entry)
            used_entries.add(line)

    return parsed

def save_performance_report(model_name, version, logs, request_type, input_prompt, start_time, used_entries):
    parsed_logs = parse_vllm_logs(logs, start_time, used_entries)

    last = parsed_logs[-1] if parsed_logs else {}
    max_prompt = max([x["prompt_tokens_per_sec"] for x in parsed_logs], default=0)
    max_generation = max([x["generation_tokens_per_sec"] for x in parsed_logs], default=0)

    file_exists = os.path.isfile("performance_report.csv")
    with open("performance_report.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "model",
                "vllm_version",
                "request_type",
                "input_prompt",
                "last_prompt_tokens_per_sec",
                "last_generation_tokens_per_sec",
                "max_prompt_tokens_per_sec",
                "max_generation_tokens_per_sec"
                ])
        writer.writerow([
            model_name,
            version,
            request_type,
            input_prompt,
            last.get("prompt_tokens_per_sec", 0),
            last.get("generation_tokens_per_sec", 0),
            max_prompt,
            max_generation
            ])

@pytest.mark.falcon
@pytest.mark.smoke
@pytest.mark.parametrize("deployment_type", DEPLOYMENT_TYPES)
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_falcon3_7b_instruct_simple(client: DynamicClient,
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
        # Forward port to access the service locally
        cmd = f"oc -n {namespace_name} port-forward pod/{predictor_pod.name} 8080:8080"
        run_static_command(cmd)
        url = "http://localhost:8080"

        openai_client = OpenAIClient(host=url, model_name=model_name)

        #Get vLLM version
        vllm_version = get_vllm_version(namespace_name, predictor_pod.name)
        
        #Completion
        used_entries = set()
        start_time = time.strftime("%H:%M:%S")
        completion_response = openai_client.request_http(endpoint="/v1/completions", query=COMPLETION_QUERY,
                                                             extra_param={'temperature': 0})
        time.sleep(2)
        completion_logs = get_vllm_throughput_logs(namespace_name, predictor_pod.name)
        save_performance_report(model_name, vllm_version, completion_logs, "completion", COMPLETION_QUERY["text"], start_time, used_entries)

        #Chat
        start_time = time.strftime("%H:%M:%S")
        chat_response = openai_client.request_http(endpoint="/v1/chat/completions", query=CHAT_QUERY,
                                                             extra_param={'temperature': 0})
        time.sleep(2)
        chat_logs = get_vllm_throughput_logs(namespace_name, predictor_pod.name)
        save_performance_report(model_name, vllm_version, chat_logs, "chat", CHAT_QUERY[0]["content"], start_time, used_entries)

        #assert completion_response == response_snapshot
        time.sleep(300)
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

@pytest.mark.falcon
@pytest.mark.smoke
@pytest.mark.xfail(reason="This test is expected to fail with the error input tokens (12) plus prefix length (0) must "
                          "be < 10.for grpc endpoint. For openai endpoint it will throw request error with http status "
                          "400")
@pytest.mark.parametrize("deployment_type", DEPLOYMENT_TYPES)
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_falcon3_7b_instruct_seq_len(client: DynamicClient,
                                        run_static_command: Callable[[str], None],
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
    Test function for validating the deployment and serving of a model with small model length

    Args:
        client (DynamicClient): The client used to interact with the Kubernetes cluster.
        run_static_command (Callable[[str], None]): A function to execute static commands in the environment.
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
    create_isvc_manifest_from_template(deployment_type, model_name, accelerator_type=accelerator_type,
                                       new_args=["--max-model-len=10"])
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
    if inference_service.instance.status.modelStatus.states.activeModelState != "Loaded":
        pytest.fail("Model is not in Loaded state")

    if deployment_type.lower() == "rawdeployment":
        cmd = f"oc -n {namespace_name} port-forward pod/{predictor_pod.name} 8033:8033"
        run_static_command(cmd)
        url = "localhost:8033"
        tgis_client = TGISGRPCPlugin(host=url, model_name=model_name, streaming=True)
        model_info = tgis_client.get_model_info()
        LOGGER.info(model_info)
        try:
            all_token = tgis_client.make_grpc_request(COMPLETION_QUERY)
            LOGGER.info(all_token)
            assert response is not None
        except grpc.RpcError as e:
            error_message = e.details()
            if "input tokens (12) plus prefix length (0) must be < 10" in error_message:
                pytest.xfail(f"Expected failure occurred: {error_message}")
            else:
                pytest.fail(f"Unexpected gRPC error: {error_message}")
    elif deployment_type.lower() == "serverless":
        url = inference_service.instance.status.url
        LOGGER.info(url)
        openai_client = OpenAIClient(host=url + ":443", model_name=model_name)
        chat_response = openai_client.request_http(endpoint="/v1/chat/completions", query=CHAT_QUERY)
    else:
        LOGGER.warning("Deployment type is not provided correctly.")
