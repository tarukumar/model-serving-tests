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

MODEL_NAMES = ["granite-2b-instruct-4k"]
DEPLOYMENT_TYPES = ["RawDeployment", "Serverless"]

COMPLETION_QUERY = [{
    "text": "List the top five breeds of dogs and their characteristics.",
},
    {"text": "Translate the following English sentence into Japanese, French, and Swahili: 'The early bird catches "
             "the worm.'"},
    {"text": "Write a short story about a robot that dreams for the first time."},
    {"text": "Explain the cultural significance of the Mona Lisa painting, and how its perception might vary in "
             "Western versus Eastern societies."},
    {
        "text": "Compare and contrast artificial intelligence with human intelligence in terms of processing information."},
    {"text": "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020."}
]
CHAT_QUERY = [[
    {
        "role": "user",
        "content": "Write python code to find even number"
    }
], [
    {"role": "system",
     "content": "Given a target sentence, construct the underlying meaning representation of the input sentence as a "
                "single function with attributes and attribute values."},
    {"role": "user",
     "content": "SpellForce 3 is a pretty bad game. The developer Grimlore Games is clearly a bunch of no-talent "
                "hacks, and 2017 was a terrible year for games anyway."}
]]


@pytest.mark.smoke
@pytest.mark.granite4k
@pytest.mark.parametrize("deployment_type", DEPLOYMENT_TYPES)
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_granite_2b_instruct_4k_simple(client: DynamicClient,
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
        model_info = tgis_client.get_model_info()
        LOGGER.info(model_info)
        all_token = []
        stream = []
        for query in COMPLETION_QUERY:
            all_tokens = tgis_client.make_grpc_request(query)
            LOGGER.info(all_tokens)
            all_token.append(all_tokens)
            streams = tgis_client.make_grpc_request_stream(query)
            LOGGER.info(streams)
            stream.append(streams)

        assert all_token == response_snapshot
        assert model_info == response_snapshot
        assert stream == response_snapshot
        # Forward port to access the service locally
        cmd = f"oc -n {namespace_name} port-forward pod/{predictor_pod.name} 8080:8080"
        run_static_command(cmd)
        url = "http://localhost:8080"
        completion_response = []
        openai_client = OpenAIClient(host=url, model_name=model_name)
        for query in COMPLETION_QUERY:
            completion_responses = openai_client.request_http(endpoint="/v1/completions", query=query)
            completion_response.append(completion_responses)
        chat_response = []
        for query in CHAT_QUERY:
            chat_responses = openai_client.request_http(endpoint="/v1/chat/completions", query=query)
            chat_response.append(chat_responses)

        assert completion_response == response_snapshot
        assert chat_response == response_snapshot

    elif deployment_type.lower() == "serverless":
        url = inference_service.instance.status.url
        LOGGER.info(url)

        openai_client = OpenAIClient(host=url + ":443", model_name=model_name)
        completion_response = openai_client.request_http(endpoint="/v1/completions", query=COMPLETION_QUERY[0])
        chat_response = []
        for query in CHAT_QUERY:
            chat_responses = openai_client.request_http(endpoint="/v1/chat/completions", query=query)
            chat_response.append(chat_responses)

        assert completion_response == response_snapshot
        assert chat_response == response_snapshot
    else:
        LOGGER.warning("Deployment type is not provided correctly.")


@pytest.mark.multigpu
@pytest.mark.granite4k
@pytest.mark.parametrize("deployment_type", DEPLOYMENT_TYPES)
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_granite_2b_instruct_4k_multi_gpu(client: DynamicClient,
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
    Test function for validating the deployment and serving of a model with multi-GPU configuration in a Kubernetes environment.

    This function performs similar steps to the simple test, but with a multi-GPU setup and additional configuration.

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
    create_isvc_manifest_from_template(deployment_type, model_name, accelerator_type=accelerator_type, gpu_count=2)
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
        #grpc
        cmd = f"oc -n {namespace_name} port-forward pod/{predictor_pod.name} 8033:8033"
        run_static_command(cmd)
        url = "localhost:8033"
        tgis_client = TGISGRPCPlugin(host=url, model_name=model_name, streaming=True)
        model_info = tgis_client.get_model_info()
        LOGGER.info(model_info)
        all_token = []
        stream = []
        for query in COMPLETION_QUERY:
            all_tokens = tgis_client.make_grpc_request(query)
            LOGGER.info(all_tokens)
            all_token.append(all_tokens)
            streams = tgis_client.make_grpc_request_stream(query)
            LOGGER.info(streams)
            stream.append(streams)

        assert all_token == response_snapshot
        assert model_info == response_snapshot
        assert stream == response_snapshot
        # Forward port to access the service locally
        cmd = f"oc -n {namespace_name} port-forward pod/{predictor_pod.name} 8080:8080"
        run_static_command(cmd)
        url = "http://localhost:8080"
        completion_response = []
        openai_client = OpenAIClient(host=url, model_name=model_name)
        for query in COMPLETION_QUERY:
            completion_responses = openai_client.request_http(endpoint="/v1/completions", query=query)
            completion_response.append(completion_responses)
        chat_response = []
        for query in CHAT_QUERY:
            chat_responses = openai_client.request_http(endpoint="/v1/chat/completions", query=query)
            chat_response.append(chat_responses)

    elif deployment_type.lower() == "serverless":
        url = inference_service.instance.status.url
        LOGGER.info(url)

        openai_client = OpenAIClient(host=url + ":443", model_name=model_name)
        completion_response = openai_client.request_http(endpoint="/v1/completions", query=COMPLETION_QUERY[0])
        chat_response = openai_client.request_http(endpoint="/v1/chat/completions", query=CHAT_QUERY[0])

        assert completion_response == response_snapshot
        assert chat_response == response_snapshot

    else:
        LOGGER.warning("Deployment type is not provided correctly.")


@pytest.mark.smoke
@pytest.mark.granite4k
@pytest.mark.xfail(reason="This test is expected to fail with the error input tokens (14) plus prefix length (0) must "
                          "be < 4 for grpc endpoint. For openai endpoint it will throw request error with http status "
                          "400")
@pytest.mark.parametrize("deployment_type", DEPLOYMENT_TYPES)
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_granite_2b_instruct_4k_seq_len(client: DynamicClient,
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
    Test function for validating the deployment and serving of a model with small model length (4)

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
                                       new_args=["--max-model-len=4"])
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
        all_token = tgis_client.make_grpc_request(COMPLETION_QUERY[0])
        LOGGER.info(all_token)

    elif deployment_type.lower() == "serverless":
        url = inference_service.instance.status.url
        LOGGER.info(url)

        openai_client = OpenAIClient(host=url + ":443", model_name=model_name)
        chat_response = openai_client.request_http(endpoint="/v1/chat/completions", query=CHAT_QUERY[0])
    else:
        LOGGER.warning("Deployment type is not provided correctly.")


@pytest.mark.smoke
@pytest.mark.granite4k
@pytest.mark.xfail(reason="This test is expected to fail with message Using beam search as a sampling parameter is "
                          "deprecated, and will be removed in the future release. Please use the "
                          "`vllm.LLM.use_beam_search` method for dedicated beam search instead, or set the "
                          "environment variable `VLLM_ALLOW_DEPRECATED_BEAM_SEARCH=1` to suppress this error.")
@pytest.mark.parametrize("deployment_type", [DEPLOYMENT_TYPES[0]])
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_granite_2b_instruct_4k_beam_search(client: DynamicClient,
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
    Test function for validating the deployment and serving of a model with small model length (4)

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
    create_isvc_manifest_from_template(deployment_type, model_name, accelerator_type=accelerator_type)
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
        cmd = f"oc -n {namespace_name} port-forward pod/{predictor_pod.name} 8080:8080"
        run_static_command(cmd)
        url = "http://localhost:8080"
        completion_response = []
        openai_client = OpenAIClient(host=url, model_name=model_name)
        completion_response = openai_client.request_http(endpoint="/v1/completions", query=COMPLETION_QUERY[0],
                                                             extra_param={'use_beam_search': True, 'best_of': 3})
        LOGGER.info(completion_response)
    else:
        LOGGER.warning("Deployment type is not provided correctly.")
