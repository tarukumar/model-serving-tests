import time
import subprocess
import signal
import os
import pytest
from syrupy.extensions.json import JSONSnapshotExtension
from ocp_resources.serving_runtime import ServingRuntime
from ocp_resources.inference_service import InferenceService
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.resource import get_client
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
import logging
from model_serving_tests.tests.constant import INFERE_DIR, RUNTIME_DIR, STORAGE_DIR

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
# Define constants for timeouts
DELETE_TIMEOUT = 600


def pytest_addoption(parser):
    parser.addoption(
        "--runtime-image",
        action="store",
        default="quay.io/modh/vllm@sha256:94e2d256da29891a865103f7e92a1713f0fd385ef611c6162526f4a297e70916",
        help="Specify the runtime image to use for the tests"
    )

    parser.addoption(
        "--runtime",
        action="store",
        default="vLLM",
        help="Specify the runtime folder name"
    )

    parser.addoption(
        "--accelerator_type",
        action="store",
        default="Nvidia",
        help="Specify the Accelerator type"
    )

    parser.addoption(
        "--runtime_name",
        action="store",
        default= "serving_runtime",
        help="Specify the runtime file name"
    )


@pytest.fixture(scope="session")
def runtime_image(request):
    """Fixture to get the runtime image option."""
    return request.config.getoption("--runtime-image")


@pytest.fixture(scope="session")
def runtime(request):
    """Fixture to get the runtime folder name option."""
    return request.config.getoption("--runtime")


@pytest.fixture(scope="session")
def accelerator_type(request):
    """Fixture to get the GPU locator option."""
    return request.config.getoption("--accelerator_type")


@pytest.fixture(scope="session")
def runtime_name(request):
    """Fixture to get the runtime folder name option."""
    return request.config.getoption("--runtime_name")


@pytest.fixture(scope="session")
def client() -> DynamicClient:
    yield get_client()


@pytest.fixture
def create_namespace(client: DynamicClient):
    """
    Factory to create and delete Namespaces needed in a test class
    """

    created_namespaces = {}

    def _create_namespace(name):
        if name in created_namespaces.keys():
            return created_namespaces[name]
        LOGGER.info("CREATING NEW NAMESPACE")
        ns = Namespace(client=client, name=name, delete_timeout=DELETE_TIMEOUT)
        ns.create()
        LOGGER.info("WAITING FOR NS STATUS TO BECOME ACTIVE")
        ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=120)
        created_namespaces[name] = ns
        LOGGER.info(f"NS {name} IS ACTIVE, RETURNING")
        return ns

    yield _create_namespace

    for ns in created_namespaces.values():
        ns.delete(wait=True)


@pytest.fixture
def create_serving_runtime_from_file(client: DynamicClient):
    _runtimes = []

    def _create_runtime(namespace, path="vLLM", runtime_name="serving_runtime"):
        LOGGER.info("CREATING NEW RUNTIME")
        runtime_yaml = RUNTIME_DIR / path / f'{runtime_name}_updated.yaml'
        runtime = ServingRuntime(client=client, namespace=namespace, yaml_file=runtime_yaml)
        runtime.create()
        _runtimes.append(runtime)
        return runtime

    yield _create_runtime
    for runtime in _runtimes:
        runtime.delete(wait=True)


@pytest.fixture
def create_secret_from_file(client: DynamicClient):
    _secret = []

    def _create_secret(namespace, path="storage_config", file_name="s3_seceret"):
        LOGGER.info("CREATING secret")
        s3_path = STORAGE_DIR / f'{file_name}.yaml'
        s3_secret = Secret(client=client, namespace=namespace, yaml_file=s3_path)
        s3_secret.create()
        _secret.append(s3_secret)
        return s3_secret

    yield _create_secret
    for s3 in _secret:
        s3.delete(wait=True)


@pytest.fixture
def create_service_account(client: DynamicClient):
    _service = []

    def _sa(namespace):
        LOGGER.info("CREATING SA")
        service_account = ServiceAccount(client=client, name="modelmesh-serving-sa", namespace=namespace,
                                         secrets=[{'name': 'models-bucket-secret'}])
        service_account.create()
        _service.append(service_account)
        return service_account

    yield _sa
    for sa in _service:
        sa.delete(wait=True)


@pytest.fixture
def create_isvc_from_file(client: DynamicClient):
    _isvc = []

    def _create_isvc(namespace, model_name, path='vLLM'):
        LOGGER.info("CREATING INFERENCE SERVICE")
        inference_yaml = INFERE_DIR / model_name / f'{model_name}_updated.yaml'
        isvc = InferenceService(client=client, namespace=namespace, yaml_file=inference_yaml)
        isvc.create()
        _isvc.append(isvc)
        return isvc

    yield _create_isvc
    for runtime in _isvc:
        runtime.delete(wait=True)


@pytest.fixture
def response_snapshot(snapshot):
    return snapshot.use_extension(JSONSnapshotExtension)


@pytest.fixture
def run_static_command():
    processes = []

    def run_cmd(cmd):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setpgrp,
                                   shell=True)
        LOGGER.info(f"Running command: {cmd}")
        processes.append(process)
        time.sleep(10)
        return process

    yield run_cmd

    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)  # Send SIGTERM to the process group
        except OSError as e:
            LOGGER.debug(f"Error terminating process: {e}")

        # Ensure the process is fully terminated
        process.stdout.close()
        process.stderr.close()
        try:
            process.wait(timeout=5)  # Wait for process to exit
        except subprocess.TimeoutExpired:
            LOGGER.debug(f"Process {process.pid} did not exit in time, force killing...")
            process.kill()  # Force kill if it didn't exit in time
            process.wait()  # Ensure it has terminated
        except Exception as e:
            LOGGER.debug(f"Exception occurred: {e}")


