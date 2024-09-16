# model-serving-tests
[WIP]Framework for Model serving functional integration tests.

## Overview
- Leverages [kubernetes python client](https://github.com/kubernetes-client/python) and [openshift-python-wrapper](https://github.com/RedHatQE/openshift-python-wrapper)
- The idea is to have the flexibility to do anything we would do through the command line, but programmatically through the K8S/OpenShift APIs, and to be able to work with cluster resources using simple Python objects.
- In this PoC, pytest was used (since pytest fixtures integrate nicely with openshift-python-wrapper), but similar results could be achieved using other testing frameworks
- These tests in principle could be run in different environments (vanilla K8S and OpenShift with OpenDataHub or OpenShift AI) with minimum effort, since they use the K8S/OpenShift API directly

## Directory
- [endpoint_utility](https://github.com/tarukumar/model-serving-tests/tree/main/model_serving_tests/endpoint_utility): Openai and Tgis endpoint to query the model
- [model_inference](https://github.com/adolfo-ab/trustyai-tests/tree/main/resources): Model Inferece template for model, This is based on Jinja template
- [runtimes](https://github.com/tarukumar/model-serving-tests/blob/main/model_serving_tests/model_config/runtimes): Runtime manifest template based on Jinja. As of now template for vllm is created
- [storage_config](https://github.com/tarukumar/model-serving-tests/blob/main/model_serving_tests/storage_config): s3 manifest for the model to be downloaded
- [tests](https://github.com/tarukumar/model-serving-tests/tree/main/model_serving_tests/tests): tests,constant,utils and pytest fixtures used in the PoC.
## Running the tests
- `export KUBECONFIG=${path to the kubeconfig of your cluster}`, or alternatively `oc login` into your cluster.
- Make sure you have Poetry installed.
- Install the project's dependencies with `poetry install`.
- Configure `pre-commit`
- When adding a new model. kindly run the test suite using ` poetry run pytest tests/your_tests.py --snapshot-update `. With this snapshot will be automatclly created for each condition of the output comparison. This needs to be done only once during intial development. 
- Run all the tests with `poetry run pytest`
- To run test with specfic runtime image with diffrent accelerator(supported: nvidia,amd,intel) run below command :

   `poetry run pytest -m smoke --runtime-image=quay.io/opendatahub/vllm:stable --accelerator_type=habana`
