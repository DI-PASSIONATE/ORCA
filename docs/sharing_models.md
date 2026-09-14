---
title: Sharing Models on Hugging Face
description: >-
  Publish an ORCA-trained ONNX surrogate model of an RF passive to the Hugging
  Face Hub with the orca-surrogate tag so COBRA and other designers can reuse it
  without re-running EM simulations.
---

# Sharing Models on Hugging Face

After training a surrogate model with ORCA, you can publish it to [Hugging Face](https://huggingface.co) so that COBRA — or anyone else — can discover and use it directly. Sharing models reduces redundant EM simulations across the community.

## Requirements

- A Hugging Face account
- The `huggingface_hub` Python package: `pip install huggingface_hub`

## File structure

Each model repository must contain exactly two files named after the model:

| File | Description |
|------|-------------|
| `<model_name>.onnx` | The exported ONNX surrogate model produced by `OnnxExporter` |
| `<model_name>.py` | The Python geometry class (subclass of `BaseGeometry`) used to generate and train the model |

The geometry class file is required so that COBRA can reconstruct the parameter space, call back into the geometry for EM verification, and correctly pre-process inference inputs.

## Step-by-step upload

1. **Create a new model repository** at [https://huggingface.co/new](https://huggingface.co/new).
   Set visibility to **Public** and note the repository ID (e.g. `your-username/tf-octa-c-ports`). Click on "Create model".

2. Create a **Model Card** (essentially just a structured README) for your repository. Click on "Add Model Card". From there, add the tag `orca-surrogate` to make it discoverable by COBRA and other users looking for ORCA models. The model card should then include this section:

    ```markdown
    ---
    tags:
    - orca-surrogate
    ```

3. **Upload the files** using the `huggingface_hub` library:

    ```python
    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = "your-username/tf-octa-c-ports"  # replace with your repo

    api.upload_file(path_or_fileobj="tf_octa_c_ports.onnx", path_in_repo="tf_octa_c_ports.onnx", repo_id=repo_id)
    api.upload_file(path_or_fileobj="tf_octa_c_ports.py",   path_in_repo="tf_octa_c_ports.py",   repo_id=repo_id)
    ```

    Or via the Hugging Face web interface: go to your repository → **Files** → **Add file → Upload files**.

4. **Verify** the repository contains both `<model_name>.onnx` and `<model_name>.py` and is tagged `orca-surrogate`.

## Using a shared model in COBRA

Once uploaded, COBRA can query all public `orca-surrogate` models or load a specific one directly by its Hugging Face repository ID. Refer to the [COBRA documentation](https://github.com/DI-PASSIONATE/COBRA) for details on how to point COBRA at a Hugging Face model repository.
