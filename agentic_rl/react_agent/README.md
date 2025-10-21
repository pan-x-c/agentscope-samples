# Training ReAct Agent with RL Example

This example demonstrates how to train a **ReAct** agent with RL using the AgentScope `learn` module.


## How to run

After implementing the workflow function, follow these steps to run the training:

1. Prerequisites

    - At least 2 NVIDIA GPUs with CUDA 12.4 or newer.
    - Follow the Trinity-RFT [installation guide](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/trinity_installation.html) to install a compatible version (Trinity-RFT >= 0.3.1).
    - Adjust the configuration file ([config.yaml](./config.yaml)) following [configuration guide](https://modelscope.github.io/Trinity-RFT/en/main/tutorial/trinity_configs.html).
    - Download the GSM8K dataset and Qwen/Qwen3-8B model checkpoints (example):

      ```bash
      huggingface-cli download openai/gsm8k --repo-type dataset
      huggingface-cli download Qwen/Qwen3-8B
      ```

2. Set up a [Ray](https://github.com/ray-project/ray) cluster

    ```bash
    ray start --head
    # for multi-node setup, run the following command on worker nodes
    # ray start --address=<master_address>
    ```

3. Run the training script

    ```bash
    python main.py
    ```
