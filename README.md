# State Representation Learning for Goal-Conditioned Reinforcement Learning

**[Open in Google Colab](https://colab.research.google.com/github/lorenzosteccanella/SRL/blob/main/Example_colab.ipynb)**

This repository contains code for learning state representations that are effective for goal-conditioned reinforcement learning (GCRL). It implements algorithms described in the following papers:

*   State Representation Learning for Goal-Conditioned Reinforcement Learning
    Lorenzo Steccanella and Anders Jonsson
    *Joint European Conference on Machine Learning and Knowledge Discovery in Databases*, 2022.

*   Asymmetric Norms to Approximate the Minimum Action Distance
    Lorenzo Steccanella, Anders Jonsson
    *Workshop Submission (details to be updated upon publication)*

**Key Idea:**

The core of the approach is learning a latent space where distances reflect the minimum number of actions required to transition between states. The code includes implementations for both symmetric and asymmetric norms. The asymmetric norm parametrization enables accurate approximations of minimum action distances in environments with inherent asymmetry. This learned representation allows for efficient planning in GCRL tasks. The repository provides implementations of distance metric learning and action encoding, combined into a learned model that can be used for planning.

**Repository Structure:**

*   `Data`: Contains example trajectories for different environments.
*   `Envs`:  Environment definitions (e.g., `PointmassEnv`, `GridWorldEnv`).
*   `ExpReplay`:  Experience replay buffer implementation for distance learning.
*   `Models`: PyTorch modules for distance encoders, action encoders, and learned models.  Includes implementations for both symmetric and asymmetric norms.
*   `Planning`: Implementation of the planning algorithm (`Planning_alg.py`).
*   `Utils`: Utility functions for data collection, plotting, etc.

**Getting Started:**

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/lorenzosteccanella/SRL.git
    cd SRL
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Example Usage:**

    *   **Using Google Colab:** You can use the `Example_colab.ipynb` notebook in Google Colab, which allows you to run the code without local setup. A direct link is provided: [Example_colab.ipynb](https://colab.research.google.com/github/lorenzosteccanella/SRL/blob/main/Example_colab.ipynb)

    *   **Running the examples locally:** The `Example.ipynb` notebook provides a complete pipeline, from training a model to performing planning for a goal-conditioned task.  Open it using Jupyter:

        ```bash
        jupyter notebook Example.ipynb
        ```
