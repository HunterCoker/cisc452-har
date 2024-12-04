# About this project

We are using the **UCI WISDM HAR** dataset for this project.

Technologies used:
- Visual Studio Code
- Anaconda3 with Python 3.10

# Project Scripts

There are two main scripts that this project is built around.
- `scripts/train.py`: Allows you to generate new models using the desired training configuration.
  > The user does not have access to the model's internal paraemeters from this script, but the user can tune other parameters, such as batch size, learning rate, and the maximum number of epochs.

- `scripts/test.py`: Allows you to test a specific model against the testing dataset.
  > The model for testing is chosen in the testing configuration. Saved models can be found in the `models` directory. The models that come with this project are named corresponding to their accuracy against the validation data after training, not the testing data.

Generate and test your own model!

# Project Setup

Before you can run the scripts, you will need to set up your project environment. To do so, follow these steps:
1. Download and Install Anaconda3 if you haven't already.

   > Find instructions here:
   > [Anaconda3 installation docs](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

   I recommend leaving all options default when clicking through the prompts.

2. Activate the `base` conda environment in your terminal.

   For **Windows 10/11** users, run the following command in a **PowerShell** terminal:

   ```bash
   ~\Anaconda3\Scripts\activate.bat base
   ```

3. Clone this repository and cd into it.

   Open a terminal and change directory (`cd`) into some project directory. Use `git` to clone this
   repository by running the following commands:

   ```bash
   git clone https://github.com/HunterCoker/cisc452-har.git
   cd cisc452-har
   ```

4. Use the provided `environment.yml` file to create the conda environment for this project.

   Run the following command:

   ```bash
   conda env create --file=environment.yml
   ```

5. Open Visual Studio Code in the root of the repo as shown below:

   ```bash
   code .  # make sure working directory is project-dir/cisc452-har
   ```

6. Make sure Visual Studio Code is using your new environment.

   - Use the keyboard command `Ctrl+Shft+P` to open the **Command Palette**

   - Search for and select the `Python: Select Interpreter` option.

   - Search for and select the `cisc452-har` environment we just made.

Once you have completed all of the above steps, Visual Studio Code should be able to find all of the modules installed in the environment. You may have to restart Visual Studio Code to see these effects.
