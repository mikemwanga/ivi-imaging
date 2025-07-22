# IVI-imaging

Repository for the collaborative work on virus infection fluorescent imaging data between the Institute of Virology (IVI) and the Data Science Lab (DSL) at the University of Bern.

## Requirements

To use the CellAnalyzer, you need to have the software installed as listed in the `environment.yml` file. Using conda, you can create an environment with the required packages by running:

```bash
conda env create -f environment.yml
```

- Make sure to deactivate any running conda environment before executing the command.
- If not already done, change directory to the root of the repository before running the command.
- You can give the environment a custom name by adding `-n your_env_name` to the command (default name is ca-env, for CellAnalyzer).