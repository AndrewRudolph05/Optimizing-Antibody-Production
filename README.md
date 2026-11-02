# Repository Structure

`datasets/`: Store raw data in this directory. Delete `.gitkeep` once you add your own files to this directory.

`figures/`: Export figures created by your code to this directory. Delete `.gitkeep` once you add your own files to this
directory.

`src/`: Store all Python code, except main.py, in this directory.

`tables/`: Export tables created by your code to this directory. Delete `.gitkeep` once you add your own files to this
directory.

`.gitignore`: Contains files to be ignored by Git. You can copy the `.gitignore` file from this repository into your own
project.

`environment.yaml`: Contains information about your conda environment. Run the following command:
`conda export > environment.yaml` to generate this file for your project. You can delete the last line in this file that
says `prefix`.

`main.py`: This is the only Python file that will be run. It should be kept relatively clean and mainly execute code
from `src/`.

`README.md`: This file, which contains information about the repository.