# Face-Recognition

1. Make the virtual environment
    Need the Python Extension for this.
    Use VS Code Command Line (Ctrl + Shift + P) -> Type `Python: Create Environment` -> `venv`
    If prompted, select the `requirements.txt` file and ignore the second step.

2. Install the requirements
    First activate the environment, run in terminal: 
    ```bash
        .venv/Scripts/activate
    ```
    Then run this command:
    ```bash
        pip install -r requirements.txt
    ```

3. Ensure ou have the folders: `db` and `images` in the same directory

3. Put the images of the people with the name in `images` folder

4. Run the file `load_faces.py`

5. Run the file `main.py`