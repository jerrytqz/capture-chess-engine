# capture-chess-engine

## Engine Setup & Usage

The chess engine is implemented in a single file: **`engine.py`**.

### Requirements

All required Python dependencies are listed in **`requirements.txt`**.

The engine was developed and tested using **Python 3.8.10** on WSL Ubuntu, but newer Python versions should also be compatible.

### Installation

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS / WSL
   # venv\Scripts\activate    # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Engine

Once dependencies are installed, run the engine directly:

```bash
python engine.py
```

No additional configuration is required, though you may also have to set permissions to make the engine executable.

## Sources

- [Chess Programming Wiki](https://www.chessprogramming.org/)
    - Search related info:
        - https://www.chessprogramming.org/Alpha-Beta
        - https://www.chessprogramming.org/Negamax
        - https://www.chessprogramming.org/Horizon_Effect
        - https://www.chessprogramming.org/Quiescence_Search
        - https://www.chessprogramming.org/Transposition_Table
    - Evaluation
        - A lot of ideas listed in https://www.chessprogramming.org/Evaluation#Basic_Evaluation_Features