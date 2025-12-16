# capture-chess-engine

## Engine Setup & Usage

The chess engine is implemented in a single file: **`engine.py`**. It is tested to be compatible with XBoard 4.9.1. Newer versions should also work but are not explicitly tested.

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
    - Search
        - https://www.chessprogramming.org/Alpha-Beta  
          Used in `search()` and `quiesce()` through alpha–beta pruning to reduce the search space.

        - https://www.chessprogramming.org/Negamax  
          The core search formulation. Child positions are evaluated as `score = -search(...)`.

        - https://www.chessprogramming.org/Horizon_Effect  
          Addressed by switching to quiescence search at depth 0 so tactical capture sequences are fully resolved.

        - https://www.chessprogramming.org/Quiescence_Search  
          Implemented in `quiesce()`, which evaluates only capture moves (up to a fixed depth) before returning a stable position score.

        - https://www.chessprogramming.org/Transposition_Table  
          Positions are cached using Zobrist hashes to avoid re-searching identical positions.

    - Evaluation
        - A lot of ideas listed in https://www.chessprogramming.org/Evaluation#Basic_Evaluation_Features  
          Used as guidance for material scoring, pawn structure (doubled/isolated/passed pawns), bishop pair bonuses, and basic endgame king activity.

- Piece-Square Tables taken from https://github.com/dimdano/numbfish/blob/main/numbfish.py  
  PST values are applied in `evaluate()` to reward good piece placement. Black pieces use mirrored squares so one table works for both colors.

- [XBoard / WinBoard Engine Protocol](https://www.gnu.org/software/xboard/engine-intf.html)  
  Used to communicate with the GUI.

