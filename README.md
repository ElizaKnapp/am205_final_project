# Spectral Mixing Times in Chord-Transition Markov Chains
Sidd and Eliza's AM 205 project!

## Overview

This project analyzes spectral mixing times in chord-transition Markov chains derived from popular music. We model songs as Markov chains where states represent chord progressions, and analyze the spectral gap and mixing time properties of these chains. The analysis uses Billboard Hot 100 chart data from 1959-1991 and chord annotations from the McGill Billboard dataset.

## Project Description

The project constructs Markov chains from chord progressions in popular songs, where each state represents a unique chord sequence. We then compute:
- **Spectral gap**: The difference between the first and second largest eigenvalues of the transition matrix
- **Spectral mixing time (t_spec)**: An estimate of mixing time based on the spectral gap
- **Power method iterations (t_PM)**: The number of iterations required for the power method to converge to the stationary distribution

The analysis explores relationships between these quantities and investigates trends over time periods.

## Installation

### Dependencies

Install dependencies using pip in a new conda environment:
```bash
pip install requirements.txt
```

### Data

The project uses:
- **Billboard Hot 100 data**: Provided in `all_songs_data.csv` which can be downloaded [here](https://drive.google.com/file/d/105zVhaCvUr4Dki_9PpdS4omoqQi_vj5b/view?usp=sharing)
- **McGill Billboard dataset**: Automatically downloaded via `mirdata` when running the analysis

## Usage

### Running the Analysis

The project provides two main execution modes:

1. **Run analysis on existing data** (default):
   ```bash
   make run
   ```
   This loads previously computed results from `song_results.csv` and generates visualizations and statistics.

2. **Gather new data and run analysis**:
   ```bash
   make gather-and-run
   ```
   This performs the full pipeline:
   - Loads the Billboard dataset
   - Matches songs to chord annotations
   - Constructs Markov chains for each song
   - Computes spectral gaps and mixing times
   - Generates all visualizations and statistics

### Command Line Options

You can also run `main.py` directly:
```bash
python main.py                    # Run without gathering data
python main.py --gather-data      # Run with data gathering
```

## Project Structure

- `main.py` - Main entry point that orchestrates the analysis pipeline
- `billboard.py` - Billboard dataset loader and song matching functionality
- `song_analysis.py` - Core analysis class that constructs Markov chains and computes spectral properties
- `results.py` - Visualization and statistical analysis functions
- `utils.py` - Utility functions for chord parsing and string matching
- `Makefile` - Convenience commands for running the analysis

## Configuration

Key parameters can be adjusted in `main.py`:
- `YEAR_START` / `YEAR_END`: Time range for analysis (default: 1959-1991)
- `MAX_SONGS_PER_YEAR`: Maximum songs to analyze per year (default: 20)
- `MAX_CANDIDATES_PER_YEAR`: Maximum candidate songs to consider per year (default: 100)
- `POWER_METHOD_TOL`: Convergence tolerance for power method (default: 1e-6)
- `ALPHA`: Smoothing parameter for transition probabilities (default: 1e-3)
- `MIN_STATES`: Minimum number of states required for a song to be analyzed (default: 3)

## Output Files

### Data Files
- `song_results.csv` - Per-song analysis results including spectral gaps, mixing times, and convergence status
- `aggregate_song_results.csv` - Year-level aggregated statistics

### Visualizations
- `spectral_mixing_vs_power_iterations.png` - Log-log scatter plot of t_spec vs t_PM, colored by number of states
- `spectral_gap_distribution.png` - Histogram of spectral gaps across all songs
- `power_iterations_distribution.png` - Histogram of power method iterations
- `spectral_gap_vs_states.png` - Scatter plot showing relationship between spectral gap and number of chord states
- `year_trends.png` - Trends in median spectral gap and power iterations over all years
- `year_trends_1960_1975.png` - Trends for early period (1960-1975)
- `year_trends_1975_1990.png` - Trends for later period (1975-1990)

## Key Analysis Results

The pipeline generates:
1. **Regression analysis**: Linear regression of power method iterations vs spectral mixing time estimate
2. **Problematic songs**: List of songs with smallest spectral gaps (potential convergence issues)
4. **Temporal trends**: Analysis of how spectral properties evolved over time
