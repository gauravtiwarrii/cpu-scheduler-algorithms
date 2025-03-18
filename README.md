# Intelligent CPU Scheduler Simulator

## Overview
This project is an implementation of various CPU scheduling algorithms using Python. It includes both preemptive and non-preemptive scheduling techniques, with an additional intelligent scheduler that dynamically selects an algorithm based on the average burst time of processes.

## Features
- **First-Come, First-Served (FCFS) Scheduling**
- **Shortest Job First (SJF) - Non-Preemptive & Preemptive (SRTF)**
- **Round Robin (RR) Scheduling**
- **Priority Scheduling - Non-Preemptive & Preemptive**
- **Intelligent Scheduler** that selects between SJF and RR based on burst time analysis
- **Performance Metrics Calculation** including:
  - Average Waiting Time
  - Average Turnaround Time
  - CPU Utilization
  - Throughput

## Installation & Setup
### Prerequisites
- Python 3.7+
- NumPy library (for numerical operations)

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/gauravtiwarrii/cpu-scheduler-algorithms.git
   cd cpu-scheduler-algorithms
   ```
2. Install dependencies:
   ```sh
   pip install numpy
   ```

## Usage
Run the script to test different scheduling algorithms:
```sh
python cpu-scheduler-algorithms.py
```

Modify the process list in `cpu-scheduler-algorithms.py` to test custom inputs:
```python
processes = [Process("P1", 0, 5, 2), Process("P2", 1, 3, 1), Process("P3", 2, 8, 3)]
```

## Example Output
```sh
Algorithm: FCFS (Non-Preemptive)
P1: Start=0, End=5, Waiting=0
P2: Start=5, End=8, Waiting=4
P3: Start=8, End=16, Waiting=6
```

## License
This project is LPU Operating Systems Project.

## Author
- **Gaurav Tiwari** - [LinkedIn](https://www.linkedin.com/in/gauravtiwarrii/)

