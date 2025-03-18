from typing import List, Tuple
from enum import Enum
import heapq
import numpy as np

class Algorithm(Enum):
    """Enumeration of scheduling algorithm names."""
    FCFS = "FCFS (Non-Preemptive)"
    SJF_NP = "SJF (Non-Preemptive)"
    SJF_P = "SJF (Preemptive - SRTF)"
    RR = "Round Robin"
    PR_NP = "Priority (Non-Preemptive)"
    PR_P = "Priority (Preemptive)"
    INTELLIGENT = "Intelligent"

class Process:
    """Represents a process in the CPU scheduling simulation."""
    def __init__(self, pid: str, arrival_time: int, burst_time: int, priority: int = 0, dependencies: List[str] = None):
        """Initialize a Process object.
        
        Args:
            pid (str): Process identifier.
            arrival_time (int): Time when the process arrives (>= 0).
            burst_time (int): CPU time required by the process (> 0).
            priority (int, optional): Priority value (default is 0).
            dependencies (List[str], optional): List of PIDs this process depends on.
        """
        if burst_time <= 0:
            raise ValueError("Burst time must be positive")
        if arrival_time < 0:
            raise ValueError("Arrival time cannot be negative")
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.priority = priority
        self.dependencies = dependencies or []
        self.remaining_time = burst_time
        self.start_time = 0
        self.end_time = 0
        self.waiting_time = 0
        self.turnaround_time = 0

def fcfs_scheduler(processes: List[Process]) -> Tuple[List[Process], str, List[Tuple[str, int, int]]]:
    """Implements First-Come-First-Serve scheduling algorithm with dependency support.
    
    Args:
        processes (List[Process]): List of Process objects to schedule.
    
    Returns:
        Tuple: (processed list, algorithm name, timeline as list of (pid, start, end) tuples).
    
    Raises:
        ValueError: If the process list is empty.
    """
    if not processes:
        raise ValueError("Process list cannot be empty")
    
    processes.sort(key=lambda x: x.arrival_time)
    current_time = 0
    timeline = [None] * len(processes)
    completed = {}
    remaining = processes.copy()
    i = 0
    
    while remaining:
        for p in remaining[:]:
            if all(dep in completed for dep in p.dependencies):
                p.start_time = max(current_time, p.arrival_time)
                p.end_time = p.start_time + p.burst_time
                p.waiting_time = p.start_time - p.arrival_time
                p.turnaround_time = p.end_time - p.arrival_time
                timeline[i] = (p.pid, p.start_time, p.end_time)
                current_time = p.end_time
                completed[p.pid] = p
                remaining.remove(p)
                i += 1
                break
        else:
            current_time += 1  # No process ready
    
    return list(completed.values()), Algorithm.FCFS.value, timeline

def sjf_non_preemptive(processes: List[Process]) -> Tuple[List[Process], str, List[Tuple[str, int, int]]]:
    """Implements Shortest Job First (Non-Preemptive) scheduling algorithm.
    
    Args:
        processes (List[Process]): List of Process objects to schedule.
    
    Returns:
        Tuple: (processed list, algorithm name, timeline).
    """
    if not processes:
        raise ValueError("Process list cannot be empty")
    
    processes.sort(key=lambda x: (x.arrival_time, x.burst_time))
    current_time = 0
    completed = []
    timeline = [None] * len(processes)
    remaining = processes.copy()
    i = 0
    
    while remaining:
        available = [p for p in remaining if p.arrival_time <= current_time]
        if not available:
            current_time += 1
            continue
        p = available[0]  # Already sorted by burst_time
        p.start_time = current_time
        p.end_time = p.start_time + p.burst_time
        p.waiting_time = p.start_time - p.arrival_time
        p.turnaround_time = p.end_time - p.arrival_time
        timeline[i] = (p.pid, p.start_time, p.end_time)
        current_time = p.end_time
        completed.append(p)
        remaining.remove(p)
        i += 1
    
    return completed, Algorithm.SJF_NP.value, timeline

def sjf_preemptive(processes: List[Process]) -> Tuple[List[Process], str, List[Tuple[str, int, int]]]:
    """Implements Shortest Job First (Preemptive - SRTF) scheduling algorithm using a heap.
    
    Args:
        processes (List[Process]): List of Process objects to schedule.
    
    Returns:
        Tuple: (processed list, algorithm name, timeline).
    """
    if not processes:
        raise ValueError("Process list cannot be empty")
    
    processes.sort(key=lambda x: x.arrival_time)
    current_time = processes[0].arrival_time
    completed = []
    timeline = []
    heap = []  # (remaining_time, pid, process)
    remaining = processes.copy()
    i = 0
    
    while remaining or heap:
        while i < len(remaining) and remaining[i].arrival_time <= current_time:
            p = remaining[i]
            heapq.heappush(heap, (p.remaining_time, p.pid, p))
            i += 1
        
        if not heap:
            current_time += 1
            continue
        
        _, _, p = heapq.heappop(heap)
        if p.start_time == 0:
            p.start_time = current_time
        current_time += 1
        p.remaining_time -= 1
        timeline.append((p.pid, current_time - 1, current_time))
        
        if p.remaining_time == 0:
            p.end_time = current_time
            p.turnaround_time = p.end_time - p.arrival_time
            p.waiting_time = p.turnaround_time - p.burst_time
            completed.append(p)
        else:
            heapq.heappush(heap, (p.remaining_time, p.pid, p))
    
    return completed, Algorithm.SJF_P.value, timeline

def rr_scheduler(processes: List[Process], quantum: int) -> Tuple[List[Process], str, List[Tuple[str, int, int]]]:
    """Implements Round Robin scheduling algorithm.
    
    Args:
        processes (List[Process]): List of Process objects to schedule.
        quantum (int): Time quantum for each process (> 0).
    
    Returns:
        Tuple: (processed list, algorithm name, timeline).
    """
    if not processes:
        raise ValueError("Process list cannot be empty")
    if quantum <= 0:
        raise ValueError("Quantum must be positive")
    
    queue = processes.copy()
    queue.sort(key=lambda x: x.arrival_time)
    current_time = queue[0].arrival_time if queue else 0
    timeline = []
    
    while queue:
        p = queue.pop(0)
        if p.start_time == 0:
            p.start_time = current_time
        exec_time = min(quantum, p.remaining_time)
        timeline.append((p.pid, current_time, current_time + exec_time))
        current_time += exec_time
        p.remaining_time -= exec_time
        
        arrived = [proc for proc in processes if proc.arrival_time <= current_time and proc not in queue and proc.remaining_time > 0]
        queue.extend(arrived)
        
        if p.remaining_time > 0:
            queue.append(p)
        else:
            p.end_time = current_time
            p.turnaround_time = p.end_time - p.arrival_time
            p.waiting_time = p.turnaround_time - p.burst_time
    
    return processes, Algorithm.RR.value, timeline

def priority_non_preemptive(processes: List[Process]) -> Tuple[List[Process], str, List[Tuple[str, int, int]]]:
    """Implements Priority (Non-Preemptive) scheduling algorithm.
    
    Args:
        processes (List[Process]): List of Process objects to schedule.
    
    Returns:
        Tuple: (processed list, algorithm name, timeline).
    """
    if not processes:
        raise ValueError("Process list cannot be empty")
    
    processes.sort(key=lambda x: (x.arrival_time, x.priority))
    current_time = 0
    completed = []
    timeline = [None] * len(processes)
    remaining = processes.copy()
    i = 0
    
    while remaining:
        available = [p for p in remaining if p.arrival_time <= current_time]
        if not available:
            current_time += 1
            continue
        p = min(available, key=lambda x: x.priority)
        p.start_time = current_time
        p.end_time = p.start_time + p.burst_time
        p.waiting_time = p.start_time - p.arrival_time
        p.turnaround_time = p.end_time - p.arrival_time
        timeline[i] = (p.pid, p.start_time, p.end_time)
        current_time = p.end_time
        completed.append(p)
        remaining.remove(p)
        i += 1
    
    return completed, Algorithm.PR_NP.value, timeline

def priority_preemptive(processes: List[Process]) -> Tuple[List[Process], str, List[Tuple[str, int, int]]]:
    """Implements Priority (Preemptive) scheduling algorithm using a heap.
    
    Args:
        processes (List[Process]): List of Process objects to schedule.
    
    Returns:
        Tuple: (processed list, algorithm name, timeline).
    """
    if not processes:
        raise ValueError("Process list cannot be empty")
    
    processes.sort(key=lambda x: x.arrival_time)
    current_time = processes[0].arrival_time
    completed = []
    timeline = []
    heap = []  # (priority, pid, process)
    remaining = processes.copy()
    i = 0
    
    while remaining or heap:
        while i < len(remaining) and remaining[i].arrival_time <= current_time:
            p = remaining[i]
            heapq.heappush(heap, (p.priority, p.pid, p))
            i += 1
        
        if not heap:
            current_time += 1
            continue
        
        _, _, p = heapq.heappop(heap)
        if p.start_time == 0:
            p.start_time = current_time
        current_time += 1
        p.remaining_time -= 1
        timeline.append((p.pid, current_time - 1, current_time))
        
        if p.remaining_time == 0:
            p.end_time = current_time
            p.turnaround_time = p.end_time - p.arrival_time
            p.waiting_time = p.turnaround_time - p.burst_time
            completed.append(p)
        else:
            heapq.heappush(heap, (p.priority, p.pid, p))
    
    return completed, Algorithm.PR_P.value, timeline

def intelligent_scheduler(processes: List[Process], quantum: int = 2) -> Tuple[List[Process], str, List[Tuple[str, int, int]]]:
    """Implements an intelligent scheduling algorithm based on burst time statistics.
    
    Args:
        processes (List[Process]): List of Process objects to schedule.
        quantum (int, optional): Time quantum for Round Robin (default is 2).
    
    Returns:
        Tuple: (processed list, algorithm name, timeline).
    """
    if not processes:
        raise ValueError("Process list cannot be empty")
    
    avg_burst = sum(p.burst_time for p in processes) / len(processes)
    burst_variance = sum((p.burst_time - avg_burst) ** 2 for p in processes) / len(processes)
    processes_copy = [Process(p.pid, p.arrival_time, p.burst_time, p.priority, p.dependencies) for p in processes]
    
    if burst_variance > 10:  # High variability
        return sjf_preemptive(processes_copy)
    elif avg_burst < 5:     # Short bursts
        return sjf_non_preemptive(processes_copy)
    else:                    # Long or uniform bursts
        return rr_scheduler(processes_copy, quantum)

def calculate_metrics(processes: List[Process]) -> Tuple[float, float, float, float]:
    """Calculates performance metrics for a list of scheduled processes.
    
    Args:
        processes (List[Process]): List of Process objects with scheduling data.
    
    Returns:
        Tuple: (avg_waiting_time, avg_turnaround_time, cpu_utilization, throughput).
    
    Raises:
        ValueError: If the process list is empty or lacks required attributes.
    """
    if not processes:
        raise ValueError("Process list cannot be empty")
    if not all(hasattr(p, 'waiting_time') for p in processes):
        raise ValueError("All processes must have scheduling data")
    
    avg_waiting = sum(p.waiting_time for p in processes) / len(processes)
    avg_turnaround = sum(p.turnaround_time for p in processes) / len(processes)
    total_burst = sum(p.burst_time for p in processes)
    total_time = max(p.end_time for p in processes) if processes else 0
    cpu_utilization = (total_burst / total_time * 100) if total_time > 0 else 0
    throughput = len(processes) / total_time if total_time > 0 else 0
    
    return avg_waiting, avg_turnaround, cpu_utilization, throughput

if __name__ == "__main__":
    # Test code with dependencies
    processes = [
        Process("P1", 0, 4),
        Process("P2", 1, 3, dependencies=["P1"]),
        Process("P3", 2, 2)
    ]
    result, name, timeline = fcfs_scheduler(processes)
    print(f"Algorithm: {name}")
    for p in result:
        print(f"{p.pid}: Start={p.start_time}, End={p.end_time}, Waiting={p.waiting_time}")
