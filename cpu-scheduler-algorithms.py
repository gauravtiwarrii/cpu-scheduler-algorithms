# scheduler_algorithms.py
import numpy as np

class Process:
    def __init__(self, pid, arrival_time, burst_time, priority=0):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time 
        self.priority = priority
        self.remaining_time = burst_time
        self.start_time = 0
        self.end_time = 0
        self.waiting_time = 0
        self.turnaround_time = 0

def fcfs_scheduler(processes):
    processes.sort(key=lambda x: x.arrival_time)
    current_time = 0
    timeline = []
    for p in processes:
        p.start_time = max(current_time, p.arrival_time)
        p.end_time = p.start_time + p.burst_time
        p.waiting_time = p.start_time - p.arrival_time
        p.turnaround_time = p.end_time - p.arrival_time
        timeline.append((p.pid, p.start_time, p.end_time))
        current_time = p.end_time
    return processes, "FCFS (Non-Preemptive)", timeline

def sjf_non_preemptive(processes):
    processes.sort(key=lambda x: (x.arrival_time, x.burst_time))
    current_time = 0
    completed = []
    remaining = processes.copy()
    timeline = []
    while remaining:
        available = [p for p in remaining if p.arrival_time <= current_time]
        if not available:
            current_time += 1
            continue
        p = min(available, key=lambda x: x.burst_time)
        p.start_time = current_time
        p.end_time = p.start_time + p.burst_time
        p.waiting_time = p.start_time - p.arrival_time
        p.turnaround_time = p.end_time - p.arrival_time
        current_time = p.end_time
        timeline.append((p.pid, p.start_time, p.end_time))
        completed.append(p)
        remaining.remove(p)
    return completed, "SJF (Non-Preemptive)", timeline

def sjf_preemptive(processes):
    processes.sort(key=lambda x: x.arrival_time)
    current_time = processes[0].arrival_time if processes else 0
    completed = []
    timeline = []
    remaining = processes.copy()
    while remaining or completed:
        available = [p for p in remaining if p.arrival_time <= current_time]
        if not available and not completed:
            current_time += 1
            continue
        if available:
            p = min(available, key=lambda x: x.remaining_time)
            if p.start_time == 0:
                p.start_time = current_time
            current_time += 1
            p.remaining_time -= 1
            timeline.append((p.pid, current_time-1, current_time))
            if p.remaining_time == 0:
                p.end_time = current_time
                p.turnaround_time = p.end_time - p.arrival_time
                p.waiting_time = p.turnaround_time - p.burst_time
                completed.append(p)
                remaining.remove(p)
        else:
            current_time += 1
    return completed, "SJF (Preemptive - SRTF)", timeline

def rr_scheduler(processes, quantum):
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
    return processes, "Round Robin", timeline

def priority_non_preemptive(processes):
    processes.sort(key=lambda x: (x.arrival_time, x.priority))
    current_time = 0
    completed = []
    remaining = processes.copy()
    timeline = []
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
        current_time = p.end_time
        timeline.append((p.pid, p.start_time, p.end_time))
        completed.append(p)
        remaining.remove(p)
    return completed, "Priority (Non-Preemptive)", timeline

def priority_preemptive(processes):
    processes.sort(key=lambda x: x.arrival_time)
    current_time = processes[0].arrival_time if processes else 0
    completed = []
    timeline = []
    remaining = processes.copy()
    while remaining or completed:
        available = [p for p in remaining if p.arrival_time <= current_time]
        if not available and not completed:
            current_time += 1
            continue
        if available:
            p = min(available, key=lambda x: x.priority)
            if p.start_time == 0:
                p.start_time = current_time
            current_time += 1
            p.remaining_time -= 1
            timeline.append((p.pid, current_time-1, current_time))
            if p.remaining_time == 0:
                p.end_time = current_time
                p.turnaround_time = p.end_time - p.arrival_time
                p.waiting_time = p.turnaround_time - p.burst_time
                completed.append(p)
                remaining.remove(p)
        else:
            current_time += 1
    return completed, "Priority (Preemptive)", timeline

def intelligent_scheduler(processes, quantum=2):
    avg_burst = sum(p.burst_time for p in processes) / len(processes)
    processes_copy = [Process(p.pid, p.arrival_time, p.burst_time, p.priority) for p in processes]
    if avg_burst < 5:
        return sjf_non_preemptive(processes_copy)
    else:
        return rr_scheduler(processes_copy, quantum)

def calculate_metrics(processes):
    if not processes or not all(hasattr(p, 'waiting_time') for p in processes):
        raise ValueError("Invalid process list passed to calculate_metrics")
    avg_waiting = sum(p.waiting_time for p in processes) / len(processes)
    avg_turnaround = sum(p.turnaround_time for p in processes) / len(processes)
    total_burst = sum(p.burst_time for p in processes)
    total_time = max(p.end_time for p in processes) if processes else 0
    cpu_utilization = (total_burst / total_time * 100) if total_time > 0 else 0
    throughput = len(processes) / total_time if total_time > 0 else 0
    return avg_waiting, avg_turnaround, cpu_utilization, throughput

if __name__ == "__main__":
    # Test code for standalone execution
    processes = [Process(f"P{i}", i, 4-i) for i in range(1, 4)]
    result, name, timeline = fcfs_scheduler(processes)
    print(f"Algorithm: {name}")
    for p in result:
        print(f"{p.pid}: Start={p.start_time}, End={p.end_time}, Waiting={p.waiting_time}")
