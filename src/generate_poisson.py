import matplotlib.pyplot as plt
import numpy as np


def generate_poisson_events(rate, size):

    # Generate random exponential inter-arrival times
    inter = np.random.exponential(scale=rate, size=size)

    # Calculate cumulative sum of inter-arrival times
    event_times = np.cumsum(inter)

    return event_times


def plot_poisson(event_times,
                 rate,
                 time_duration):

    num_events = len(event_times)
    inter_arrival_times = np.diff(event_times)

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f'Poisson Process Simulation (λ = {rate}, '
        f'Duration = {time_duration} seconds)\n', fontsize=16)

    axs[0].step(event_times, np.arange(1, num_events + 1),
                where='post', color='blue')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Event Number')
    axs[0].set_title(
        f'Poisson Process Event Times\nTotal: {num_events} events\n')
    axs[0].grid(True)

    axs[1].hist(inter_arrival_times, bins=20, color='green', alpha=0.5)
    axs[1].set_xlabel('Inter-Arrival Time')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title(
        f'Histogram of Inter-Arrival Times\nMEAN: '
        f'{np.mean(inter_arrival_times):.2f} | '
        f'STD: {np.std(inter_arrival_times):.2f}\n')
    axs[1].grid(True, alpha=0.5)

    plt.tight_layout()
    plt.show()
