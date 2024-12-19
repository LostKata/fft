import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys

def print_with_explanation(variable_name, value, explanation):
    """Helper function to print variables with explanations."""
    print(f"{variable_name}: {value}\nExplanation: {explanation}\n")

def generate_maple_code(fft_frequencies, fft_magnitude, max_points=50, freq_limit=400):
    """Generate Maple 2024 code for FFT data, limiting the number of points and frequency range."""
    # Filter frequencies within the limit
    within_limit = fft_frequencies <= freq_limit
    fft_frequencies = fft_frequencies[within_limit]
    fft_magnitude = fft_magnitude[within_limit]

    # Reduce the size of data by sampling only `max_points` evenly spaced points
    indices = np.linspace(0, len(fft_frequencies) - 1, num=min(max_points, len(fft_frequencies)), dtype=int)
    reduced_frequencies = fft_frequencies[indices]
    reduced_magnitudes = fft_magnitude[indices]


def perform_fft(file_path):
    try:
        # Step 1: Load the audio file
        samplerate, data = wavfile.read(file_path)
        print_with_explanation("samplerate", samplerate, "The number of samples per second in the audio file.")

        # Ensure mono audio for simplicity (if stereo, take one channel)
        if len(data.shape) > 1:
            data = data[:, 0]  # Take the first channel
            print_with_explanation("data", data, "Stereo audio detected. Taking the first channel for processing.")
        else:
            print_with_explanation("data", data, "Audio data is mono.")

        # Step 2: Normalize the data
        normalized_data = data / np.max(np.abs(data))
        print_with_explanation(
            "normalized_data", normalized_data,
            "The audio data is normalized to lie between -1 and 1.")

        # Step 3: Perform FFT
        fft_result = np.fft.fft(normalized_data)
        print_with_explanation(
            "fft_result", fft_result,
            "The result of the Fast Fourier Transform, showing frequency components as complex numbers.")

        # Step 4: Compute the frequency bins
        fft_magnitude = np.abs(fft_result)  # Magnitude of FFT
        print_with_explanation(
            "fft_magnitude", fft_magnitude,
            "The magnitude of the FFT result, representing the strength of each frequency component.")

        fft_frequencies = np.fft.fftfreq(len(normalized_data), 1 / samplerate)
        print_with_explanation(
            "fft_frequencies", fft_frequencies,
            "The corresponding frequencies for each FFT magnitude value.")

       # Step 6: Visualize the result (limited to 400 Hz)
        plt.figure(figsize=(12, 6))
        within_limit = fft_frequencies[:len(fft_frequencies)//2] <= 400
        plt.plot(fft_frequencies[:len(fft_frequencies)//2][within_limit], fft_magnitude[:len(fft_magnitude)//2][within_limit])
        plt.title("FFT of the Audio Signal (Limited to 400 Hz)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid()
        plt.show()

    except FileNotFoundError:
        print("Error: The file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fft_analysis.py <path_to_audio_file>")
    else:
        file_path = sys.argv[1]
        perform_fft(file_path)
