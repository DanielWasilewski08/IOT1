import math

class EdgeSignalProcessor:
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = [0] * window_size
        self.current_sum = 0
        self.pointer = 0
        self.count = 0

    def push(self, sample: float) -> float: # Dodaje próbkę w O(1) i zwraca wygładzoną wartość
        # Usunięcie najstarszej wartości z sumy i nadpisanie jej nową próbką
        self.current_sum -= self.buffer[self.pointer]
        self.buffer[self.pointer] = sample
        self.current_sum += sample

        # ustawienie nowej pozycji do nadpisania w buforze
        self.pointer = (self.pointer + 1) % self.window_size

        # count do wyliczenia średniej poprzez uwzględnienie początkowego zapełniania okna
        if self.count < self.window_size:
            self.count += 1

        # Obliczenie i zwrócenie wygładzonej wartości
        smoothed_value = self.current_sum / self.count
        return smoothed_value


    def extract_features(self) -> tuple:    #Zwraca wektor cech (Mean, Std, RMS, PTP) zebranych w buforze
        if self.count == 0:
            return (0.0, 0.0, 0.0, 0.0)  # Zwraca zerowy wektor cech, jeśli bufor jest pusty
        mean = self.current_sum / self.count
        sq_sum = 0
        sq_diff_sum = 0
        min_val = self.buffer[0]
        max_val = self.buffer[0]

        for i in range(self.count):
            val = self.buffer[i]
            sq_sum += val ** 2
            sq_diff_sum += (val - mean) ** 2
            if val < min_val: min_val = val
            if val > max_val: max_val = val

        std = math.sqrt(sq_diff_sum / self.count) # pierwiastek z wariancji
        rms = math.sqrt(sq_sum / self.count) # pierwiastek z wartości średniej kwadratowej
        ptp = max_val - min_val # amplituda

        return (round(mean, 2), round(std, 2), round(rms, 2), round(ptp, 2))

#-----------------------------
processor = EdgeSignalProcessor(window_size=5)
raw_ppg = [70, 72, 130, 71, 73, 75, 140, 74, 72, 71]
for sample in raw_ppg:
    smoothed = processor.push(sample)
    print(smoothed)

print(f"pojedyncza krotka z bufora przekazywana do ml: {processor.extract_features()}")