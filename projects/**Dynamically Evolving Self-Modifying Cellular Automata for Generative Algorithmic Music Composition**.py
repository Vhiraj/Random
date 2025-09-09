This project implements a **Dynamically Evolving Self-Modifying Cellular Automata for Generative Algorithmic Music Composition**.

The system operates by simulating a Cellular Automaton (CA) whose evolution rules are not static but dynamically mutate over time. The changing patterns and states within the CA grid are then translated into musical notes and played back in real-time via a virtual MIDI port. This creates a continuously evolving, algorithmic musical piece driven by the emergent properties of the self-modifying CA.

**Key Components:**

1.  **Cellular Automaton (`cellular_automata.py`)**: Manages a 2D grid of cells, each with a state (e.g., 0, 1, 2, 3). It evolves the grid based on a set of rules and the states of neighboring cells (Moore or Von Neumann neighborhood).
2.  **Rule Evolver (`rule_evolver.py`)**: Responsible for generating, storing, and dynamically mutating the CA's evolution rules. Rules are represented as a dictionary mapping `(current_cell_state, number_of_live_neighbors)` to `next_cell_state`. Periodically, this component introduces random changes to these rules, causing the CA's behavior to shift.
3.  **Music Composer (`music_composer.py`)**: Interprets the current state of the CA grid and translates its patterns into MIDI messages (note on/off). It sends these messages to a virtual MIDI output port, allowing users to connect a DAW, synthesizer, or MIDI visualizer to hear the generated music in real-time. The mapping defines which CA states correspond to specific notes, durations, and how grid positions influence pitch and timing.
4.  **Configuration (`config.py`)**: A centralized place for all customizable parameters, including grid dimensions, CA states, MIDI settings (BPM, port name, note mappings), and rule evolution parameters.
5.  **Main Orchestration (`main.py`)**: Ties all components together, running the simulation loop, triggering rule evolutions, and initiating music composition for each CA generation.

**How it works musically:**
Each generation of the CA grid is interpreted as a musical "phrase" or "measure".
-   Rows of the CA grid are mapped to sequential time segments within this phrase.
-   Within each row, "active" cells (states mapped to notes) trigger MIDI notes.
-   The column index of a cell adds an offset to its base pitch, creating harmonic or melodic variations.
-   Cell state also determines the note's duration.
-   The `MusicComposer` ensures polyphony by scheduling all note_on and note_off events for a generation based on calculated real-time delays, then sending them precisely via the MIDI port.

**To run this project:**

1.  **Install Dependencies:**
    ```bash
    pip install numpy mido
    ```
2.  **Set up a Virtual MIDI Port:**
    This script outputs MIDI to a virtual port. You need to create one if your system doesn't have it automatically:
    *   **Windows**: Download and install `loopMIDI` (https://www.tobias-erichsen.de/software/loopmidi.html). Create a virtual MIDI cable.
    *   **macOS**: Virtual ports are built-in. Open 'Audio MIDI Setup' (Applications -> Utilities), then 'Window -> Show MIDI Studio'. Double-click the 'IAC Driver' icon, check 'Device is online', and create a port if needed.
    *   **Linux**: You can use `aconnect` or `qjackctl`. You might need to load the `snd-virmidi` kernel module: `sudo modprobe snd-virmidi`. Then, use `aconnect -o` to see available output ports.

3.  **Run the Script:**
    ```bash
    python main.py
    ```

4.  **Connect a MIDI Receiver:**
    Open your Digital Audio Workstation (DAW), a synthesizer application, or a MIDI visualizer. Configure it to receive MIDI input from the virtual port named "SelfModCA_Music" (or whatever name is specified in `config.py`). You should start hearing and/or seeing the generated music!

---

**Project Structure:**

```
self_mod_ca_music/
├── main.py
├── cellular_automata.py
├── rule_evolver.py
├── music_composer.py
├── config.py
└── requirements.txt
```

---

**`config.py`**

```python
"""
Configuration settings for the Dynamically Evolving Self-Modifying Cellular Automata
for Generative Algorithmic Music Composition.
"""

class Config:
    # --- Cellular Automata Settings ---
    GRID_WIDTH = 20
    GRID_HEIGHT = 10
    NUM_STATES = 4 # Number of possible states for each cell (0, 1, 2, ..., NUM_STATES-1)
                   # State 0 typically represents "off" or "dead" for musical mapping.
    INITIAL_DENSITY = 0.3 # Probability of a cell being in a non-zero state initially

    # --- Evolution Control ---
    MAX_GENERATIONS = 200 # Total CA generations to simulate before stopping
    TICKS_PER_RULE_CHANGE = 15 # How often the CA rule set mutates (in generations)

    # --- Music Related Settings ---
    MIDI_PORT_NAME = "SelfModCA_Music" # Name of the virtual MIDI port to create/use
    MIDI_BPM = 120 # Tempo of the generated music
    DEFAULT_VELOCITY = 80 # MIDI velocity for notes (0-127)
    MIDI_CHANNEL = 0 # MIDI channel to send messages on (0-15)
    
    # Map CA states (non-zero) to base MIDI note numbers (e.g., C4=60, E4=64, G4=67)
    # State 0 is generally treated as 'off' or 'rest' and won't trigger a note.
    NOTE_MAP = {
        1: 60, # C4
        2: 64, # E4
        3: 67, # G4
    }
    
    # Map CA states to note durations in beats (e.g., 0.25=quarter, 0.5=half, 1.0=whole)
    DURATION_MAP = {
        1: 0.25, # Short rhythmic element
        2: 0.5,  # Medium duration
        3: 1.0,  # Long duration
    }
    DEFAULT_DURATION = 0.25 # Default duration if a state is not in DURATION_MAP
    
    # How long (in musical beats) one full CA grid generation's music takes to play.
    # This controls the overall pace of the musical progression relative to CA evolution.
    GENERATION_MUSIC_DURATION_BEATS = 4.0 # e.g., one measure

    # --- Rule Evolution Parameters ---
    RULE_MUTATION_RATE = 0.1 # Probability for a single rule entry to mutate during evolution
    NEIGHBORHOOD_TYPE = 'moore' # 'moore' (8 neighbors) or 'von_neumann' (4 neighbors)
    
    # Defines what constitutes a "live" neighbor for rule calculation (e.g., any non-zero state)
    LIVE_STATE_THRESHOLD = 0
```

---

**`cellular_automata.py`**

```python
"""
Defines the CellularAutomaton class, responsible for managing the grid,
cell states, and applying evolution rules.
"""

import numpy as np
import random
from config import Config

class CellularAutomaton:
    """
    A class representing a Cellular Automaton.
    Manages the grid of cells, their states, and applies evolution rules
    to transition the grid to its next generation.
    """

    def __init__(self, width, height, num_states, initial_density, neighborhood_type='moore'):
        """
        Initializes the Cellular Automaton.

        Args:
            width (int): The width of the CA grid.
            height (int): The height of the CA grid.
            num_states (int): The number of possible states each cell can have (0 to num_states-1).
            initial_density (float): Probability (0.0-1.0) for a cell to be initialized
                                     to a non-zero state.
            neighborhood_type (str): The type of neighborhood to consider ('moore' or 'von_neumann').
        """
        self.width = width
        self.height = height
        self.num_states = num_states
        self.neighborhood_type = neighborhood_type
        self.grid = self._initialize_grid(initial_density)
        self.rule = None # The evolution rule, set externally by RuleEvolver

    def _initialize_grid(self, initial_density):
        """
        Initializes the CA grid with random states based on the given density.
        Cells are assigned state 0 or a random non-zero state.

        Args:
            initial_density (float): Probability for a cell to be in a non-zero state.

        Returns:
            np.ndarray: The initialized 2D grid.
        """
        grid = np.zeros((self.height, self.width), dtype=int)
        for r in range(self.height):
            for c in range(self.width):
                if random.random() < initial_density:
                    # Assign a random non-zero state
                    grid[r, c] = random.randint(1, self.num_states - 1)
        return grid

    def _get_neighbors(self, r, c):
        """
        Retrieves the states of the neighbors for a given cell (r, c).
        Uses toroidal (wrapping) boundaries.

        Args:
            r (int): Row index of the cell.
            c (int): Column index of the cell.

        Returns:
            list: A list of integer states of the neighboring cells.
        """
        neighbors = []
        if self.neighborhood_type == 'moore':
            # Moore neighborhood: 8 cells (all surrounding)
            dr = [-1, -1, -1, 0, 0, 1, 1, 1]
            dc = [-1, 0, 1, -1, 1, -1, 0, 1]
        elif self.neighborhood_type == 'von_neumann':
            # Von Neumann neighborhood: 4 cells (up, down, left, right)
            dr = [-1, 0, 0, 1]
            dc = [0, -1, 1, 0]
        else:
            raise ValueError(f"Unknown neighborhood type: {self.neighborhood_type}")

        for i in range(len(dr)):
            nr, nc = (r + dr[i]) % self.height, (c + dc[i]) % self.width
            neighbors.append(self.grid[nr, nc])
        return neighbors

    def _apply_rule(self, current_state, live_neighbor_count):
        """
        Applies the current CA rule to determine the next state of a cell.
        The rule is a dictionary mapping `(current_state, live_neighbor_count)` to `next_state`.

        Args:
            current_state (int): The current state of the cell.
            live_neighbor_count (int): The number of 'live' neighbors.

        Returns:
            int: The next state of the cell.
        """
        key = (current_state, live_neighbor_count)
        # If a specific rule is not found, default to the current state to prevent errors.
        return self.rule.get(key, current_state)

    def evolve(self):
        """
        Computes and updates the CA grid to its next generation by applying the rule
        to each cell based on its current state and its neighbors.
        """
        if self.rule is None:
            raise ValueError("CA rule has not been set. Call set_rule() first.")

        new_grid = np.copy(self.grid) # Create a copy to build the new generation
        for r in range(self.height):
            for c in range(self.width):
                current_state = self.grid[r, c]
                neighbors = self._get_neighbors(r, c)
                
                # Count 'live' neighbors based on the configured threshold
                # A 'live' neighbor is any neighbor whose state is above Config.LIVE_STATE_THRESHOLD
                live_neighbor_count = sum(1 for n_state in neighbors if n_state > Config.LIVE_STATE_THRESHOLD)
                
                # Determine the new state using the current rule
                new_grid[r, c] = self._apply_rule(current_state, live_neighbor_count)
        self.grid = new_grid # Update the grid to the new generation

    def set_rule(self, new_rule):
        """
        Sets the current evolution rule for the Cellular Automaton.

        Args:
            new_rule (dict): A dictionary representing the CA rule.
                             Keys: (current_state, live_neighbor_count)
                             Values: next_state
        """
        self.rule = new_rule

    def get_grid(self):
        """
        Returns the current state of the CA grid.

        Returns:
            np.ndarray: The 2D numpy array representing the current grid.
        """
        return self.grid

    def __str__(self):
        """
        Provides a string representation of the current CA grid for debugging or display.
        """
        s = ""
        for r in range(self.height):
            s += " ".join(str(self.grid[r, c]) for c in range(self.width)) + "\n"
        return s
```

---

**`rule_evolver.py`**

```python
"""
Defines the RuleEvolver class, responsible for dynamically creating and modifying
the Cellular Automata rules.
"""

import random
from config import Config

class RuleEvolver:
    """
    Manages the dynamic evolution and mutation of Cellular Automata rules.
    Rules are represented as a dictionary: `(current_state, live_neighbor_count) -> next_state`.
    """

    def __init__(self, num_states, neighborhood_type, mutation_rate):
        """
        Initializes the RuleEvolver.

        Args:
            num_states (int): The number of possible states for CA cells.
            neighborhood_type (str): The type of neighborhood used by the CA ('moore' or 'von_neumann').
            mutation_rate (float): Probability (0.0-1.0) for a single rule entry to mutate.
        """
        self.num_states = num_states
        self.mutation_rate = mutation_rate
        self.max_neighbors = self._get_max_neighbors(neighborhood_type)
        self.current_rule = self._generate_random_rule() # Start with a random rule set

    def _get_max_neighbors(self, neighborhood_type):
        """
        Determines the maximum possible number of neighbors based on the neighborhood type.

        Args:
            neighborhood_type (str): The type of neighborhood ('moore' or 'von_neumann').

        Returns:
            int: The maximum number of neighbors (e.g., 8 for Moore, 4 for Von Neumann).
        """
        if neighborhood_type == 'moore':
            return 8
        elif neighborhood_type == 'von_neumann':
            return 4
        else:
            raise ValueError(f"Unknown neighborhood type: {neighborhood_type}")

    def _generate_random_rule(self):
        """
        Generates a complete initial random rule set for the CA.
        For every possible combination of `(current_state, live_neighbor_count)`,
        a random `next_state` is assigned.

        Returns:
            dict: The newly generated random rule set.
        """
        rule = {}
        for current_state in range(self.num_states):
            for live_neighbor_count in range(self.max_neighbors + 1):
                # The next state can be any of the possible states
                next_state = random.randint(0, self.num_states - 1)
                rule[(current_state, live_neighbor_count)] = next_state
        return rule

    def evolve_rule(self):
        """
        Mutates the current rule set. For each entry in the rule dictionary,
        there's a `mutation_rate` probability that its `next_state` value
        will be changed to a new random state.
        """
        new_rule = self.current_rule.copy()
        for key in new_rule:
            if random.random() < self.mutation_rate:
                # Mutate this specific rule entry by assigning a new random target state
                new_rule[key] = random.randint(0, self.num_states - 1)
        self.current_rule = new_rule
        return self.current_rule

    def get_current_rule(self):
        """
        Returns the current CA rule dictionary.

        Returns:
            dict: The currently active CA rule set.
        """
        return self.current_rule

    def __str__(self):
        """
        Provides a string representation of the current rule set for debugging.
        """
        s = "Current CA Rule:\n"
        # Sort keys for consistent output
        sorted_keys = sorted(self.current_rule.keys())
        for key in sorted_keys:
            s += f"  {key} -> {self.current_rule[key]}\n"
        return s
```

---

**`music_composer.py`**

```python
"""
Defines the MusicComposer class, responsible for translating Cellular Automata
grid states into real-time MIDI musical events.
"""

import time
from mido import Message, open_output, get_output_names
from config import Config

class MusicComposer:
    """
    Translates Cellular Automata grid states into musical events (MIDI messages)
    and sends them to a virtual MIDI port for real-time playback.
    """

    def __init__(self, config):
        """
        Initializes the MusicComposer.

        Args:
            config (Config): An instance of the Config class containing all settings.
        """
        self.config = config
        self.output_port = None
        self._open_midi_port()

    def _open_midi_port(self):
        """
        Attempts to open a virtual MIDI output port.
        Prints an error message if the port cannot be opened.
        """
        try:
            # Attempt to open a virtual port. This requires OS-level support (e.g., loopMIDI, IAC Driver).
            self.output_port = open_output(self.config.MIDI_PORT_NAME, virtual=True)
            print(f"Opened virtual MIDI port: '{self.config.MIDI_PORT_NAME}'")
        except Exception as e:
            print(f"Error opening MIDI port '{self.config.MIDI_PORT_NAME}': {e}")
            print("Available MIDI output ports:", get_output_names())
            print("Please ensure a virtual MIDI interface is properly configured on your system.")
            self.output_port = None # Set to None to prevent further MIDI operations

    def _beats_to_seconds(self, beats):
        """
        Converts a duration in musical beats to real-time seconds based on the configured BPM.

        Args:
            beats (float): Duration in beats.

        Returns:
            float: Duration in seconds.
        """
        return beats * (60 / self.config.MIDI_BPM)

    def compose_and_play(self, ca_grid):
        """
        Interprets the current CA grid and generates a sequence of MIDI messages.
        It schedules and plays notes to create polyphonic music for the current generation.

        The musical interpretation:
        - Each row of the CA grid corresponds to a sequential time segment within the
          `GENERATION_MUSIC_DURATION_BEATS`.
        - Within each row, 'active' cells (states mapped in NOTE_MAP) trigger notes.
        - The column index `c` adds an offset to the base MIDI note, creating harmonic/melodic variation.
        - The cell's state determines its base note and duration.

        Args:
            ca_grid (np.ndarray): The current 2D grid state of the Cellular Automaton.
        """
        if self.output_port is None:
            print("MIDI port not open, skipping music composition for this generation.")
            return

        height, width = ca_grid.shape
        
        # Calculate the duration (in beats) for each row to play musically
        row_duration_beats = self.config.GENERATION_MUSIC_DURATION_BEATS / height
        
        # List to store (midi_message, target_real_time_offset_seconds)
        messages_to_schedule = []
        
        # Iterate through the grid to determine all notes to play in this generation
        for r in range(height):
            # Calculate the start time (in beats, relative to generation start) for the current row
            current_row_start_beat = r * row_duration_beats
            
            for c in range(width):
                state = ca_grid[r, c]
                if state in self.config.NOTE_MAP: # Only play if the state maps to a note
                    base_midi_note = self.config.NOTE_MAP[state]
                    
                    # Offset the base note using the column index for melodic variation
                    # Multiplying by 2 or 3 spreads the notes out more harmonically
                    midi_note = base_midi_note + (c * 2) 
                    
                    # Ensure the MIDI note is within the valid range (0-127)
                    midi_note = max(0, min(127, midi_note))

                    note_duration_beats = self.config.DURATION_MAP.get(state, self.config.DEFAULT_DURATION)
                    
                    # Ensure note duration is reasonable:
                    # - At least a small minimum for audibility (e.g., 0.05 beats)
                    # - Not longer than the row's allocated time minus a small gap to avoid overlap with next row
                    actual_note_duration_beats = max(0.05, min(note_duration_beats, row_duration_beats - 0.01))
                    
                    # Schedule note_on message
                    messages_to_schedule.append(
                        (Message('note_on', note=midi_note, velocity=self.config.DEFAULT_VELOCITY, channel=self.config.MIDI_CHANNEL),
                         self._beats_to_seconds(current_row_start_beat))
                    )
                    # Schedule note_off message
                    messages_to_schedule.append(
                        (Message('note_off', note=midi_note, velocity=0, channel=self.config.MIDI_CHANNEL),
                         self._beats_to_seconds(current_row_start_beat + actual_note_duration_beats))
                    )
        
        # Sort all messages by their target real-time offset for precise playback
        messages_to_schedule.sort(key=lambda x: x[1])

        # Play scheduled messages
        start_generation_real_time = time.time() # Absolute real time when this generation's music starts
        
        for msg, target_delay_seconds in messages_to_schedule:
            # Calculate the absolute real time when this message should be sent
            target_send_time_abs = start_generation_real_time + target_delay_seconds
            
            # Wait if necessary until it's time to send this message
            current_real_time = time.time()
            if current_real_time < target_send_time_abs:
                time.sleep(target_send_time_abs - current_real_time)
            
            # Send the MIDI message
            self.output_port.send(msg)
            
        # After all messages for the current generation are sent, ensure that the CA evolution
        # doesn't proceed until the configured musical duration for this generation has passed.
        end_generation_music_real_time = start_generation_real_time + self._beats_to_seconds(self.config.GENERATION_MUSIC_DURATION_BEATS)
        current_real_time = time.time()
        if current_real_time < end_generation_music_real_time:
            time.sleep(end_generation_music_real_time - current_real_time)
                
    def close(self):
        """
        Closes the MIDI output port, releasing system resources.
        Also sends an 'all notes off' message for safety.
        """
        if self.output_port:
            try:
                # Send 'All Notes Off' control message for all channels
                # Control Change 123 (All Notes Off) for channel 0, value 0
                self.output_port.send(Message('control_change', channel=self.config.MIDI_CHANNEL, control=123, value=0))
                self.output_port.close()
                print(f"Closed MIDI port: '{self.config.MIDI_PORT_NAME}'")
            except Exception as e:
                print(f"Error closing MIDI port: {e}")

```

---

**`main.py`**

```python
"""
This script orchestrates the Dynamically Evolving Self-Modifying Cellular Automata
for Generative Algorithmic Music Composition.

It brings together the Cellular Automaton, Rule Evolver, and Music Composer
to generate real-time evolving music based on a self-modifying CA.

To run this script:
1.  Install dependencies: `pip install numpy mido`
2.  Set up a virtual MIDI port on your operating system (e.g., 'loopMIDI' on Windows,
    'IAC Driver' on macOS, 'snd-virmidi' with 'aconnect' on Linux).
3.  Run the script: `python main.py`
4.  Connect your DAW, synthesizer, or MIDI visualizer to the virtual MIDI port
    named "SelfModCA_Music" (or as configured in `config.py`).
"""

# Import necessary classes from our project modules
from config import Config
from cellular_automata import CellularAutomaton
from rule_evolver import RuleEvolver
from music_composer import MusicComposer
import time

def main():
    """
    The main function to initialize, run, and manage the CA and music generation process.
    """
    print("Initializing Generative Algorithmic Music Composer...")
    config = Config() # Load all configuration settings

    # --- 1. Initialize Cellular Automaton ---
    ca = CellularAutomaton(
        width=config.GRID_WIDTH,
        height=config.GRID_HEIGHT,
        num_states=config.NUM_STATES,
        initial_density=config.INITIAL_DENSITY,
        neighborhood_type=config.NEIGHBORHOOD_TYPE
    )
    print(f"Cellular Automaton initialized: {config.GRID_WIDTH}x{config.GRID_HEIGHT} grid, {config.NUM_STATES} states.")

    # --- 2. Initialize Rule Evolver ---
    rule_evolver = RuleEvolver(
        num_states=config.NUM_STATES,
        neighborhood_type=config.NEIGHBORHOOD_TYPE,
        mutation_rate=config.RULE_MUTATION_RATE
    )
    # Set the initial rule for the CA
    ca.set_rule(rule_evolver.get_current_rule())
    print(f"Rule Evolver initialized. Initial rule established.")

    # --- 3. Initialize Music Composer ---
    composer = MusicComposer(config)
    
    # Exit if MIDI port could not be opened
    if composer.output_port is None:
        print("MIDI port could not be opened. Please check your MIDI setup. Exiting.")
        return

    try:
        print("\n--- Starting Simulation and Music Generation ---")
        print(f"BPM: {config.MIDI_BPM}, Generation Music Duration: {config.GENERATION_MUSIC_DURATION_BEATS} beats.")
        print(f"CA rules will evolve every {config.TICKS_PER_RULE_CHANGE} generations.")
        print("Connect your MIDI receiver to the virtual port specified in config.py.")
        print("Press Ctrl+C to stop the process gracefully.\n")

        for generation in range(config.MAX_GENERATIONS):
            print(f"--- Generation {generation + 1}/{config.MAX_GENERATIONS} ---")
            
            # Dynamically evolve CA rules at specified intervals
            if (generation + 1) % config.TICKS_PER_RULE_CHANGE == 0:
                print(f"  Evolving CA rules (Mutation event at Generation {generation + 1})...")
                rule_evolver.evolve_rule()
                ca.set_rule(rule_evolver.get_current_rule())
                # print(f"  New Rule:\n{rule_evolver.get_current_rule()}") # Uncomment to see the new rule

            current_grid = ca.get_grid()
            # Uncomment the line below to print the CA grid to the console for each generation
            # print("  Current CA Grid:\n" + str(ca)) 

            # Generate and play music based on the current CA state
            composer.compose_and_play(current_grid)

            # Evolve the CA for the next generation
            ca.evolve()

    except KeyboardInterrupt:
        print("\n--- Stopping Simulation ---")
    finally:
        # Ensure the MIDI port is closed and any lingering notes are turned off
        composer.close()
        print("Program finished.")

if __name__ == "__main__":
    main()
```

---

**`requirements.txt`**

```
numpy==1.26.4
mido==1.3.0
```