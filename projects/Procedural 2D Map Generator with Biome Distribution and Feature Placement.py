```python
import random
import math

# --- 1. Explanation Block ---
"""
Procedural 2D Map Generator with Biome Distribution and Feature Placement

This script generates a square or rectangular 2D map using a custom Perlin noise
implementation to simulate terrain height variations. Based on these height values,
different biomes (e.g., Ocean, Grassland, Forest, Mountain, Snow) are distributed
across the map, creating distinct geographical regions.

After biome distribution, the script intelligently places various features (e.g.,
Lakes, Forests, Mountain Peaks) within suitable biomes. This adds granular detail
and makes the generated maps more interesting and visually diverse.

The output is a text-based representation of the generated map, showcasing biomes
and features using different ASCII symbols and ANSI terminal colors for enhanced readability.

Key Features:
-   **Perlin Noise:** A custom implementation of Perlin noise generates smooth, natural-looking
    height maps.
-   **Configurable Parameters:** Map dimensions, noise complexity (octaves, persistence,
    lacunarity, scale), and a random seed are all customizable via a `MapConfig` class.
-   **Biome Distribution:** Biomes are defined with specific height ranges and are assigned
    to map cells based on their generated height values.
-   **Feature Placement:** Features have names, symbols, probabilities, and can only spawn
    within specified biome types, adding logical detail.
-   **Modular Design:** The code is structured with classes (`MapConfig`, `Biome`, `Feature`,
    `PerlinNoiseGenerator`, `MapGenerator`) to promote readability, reusability, and
    ease of modification.
-   **Text-based Visualization:** Maps are rendered using ASCII characters and optional ANSI
    color codes, making them viewable directly in the terminal.
-   **Self-contained:** The entire project is within a single Python file, requiring only
    standard library modules (`random`, `math`).
-   **Map Statistics:** Provides a summary of biome and feature counts after generation.
"""

# --- 2. Configuration Class ---
class MapConfig:
    """
    Configuration settings for the map generator.
    Encapsulates all adjustable parameters for map generation.
    """
    def __init__(self,
                 width: int = 120,
                 height: int = 40,
                 seed: int = None,
                 noise_scale: float = 0.04,
                 noise_octaves: int = 5,
                 noise_persistence: float = 0.4,
                 noise_lacunarity: float = 2.5,
                 feature_density: float = 0.015):
        """
        Initializes map generation configuration.

        Args:
            width (int): The width of the map in cells.
            height (int): The height of the map in cells.
            seed (int, optional): Seed for the random number generator. If None, a random
                                  seed is used, ensuring different maps each run.
            noise_scale (float): Determines the "zoom" level of the Perlin noise.
                                 Smaller values produce larger, smoother terrain features.
            noise_octaves (int): Number of layers of noise combined. More octaves add
                                 more fine-grained detail.
            noise_persistence (float): Amplitude multiplier for each successive octave.
                                       Controls how much detail each new octave adds.
            noise_lacunarity (float): Frequency multiplier for each successive octave.
                                      Controls how quickly detail frequency increases.
            feature_density (float): A global probability multiplier (0.0-1.0) for
                                     feature placement. Lower values result in fewer features.
        """
        self.width = width
        self.height = height
        # Use a specified seed or generate a random one
        self.seed = seed if seed is not None else random.randint(0, 100000)
        self.noise_scale = noise_scale
        self.noise_octaves = noise_octaves
        self.noise_persistence = noise_persistence
        self.noise_lacunarity = noise_lacunarity
        self.feature_density = feature_density

# --- 3. Biome Class ---
class Biome:
    """
    Represents a specific biome type with its properties.
    Biomes are defined by a height range and a list of features they can host.
    """
    def __init__(self, name: str, symbol: str, min_height: float, max_height: float,
                 features_allowed: list[str], color_code: str = None):
        """
        Initializes a Biome object.

        Args:
            name (str): The descriptive name of the biome (e.g., "Ocean", "Forest Biome").
            symbol (str): The character symbol used for text representation of this biome
                          when no feature is present.
            min_height (float): The minimum normalized height (0.0-1.0) for this biome to appear.
            max_height (float): The maximum normalized height (0.0-1.0) for this biome to appear.
            features_allowed (list[str]): A list of feature names (strings) that are allowed
                                          to spawn within this biome type.
            color_code (str, optional): An ANSI escape code for terminal color (e.g., "\033[94m"
                                        for blue). Used for coloring the biome symbol.
        """
        self.name = name
        self.symbol = symbol
        self.min_height = min_height
        self.max_height = max_height
        self.features_allowed = features_allowed
        # Default to empty string if no color code is provided, to avoid printing "None"
        self.color_code = color_code if color_code else ""

    def __repr__(self):
        """Returns a string representation of the Biome object."""
        return (f"Biome(name='{self.name}', symbol='{self.symbol}', "
                f"height=[{self.min_height:.2f}-{self.max_height:.2f}])")

# --- 4. Feature Class ---
class Feature:
    """
    Represents a specific map feature that can be placed in biomes.
    Features are smaller, localized elements like lakes, forests, or peaks.
    """
    def __init__(self, name: str, symbol: str, probability: float, size: int = 1):
        """
        Initializes a Feature object.

        Args:
            name (str): The descriptive name of the feature (e.g., "Lake", "Mountain Peak").
            symbol (str): The character symbol used for text representation of this feature.
            probability (float): The base probability (0.0-1.0) of this feature attempting
                                 to spawn per suitable cell. This is multiplied by the
                                 `MapConfig.feature_density`.
            size (int): The "radius" or spread of the feature, indicating how many cells
                        it might occupy or influence. (Currently simplified to 1 for this
                        single-file implementation, but extensible).
        """
        self.name = name
        self.symbol = symbol
        self.probability = probability
        self.size = size # For future expansion, e.g., to generate multi-cell features

    def __repr__(self):
        """Returns a string representation of the Feature object."""
        return (f"Feature(name='{self.name}', symbol='{self.symbol}', "
                f"prob={self.probability:.3f}, size={self.size})")

# --- 5. Perlin Noise Generator (Self-contained) ---
class PerlinNoiseGenerator:
    """
    Generates 2D Perlin noise using a pure Python implementation of Ken Perlin's Improved Noise.
    This class provides a smooth, gradient-based noise function.
    """
    def __init__(self, seed: int):
        """
        Initializes the Perlin Noise generator with a specific seed.
        The seed is used to shuffle the permutation table, ensuring reproducible noise.
        """
        self._seed = seed
        random.seed(self._seed) # Seed for permutation table
        self._p = list(range(256))
        random.shuffle(self._p)
        self._p += self._p # Duplicate permutation table for fast modulo operation (p[i & 255])

    def _fade(self, t: float) -> float:
        """
        Perlin's fade function (6t^5 - 15t^4 + 10t^3).
        Smooths the interpolation between gradient vectors, removing sharp edges.
        """
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _lerp(self, a: float, b: float, t: float) -> float:
        """
        Linear interpolation (Linear Interpolate) between two values 'a' and 'b' by 't'.
        """
        return a + t * (b - a)

    def _grad(self, hash_val: int, x: float, y: float) -> float:
        """
        Calculates the dot product between a pseudorandom gradient vector and the
        distance vector from the corner to the point (x,y).
        """
        h = hash_val & 0xF # Get the last 4 bits of the hash
        
        # Simplified gradient vector calculation:
        # Each of the 16 hash values corresponds to one of 8 direction vectors
        # (vectors to midpoints of edges of a hypercube, or 0-vectors).
        # This implementation uses the common simplification:
        u = x if h < 8 else y
        v = y if h < 4 or h == 12 or h == 13 else x # This pattern generates a variety of vectors.
        
        # Apply +/- based on hash for dot product
        if h & 1 == 0: u = -u
        if h & 2 == 0: v = -v
        
        return u + v

    def noise(self, x: float, y: float) -> float:
        """
        Generates a 2D Perlin noise value for a given (x, y) coordinate.
        The output value is typically between -1.0 and 1.0.
        """
        # Find the integer grid cell coordinates for the point
        X = math.floor(x) & 255 # & 255 wraps to 0-255, useful for permutation table
        Y = math.floor(y) & 255

        # Get the fractional parts of the coordinates within the grid cell
        x -= math.floor(x)
        y -= math.floor(y)

        # Compute fade curves for the fractional coordinates
        u = self._fade(x)
        v = self._fade(y)

        # Hash coordinates of the 4 corners of the grid cell
        A = self._p[X] + Y
        AA = self._p[A]
        AB = self._p[A + 1]
        B = self._p[X + 1] + Y
        BA = self._p[B]
        BB = self._p[B + 1]

        # Interpolate between the 4 gradients from the corners
        # This combines the contribution of each corner's gradient vector
        res = self._lerp(
            self._lerp(self._grad(AA, x, y),      # Bottom-left corner
                       self._grad(BA, x - 1, y),   # Bottom-right corner
                       u),
            self._lerp(self._grad(AB, x, y - 1),   # Top-left corner
                       self._grad(BB, x - 1, y - 1), # Top-right corner
                       u),
            v
        )
        return res

    def generate_noise_map(self, width: int, height: int, scale: float,
                           octaves: int, persistence: float, lacunarity: float) -> list[list[float]]:
        """
        Generates a 2D noise map of specified dimensions using multi-octave Perlin noise.

        Args:
            width (int): Width of the map grid.
            height (int): Height of the map grid.
            scale (float): Scaling factor for the noise coordinates. A smaller scale
                           produces larger, more stretched noise features.
            octaves (int): The number of noise layers to combine. Each octave adds
                           finer detail.
            persistence (float): The amplitude multiplier for each subsequent octave.
                                 Controls the falloff of detail.
            lacunarity (float): The frequency multiplier for each subsequent octave.
                                Controls how much more "zoomed in" each octave is.

        Returns:
            list[list[float]]: A 2D list (height x width) of normalized noise values,
                               ranging from 0.0 to 1.0.
        """
        noise_map = [[0.0 for _ in range(width)] for _ in range(height)]
        max_noise_height = float('-inf')
        min_noise_height = float('inf')

        # To ensure different seeds generate distinct starting points in the noise field,
        # we can add an arbitrary offset based on the seed.
        # This prevents identical patterns from appearing if only scale/octaves change.
        # Use a consistent offset based on seed, rather than a fixed (0,0) or random.
        # For simplicity, we'll just use 0,0 for this specific implementation, as the
        # internal permutation table already varies with the seed.
        x_offset, y_offset = 0, 0 

        for y in range(height):
            for x in range(width):
                amplitude = 1.0
                frequency = 1.0
                noise_height = 0.0

                # Combine multiple octaves (layers) of noise
                for i in range(octaves):
                    # Calculate sample coordinates for the current octave
                    sample_x = (x + x_offset) / scale * frequency
                    sample_y = (y + y_offset) / scale * frequency
                    
                    # Get Perlin noise value (typically -1.0 to 1.0)
                    noise_value = self.noise(sample_x, sample_y)
                    
                    # Accumulate noise with current amplitude
                    noise_height += noise_value * amplitude

                    # Update amplitude and frequency for the next octave
                    amplitude *= persistence
                    frequency *= lacunarity
                
                noise_map[y][x] = noise_height

                # Track min/max noise values for normalization
                if noise_height > max_noise_height:
                    max_noise_height = noise_height
                if noise_height < min_noise_height:
                    min_noise_height = noise_height

        # Normalize the entire noise map to values between 0.0 and 1.0
        normalized_map = [[0.0 for _ in range(width)] for _ in range(height)]
        height_range = max_noise_height - min_noise_height
        
        # Handle cases where all noise values are identical (e.g., very low octaves or scale)
        if height_range == 0:
            return [[0.5 for _ in range(width)] for _ in range(height)] # All values become 0.5

        for y in range(height):
            for x in range(width):
                normalized_map[y][x] = (noise_map[y][x] - min_noise_height) / height_range

        return normalized_map


# --- 6. Map Generator Class ---
class MapGenerator:
    """
    The main class responsible for orchestrating the map generation process.
    It combines height map generation, biome distribution, and feature placement.
    """
    def __init__(self, config: MapConfig, biomes: list[Biome], features: list[Feature]):
        """
        Initializes the MapGenerator.

        Args:
            config (MapConfig): An instance of `MapConfig` holding all generation parameters.
            biomes (list[Biome]): A list of all available `Biome` types.
            features (list[Feature]): A list of all available `Feature` types.
        """
        self.config = config
        # Sort biomes by min_height to ensure correct assignment (from lowest to highest height)
        self.biomes = sorted(biomes, key=lambda b: b.min_height)
        # Store features in a dictionary for efficient lookup by name
        self.features = {f.name: f for f in features}

        random.seed(self.config.seed) # Seed the global random for feature placement
        self.noise_generator = PerlinNoiseGenerator(self.config.seed) # Noise generator also uses the seed

        # Initialize map layers
        self.height_map: list[list[float]] = [] # Stores normalized height values
        self.biome_map: list[list[Biome]] = [] # Stores Biome objects
        self.feature_map: list[list[Feature | None]] = [] # Stores Feature objects or None

    def _generate_height_map(self) -> None:
        """
        Generates the terrain height map using the configured Perlin noise parameters.
        The result is stored in `self.height_map`.
        """
        print(f"Generating height map (Seed: {self.config.seed})...")
        self.height_map = self.noise_generator.generate_noise_map(
            self.config.width,
            self.config.height,
            self.config.noise_scale,
            self.config.noise_octaves,
            self.config.noise_persistence,
            self.config.noise_lacunarity
        )
        print("Height map generated.")

    def _distribute_biomes(self) -> None:
        """
        Distributes biomes across the map based on the generated height values.
        Each cell in `self.biome_map` is assigned a `Biome` object.
        """
        print("Distributing biomes...")
        # Initialize biome map with a default biome (e.g., the lowest height biome)
        # This simplifies subsequent assignment as every cell will have a biome object.
        self.biome_map = [[self.biomes[0] for _ in range(self.config.width)] for _ in range(self.config.height)]
        
        for y in range(self.config.height):
            for x in range(self.config.width):
                height = self.height_map[y][x]
                # Iterate through sorted biomes to find the one matching the current height
                for biome in self.biomes:
                    if biome.min_height <= height <= biome.max_height:
                        self.biome_map[y][x] = biome
                        break # Found the biome, move to the next cell
        print("Biomes distributed.")

    def _place_features(self) -> None:
        """
        Places features randomly within suitable biomes across the map.
        Features are stored in `self.feature_map`.
        """
        print("Placing features...")
        self.feature_map = [[None for _ in range(self.config.width)] for _ in range(self.config.height)]

        for y in range(self.config.height):
            for x in range(self.config.width):
                current_biome = self.biome_map[y][x]
                
                # Only attempt to place a feature if the cell doesn't already have one
                # This simple approach prevents features from overwriting each other directly.
                if self.feature_map[y][x] is None:
                    # Iterate through the features allowed in the current biome
                    for feature_name in current_biome.features_allowed:
                        feature = self.features.get(feature_name)
                        # Check if feature exists and if random chance succeeds
                        if (feature and
                            random.random() < feature.probability * self.config.feature_density):
                            
                            # Place the feature. This simplified version places it only at (x,y).
                            # A more advanced version would use feature.size to affect an area.
                            self._apply_feature_at_cell(x, y, feature)
                            
                            # Break after placing one feature to prevent stacking multiple features
                            # on the same cell, unless _apply_feature_at_cell handles it.
                            break
        print("Features placed.")

    def _apply_feature_at_cell(self, x: int, y: int, feature: Feature) -> None:
        """
        Applies a single feature to a specific cell (x, y) on the map.

        Args:
            x (int): X-coordinate of the cell.
            y (int): Y-coordinate of the cell.
            feature (Feature): The feature object to place.
        """
        # For this example, feature.size is largely ignored, and features are single-cell.
        # To implement multi-cell features (e.g., a lake of size 3):
        # for dy in range(-feature.size // 2, feature.size // 2 + 1):
        #     for dx in range(-feature.size // 2, feature.size // 2 + 1):
        #         nx, ny = x + dx, y + dy
        #         # Check bounds and biome compatibility (e.g., only place lake in grassland)
        #         if 0 <= nx < self.config.width and 0 <= ny < self.config.height and \
        #            self.biome_map[ny][nx].name == "Grassland":
        #             self.feature_map[ny][nx] = feature
        
        # Simple placement: just set the feature at the given cell.
        self.feature_map[y][x] = feature


    def generate_map(self) -> tuple[list[list[float]], list[list[Biome]], list[list[Feature | None]]]:
        """
        Generates the complete map by sequentially calling the generation steps.

        Returns:
            tuple: A tuple containing the generated `height_map`, `biome_map`,
                   and `feature_map` (in that order).
        """
        self._generate_height_map()
        self._distribute_biomes()
        self._place_features()
        return self.height_map, self.biome_map, self.feature_map

    def render_text_map(self) -> str:
        """
        Generates a text-based representation of the map, prioritizing features over biomes
        for display. Uses ANSI escape codes for basic terminal coloring if supported.

        Returns:
            str: A multi-line string representing the generated map with a legend.
        """
        output = []
        RESET_COLOR = "\033[0m" # ANSI escape code to reset color to default

        # --- Generate Map Legend ---
        output.append("--- Map Legend ---")
        
        # Biome legend
        output.append("Biomes:")
        seen_biome_symbols = set()
        for biome in self.biomes:
            if biome.symbol not in seen_biome_symbols: # Avoid duplicate symbols in legend
                output.append(f"{biome.color_code}{biome.symbol}{RESET_COLOR}: {biome.name}")
                seen_biome_symbols.add(biome.symbol)
        
        # Feature legend
        output.append("\nFeatures:")
        seen_feature_symbols = set()
        for feature_name in sorted(self.features.keys()): # Sort for consistent legend order
            feature = self.features[feature_name]
            if feature.symbol not in seen_feature_symbols: # Avoid duplicate symbols in legend
                output.append(f"{feature.symbol}: {feature.name}")
                seen_feature_symbols.add(feature.symbol)
        output.append("-" * self.config.width + "\n") # Separator line

        # --- Render the Map Grid ---
        for y in range(self.config.height):
            row_symbols = []
            for x in range(self.config.width):
                feature = self.feature_map[y][x]
                biome = self.biome_map[y][x]
                
                # Features take display precedence over biomes
                if feature:
                    row_symbols.append(feature.symbol)
                else:
                    # Apply biome color code, then its symbol, then reset color
                    row_symbols.append(f"{biome.color_code}{biome.symbol}{RESET_COLOR}")
            output.append("".join(row_symbols))
        return "\n".join(output)

# --- 7. Main Execution Block ---
if __name__ == "__main__":
    # --- Define Biomes ---
    # Each biome is defined with a name, a display symbol, a height range (0.0-1.0),
    # a list of features it can host, and an optional ANSI color code for terminals.
    # ANSI color codes for terminal output:
    # Blue: \033[94m, Cyan: \033[96m, Green: \033[92m, Yellow: \033[93m, Red: \033[91m, White: \033[97m
    # Dark Green: \033[32m, Dark Gray: \033[90m
    # Reset Color: \033[0m
    
    OCEAN = Biome("Ocean", "~", 0.0, 0.15, [], color_code="\033[94m") # Deep blue water
    BEACH = Biome("Beach", ".", 0.15, 0.20, [], color_code="\033[93m") # Sandy yellow
    GRASSLAND = Biome("Grassland", ":", 0.20, 0.50, ["Forest", "Lake"], color_code="\033[92m") # Lush green
    FOREST_BIOME = Biome("Forest Biome", ";", 0.50, 0.70, ["Forest", "Bush"], color_code="\033[32m") # Darker green
    MOUNTAIN = Biome("Mountain", "^", 0.70, 0.85, ["Mountain Peak"], color_code="\033[90m") # Gray/Dark
    SNOW = Biome("Snow", "*", 0.85, 1.0, [], color_code="\033[97m") # White

    all_biomes = [OCEAN, BEACH, GRASSLAND, FOREST_BIOME, MOUNTAIN, SNOW]

    # --- Define Features ---
    # Each feature has a name, a display symbol, a base probability, and a size (currently
    # simplified to 1 for single-cell placement in this example).
    # The probability is multiplied by MapConfig.feature_density.
    LAKE = Feature("Lake", "L", 0.005, size=3) # Small chance, often in low areas
    FOREST = Feature("Forest", "T", 0.015, size=2) # Moderate chance, for woods
    MOUNTAIN_PEAK = Feature("Mountain Peak", "A", 0.008, size=1) # Rare, for highest points
    BUSH = Feature("Bush", "o", 0.02, size=1) # Common, small vegetation

    all_features = [LAKE, FOREST, MOUNTAIN_PEAK, BUSH]

    # --- Configure Map Generation ---
    # Adjust these parameters to create different map styles.
    map_config = MapConfig(
        width=120,           # Map width in characters
        height=40,           # Map height in characters
        seed=12345,          # Set to None for a random map each run
        noise_scale=0.04,    # Smaller values => larger landmasses/features
        noise_octaves=5,     # More octaves => more detailed terrain
        noise_persistence=0.4, # Lower persistence => smoother transitions
        noise_lacunarity=2.5, # Higher lacunarity => finer details with each octave
        feature_density=0.015 # Overall multiplier for feature spawn probability
    )

    # --- Generate Map ---
    print("\n--- Starting Map Generation ---")
    generator = MapGenerator(map_config, all_biomes, all_features)
    # The generate_map method returns the internal map data structures,
    # which can be used for further processing or analysis if needed.
    height_map, biome_map, feature_map = generator.generate_map()
    print("--- Map Generation Complete ---")

    # --- Render and Print Map to Console ---
    print(generator.render_text_map())

    # --- Optional: Print Map Statistics ---
    print("\n--- Map Statistics ---")
    
    # Biome counts
    biome_counts = {biome.name: 0 for biome in all_biomes}
    for y in range(map_config.height):
        for x in range(map_config.width):
            biome_counts[biome_map[y][x].name] += 1
    
    total_cells = map_config.width * map_config.height
    print("Biome Distribution:")
    for name, count in sorted(biome_counts.items()):
        print(f"  {name}: {count} cells ({count/total_cells:.2%})")
    
    # Feature counts
    feature_counts = {feature.name: 0 for feature in all_features}
    for y in range(map_config.height):
        for x in range(map_config.width):
            if feature_map[y][x]:
                feature_counts[feature_map[y][x].name] += 1
    
    print("\nFeature Occurrences:")
    any_features_placed = False
    for name, count in sorted(feature_counts.items()):
        if count > 0:
            print(f"  {name}: {count} occurrences")
            any_features_placed = True
    if not any_features_placed:
        print("  No features were placed on the map (try increasing feature_density or probabilities).")
    
    print("\nDone.")
```