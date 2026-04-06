"""Module for drawing tensor networks in a human-readable format."""

from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class TNSketch:
    """A class for drawing tensor networks in a human-readable format.

    Attributes:
        network (TensorNetworkCircuit): The tensor network circuit to draw.
        n_modes (int): The number of modes in the tensor network.
        label_mode (str): The labeling mode for the gates. Can be "full" for full labels,
        "short" for names only, "no_values" for labels without parameter values,
        or "minimal" for minimal labels.
    """
    def __init__(self, n_modes: int, name: Optional[str]=None, label_mode: Optional[str]="full"):
        self.n_modes = n_modes
        self.name = name if name is not None else "unnamed_network"
        if label_mode not in {"full", "short", "no_values", "minimal"}:
            raise ValueError(f"Unknown label mode: {label_mode}. Supported modes are 'full', \
                'short', 'no_values', and 'minimal'.")
        self.label_mode = label_mode
        self._grid = {i: {"top": "", "mid": f"m{i}: ──", "bot": ""} for i in range(n_modes)}

    def _generate_label(self,
                        gate_name: str,
                        single_mode: bool,
                        tags: Optional[set[str]]=None,
                        params: Optional[dict]=None) -> str:
        """Generate a label for a gate based on its name, tags, and parameters.

        Args:
            gate_name (str): The name of the gate.
            tags (Optional[set[str]], optional): Tags associated with the gate. Defaults to None.
            params (Optional[dict], optional): The parameters of the gate. Defaults to None.
        
        Returns:
            tuple[str, str]: The generated labels for the gate.
        """
        tag_labels = {
            'bs': "BS",
            'ps': "PS",
            'input': "Input",
            'output': "Output",
            'contracted': "⧉ contracted"
        }
        if self.label_mode == "minimal":
            label1 = gate_name[:4]  # take first 4 characters of the gate name as label
            label2 = " " * len(label1)  # empty label of the same length for alignment
        elif self.label_mode == "short":
            label1 = gate_name  # use the full gate name as label, without parameter values or tags
            label2 = " " * len(label1)
        else:
            label1 = gate_name
            label2 = ", ".join([tag_labels.get(tag, tag) for tag in tags]) if tags is not None else None

            if self.label_mode == "full":
                if params is not None and params != {}:
                    if label2 is not None:
                        label2 += ": "

                    if 'phi' in params and 'theta' in params:
                        label2 += f"φ={params['phi']:.2f}, θ={params['theta']:.2f}"
                    elif 'angle' in params:
                        label2 += f"α={params['angle']:.2f}"
                    else:
                        label2 += ", ".join([f"{k}={v}" for k, v in params.items()])
            elif self.label_mode == "no_values":
                if params is not None and params != {}:
                    if label2 is not None:
                        label2 += ": "

                    if 'phi' in params and 'theta' in params:
                        label2 += "φ, θ"
                    elif 'angle' in params:
                        label2 += "α"
                    else:
                        label2 += ", ".join([f"{k}" for k in params.keys()])


            else:
                raise ValueError(f"Unknown label mode: {self.label_mode}")
            
            if single_mode:  # single mode gate
                label1 = label1 + "  (" + label2 + ")" if label2 is not None else label1
                label2 = None  # only use one label for single mode gates
            
            label_len = max(len(label1), len(label2) if label2 is not None else 0)
            label1 = label1.center(label_len)
            if label2 is not None:
                label2 = label2.center(label_len)
        
        return label1, label2
    
    def add_gate(self,
                 gate_name: str,
                 startmode: int,
                 tags: Optional[set[str]]=None,
                 params: Optional[dict]=None,
                 endmode: Optional[int]=None):
        """Add a gate to the tensor network.

        Args:
            gate_name (str): The name of the gate.
            startmode (int): The starting mode of the gate.
            endmode (Optional[int], optional): The ending mode of the gate. Defaults to None.
        """
        
        if endmode is None:
            endmode = startmode

        single_mode = (startmode == endmode)
        label1, label2 = self._generate_label(gate_name, single_mode, tags, params)

        max_char_length = max(*[len(self._grid[i]["top"]) for i in range(startmode, endmode + 1)],
                              *[len(self._grid[i]["mid"]) for i in range(startmode, endmode + 1)],
                              *[len(self._grid[i]["bot"]) for i in range(startmode, endmode + 1)])
        for i in range(startmode, endmode + 1):
            self._grid[i]["top"] += " " * (max_char_length - len(self._grid[i]["top"]))
            self._grid[i]["mid"] += "─" * (max_char_length - len(self._grid[i]["mid"]))
            self._grid[i]["bot"] += " " * (max_char_length - len(self._grid[i]["bot"]))

        if (endmode - startmode) % 2 == 0: # odd number of modes
            for i in range(startmode, endmode + 1):
                if i == startmode:
                    self._grid[i]["top"] += "┌" + "─" * (len(label1) + 2) + "┐ "
                else:
                    self._grid[i]["top"] += "│" + " " * (len(label1) + 2) + "│ "

                if i == (startmode + endmode) // 2:
                    self._grid[i]["mid"] += f"┤ {label1} ├─"
                else:
                    self._grid[i]["mid"] += "┤" + " " * (len(label1) + 2) + "├─"

                if i == endmode:
                    self._grid[i]["bot"] += "└" + "─" * (len(label1) + 2) + "┘ "
                elif i == (startmode + endmode) // 2:
                    self._grid[i]["bot"] += f"│ {label2} │ "
                else:
                    self._grid[i]["bot"] += "│" + " " * (len(label1) + 2) + "│ "
        else: # even number of modes
            for i in range(startmode, endmode + 1):
                if i == startmode:
                    self._grid[i]["top"] += "┌" + "─" * (len(label1) + 2) + "┐ "
                elif i == (startmode + endmode) // 2 + 1:
                    self._grid[i]["top"] += f"│ {label2} │ "
                else:
                    self._grid[i]["top"] += "│" + " " * (len(label1) + 2) + "│ "
                
                self._grid[i]["mid"] += "┤" + " " * (len(label1) + 2) + "├─"

                if i == endmode:
                    self._grid[i]["bot"] += "└" + "─" * (len(label1) + 2) + "┘ "
                elif i == (startmode + endmode) // 2:
                    self._grid[i]["bot"] += f"│ {label1} │ "
                else:
                    self._grid[i]["bot"] += "│" + " " * (len(label1) + 2) + "│ "

    def draw(self):
        """Draw the tensor network."""
        max_char_length = max(*[len(self._grid[i]["top"]) for i in range(self.n_modes)],
                              *[len(self._grid[i]["mid"]) for i in range(self.n_modes)],
                              *[len(self._grid[i]["bot"]) for i in range(self.n_modes)])
        for i in range(self.n_modes):
            self._grid[i]["top"] += " " * (max_char_length - len(self._grid[i]["top"]))
            self._grid[i]["mid"] += "─" * (max_char_length - len(self._grid[i]["mid"]))
            self._grid[i]["bot"] += " " * (max_char_length - len(self._grid[i]["bot"]))

        result = [self._grid[i]["top"] + "\n" + self._grid[i]["mid"] + "\n" + self._grid[i]["bot"] for i in range(self.n_modes)]
        print(f"Tensor Network: {self.name}")
        print("\n".join(result))

class TNPlot:
    """Draw a tensor network using Matplotlib.

    Attributes:
        n_modes (int): The number of modes in the tensor network.
        name (str): The name of the tensor network.
        label_mode (str): The mode for labeling gates ("full", "short", "no_values", "minimal").
        fig (matplotlib.figure.Figure): The Matplotlib figure object for drawing.
        ax (matplotlib.axes.Axes): The Matplotlib axes object for drawing.
        curr_x (float): The current x-coordinate for placing the next gate in the drawing.
    """
    def __init__(self, n_modes: int, name: str = None, label_mode: Optional[str]="full"):
        self.n_modes = n_modes
        self.name = name if name is not None else "unnamed_network"
        self.label_mode = label_mode
        self.curr_mode_x = {i: 0 for i in range(n_modes)}  # track the current x position for each mode

        # Initialize the Matplotlib figure and axes
        self.fig, self.ax = plt.subplots(figsize=(12, n_modes * 1.2))
        self.ax.set_title(f"Tensor Network: {self.name}", fontsize=14, pad=20)

        # Wires drawing
        for i in range(n_modes):
            self.ax.hlines(i, -0.5, 0.5, colors='black', linewidth=1.5, zorder=1)
            self.ax.text(-0.8, i, f'm{i}', va='center', ha='right', fontsize=12, fontweight='bold')

    def _generate_label(self,
                        gate_name: str,
                        tags: Optional[set[str]]=None,
                        params: Optional[dict]=None) -> str:
        """Generate a label for a gate based on its name, tags, and parameters.

        Args:
            gate_name (str): The name of the gate.
            tags (Optional[set[str]], optional): The tags associated with the gate. Defaults to None.
            params (Optional[dict], optional): The parameters of the gate. Defaults to None.
        
        Returns:
            tuple[str, str]: The generated labels for the gate.
        """
        tag_labels = {
            'bs': "BS",
            'ps': "PS",
            'input': "Input",
            'output': "Output",
            'contracted': r"$\boxtimes$ contracted"
        }
        if self.label_mode == "minimal":
            label1 = gate_name[:4]  # take first 4 characters of the gate name as label
            label2 = None
        elif self.label_mode == "short":
            label1 = gate_name  # use the full gate name as label, but do not include parameter values or tags
            label2 = None
        else:
            label1 = gate_name
            label2 = ", ".join([tag_labels.get(tag, tag) for tag in tags]) if tags is not None else None

            if self.label_mode == "full":
                if params is not None and params != {}:
                    if label2 is not None:
                        label2 += ": "

                    if 'phi' in params and 'theta' in params:
                        label2 += fr"$\phi$={params['phi']:.2f}, $\theta$={params['theta']:.2f}"
                    elif 'angle' in params:
                        label2 += fr"$\alpha$={params['angle']:.2f}"
                    else:
                        label2 += ", ".join([f"{k}={v}" for k, v in params.items()])
            elif self.label_mode == "no_values":
                if params is not None and params != {}:
                    if label2 is not None:
                        label2 += ": "

                    if 'phi' in params and 'theta' in params:
                        label2 += r"$\phi, \theta$"
                    elif 'angle' in params:
                        label2 += r"$\alpha$"
                    else:
                        label2 += ", ".join([f"{k}" for k in params.keys()])


            else:
                raise ValueError(f"Unknown label mode: {self.label_mode}")
            
            # if single_mode:  # single mode gate
            #     label1 = label1 + "  (" + label2 + ")" if label2 is not None else label1
            #     label2 = None  # only use one label for single mode gates
            
            label_len = max(len(label1), len(label2) if label2 is not None else 0)
            label1 = label1.center(label_len)
            if label2 is not None:
                label2 = label2.center(label_len)
        
        return label1, label2

    def add_gate(self, gate_name, startmode, tags=None, params=None, endmode=None):
        """Add a gate to the tensor network drawing.
        
        Args:            gate_name (str): The name of the gate.
            startmode (int): The starting mode of the gate.
            tags (Optional[set[str]], optional): The tags associated with the gate. Defaults to
                None.
            params (Optional[dict], optional): The parameters of the gate. Defaults to None.
            endmode (Optional[int], optional): The ending mode of the gate. Defaults to None
                (same as startmode).
        """
        if endmode is None:
            endmode = startmode

        # 1. Label generation
        l1, l2 = self._generate_label(gate_name, tags, params)
        full_label = f"{l1}\n{l2}" if (l2 is not None and l2 != "") else l1

        # 2. Box drawing
        m_min, m_max = min(startmode, endmode), max(startmode, endmode)
        box_height = (m_max - m_min) + 0.7
        # Adjust box width based on label length (with some padding)
        label_length = max(len(l1), len(l2) if l2 else 0)
        box_width = label_length * 0.1 + 0.6
        
        # 3. X positionning
        x = max(self.curr_mode_x[i] for i in range(startmode, endmode + 1)) + 0.25 + box_width/2

        # Determine the color of the box based on tags
        color = "#26B2F3" # Light blue by default
        if tags and 'contracted' in tags:
            color = "#BB49CD" # Purple for contractions
        if tags and ('input' in tags or 'output' in tags):
            color = "#23EA33" # Green for inputs and outputs
        if tags and ('bs' in tags or 'ps' in tags):
            color = "#FF933B" # Orange for BS and PS

        rect = patches.Rectangle((x - box_width/2, m_min - 0.35), box_width, box_height, 
                                 facecolor=color, edgecolor='black', linewidth=1.5, zorder=3)
        self.ax.add_patch(rect)

        # 4. Gate Text
        self.ax.text(x, (m_min + m_max)/2, full_label, 
                     ha='center', va='center', zorder=4, fontsize=9, wrap=True)

        # 5. Wires drawing
        for i in range(self.n_modes):
            # Draw the wire from the previous position to the current position + a bit after the box
            self.ax.hlines(i, self.curr_mode_x[i], x + box_width/2 + 0.25, colors='black', linewidth=1.5, zorder=1)

        # Update the current x position for the modes involved in the gate
        for i in range(startmode, endmode + 1):
            self.curr_mode_x[i] = x + box_width/2 + 0.25

    def finalize(self):
        """Finalize the drawing by setting the limits and displaying the plot."""
        self.ax.set_xlim(-1.5, max(self.curr_mode_x.values()) + 1)
        self.ax.set_ylim(-0.7, self.n_modes - 0.3)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        plt.gca().invert_yaxis() # Mode 0 en haut comme dans ta grille
        plt.tight_layout()
        plt.show()