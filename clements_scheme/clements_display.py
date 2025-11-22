import plotly.graph_objects as go
import numpy as np
import clements_scheme
from rnd_unitary import random_unitary


clements_graph = go.Figure()

def draw_clements_scheme(D, decomposition, N=None):
    """Draw a Clements scheme using Plotly."""
    spacing = 100
    radius = 15

    if N is None:
        # infer N from the decomposition
        max_mode = 0
        for m, n, phi, theta in decomposition:
            max_mode = max(max_mode, m, n)
        N = max_mode + 1

    # lines
    for i in range(N):
        flat_x = N-i if i % 2 == 0 else i + 1
        flat_y = 0 if i % 2 == 0 else N-1
        clements_graph.add_trace(go.Scatter(
            x=[0, 1, flat_x, flat_x+1, N+1, N+2, None] * spacing,
            y = [N-i-1, N-i-1, flat_y, flat_y, i, i, None] * spacing,
            mode='lines',
            line={"color":'Black', "width":2},
            connectgaps=False,
            hoverinfo='skip'
        ))

    # beamsplitters (hover info)
    bs_index = 0
    for i in range(2, N, 2):
        for j in range(i):
            m, n, phi, theta = decomposition[bs_index + j]
            y_coord = (N - m + N - n - 2) / 2
            x_coord = y_coord + (N - i + 1)
            clements_graph.add_trace(go.Scatter(
                x = [x_coord], y = [y_coord],
                mode='markers',
                marker_opacity=0,
                hovertext=[f"BS between modes {m+1} and {n+1}<br>φ={phi:.3f}, θ={theta:.3f}"],
                hoverinfo='text'
            ))
        bs_index += i
    start = N-1 if N % 2 == 0 else N-2
    for i in range(start, 0, -2):
        for j in range(i):
            m, n, phi, theta = decomposition[bs_index + j]
            y_coord = (N - m + N - n - 2) / 2
            x_coord = y_coord + (i - N + 2)
            clements_graph.add_trace(go.Scatter(
                x = [x_coord], y = [y_coord],
                mode='markers',
                marker_opacity=0,
                hovertext=[f"BS between modes {m+1} and {n+1}<br>φ={phi:.3f}, θ={theta:.3f}"],
                hoverinfo='text'
            ))
        bs_index += i

    # phase shifters (hover info)
    for i in range(N):
        clements_graph.add_trace(go.Scatter(
            x = [N+1.5], y = [N - i - 1],
            mode='markers',
            marker_opacity=0,
            hovertext=[f"Phase shifter on mode {i+1}<br>φ={np.angle(D[i,i]):.3f}" ],
            hoverinfo='text'
        ))

    # U_text = [[f"{x.real:.2f}+{x.imag:.2f}j" for x in row] for row in U]
    # clements_graph.add_trace(go.Table(
    #     # header=dict(values=[f"Col {i+1}" for i in range(U.shape[1])],
    #     #             fill_color='lightgrey', align='center'),
    #     cells=dict(values=[list(col) for col in zip(*U_text)],
    #             fill_color='white', align='center')
    # ))

    clements_graph.update_layout(
        title="Clements Scheme of U",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white')
    clements_graph.update_xaxes(visible=False,)
    clements_graph.update_yaxes(dtick=1, ticktext=[str(N-i) for i in range(N)],
                                tickvals=list(range(N)))
    clements_graph.show()

U = random_unitary(6)
decomposition, D = clements_scheme.full_clements(U, project=True)
#print(decomposition)
draw_clements_scheme(U, decomposition)