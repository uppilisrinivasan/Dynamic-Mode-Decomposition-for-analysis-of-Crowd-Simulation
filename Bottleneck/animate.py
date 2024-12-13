import re
import json
import numpy as np
import pandas as pd
import PyQt5
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as anim



def read_density(scenario_id, run_id):
    density_df = pd.read_csv("./result_df2.csv", index_col=[0,1])
    
    # only extract the values for the set scenario and run
    density_df = density_df.loc[pd.IndexSlice[scenario_id, run_id], :].set_index(["timeStep", "faceId"])
    return density_df

# set folder
def read_and_init_mesh(filepath_mesh):
    # read vertices in pd.Dataframe -- everything between one after #nVertices and #nBorders

    with open(filepath_mesh) as f:
        entire_file = f.read()

    # nVertices dimension boundaryMarker targetMarker nAttributes

    # nBorders
    start_string = "#nVertices dimension boundaryMarker targetMarker nAttributes"
    end_string = "# nBorders"

    pattern = re.escape(start_string) + r"(.*?)" + re.escape(end_string)
    matches_vertices = (
        re.search(pattern, entire_file, re.DOTALL).group(1).strip().split("\n")[1:]
    )

    vertices_rows = np.genfromtxt(matches_vertices, dtype=float)
    vertices = pd.DataFrame(
        vertices_rows[:, -2:], index=vertices_rows[:, 0].astype(int), columns=["x", "y"]
    )

    start_string = "# nVertices vertexIds"
    end_string = "# nHoles"

    pattern = re.escape(start_string) + r"(.*?)" + re.escape(end_string)
    matches_edges = (
        re.search(pattern, entire_file, re.DOTALL).group(1).strip().split("\n")
    )

    edges_rows = np.genfromtxt(matches_edges, dtype=int)
    connections = pd.DataFrame(edges_rows[:, -3:] - 1, columns=["v1", "v2", "v3"])

    triag = tri.Triangulation(*vertices.to_numpy().T, triangles=connections)

    f, ax = plt.subplots()

    tripcolor = ax.tripcolor(
        triag,
        facecolors=np.zeros(connections.shape[0]),
        vmin=0,
        vmax=float(density.max().iloc[0]),
        #vmax=float(density.max()[0]),
        cmap="Reds",
    )
    ax.triplot(triag, c="black")
    f.colorbar(tripcolor)
    ax.axis("equal")

    return f, ax, tripcolor


def read_width_and_midpoint(scenario_id, run_id):
    lookup = pd.read_csv("./lookup_df2.csv", header=[0,1], index_col=[0,1])
    
    xmax_str,ymin_str, ymax_str, midpoint_str, exit_witdh_str = lookup.loc[(scenario_id, run_id), ("Parameter", "description")].split(",")

    xmax = float(xmax_str.split("=")[1].strip())
    ymin = float(ymin_str.split("=")[1].strip())
    ymax = float(ymax_str.split("=")[1].strip())
    midpoint = float(midpoint_str.split("=")[1].strip())
    width = float(exit_witdh_str.split("=")[1].strip())
    
    return xmax, ymin, ymax, midpoint, width

scenario_id = 0
run_id = 0

density = read_density(scenario_id, run_id)

xmax, ymin, ymax, midpoint, width = read_width_and_midpoint(scenario_id, run_id)

f, ax, tripcolor = read_and_init_mesh(f"./mesh.txt")

ax.vlines(xmax, ymin=midpoint+width/2, ymax=midpoint*2, color="black", linewidth=3)
ax.vlines(xmax, ymin=0, ymax=midpoint-width/2, color="black", linewidth=3)


def update(frame):
    if frame + 1 <= 1251: #in density.index.get_level_values('timeStep'):
        new_density = density.loc[pd.IndexSlice[frame + 1, :], :].to_numpy().flatten()
        tripcolor.set_array(new_density)
        return tripcolor, f
    else:
        # Handle the case where frame + 1 is out of bounds
        # For example, setting a default value or taking alternative action
        pass

ani = anim.FuncAnimation(fig=f, func=update, frames= 300 - 1, interval=200)  #density.shape[0]
#ani
writer = anim.PillowWriter(fps=30)  # Adjust fps as needed
ani.save('filename3.gif', writer=writer)

#plt.show()

print ('hello')
