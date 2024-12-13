from collections.abc import MutableMapping
from suqc.CommandBuilder.JarCommand import JarCommand
from suqc import DictVariation, PostScenarioChangesBase
from suqc.parameter.postchanges import (
    ChangeRealTimeSimTimeRatio,
    ChangeScenarioName,
    AlwaysEnableMetaData,
)
from pathlib import Path
import numpy as np

# refert to string because suqc cannot handle Path objects
path2model = str(Path("./vadere-console.jar").absolute())
path2scenario = str(Path("./scenarios/bottleneck_forward.scenario").absolute())
path2script = str(Path(__file__).resolve().parent)
path2output= str(Path(path2script) / "output")

jar_command = (
    JarCommand(jar_file=path2model).add_option("-enableassertions").main_class("suq")
)

# see values in get_obstacle
xmax = 52.5
ymin = 1.5
ymax = 31.5
room_width = 30

y_midpoint = ymin + room_width / 2


def get_obstacle(width):
    min_width = 1.0  # TODO: make sensible values!
    max_width = 28.0

    if not (min_width <= width <= max_width):
        raise ValueError(
            f"Width is outside valid region ({min_width=} <= {width=} <= {max_width=})"
        )

    # these values are set below
    y_up = y_midpoint + width / 2.0
    y_down = y_midpoint - width / 2.0
    temp = 8.0 + width

    sampled_obstacle = [ 
        {
        "id" : 1,
        "shape" : {
          "x" : 10.0,
          "y" : 10.0,
          "width" : 1.0,
          "height" : 40.0,
          "type" : "RECTANGLE"
        },
        "visible" : True
      }, {
        "id" : 2,
        "shape" : {
          "x" : 10.0,
          "y" : 50.0,
          "width" : 59.0,
          "height" : 1.0,
          "type" : "RECTANGLE"
        },
        "visible" : True
      }, {
        "id" : 3,
        "shape" : {
          "x" : 10.0,
          "y" : 10.0,
          "width" : 59.0,
          "height" : 1.0,
          "type" : "RECTANGLE"
        },
        "visible" : True
      }, {
        "id" : 4,
        "shape" : {
          "x" : 71.0,
          "y" : 50.0,
          "width" : 19.0,
          "height" : 1.0,
          "type" : "RECTANGLE"
        },
        "visible" : True
      }, {
        "id" : 5,
        "shape" : {
          "x" : 71.0,
          "y" : 10.0,
          "width" : 19.0,
          "height" : 1.0,
          "type" : "RECTANGLE"
        },
        "visible" : True
      }, {
        "id" : 6,
        "shape" : {
          "x" : 90.0,
          "y" : 32.0,
          "width" : 1.0,
          "height" : 19.0,
          "type" : "RECTANGLE"
        },
        "visible" : True
      }, {
        "id" : 7,
        "shape" : {
          "x" : 90.0,
          "y" : 10.0,
          "width" : 1.0,
          "height" : 19.0,
          "type" : "RECTANGLE"
        },
        "visible" : True
      }, {
        "id" : 8,
        "shape" : {
          "x" : 86.7,
          "y" : 50.0,
          "width" : 4.1,
          "height" : temp,
          "type" : "RECTANGLE"
        },
        "visible" : True
      }, {
        "id" : 9,
        "shape" : {
          "x" : 87.3,
          "y" : 0.0,
          "width" : 3.5,
          "height" : temp,
          "type" : "RECTANGLE"
        },
        "visible" : True
      } ]

    return sampled_obstacle

def attribute_sim(random_seed):
    sampled_attributes =  [ 
        {
        "finishTime" : 500.0,
        "simTimeStepLength" : 0.4,
        "realTimeSimTimeRatio" : 0.0,
        "writeSimulationData" : True,
        "visualizationEnabled" : True,
        "printFPS" : False,
        "digitsPerCoordinate" : 2,
        "useFixedSeed" : False,
        "fixedSeed" : random_seed,
        "simulationSeed" : 0
        }
        ]
    

    return sampled_attributes


widths_samples = [2.0]
#widths_samples = np.arange(2, 25, 3)
np.random.seed(33)
# number of agents must for now be specified in to the basis file (bottleneck_forward.scenario)
#random_seeds_numpy = [100]
# hardcode some information for animate.py - used to visualize the width of the exit
parameter = [
    {
        #"attributesSimulation" : attribute_sim(seed),
        "obstacles": get_obstacle(exit_width),
        "description": f"{xmax=},{ymin=},{ymax=},{y_midpoint=},{exit_width=}",
        
    }
    #for seed in random_seeds_numpy
    for exit_width in widths_samples
]

post_changes = PostScenarioChangesBase(apply_default=False)
post_changes = post_changes.add_scenario_change(ChangeRealTimeSimTimeRatio())
post_changes = post_changes.add_scenario_change(AlwaysEnableMetaData())
post_changes = post_changes.add_scenario_change(ChangeScenarioName())

setup = DictVariation(
    scenario_path=path2scenario,
    parameter_dict_list=parameter,
    qoi="density_mesh.txt",
    model=jar_command,
    scenario_runs=1,
    post_changes=post_changes,
    output_path=path2script,
    output_folder="output",
    remove_output=False,
)


lookup_df, result_df = setup.run()


lookup_df.to_csv(Path(path2script) / "lookup_df4.csv")

# subsample time interval if needed
# remove steady state solution (all zero)
result_df = result_df.loc[(result_df!=0).any(axis=1)]
result_df.to_csv(Path(path2script) / "result_df4.csv")

# copy a mesh from one of the results along the two data frames -- it is assumed that they are all the same

import shutil
src = Path(path2output) / "vadere_output" / "0_0_output" / "mesh.txt"
dst = Path(path2script) / "mesh.txt"
shutil.copyfile(src, dst)
