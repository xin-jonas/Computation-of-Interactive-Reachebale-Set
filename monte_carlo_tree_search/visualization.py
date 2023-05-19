import logging
import os
from pathlib import Path
from typing import Tuple, Union, List

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.scenario import Scenario
from commonroad.visualization.mp_renderer import MPRenderer


from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.trajectory import State
from commonroad.prediction.prediction import TrajectoryPrediction
from actions import Actions
from vehiclemodels import parameters_vehicle3
from commonroad.scenario.obstacle import ObstacleType
from commonroad.geometry.shape import Rectangle
import copy
import numpy as np

logger = logging.getLogger(__name__)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def plot_scenario(scenario: Scenario, figsize: Tuple = (25, 15),
                  step_start: int = 0, step_end: int = 10, steps: List[int] = None,
                  plot_limits: List = None, path_output: str = None,
                  save_gif: bool = True, duration: float = None,
                  save_plots: bool = True, show_lanelet_label: bool = False):
    """
    Plots scenarios with predicted motions.
    """
    path_output = path_output or "./output"
    Path(path_output).mkdir(parents=True, exist_ok=True)

    plot_limits = plot_limits if plot_limits else compute_plot_limits_from_lanelet_network(scenario.lanelet_network)
    if steps:
        steps = [step for step in steps if step <= step_end + 1]
    else:
        steps = range(step_start, step_end + 1)
    duration = duration if duration else scenario.dt

    renderer = MPRenderer(plot_limits=plot_limits, figsize=figsize)
    for step in steps:
        time_step = step
        if save_plots:
            # clear previous plot
            plt.cla()
        else:
            # create new figure
            plt.figure(figsize=figsize)
            renderer = MPRenderer(plot_limits=plot_limits)

        # plot scenario and planning problem
        scenario.draw(renderer, draw_params={"dynamic_obstacle": {"draw_icon": True},
                                             "trajectory": {"draw_trajectory": True},
                                             "time_begin": time_step,
                                             "lanelet": {"show_label": show_lanelet_label}})

        # settings and adjustments
        plt.rc("axes", axisbelow=True)
        ax = plt.gca()
        ax.set_aspect("equal")
        ax.set_title(f"$t = {time_step / 10.0:.1f}$ [s]", fontsize=28)
        ax.set_xlabel(f"$s$ [m]", fontsize=28)
        ax.set_ylabel("$d$ [m]", fontsize=28)
        plt.margins(0, 0)
        renderer.render()

        if save_plots:
            save_fig(save_gif, path_output, step)

        else:
            plt.show()

    if save_plots and save_gif:
        make_gif(path_output, "png_scenario_", steps, str(scenario.scenario_id), duration)


def compute_plot_limits_from_lanelet_network(lanelet_network: LaneletNetwork, margin: int = 10):
    list_vertices_x = list()
    list_vertices_y = list()
    for lanelet in lanelet_network.lanelets:
        vertex_center = lanelet.center_vertices
        list_vertices_x.extend(list(vertex_center[:, 0]))
        list_vertices_y.extend(list(vertex_center[:, 1]))

    x_min, x_max = min(list_vertices_x), max(list_vertices_x)
    y_min, y_max = min(list_vertices_y), max(list_vertices_y)
    plot_limits = [x_min - margin, x_max + margin, y_min - margin, y_max + margin]

    return plot_limits


def save_fig(save_gif: bool, path_output: str, time_step: int):
    if save_gif:
        # save as png
        print("\tSaving", os.path.join(path_output, f'{"png_scenario"}_{time_step:05d}.png'))
        plt.savefig(os.path.join(path_output, f'{"png_scenario"}_{time_step:05d}.png'), format="png",
                    bbox_inches="tight",
                    transparent=False)

    else:
        # save as svg
        print("\tSaving", os.path.join(path_output, f'{"svg_scenario"}_{time_step:05d}.svg'))
        plt.savefig(f'{path_output}{"svg_scenario"}_{time_step:05d}.svg', format="svg", bbox_inches="tight",
                    transparent=False)


def make_gif(path: str, prefix: str, steps: Union[range, List[int]],
             file_save_name="animation", duration: float = 0.1):
    images = []
    filenames = []

    for step in steps:
        im_path = os.path.join(path, prefix + "{:05d}.png".format(step))
        filenames.append(im_path)

    for filename in filenames:
        images.append(imageio.imread(filename))
    file_save_name = 'a_' + file_save_name
    imageio.mimsave(os.path.join(path, file_save_name + ".gif"), images, duration=duration)


def plot_trajectory_velocity(scenario: Scenario):
    ob_list = scenario.dynamic_obstacles
    path_output = "./output"
    Path(path_output).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 6))
    # trajectory of dynamic obstacle
    plt.subplot(2, 1, 1)
    plt.xlabel("x")
    plt.ylabel("y")
    labels = []
    plts = []
    for ob in ob_list:
        ob_path = ob.prediction.trajectory.state_list
        x = [state.position[0] for state in ob_path]
        y = [state.position[1] for state in ob_path]
        # print(x)
        # print(y)
        label = 'obstacle_id: ' + str(ob.obstacle_id)
        # plt.plot(x, y, label=label)
        temp, = plt.plot(x, y, label=label)
        plts.append(temp)
        labels.append(label)
    plt.legend(handles=plts, labels=labels)
    # velocity of dynamic obstacle
    plt.subplot(2, 1, 2)
    plt.xlabel("time_step")
    plt.ylabel("velocity")
    labels = []
    plts = []
    for ob in ob_list:
        ob_path = ob.prediction.trajectory.state_list
        x = [state.time_step for state in ob_path]
        y = [state.velocity for state in ob_path]
        # print(x)
        # print(y)
        label = 'obstacle_id: ' + str(ob.obstacle_id)
        # plt.plot(x, y, label=label)
        temp, = plt.plot(x, y, label=label)
        plts.append(temp)
        labels.append(label)
    plt.savefig(os.path.join(path_output, "a_s_v_obs" + str(scenario.scenario_id)+".png"))
    plt.show()


def generate_vehicle(scenario, start_position, time_length):
    # add ego object into scenario
    # ego shape
    vehicle3 = parameters_vehicle3.parameters_vehicle3()
    ego_vehicle_shape = Rectangle(length=vehicle3.l, width=vehicle3.w)

    ego_initial_state = State(position = np.array(start_position),
                                       velocity = 7,
                                       orientation = 1.57,
                                       acceleration = 0,
                                       yaw_rate = 0,
                                       slip_angle = 0,
                                       time_step = 0)
    # generate the states for the obstacle for time steps 1 to 40 by assuming constant velocity
    state_list = [ego_initial_state]
    new_state = copy.deepcopy(ego_initial_state)
    # action = 0
    for i in range(1, time_length):
        action = np.random.choice([0, 1, 2])
        new_state = Actions(new_state, action, scenario.dt).take_action()
        state_list.append(new_state)

    # create the planned trajectory starting at time step 1
    ego_vehicle_trajectory = Trajectory(initial_time_step=1, state_list=state_list[1:])
    # create the prediction using the planned trajectory and the shape of the ego vehicle
    ego_vehicle_prediction = TrajectoryPrediction(trajectory=ego_vehicle_trajectory,
                                                  shape=ego_vehicle_shape)
    # the ego vehicle can be visualized by converting it into a DynamicObstacle
    ego_vehicle_type = ObstacleType.CAR
    # generate the dynamic obstacle according to the specification
    ego_obstacle_id = scenario.generate_object_id()
    print(ego_obstacle_id)
    ego_vehicle = DynamicObstacle(obstacle_id=ego_obstacle_id, obstacle_type=ego_vehicle_type,
                                  obstacle_shape=ego_vehicle_shape, initial_state=ego_initial_state,
                                  prediction=ego_vehicle_prediction)
    # add ego vehicle to the scenario
    scenario.add_objects(ego_vehicle)
    return scenario

