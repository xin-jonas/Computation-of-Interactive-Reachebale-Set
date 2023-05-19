import commonroad_reach.utility.logger as util_logger
from commonroad_reach.data_structure.configuration_builder import ConfigurationBuilder
from commonroad_reach.data_structure.reach.reach_interface import ReachableSetInterface
from commonroad_reach.utility import visualization as util_visual
import numpy as np

import os
from commonroad.common.file_reader import CommonRoadFileReader
# import necessary classes from different modules
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import StaticObstacle, ObstacleType
# import necessary classes from different modules
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction
from actions_range import ActionsRange as Actions
from vehiclemodels import parameters_vehicle3
import copy
from mcts_idm_sim_reach import MonteCarloTreeSearchIRS
import visualization as vs
import time


def main():
    # ==== specify scenario
    # name_scenario = "DEU_Test-1_1_T-1"
    # name_scenario = "ZAM_Over-1_1"
    # name_scenario = "ARG_Carcarana-1_1_T-1"
    # name_scenario = "USA_US101-6_1_T-1"
    name_scenario = "ZAM_Intersection-1_2_T-1"
    # name_scenario = 'USA_US101-15_1_T-1'
    # name_scenario = 'DEU_IV21-1_1_T-1'
    # name_scenario = 'ZAM_Zip-1_6_T-1'
    # name_scenario = 'DEU_Guetersloh-39_1_T-1' # !!
    # name_scenario = 'DEU_Flensburg-61_1_T-1'
    # name_scenario = 'DEU_Lohmar-13_1_T-1'
    # name_scenario = 'DEU_Lohmar-20_1_T-1'
    # name_scenario = 'DEU_Lohmar-32_1_T-1'
    # name_scenario = 'DEU_Moelln-7_1_T-1'
    # name_scenario = 'ESP_Almansa-6_1_T-1'
    # name_scenario = 'USA_US101-1_1_T-1'
    # name_scenario = 'USA_US101-11_4_T-1'
    # name_scenario = 'USA_US101-15_3_T-1'
    file_path = '/home/xinzhang/MCTS/demo/IDM_demo/commonroad-motion-planning-library/scenario'
    file_path = os.path.join(file_path, name_scenario + '.xml')
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    planning_problem.initial_state.acceleration = 0
    # add ego object into scenario
    # ego shape
    vehicle3 = parameters_vehicle3.parameters_vehicle3()
    ego_vehicle_shape = Rectangle(length=vehicle3.l, width=vehicle3.w)

    ego_initial_state = planning_problem.initial_state
    # generate the states for the obstacle for time steps 1 to 40 by assuming constant velocity
    state_list = [ego_initial_state]
    new_state = copy.deepcopy(ego_initial_state)
    # action = 0
    for i in range(1, 39):
        action = np.random.choice([0, 1, 2])
        new_state = Actions(new_state, action, scenario.dt, sim=True).take_action()
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
    # print(ego_obstacle_id)
    ego_vehicle = DynamicObstacle(obstacle_id=ego_obstacle_id, obstacle_type=ego_vehicle_type,
                                  obstacle_shape=ego_vehicle_shape, initial_state=ego_initial_state,
                                  prediction=ego_vehicle_prediction)
    # print(ego_vehicle.obstacle_shape)
    # add ego vehicle to the scenario
    scenario.add_objects(ego_vehicle)

    MCTS = MonteCarloTreeSearchIRS(scenario, planning_problem, ego_obstacle_id)
    # print('------------------------------start MCTS computation--------------------------- ')
    # time.sleep(4)
    search_path, new_scenario = MCTS.execute_search()
    # ==== calculate a_l, c_l
    a_lon = [[s[0].acceleration, s[-1].acceleration] for s in search_path]
    print(a_lon)
    a_lon.pop(0)
    count = 1
    a_l = []
    c_l = []
    i = 1
    while i < len(a_lon):
        if a_lon[i] == a_lon[i - 1]:
            count += 1
            i += 1
        else:
            a_l.append(a_lon[i - 1])
            c_l.append(count)
            i += 1
            count = 1

        if i == len(a_lon) - 1:
            count += 1
            a_l.append(a_lon[i - 1])
            c_l.append(count)
    # ===
    print(a_l)
    print(c_l)
    for i, a in enumerate(a_l):
        if abs(a[1] - a[0]) <= 0.5:
           a_l[i][0] -= 0.25
           a_l[i][1] += 0.25
    # a_l = [[-3, -1], [-1, 1], [-3, -1], [-1, 1], [-6, -3], [-6, -3], [0.0, -3], [1, 7.5]]
    # c_l = [5, 5, 10, 5, 2, 1, 2, 10]
    print('------------------------------start reach set computation----------------------- ')
    # ==== build configuration
    config = ConfigurationBuilder.build_configuration(name_scenario)
    # ====
    # a_l: a_lon list
    # c_l: count of a_lon in each stage
    # a_l = [[-6, 7]]
    # c_l = [30]
    # config.vehicle.ego.a_lat_max = -1
    # config.vehicle.ego.a_lat_min = 1
    config.planning.steps_computation = np.sum(c_l)
    print(np.sum(c_l))
    config.reachable_set.mode_computation = 1
    config.update()
    reach_interface = ReachableSetInterface(config)
    # start_s = 0
    end_s = 0
    config.print_configuration_summary()
    last_step = False
    for a_lon, count in zip(a_l, c_l):
        start_s = end_s + 1
        print('start:')
        print(start_s)
        print('end:')
        end_s = start_s + count - 1
        print(end_s)
        print('a_lon:')
        print(a_lon)
        if end_s + 1 == config.planning.steps_computation:
            last_step = True

        reach_interface.compute_reachable_sets(step_start=start_s, step_end=end_s, a_lon=a_lon, last_step=last_step)
    config.debug.save_plots = 1
    fig_steps = np.arange(1, np.sum(c_l)).tolist()
    util_visual.plot_scenario_with_reachable_sets(reach_interface, steps=fig_steps, figsize=(7, 7))


if __name__ == "__main__":
    # warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
