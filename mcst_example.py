import os
import numpy as np
import copy

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import ObstacleType
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction
from vehiclemodels import parameters_vehicle3

from commonroad_reach.data_structure.configuration_builder import ConfigurationBuilder
from monte_carlo_tree_search.actions import Actions
from monte_carlo_tree_search.mcts_idm_sim_v5 import MonteCarloTreeSearchV
import monte_carlo_tree_search.visualization as vs

# scenario_name = 'ZAM_Intersection-1_2_T-1.xml'
# file_path = os.path.abspath(
#     os.path.join(
#         os.path.abspath(__file__),
#         '../configurations/',
#         scenario_name,
#     )
# )
# name_scenario = "USA_US101-6_1_T-1"
name_scenario = "ZAM_Intersection-1_2_T-1"
# name_scenario = 'USA_US101-15_1_T-1'
# name_scenario = 'DEU_IV21-1_1_T-1'
# name_scenario = 'ZAM_Zip-1_6_T-1'
# name_scenario = 'DEU_Guetersloh-39_1_T-1' # !!
# name_scenario = 'DEU_Flensburg-61_1_T-1'
# name_scenario = 'DEU_Lohmar-13_1_T-1'
name_scenario = 'DEU_Lohmar-20_1_T-1'
# name_scenario = 'DEU_Lohmar-32_1_T-1'
# name_scenario = 'DEU_Moelln-7_1_T-1'
# name_scenario = 'ESP_Almansa-6_1_T-1'
# name_scenario = 'USA_US101-1_1_T-1'
# name_scenario = 'USA_US101-11_4_T-1'
# name_scenario = 'USA_US101-15_3_T-1'
file_path = '/home/xinzhang/MCTS/demo/IDM_demo/commonroad-motion-planning-library/scenario'
file_path = os.path.join(file_path, name_scenario + '.xml')

scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
print(str(scenario.scenario_id))
# vehicle_38 = scenario.obstacle_by_id(38) # 31
# scenario.remove_obstacle(vehicle_38)
planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
# planning_problem.initial_state.position = np.array([50.1054, -44.6537])
# planning_problem.initial_state.position = np.array([44.5, -17])
planning_problem.initial_state.acceleration = 0
# dynamic_obstacle = scenario.dynamic_obstacles
# for dynamic_ob in dynamic_obstacle:
#     if dynamic_ob.obstacle_id != 317:
#         scenario.remove_obstacle(dynamic_ob)
# change the velocity to velocity_y based on interaction scenario
# for pp_id, planning_problem in planning_problem_set.planning_problem_dict.items():
#     print(planning_problem.initial_state)
#     planning_problem.initial_state.position = np.array([44.5, -15])
#     # planning_problem.initial_state.position = np.array([44.5, -12])
#     # planning_problem.initial_state.position = np.array([44.5, -10])
#     # planning_problem.initial_state.position = np.array([44.5, -8])
#     planning_problem.initial_state.velocity = 7
#     planning_problem.initial_state.acceleration = 0
#     print(planning_problem.initial_state)
#     # print(planning_problem.goal.state_list)

# add ego object into scenario
# ego shape
vehicle3 = parameters_vehicle3.parameters_vehicle3()
ego_vehicle_shape = Rectangle(length=vehicle3.l, width=vehicle3.w)

ego_initial_state = planning_problem.initial_state
# generate the states for the obstacle for time steps 1 to 40 by assuming constant velocity
state_list = [ego_initial_state]
new_state = copy.deepcopy(ego_initial_state)
# action = 0
for i in range(1, 30):
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
# print(ego_obstacle_id)
ego_vehicle = DynamicObstacle(obstacle_id=ego_obstacle_id, obstacle_type=ego_vehicle_type,
                              obstacle_shape=ego_vehicle_shape, initial_state=ego_initial_state,
                              prediction=ego_vehicle_prediction)
# add ego vehicle to the scenario
scenario.add_objects(ego_vehicle)
# scenario_m = copy.deepcopy(scenario)
# scenario_m.add_objects(ego_vehicle)

MCTS = MonteCarloTreeSearchV(scenario, planning_problem, ego_obstacle_id)
search_path, new_scenario = MCTS.execute_search()
print('--------path length----')
print(len(search_path))

# path = np.array(search_path[1:]).T.tolist()[1]

# scenario = MCTS.scenario
# add ego object into scenario
# ego shape
vehicle3 = parameters_vehicle3.parameters_vehicle3()
ego_vehicle_shape = Rectangle(length=vehicle3.l, width=vehicle3.w)

# ego_initial_state = planning_problem.initial_state
# create the planned trajectory starting at time step 0
ego_vehicle_trajectory = Trajectory(initial_time_step=1, state_list=search_path[1:])
# create the prediction using the planned trajectory and the shape of the ego vehicle
ego_vehicle_prediction = TrajectoryPrediction(trajectory=ego_vehicle_trajectory,
                                              shape=ego_vehicle_shape)
# the ego vehicle can be visualized by converting it into a DynamicObstacle
ego_vehicle_type = ObstacleType.CAR
# generate the dynamic obstacle according to the specification
ego_vehicle = DynamicObstacle(obstacle_id=ego_obstacle_id, obstacle_type=ego_vehicle_type,
                              obstacle_shape=ego_vehicle_shape, initial_state=ego_initial_state,
                              prediction=ego_vehicle_prediction)
# print(ego_vehicle.obstacle_shape)
# add ego vehicle to the scenario
scenario.add_objects(ego_vehicle)
vs.plot_trajectory_velocity(scenario)
vs.plot_scenario(scenario, step_end=len(search_path))
