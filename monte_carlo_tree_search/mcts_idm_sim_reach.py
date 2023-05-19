import numpy as np
import copy
from numpy import random

from prediction.advanced_models.agent_sim.agent import clean_scenario, update_scenario
from monte_carlo_tree_search.mcts_node import MCTSNode
from monte_carlo_tree_search.basefunctions import BaseFunction
from actions_range import ActionsRange as Actions
from config import Config
import time
from commonroad.scenario.trajectory import State
from commonroad_helper_functions.utils.cubicspline import CubicSpline2D


class MonteCarloTreeSearchIRS:
    """
    Class for action series choosing using the Single-Player Monte Carlo Tree Search.
    """

    def __init__(self, scenario, planning_problem, ego_id):
        """
        Constructor to set up the search of an action search, given by the vehicle
        id in the scenario.
        :param scenario: the CommonRoad scenario
        :param planning_problem: the specified planning problem in the scenario
        :param ego_id: the vehicle id to obtain the reference trajectory from the scenario.
        """
        self.scenario = scenario
        self.sim_scenario = scenario
        np.random.seed(0)
        self.config = Config()
        MCTSNode.node_count = 0
        self.ego_id = ego_id
        # remove the ego vehicle from the scenario, so that the trajectory is excluded by the collision checker.
        self.ego_vehicle = scenario.obstacle_by_id(self.ego_id)
        # scenario.remove_obstacle(self.ego_vehicle)

        # following lines necessary to avoid an exception in SearchBaseClass
        self.ego_vehicle.initial_state.yaw_rate = 0
        self.ego_vehicle.initial_state.slip_angle = 0

        # get the base initial of ego
        planning_problem.initial_state = self.ego_vehicle.initial_state
        self.trajectory = self.ego_vehicle.prediction.trajectory
        self.time_goal = len(self.trajectory.state_list)

        # BaseFunction instance
        self.Base = BaseFunction(scenario, planning_problem, self.ego_vehicle)
        self.scenario.remove_obstacle(self.ego_vehicle)

    def execute_search(self):
        """
        Performs the whole action selection. Coordinates the search for a sequence action until the
        final state is a terminal state.
        :return: the list of trajectory states
        """
        # monte carlo search start
        list_actions = list(np.arange(-2, 3))
        root = MCTSNode(path=[[self.ego_vehicle.initial_state]], list_actions=list_actions,
                        depth_tree=0, parent=None)
        print('start simulation')
        time_start = time.time()
        # repetition of the search for the next optimal action
        for i in range(self.time_goal):
            print('step ' + str(i))
            samples = self.config.SAMPLES
            new_root = self.one_action(root, samples)
            print("Choose node: {}".format(new_root))
            if not new_root:
                print("!!!!!!!break!!!!!!")
                # no next optimal action found (root node has no unexpanded action due to immediate collisions)
                break

            if self.config.TREE_REUSE:
                root = copy.deepcopy(new_root)
                # root.path = new_root.path
                root.parent = None
            else:
                root = MCTSNode(path=new_root.path, list_actions=new_root.list_actions,
                                depth_tree=new_root.depth_tree, parent=root, id=0)

            # the next optimal node is terminal, finish simulation
            if self.is_terminal(node=root):
                print('finish simulation')
                time_end = time.time()
                print('time cost', time_end - time_start, 's')
                break

        return root.path, self.scenario

    def one_action(self, root, samples):
        """
        Executes the MCTS for the next optimal action. Repeats the four main MCTS steps for a given number of
        samples.
        :param root: The root node, from where the search starts. compute budget
        :param samples: The number of repetitions of the four MCTS steps.
        :return: The next optimal action.
        """
        for i in range(samples):
            # print('samples ' + str(i))
            # selection and expansion
            leaf = self.tree_police(root)
            # reward = self.simulation(leaf)diff_lat
            # only if the leaf node is non-terminal node, run simulation, otherwise calculate the terminal reward
            if not self.is_terminal(leaf):
                # print('-----------hier leaf not terminal----------')
                # print(leaf)
                reward = self.simulation(leaf)
            else:
                # print('-----------hier leaf terminal----------')
                reward = self.eval_terminal_reward(leaf)
            # backpropagation
            self.backpropagation(leaf, reward)

        # select the best child node
        if root.no_children:
            return None
        else:
            next_optimal_action = self.select_next_action(root)
        return next_optimal_action

    def tree_police(self, node):
        """
        while node is not terminal:
            if node has no child:
                expansion
            elif random.uniform(0,1)<.5:
                node = selection with UCT (not always expanded)
            else:
                if node is fully expanded:
                    node = selection with UCT
                else:
                    expansion
        """
        # print('--------------tree_police----------')
        # cur_node = node
        while not self.is_terminal(node, collision_check=False):
            children = node.expanded_children
            if len(children) == 0:
                # print("------------no child----------")
                cur_node = self.expansion(node)
                return cur_node
            elif random.uniform(0, 1) < 0.3:  # node = selection with UCT (not always expanded)
                # print("------------not always expand----------")
                node = self.selection(node)
            else:
                if node.fully_expanded():
                    # print("------------fully expanded so selection----------")
                    node = self.selection(node)
                else:
                    # print("------------expand----------")
                    cur_node = self.expansion(node)
                    return cur_node
        return node

    @staticmethod
    def selection(root):
        # print('--------selection---------')
        cur_node = root
        children = cur_node.expanded_children
        max_uct = -float("inf")
        for child in children:
            uct = child.uct
            if uct >= max_uct:
                max_uct = uct
                cur_node = child
        return cur_node

    def expansion(self, node):
        """
        if node has unexpanded maneuver:
            expansion with some condition within unexpanded maneuver
        elif the node is full expanded
            return random choice a node from child node list
        elif the node satisfied the terminal condition
            return current node
        """
        path = copy.deepcopy(node.path)
        # print('--------expansion---------')
        if self.is_terminal(node):
            # print('####terminal###')
            return node
        if len(node.unexpanded_actions) == 0:
            # print('------full expansion------')
            # even though all unexpanded_children are expanded, the node is not marked as fully expanded due to the
            # visit threshold, returns randomly an expanded child that has less than VISIT_THRESHOLD visits
            if node.expanded_children:
                visits = np.array([c.visits < self.config.VISIT_THRESHOLD for c in node.expanded_children])
                sub_node = np.random.choice(node.expanded_children, p=visits / sum(visits))
            else:
                print('no child')
                return node
            return sub_node
        # times = 1
        # unexpanded_actions = copy.deepcopy(node.unexpanded_actions)
        while True:
            # times += 1
            if not node.unexpanded_actions:
                # print('no unexpanded_actions')
                if node.expanded_children:
                    visits = np.array([c.visits < self.config.VISIT_THRESHOLD for c in node.expanded_children])
                    sub_node = np.random.choice(node.expanded_children, p=visits / sum(visits))
                else:
                    # print('no child')
                    return node
                return sub_node

            # print('actions len:' + str(len(node.unexpanded_actions)))
            action_successor = np.random.choice(node.unexpanded_actions)
            node.unexpanded_actions.remove(action_successor)
            # state simulation
            path_translated = self.translate_action_to_current_state2(action_successor, path)

            if path_translated is not None:
                break
            # # check collision
            # collision_free = 0
            # path = np.array(path_translated[1:]).T.tolist()
            # for i in range(0, len(path)):
            #     collision_free += self.Base.is_collision_free(path[i])
            # if collision_free > 1:
            #     print('action_successor:' + str(action_successor))
            #     break

        list_actions_current = copy.copy(node.list_actions)
        # add expanded action to the expanded_children in the node
        # print('len path translated')
        # print(len(path_translated))
        node.add_child(path_translated, list_actions_current, action=action_successor)
        path_list = self.Base.sample_path(path_translated[-6], 5, action_successor)
        node.expanded_children[-1].last_action_reward = self.eval_action_reward(path_list)
        return node.expanded_children[-1]

    def simulation(self, node):
        """
        Performs the simulation step of the MCTS process following the random default policy.
        :param node: the node from where the simulation starts
        :return: the simulation reward
        """
        # print('----------simulation------------')
        cur_path = copy.deepcopy(node.path)
        # path = np.array(cur_path[1:]).T.tolist()[1]
        # state_list before expansion, num of sample path, take action
        path_list = self.Base.sample_path(cur_path[-6], 5, node.take_action)
        # print(path_list)
        sim_result = {
            'path_id': 0,
            'is_terminal': 0,
            'simulation_reward': 0,
            'terminal_reward': 0,
            'sum_reward': 0,
        }
        path_sim_result = []
        for i, path_sample in enumerate(path_list):
            sim_result['path_id'] = i
            sim_scenario = copy.deepcopy(self.scenario)
            sim_scenario = self.Base.add_ego_back_scenario(path_sample, self.ego_id, sim_scenario)
            effect_vehicle, direct_effect_vehicle, undirect_effect_vehicle, sim_scenario = \
                self.Base.filter_out_effected_obstacles(path_sample[0].time_step, sim_scenario=sim_scenario)
            agent_list = \
                self.Base.create_idm_agent_list(effect_vehicle, direct_effect_vehicle, undirect_effect_vehicle,
                                                sim_scenario)
            sim_scenario = clean_scenario(scenario=sim_scenario, agent_list=agent_list)

            for j in range(5):
                for agent in agent_list:
                    if agent.agent_id == self.ego_id:
                        agent.step_agent_without_idm(sim_scenario=sim_scenario, state=path_sample[j])
                    else:
                        agent.step(scenario=sim_scenario)
                sim_scenario = update_scenario(scenario=sim_scenario, agent_list=agent_list)

            cur_path = sim_scenario.obstacle_by_id(self.ego_id).prediction.trajectory.state_list
            times = 3
            time_rest = self.time_goal - node.path[-1][0].time_step
            is_terminal = self.is_terminal(path=cur_path, sim_scenario=sim_scenario,\
                                           collision_check=True, sim=True, time_rest=time_rest)
            while is_terminal != 0 and times >= 0:
                # sim_scenario.remove_obstacle(self.ego_vehicle)
                # cur_action = np.random.choice(node.list_actions)
                direct_effect_vehicle_state = None
                if direct_effect_vehicle is not None:
                    direct_effect_vehicle_state = sim_scenario.obstacle_by_id(direct_effect_vehicle). \
                        prediction.trajectory.state_list[cur_path[-1].time_step]

                cur_action = self.Base.random_choose_one_action(cur_path[-1], [-2, -1, 0, 1, 2], direct_effect_vehicle_state)[0]
                # cur_action = np.random.choice(list(np.arange(-2, 3)))

                for _ in range(0, 5):
                    for agent in agent_list:
                        if agent.agent_id == self.ego_id:
                            agent.idm_parameters['action'] = cur_action
                        agent.step(scenario=sim_scenario)
                    sim_scenario = update_scenario(scenario=sim_scenario, agent_list=agent_list)
                cur_path = sim_scenario.obstacle_by_id(self.ego_id).prediction.trajectory.state_list
                is_terminal = self.is_terminal(path=cur_path, sim_scenario=sim_scenario, collision_check=True,
                                               sim=True, time_rest=time_rest)
                times -= 1
            sim_result['is_terminal'] = is_terminal
            # reward for applying the action
            simulation_reward = self.eval_action_reward(cur_path, sim_scenario)
            sim_result['simulation_reward'] = simulation_reward
            # compute the terminal reward for the simulation
            terminal_reward = self.eval_terminal_reward(path=cur_path,\
                                                        is_terminal=is_terminal, sim_scenario=sim_scenario)
            sim_result['terminal_reward'] = terminal_reward
            sim_result['sum_reward'] = terminal_reward + simulation_reward
            path_sim_result.append(sim_result)

        # return path_sim_result[0]['sum_reward']
        a = self.Base.calculate_reward_sim(path_sim_result)
        return a

    def backpropagation(self, cur_node, reward):
        """
        Performs the backpropagation step of the MCTS process.
        :param cur_node: the expanded child, the backpropagation sta
        :param reward: the reward of the simulation, to be backed-up.
        """
        # logging.log(18, "Backpropagation")
        # print("------------Backpropagation----------")
        # cur_node = node
        while cur_node:
            reward = cur_node.update(reward)
            # print(cur_node)
            cur_node = cur_node.parent

    def eval_action_reward(self, path, sim_scenario=None):
        """
        Evaluates the reward for an applied primitive by giving a penalty for the displacement error of the
        primitive to the corresponding reference trajectory segment.
        :param path: the list states after applied action
        :param sim_scenario: scenario after simulation
        :return: the reward for the applied action
        """
        # ego state in last five time step
        if sim_scenario is None:
            j_p = 0
            j_path = 0
            for p in path:
                ego_state = p[-1]
                time_step = p[-1].time_step
                j_p += self.Base.cal_distance_to_obstacle(ego_state, time_step)
                j_path += self.Base.cal_diff_ref_ego_path(p)
            J_p = j_p/len(path)
            J_path = j_path/len(path)
        else:
            ego_state = path[-1]
            time_step = path[-1].time_step
            J_p = self.Base.cal_distance_to_obstacle(ego_state, time_step, sim_scenario=sim_scenario)
            # print(J_p)
            # normalize the difference distance
            J_path = self.Base.cal_diff_ref_ego_path(path)
        # print(J_path)
        return J_p + J_path

    def eval_terminal_reward(self, node: MCTSNode = None, path=None, is_terminal=None, sim_scenario=None):
        """
        Evaluates the reward for a terminal state. The state is given either by a MCTSNode or by the path of the last
        applied primitive. The reward is computed by giving penalties for not reaching the time goal and for causing a
        collision.
        :param node: the terminal node, by default = None
        :param path: the path of the last applied primitive, by default = None
        :param sim_scenario
        :return: the reward for a terminal state
        """
        # assert node is None or self.is_terminal(node)
        if node is not None:
            path = node.path
            final_time = path[-1][0].time_step
        else:
            final_time = path[-1].time_step
        reward = 0
        if is_terminal is not None and is_terminal == 1:
            reward += self.config.COLLISION_PENALTY
        elif is_terminal is None:
            collision_free = 0
            path_list = self.Base.sample_path(path[-6], 5, node.take_action)
            for i in range(0, len(path_list)):
                collision_free += self.Base.is_collision_free(path_list[i])
            if collision_free >= 3:
                reward += self.config.COLLISION_PENALTY

        # time penalty
        if final_time < self.time_goal:
            reward += self.config.TIME_PENALTY * (self.time_goal - final_time) * self.time_goal
        return reward

    def translate_action_to_current_state2(self, action, path):
        state_groupe = path[-1]
        if len(state_groupe) == 1:
            state_groupe = state_groupe*2

        # if action == 0:
        #     acceleration = [-1, 0, 1]
        # elif action == 1:
        #     acceleration = [1, 4, 7]
        # elif action == 2:
        #     acceleration = [-3, -2, -1]
        # else:
        #     acceleration = [-6, -4.5, -3]

        # if action == 0:
        #     acceleration = [-1, 0, 1]
        # elif action == 1:
        #     acceleration = [1, 2.5, 4]
        # elif action == 2:
        #     acceleration = [4, 5.5, 7]
        # elif action == -1:
        #     acceleration = [-3, -2, -1]
        # elif action == -2:
        #     acceleration = [-6, -4.5, -3]
        acceleration = self.Base.action_system(action)

        for _ in range(1, 6):
            state_list = []
            for i, state in enumerate(state_groupe):
                acc = acceleration[i]
                velocity = state.velocity + acc*self.scenario.dt
                if velocity < 0:
                    acc = -state.velocity/self.scenario.dt
                    if acc > max(acceleration):
                        path = None
                        return path
                    velocity = 0
                # covered distance along the center line of the current lanelet
                ds = state.velocity * self.scenario.dt + 1 / 2 * acc * self.scenario.dt ** 2
                center_points = np.array(self.Base.ref_CLCS.reference_path())
                try:
                    ego_lanelet_spline = CubicSpline2D(center_points[:, 0], center_points[:, 1])
                except ValueError:
                    center_points = np.unique(center_points, axis=0)
                    ego_lanelet_spline = CubicSpline2D(center_points[:, 0], center_points[:, 1])

                # calculate the new position (arc length) travelled along the spline
                s_new = ego_lanelet_spline.get_min_arc_length(state.position)[0] + ds

                # new position
                x, y = ego_lanelet_spline.calc_position(s_new)
                position = np.array([x, y])

                # new orientation
                orientation = ego_lanelet_spline.calc_yaw(s_new)
                state = State(
                    position=position,
                    orientation=orientation,
                    velocity=velocity,
                    acceleration=acc,
                    time_step=state.time_step + 1,
                )
                state_list.append(state)

            path.append(state_list)
            state_groupe = copy.copy(state_list)
        return path

    @staticmethod
    def translate_action_to_current_state(action, path, sim=False):
        for _ in range(0, 5):
            if sim:
                next_state = Actions(path[-1], action, 0.1, sim).take_action()
            else:
                next_state = Actions(path[-1], action, 0.1).take_action()
                # print(next_state)
            path.append(next_state)
        return path

    @staticmethod
    def select_next_action(root):
        """
        Returns the next optimal primitive according to the average rewards.
        :param root: the root node
        :return: the next optimal primitive
        """
        children = root.expanded_children
        visits = np.array([c.visits for c in children])
        rewards = np.array([c.reward for c in children])
        avg_rewards = rewards / visits
        return children[np.argmax(avg_rewards)]

    def is_terminal(self, node=None, path=None, collision_check=True, sim_scenario=None, sim=False, time_rest=None ):
        """
        Returns whether a node or a path is terminal. Returns true if
        1. the time goal is reached by the final state,
        2. the node has no successor primitive or
        3. the primitive created a collision.
        :param node: by default = None
        :param path: by default = None
        :param collision_check: boolean if a collision check should be performed
        :param sim_scenario
        :param sim
        :return: true if the node is terminal otherwise false
        """
        if path is None:
            path = node.path

        # is_terminal = None
        # collision_free = True
        if collision_check and sim:
            collision_free = self.Base.is_collision_free(path, sim_scenario=sim_scenario)
            if not collision_free:
                # collision happened
                is_terminal = 1
                return is_terminal

        if time_rest is not None and time_rest <= 0 and sim:
            # goal time is reached
            is_terminal = 2
            return is_terminal
        else:
            time = path[-1].time_step >= self.time_goal if sim else path[-1][0].time_step >= self.time_goal
            if time:
                is_terminal = 2 if sim else True
                return is_terminal

        if len(path) > 2:
            if sim:
                reached = self.Base.is_reached(path)
            else:
                reached = 0
                path_list = self.Base.sample_path(path[-6], 5, node.take_action)
                for path in path_list:
                    reached += self.Base.is_reached(path)
                reached = True if reached >= 3 else False
            if reached:
                is_terminal = 0 if sim else True
                return is_terminal
            else:
                is_terminal = 3 if sim else False
                return is_terminal
        # return time or reached and collision_free

    #
    #
    # def choose_one_state(self, cur_path):
    #     # start_position = (cur_path[-1].position - cur_path[0].position)/2 + cur_path[-1].position
    #     index = np.random.choice([0, 1])
    #     return cur_path[-1][index]
