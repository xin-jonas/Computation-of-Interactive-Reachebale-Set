import agent

import numpy as np
import copy
from numpy import random
from prediction.advanced_models.agent_sim.agent import clean_scenario, update_scenario
from monte_carlo_tree_search.mcts_node import MCTSNode
from idm import IDMAgent
from basefunctions_O import BaseFunctionO as BaseFunction
from actions import Actions
from monte_carlo_tree_search.config import Config
import time
from commonroad.scenario.trajectory import State
from commonroad_helper_functions.utils.cubicspline import CubicSpline2D


class MonteCarloTreeSearchV:
    """
    Class for action series choosing using the Single-Player Monte Carlo Tree Search.
    """

    def __init__(self, scenario, planning_problem, ego_id, config=None):
        """
        Constructor to set up the search of an action search, given by the vehicle
        id in the scenario.
        :param scenario: the CommonRoad scenario
        :param planning_problem: the specified planning problem in the scenario
        :param ego_id: the vehicle id to obtain the reference trajectory from the scenario.
        """
        self.scenario = scenario
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
        self.Base = BaseFunction(scenario, planning_problem, self.ego_vehicle, config)
        self.sim_scenario = copy.deepcopy(self.scenario)
        self.expan_scenario = copy.deepcopy(self.scenario)
        for ob in self.expan_scenario.dynamic_obstacles:
            if ob.obstacle_id is not self.ego_id:
                self.expan_scenario.remove_obstacle(ob)

        self.sim_scenario.remove_obstacle(self.ego_vehicle)

    def execute_search(self):
        """
        Performs the whole action selection. Coordinates the search for a sequence action until the
        final state is a terminal state.
        :return: the list of trajectory states
        """
        self.scenario.remove_obstacle(self.ego_vehicle)
        # monte carlo search start
        list_actions = list(np.arange(0, 4))
        root = MCTSNode(path=[self.ego_vehicle.initial_state], list_actions=list_actions,
                        depth_tree=0, parent=None)
        print('start MCTS')
        time_start = time.time()
        # repetition of the search for the next optimal action
        for i in range(int(self.time_goal/5) + 1):
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

            # the next optimal primitive is in a final state. The matching is finished.
            if self.is_terminal(node=root):
                print('finish simulation')
                time_end = time.time()
                print('time cost', time_end - time_start, 's')
                print('reached:' + repr(self.Base.is_reached(root.path)))
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
            print('samples ' + str(i))
            # node = self.selection(root)
            # if node is None:
            #     # selection failed
            #     return None
            leaf = self.tree_police(root)
            # reward = self.simulation(leaf)
            if not self.is_terminal(leaf):
                # print('-----------hier leaf not terminal----------')
                # print(leaf)
                reward = self.simulation(leaf)
            else:
                # print('-----------hier leaf terminal----------')
                reward = self.eval_terminal_reward(leaf)
            self.backpropagation(leaf, reward)
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
            if not children:
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

    def selection(self, root):
        # print('--------selection---------')
        cur_node = root
        children = cur_node.expanded_children
        max_uct = -float("inf")
        for child in children:
            uct = child.uct
            if uct >= max_uct:
                max_uct = uct
                cur_node = child
        # save the reward of the last primitive for the backpropagation
        # cur_node.last_action_reward = self.eval_action_reward(cur_node.path)
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
                # print('no child')
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
            # print(node.unexpanded_actions)
            action_successor = np.random.choice(node.unexpanded_actions)
            # action_successor = self.Base.random_choose_one_action(node.path[-1], node.unexpanded_actions)
            node.unexpanded_actions.remove(action_successor)
            # state translate
            path_translated = self.translate_action_to_current_state2(action_successor, path)
            # check collision
            # if self.Base.is_collision_free(path_translated, self.sim_scenario):
            #     break
            if path_translated:
                break
            # if self.Base.is_collision_free(path_translated, self.expan_scenario):
            #     break

        list_actions_current = copy.copy(node.list_actions)
        # list_primitives_current.append(action_successor)
        # add expanded action to the expanded_children in the node
        # print('len path translated')
        # print(len(path_translated))
        node.add_child(path_translated, list_actions_current, action=action_successor)
        node.expanded_children[-1].last_action_reward = self.eval_action_reward(path_translated)
        # print(node.expanded_children[-1])
        return node.expanded_children[-1]

    def simulation(self, node):
        """
        Performs the simulation step of the MCTS process following the random default policy.
        :param node: the node from where the simulation starts
        :return: the simulation reward
        """
        # print('----------simulation------------')
        cur_path = copy.deepcopy(node.path)
        sim_scenario = copy.deepcopy(self.scenario)
        sim_scenario = self.Base.add_ego_back_scenario(cur_path, self.ego_id, sim_scenario)
        effect_vehicle, direct_effect_vehicle, undirect_effect_vehicle, sim_scenario =\
            self.Base.filter_out_effected_obstacles(node.path[-5].time_step, sim_scenario=sim_scenario)
        agent_list =\
            self.Base.create_idm_agent_list(effect_vehicle, direct_effect_vehicle, undirect_effect_vehicle, sim_scenario)
        sim_scenario = clean_scenario(scenario=sim_scenario, agent_list=agent_list)

        for _ in range(5):
            for agent in agent_list:
                if agent.agent_id == self.ego_id:
                    agent.idm_parameters['action'] = node.take_action
                # print(agent.state)
                agent.step(scenario=sim_scenario)
            sim_scenario = update_scenario(scenario=sim_scenario, agent_list=agent_list)

        cur_path = sim_scenario.obstacle_by_id(self.ego_id).prediction.trajectory.state_list
        times = 3
        while not self.is_terminal(path=cur_path, sim_scenario=self.scenario) and times >= 0:
            # sim_scenario.remove_obstacle(self.ego_vehicle)
            # cur_action = np.random.choice(node.list_actions)
            direct_effect_vehicle_state = None
            if direct_effect_vehicle:
                 direct_effect_vehicle_state = sim_scenario.obstacle_by_id(direct_effect_vehicle).\
                    prediction.trajectory.state_list[cur_path[-1].time_step]

            cur_action = self.Base.random_choose_one_action(cur_path[-1], [0, 1, 2, 3], direct_effect_vehicle_state)
            for _ in range(0, 5):
                for agent in agent_list:
                    if agent.agent_id == self.ego_id:
                        agent.idm_parameters['action'] = cur_action
                    agent.step(scenario=sim_scenario)
                sim_scenario = update_scenario(scenario=sim_scenario, agent_list=agent_list)
            cur_path = sim_scenario.obstacle_by_id(self.ego_id).prediction.trajectory.state_list
            times -= 1
        # reward for applying the action
        simulation_reward = self.eval_action_reward(cur_path, sim_scenario)
        # compute the terminal reward for the simulation
        simulation_reward += self.eval_terminal_reward(path=cur_path, sim_scenario=sim_scenario)
        return simulation_reward

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
        :param sim_scenario:
        :return: the reward for the applied action
        """
        ego_state = path[-1]
        time_step = path[-1].time_step
        J_p = self.Base.cal_distance_to_obstacle(ego_state, time_step, sim_scenario=sim_scenario)
        J_path = self.Base.cal_diff_ref_ego_path(path)
        return J_path + J_p

    def eval_terminal_reward(self, node: MCTSNode = None, path=None, sim_scenario = None):
        """
        Evaluates the reward for a terminal state. The state is given either by a MCTSNode or by the path of the last
        applied primitive. The reward is computed by giving penalties for not reaching the time goal and for causing a
        collision.
        :param node: the terminal node, by default = None
        :param path: the path of the last applied primitive, by default = None
        :return: the reward for a terminal state
        """
        assert node is None or self.is_terminal(node)
        if node is not None:
            path = node.path
        reward = 0
        final_time = path[-1].time_step
        # time penalty
        if final_time < self.time_goal:
            reward += self.config.TIME_PENALTY * (self.time_goal - final_time) * self.time_goal
        # collision penalty
        if not self.Base.is_collision_free(path, sim_scenario=self.scenario):
            reward += self.config.COLLISION_PENALTY
        else:
            reward += 1000
        return reward

    def translate_action_to_current_state(self, action, path):
        idm_parameters: dict = {
            'v_0': 8,
            's_0': 10,
            'T': 1,
            'a_max': 7,
            'a_min': -6,
            'b': 1.5,
            'delta': 4,
            'label': 0,
            'action': action,
        }
        self.expan_scenario.obstacle_by_id(self.ego_id).initial_state = copy.deepcopy(path[-1])
        self.expan_scenario.obstacle_by_id(self.ego_id).initial_state.time_step = 0
        ego_agent = IDMAgent(
                scenario=self.expan_scenario,
                agent_id=self.ego_id,
                expected_lanelets_list=self.Base.route_lanelet,
                enable_logging=False,
                debug_step=False,
                idm_parameters=idm_parameters)
        expan_scenario = clean_scenario(scenario=self.expan_scenario, agent_list=[ego_agent])
        time_step_temp = path[-1].time_step + 1
        for i in range(5):
            ego_agent.step(expan_scenario)
            state = State(
                position=ego_agent.state.position,
                orientation=ego_agent.state.orientation,
                velocity=ego_agent.state.velocity,
                acceleration=ego_agent.state.acceleration,
                time_step=time_step_temp + i,
            )
            path.append(state)
        return path

    def translate_action_to_current_state2(self, action, path):
        state = path[-1]
        if action == 0:
            acceleration = 0
        elif action == 1:
            acceleration = 7
        elif action == 2:
            acceleration = -3
        else:
            acceleration = -6
        for _ in range(5):
            velocity = state.velocity + acceleration * self.scenario.dt
            if velocity < 0:
                acceleration = -state.velocity/self.scenario.dt

            # covered distance along the center line of the current lanelet
            ds = state.velocity * self.scenario.dt + 1 / 2 * acceleration * self.scenario.dt ** 2
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
                acceleration=acceleration,
                time_step=state.time_step + 1,
            )
            path.append(state)
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

    def is_terminal(self, node=None, path=None, collision_check=True, sim_scenario=None):
        """
        Returns whether a node or a path is terminal. Returns true if
        1. the time goal is reached by the final state,
        2. the node has no successor primitive or
        3. the primitive created a collision.
        :param node: by default = None
        :param path: by default = None
        :param collision_check: boolean if a collision check should be performed
        :param sim_scenario:
        :return: true if the node is terminal otherwise false
        """
        if path is None:
            path = node.path
        time = path[-1].time_step >= self.time_goal
        collision_free = True
        if collision_check and len(path) >= 2:
            if sim_scenario:
                collision_free = self.Base.is_collision_free(path, sim_scenario=sim_scenario)
            else:
                collision_free = self.Base.is_collision_free(path)
        return time or not collision_free # or no_children
