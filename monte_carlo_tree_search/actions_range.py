import numpy as np


class ActionsRange:
    def __init__(self, pre_state, action_num, dt, sim=False):
        self.pre_state = pre_state
        self.dt = dt
        if action_num is None:
            self.action_num = 0
        else:
            self.action_num = action_num
        self.sim = sim

    # keep current velocity constant
    def take_action(self):
        if self.action_num == 0:
            up_state = self.action_1()
        elif self.action_num == 1:
            up_state = self.action_2()
        elif self.action_num == 2:
            up_state = self.action_3()
        elif self.action_num == 3:
            up_state = self.action_4()
        else:
            up_state = None

        return up_state

    # speed constant
    # the acceleration between [-1, 1]
    def action_1(self):
        if self.sim:
            update_state = self.acc_action(0, self.pre_state)
        else:
            if len(self.pre_state) == 1:
                update_state = [self.acc_action(-1, self.pre_state[0]), self.acc_action(0, self.pre_state[0]),
                                self.acc_action(1, self.pre_state[0])]
            else:
                update_state = [self.acc_action(-1, self.pre_state[0]), self.acc_action(0, self.pre_state[1]),
                                self.acc_action(1, self.pre_state[-1])]
        return update_state

    # acceleration
    # the acceleration between [1, 7.5]
    def action_2(self):
        # 1/2*ax*dt^2
        # max_acceleration/2
        if self.sim:
            update_state = self.acc_action(3.25, self.pre_state)
        else:
            if len(self.pre_state) == 1:
                update_state = [self.acc_action(1, self.pre_state[0]), self.acc_action(4.25, self.pre_state[0]),
                                self.acc_action(7.5, self.pre_state[0])]
            else:
                update_state = [self.acc_action(1, self.pre_state[0]), self.acc_action(4.25, self.pre_state[1]),
                                self.acc_action(7.5, self.pre_state[-1])]
        return update_state

    # deceleration
    # the acceleration between [-3, -1]
    def action_3(self):
        if self.sim:
            update_state = self.acc_action(-2, self.pre_state)
        else:
            if len(self.pre_state) == 1:
                update_state = [self.acc_action(-3, self.pre_state[0]), self.acc_action(-2, self.pre_state[0]),
                                self.acc_action(-1, self.pre_state[0])]
            else:
                update_state = [self.acc_action(-3, self.pre_state[0]), self.acc_action(-2, self.pre_state[1]),
                                self.acc_action(-1, self.pre_state[-1])]
        return update_state

    # braking
    # the acceleration between [-6, -3]
    def action_4(self):
        if self.sim:
            update_state = self.acc_action(-6, self.pre_state)
        else:
            if len(self.pre_state) == 1:
                update_state = [self.acc_action(-6, self.pre_state[0]), self.acc_action(-4.5, self.pre_state[0]),
                                self.acc_action(-3, self.pre_state[0])]
            else:
                update_state = [self.acc_action(-6, self.pre_state[0]), self.acc_action(-4.5, self.pre_state[1]),
                                self.acc_action(-3, self.pre_state[-1])]
        return update_state

    def acc_action(self, ax, state):
        velocity = state.velocity + ax * self.dt
        # print(velocity)
        if velocity <= 0:
            velocity = 0
            ax = - state.velocity / self.dt
        elif velocity > 25:
            velocity = 25
            ax = (25 - state.velocity) / self.dt
        r_angle = state.orientation
        # p_off = np.array(
        #    [self.state.velocity * np.cos(r_angle) * self.dt, self.state.velocity * np.sin(r_angle) * self.dt])
        p_off = np.array(
            [0, state.velocity * np.sin(r_angle) * self.dt + 0.5 * ax * np.sin(r_angle) * np.square(self.dt)])
        # rotation = self.state.orientation
        update_state = state.translate_rotate(p_off, 0)
        update_state.time_step = state.time_step + 1
        update_state.velocity = velocity
        update_state.acceleration = ax
        update_state.yaw_rate = 0
        update_state.slip_angle = 0
        return update_state
