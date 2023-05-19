from commonroad.scenario.trajectory import State
import numpy as np


class Actions:
    def __init__(self, state, action_num, dt):
        self.state = state
        self.dt = dt
        if action_num is None:
            self.action_num = 0
        else:
            self.action_num = action_num

    # keep current velocity constant
    def take_action(self, scenario=None):
        if scenario:
            if (
                    len(
                        scenario.lanelet_network.find_lanelet_by_position(
                            [self.state.position]
                        )[0]
                    )
                    > 0
            ):
                lanelet_id_lst = (
                    scenario.lanelet_network.find_lanelet_by_position(
                        [self.state.position]
                    )[0]
                )
                lanelettemp = self.scenario.lanelet_network.find_lanelet_by_id(
                    lanelet_id_lst[0]
                )
                ego_lanelet_spline = lanelet2spline(
                    lanelet=lanelettemp
                )
            else:
                pass


        if self.action_num == 0:
            up_state = self.action_1()
        elif self.action_num == 1:
            up_state = self.action_2()
        elif self.action_num == 2:
            up_state = self.action_3()
        elif self.action_num == 3:
            up_state = self.action_4()
        elif self.action_num == 4:
            up_state = self.action_5()
        elif self.action_num == 5:
            up_state = self.action_6()

        # elif self.action_num == 6:
        #     update_state = self.action_7()
        return up_state

    # speed constant
    def action_1(self):
        r_angle = self.state.orientation
        p_off = np.array([self.state.velocity*np.cos(r_angle)*self.dt, self.state.velocity*np.sin(r_angle)*self.dt])
        # print("hier")
        # print(p_off)
        # rotation = self.state.orientation
        update_state = self.state.translate_rotate(p_off, 0)
        update_state.time_step = self.state.time_step + 1
        update_state.yaw_rate = 0
        update_state.slip_angle = 0
        return update_state

    # acceleration
    def action_2(self):
        # 1/2*ax*dt^2
        # max_accelearion/2
        ax = 7.5
        update_state = self.acc_action(ax)
        return update_state

    # deceleration
    def action_3(self):
        # min_accelearion/2
        ax = -3
        update_state = self.acc_action(ax)
        return update_state

    # braking
    def action_4(self):
        ax = -6
        update_state = self.acc_action(ax)
        return update_state

    # lane change left
    def action_5(self):
        vy = 4
        update_state = self.lc_action(vy)
        return update_state

    # lane change right
    def action_6(self):
        # set bias acceleration with constant acceleration
        vy = -4
        update_state = self.lc_action(vy)
        return update_state

    """"
        # following the lead vehicle
        def action_7(self, bwd: Moveable, fwd: Moveable, dt):
            # ROUND_VEL = 1e-2
            if fwd == None:
                acc = min(MAIN_BSAVE, (SPEED_LIMIT_KMH - self.vel[0]) / (3 * dt))
                # assume 3 time intervals get to desired velocity
            else:
                v1, v2 = bwd.vel[0], fwd.vel[0]
                pos1, pos2 = bwd.pos[0], fwd.pos[0]

                if np.isclose(v1, v2, atol=self.ROUND_VEL):
                    assert pos1 < pos2  # make sure fwd is really forward
                    acc = (
                            v2 - v1 + fwd.acc[0]
                    )  # synchronize the speed, then the acc is the same
                    if bwd.distance_to(fwd) > self.model.s0:
                        acc = max(self.acc[0], acc + 0.2)  # can't exceed the maximum value
                acc = self.model.calc_acc(bwd, fwd)

            self.acc = np.array([acc, 0])
            self.accelerate(dt)  # update velocity
            self.translate(dt)  # update position
            return a
    """
    def acc_action(self, ax):
        velocity = self.state.velocity + ax * self.dt
        if velocity < 0:
            velocity = 0
            ax = - self.state.velocity / self.dt
        elif velocity > 15:
            velocity = 15
            ax = (15 - self.state.velocity) / self.dt
        r_angle = self.state.orientation
        # p_off = np.array(
        #    [self.state.velocity * np.cos(r_angle) * self.dt, self.state.velocity * np.sin(r_angle) * self.dt])
        p_off = np.array([self.state.velocity*np.cos(r_angle) * self.dt + 0.5 * ax*np.cos(r_angle)* np.square(self.dt),
                          self.state.velocity*np.sin(r_angle) * self.dt + 0.5 * ax*np.sin(r_angle)* np.square(self.dt)])
        # rotation = self.state.orientation
        update_state = self.state.translate_rotate(p_off, 0)
        update_state.time_step = self.state.time_step + 1
        update_state.velocity = velocity
        update_state.acceleration = ax
        update_state.yaw_rate = 0
        update_state.slip_angle = 0
        return update_state

    def lc_action(self, vy):
        p_off = np.array([self.state.velocity * self.dt, vy * self.dt])
        # set bias acceleration with constant acceleration
        # update_state = self.state.translate_rotate(p_off, 0)
        r_angle = np.arctan(vy/self.state.velocity)
        # print(r_angle)
        update_state = self.state.translate_rotate(p_off, 0)
        # update_state = update_state.translate_rotate(np.zeros(2), r_angle)
        update_state.time_step = self.state.time_step + 1
        # update_state.velocity = self.state.velocity + ax * self.dt
        update_state.orientation = r_angle
        update_state.velocity_y = vy
        update_state.acceleration = 0
        return update_state


