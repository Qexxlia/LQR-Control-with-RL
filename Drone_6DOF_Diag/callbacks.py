from typing import Any, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure, HParam, TensorBoardOutputFormat

matplotlib.use("Agg")


class StateCallback(BaseCallback):
    def __init__(self, args, plot_rate):
        super().__init__(0)

        self.map_limits = args.get("map_limits")
        self.max_duration = args.get("max_duration")
        self.t_step = args.get("t_step")
        self.plot_rate = plot_rate
        self.rollouts = 0

    def _on_rollout_end(self):
        # PLOT DATA
        if self.rollouts % self.plot_rate == 0:
            # Get environment
            vec_env = self.model.get_env()

            obs = vec_env.reset()

            # Initial Action
            [action, _state] = self.model.predict(obs, deterministic=True)

            # Record all actions taken
            action_record = action

            # Simulate
            done = False
            donecount = 0
            while not done:
                [obs, reward, done, info] = vec_env.step(action)
                [action, _state] = self.model.predict(obs, deterministic=True)

                action_record = np.vstack((action_record, action))

                if info[-1].get("t", [0])[-1] > self.max_duration:
                    done = True
                if info[-1].get("failed", False):
                    donecount += 1
                if donecount >= 10:
                    return True

            # actions = np.zeros((np.shape(action_record)[0]))
            # for i in range(0, np.shape(action_record)[0]):
            #     actions.append(vec_env.env_method("map_action", action))

            # Extract data
            position = info[-1].get("position", np.zeros(1))
            attitude = info[-1].get("attitude", np.zeros(1))
            velocity = info[-1].get("velocity", np.zeros(1))
            spin = info[-1].get("spin", np.zeros(1))
            t = info[-1].get("t", np.zeros(1))
            u = info[-1].get("u", np.zeros(1))
            reward = reward[-1]
            # t_step = self.t_step

            # Action time steps
            # stepped_t = np.array([0, t_step])
            # while np.shape(stepped_t)[0] < np.shape(q_weights)[0]:
            #     stepped_t = np.append(stepped_t, stepped_t[-1] + t_step)

            # Time arrays for plotting
            # q_t = np.vstack([stepped_t] * 144)
            # r_t = np.vstack([stepped_t] * 16)
            t3 = np.vstack([t] * 3)
            t4 = np.vstack([t] * 4)

            # Font sizes
            tsL = 12
            tsS = 7

            def create_plot(fig_num, title, xlabel, ylabel, x_data, y_data, labels):
                fig = plt.figure(fig_num, figsize=(10, 2))
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(x_data, y_data, label=labels)
                ax.set_title(title, fontsize=tsL)
                ax.set_xlabel(xlabel, fontsize=tsS)
                ax.set_ylabel(ylabel, fontsize=tsS)
                if labels is not None:
                    fig.legend()
                return fig

            # Q Weights
            # fig0 = create_plot(
            #     0, "Q Weights", "Time (s)", "Weight", q_t, q_weights.transpose(), None
            # )
            #
            # # R Weights
            # fig1 = create_plot(
            #     1, "R Weights", "Time (s)", "Weight", r_t, r_weights.transpose(), None
            # )

            # Voltage
            fig2 = create_plot(
                2,
                "Control",
                "Time (s)",
                "Control",
                t4.T,
                u.T,
                ["U1", "U2", "U3", "U4"],
            )

            # Position
            fig3 = create_plot(
                3,
                "Position",
                "Time (s)",
                "Distance (m)",
                t3.T,
                position.T,
                ["x", "y", "z"],
            )

            # Attitude
            fig4 = create_plot(
                4,
                "Attitude",
                "Time (s)",
                "Angle (rad)",
                t3.T,
                attitude.T,
                ["roll", "pitch", "yaw"],
            )

            # Velocity
            fig5 = create_plot(
                5,
                "Velocity",
                "Time (s)",
                "Velocity (m/s)",
                t3.T,
                velocity.T,
                ["x", "y", "z"],
            )

            # Spin
            fig6 = create_plot(
                6,
                "Spin",
                "Time (s)",
                "Spin (rad/s)",
                t3.T,
                spin.T,
                ["roll", "pitch", "yaw"],
            )
            # Log plots to tensorboard
            # self.logger.record(
            #     "plots/action/q_weights",
            #     Figure(fig0, close=True),
            #     exclude=("stdout", "log", "json", "csv"),
            # )
            # self.logger.record(
            #     "plots/action/r_weights",
            #     Figure(fig1, close=True),
            #     exclude=("stdout", "log", "json", "csv"),
            # )
            self.logger.record(
                "plots/state/control",
                Figure(fig2, close=True),
                exclude=("stdout", "log", "json", "csv"),
            )
            self.logger.record(
                "plots/state/position",
                Figure(fig3, close=True),
                exclude=("stdout", "log", "json", "csv"),
            )
            self.logger.record(
                "plots/state/attitude",
                Figure(fig4, close=True),
                exclude=("stdout", "log", "json", "csv"),
            )
            self.logger.record(
                "plots/state/velocity",
                Figure(fig5, close=True),
                exclude=("stdout", "log", "json", "csv"),
            )
            self.logger.record(
                "plots/state/spin",
                Figure(fig6, close=True),
                exclude=("stdout", "log", "json", "csv"),
            )

            # LOG DATA
            self.logger.record(
                "time/ep_time_elapsed", t[-1], exclude=("stdout", "log", "json", "csv")
            )
            self.logger.record(
                "reward/reward", reward, exclude=("stdout", "log", "json", "csv")
            )

        self.rollouts += 1
        return True

    def _on_step(self):
        return True


class HParamCallback(BaseCallback):
    def on_training_start(
        self, locals_: Dict[str, Any], globals_: Dict[str, Any]
    ) -> None:
        self.env = self.model.get_env()

        # Parameters to log
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "n steps": self.model.n_steps,
            "n epochs": self.model.n_epochs,
            "batch size": self.model.batch_size,
            "gamma": self.model.gamma,
            "entropy coefficient": self.model.ent_coef,
            "value function coefficient": self.model.vf_coef,
            "max gradient norm": self.model.max_grad_norm,
            "initial std (log)": self.model.policy_kwargs.get("log_std_init"),
            "max duration": self.env.get_attr("max_duration")[0],
            "map limits": np.array2string(self.env.get_attr("map_limits")[0]),
            "time step": self.env.get_attr("t_step")[0],
            "initial state": np.array2string(self.env.get_attr("initial_state")[0]),
            "seed": self.env.get_attr("seed")[0],
        }

        # Metrics to log
        metric_dict = {
            "train/clip_range": 0,
        }

        # Log hparams and metrics
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


class TextDataCallback(BaseCallback):
    def on_training_start(
        self, locals_: Dict[str, Any], globals_: Dict[str, Any]
    ) -> None:

        # Get TB writer
        output_formats = self.logger.output_formats
        self.tb_writer = next(
            formatter
            for formatter in output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        ).writer

        # Get environment
        self.env = self.model.get_env()

        np.printoptions(linewidth=1000)

        # Log Hparams in Text
        hparam_text = (
            "Algorithm: "
            + str(self.model.__class__.__name__)
            + "<br>"
            + "Learning Rate: "
            + str(self.model.learning_rate)
            + "<br>"
            + "N Steps: "
            + str(self.model.n_steps)
            + "<br>"
            + "N Epochs: "
            + str(self.model.n_epochs)
            + "<br>"
            + "Batch Size: "
            + str(self.model.batch_size)
            + "<br>"
            + "Gamma: "
            + str(self.model.gamma)
            + "<br>"
            + "Entropy Coefficient: "
            + str(self.model.ent_coef)
            + "<br>"
            + "Value Function Coefficient: "
            + str(self.model.vf_coef)
            + "<br>"
            + "Max Gradient Norm: "
            + str(self.model.max_grad_norm)
            + "<br>"
            + "Initial Std (log): "
            + str(self.model.policy_kwargs.get("log_std_init"))
            + "<br>"
        )

        self.tb_writer.add_text("hparams", hparam_text)

        # Log environment parameters
        env_text = (
            "Max Duration: "
            + str(self.env.get_attr("max_duration")[0])
            + "<br>"
            + "Map Limits: "
            + np.array2string(self.env.get_attr("map_limits")[0])
            + "<br>"
            + "Time Step: "
            + str(self.env.get_attr("t_step")[0])
            + "<br>"
            + "Initial State"
            + np.array2string(self.env.get_attr("initial_state")[0])
            + "<br>"
            "Max Position: "
            + str(self.env.get_attr("max_position")[0])
            + "<br>"
            + "Max Attitude: "
            + str(self.env.get_attr("max_attitude")[0])
            + "<br>"
            + "Max Velocity: "
            + str(self.env.get_attr("max_velocity")[0])
            + "<br>"
            + "Max Spin: "
            + str(self.env.get_attr("max_spin")[0])
            + "<br>"
            + "Seed: "
            + str(self.env.get_attr("seed")[0])
            + "<br>"
        )

        self.tb_writer.add_text("environment", env_text)

        # Log Reward Function
        def extract_data(filename):
            lines = []
            file = open(filename, "rt")
            for line in file:
                lines.append(line)
            file.close()

            printing = False
            output = ""

            for line in lines:
                if line.find("# $$") != -1:
                    printing = not printing

                if printing:
                    output = output + line

            output = output.replace("$$", "")
            output = output.replace("\n", "<br>")
            return output

        self.tb_writer.add_text("reward", extract_data("./drone_env.py"))

        self.tb_writer.flush()

    def _on_step(self) -> bool:
        return True


class UMaxCallback(BaseCallback):
    def __init__(self, args):
        super().__init__(0)
        self.initial_u_max = args.get("u_max_initial")
        self.final_u_max = args.get("u_max_final")
        self.step_gap = args.get("step_gap")
        self.i = 0
        self.total_timesteps = args.get("total_timesteps")
        self.u_max_list = np.linspace(
            self.initial_u_max,
            self.final_u_max,
            (int)(self.total_timesteps / self.step_gap),
        )
        self.env = None

    def _on_step(self) -> bool:
        if self.num_timesteps % self.step_gap == 0:
            self.env = self.training_env
            self.env.env_method("update_u_max", self.u_max_list[self.i])
            self.i += 1

        self.logger.record(
            "other/u_max",
            self.u_max_list[self.i],
            exclude=("stdout", "log", "json", "csv"),
        )
        return True
