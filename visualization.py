import numpy as np
import pyvista as pv
from pyvista import Plotter
from typing import Tuple, List, Dict, Optional


class RadioTime:
    def __init__(self, plotter: Plotter, num_time_step: int = 150, cell_time_size: float = 5,
                 cell_freq_size: float = 40, cell_height: float = 50, freq_separation: float = 20):
        self._current_occupancy: Dict[str, np.ndarray] = {}
        self._freq_channel_list: Optional[List[int]] = None
        self._freq_channel_num: int = 0
        self._network_operator_list: Optional[List[str]] = None
        self._time_domain_occupancy: Optional[Dict[str, np.ndarray]] = None

        self._num_time_step = num_time_step
        self._grid_x: Dict[str, np.ndarray] = {}
        self._grid_y: Dict[str, np.ndarray] = {}

        cell_margin: float = 0.01
        self._cell_unit = cell_height * np.array([[0, 1, 1, 0], [0, 1, 1, 0]])
        self._grid_unit_x = np.array([[0, cell_margin, cell_margin + cell_freq_size, 2 * cell_margin + cell_freq_size],
                                      [0, cell_margin, cell_margin + cell_freq_size, 2 * cell_margin + cell_freq_size]])
        self._grid_unit_y = np.array([[0, 0, 0, 0], [cell_time_size, cell_time_size, cell_time_size, cell_time_size]])
        self._grid_spacing_x = 2 * cell_margin + cell_freq_size + freq_separation
        self._grid_spacing_y = cell_margin + cell_time_size
        self._mesh_dict = {}
        self._plotter = plotter

    def set_freq_channel_list(self, freq_channel_list: List[int]):
        self._freq_channel_list = freq_channel_list

    def set_network_operator_list(self, network_operator_list: List[str]):
        self._network_operator_list = ['collision'] +  network_operator_list
        self._freq_channel_num = len(self._freq_channel_list)
        self._collision = [0] * self._freq_channel_num
        occupancy = np.zeros((2, 4 * self._freq_channel_num))
        self._current_occupancy = {network_operator: occupancy.copy() for network_operator in self._network_operator_list}
        time_occupancy = np.zeros((2 * self._num_time_step, 4 * self._freq_channel_num))
        self._time_domain_occupancy = {network_operator: time_occupancy.copy() for network_operator in self._network_operator_list}

        # make floor
        max_grid_x = 0
        max_grid_y = 0
        for ind, network_operator in enumerate(self._network_operator_list):
            self._grid_x[network_operator] = np.zeros_like(time_occupancy)
            self._grid_y[network_operator] = np.zeros_like(time_occupancy)
            for time in range(self._num_time_step):
                for freq in range(self._freq_channel_num):
                    self._grid_x[network_operator][2*time: 2*time+2, 4*freq: 4*freq+4] = \
                        self._grid_unit_x + self._grid_spacing_x * freq
                    self._grid_y[network_operator][2*time: 2*time+2, 4*freq: 4*freq+4] = \
                        self._grid_unit_y + self._grid_spacing_y * time
            max_grid_x = np.max((max_grid_x, np.max(self._grid_x[network_operator])))
            max_grid_y = np.max((max_grid_y, np.max(self._grid_y[network_operator])))
        self._plotter.add_mesh(pv.Box([0, max_grid_x, 0, max_grid_y, -0.05, 0.05]), color='black')

        # make packet
        color = ['red', 'blue', 'yellow', 'green', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan']
        for ind, network_operator in enumerate(self._network_operator_list):
            mesh = pv.StructuredGrid(self._grid_x[network_operator], self._grid_y[network_operator],
                                     self._time_domain_occupancy[network_operator])
            self._mesh_dict[network_operator] = mesh
            self._plotter.add_mesh(mesh, color=color[ind], opacity=1)

    def log_decode(self, ch_info):
        ch_agent = ch_info.pop('agent')
        ch_users = np.sum(list(ch_info.values()), axis=0)
        ch_collision = []
        for i in range(self._freq_channel_num):
            if ch_agent[i] >= 1 and ch_users[i] >= 1:
                ch_collision.append(1)
            else:
                ch_collision.append(0)
        ch_info['agent'] = ch_agent
        ch_info['collision'] = ch_collision

        for network_operator in ch_info:
            if network_operator == 'collision':
                for ind in range(self._freq_channel_num):
                    if ch_info[network_operator][ind] == 1:
                        self.send_packet(network_operator, ind)
                    elif ch_info[network_operator][ind] == 0:
                        self.receive_packet(network_operator, ind)
            else:
                for ind in range(self._freq_channel_num):
                    if ch_info['collision'][ind] == 1:
                        self.receive_packet(network_operator, ind)
                    else:
                        if ch_info[network_operator][ind] == 1:
                            self.send_packet(network_operator, ind)
                        elif ch_info[network_operator][ind] == 0:
                            self.receive_packet(network_operator, ind)


    def send_packet(self, network_operator: str, freq_channel: int):
        ind = self._freq_channel_list.index(freq_channel)
        self._current_occupancy[network_operator][0: 2, 4 * ind: 4 * (ind + 1)] = self._cell_unit

    def receive_packet(self, network_operator: str, freq_channel: int):
        ind = self._freq_channel_list.index(freq_channel)
        self._current_occupancy[network_operator][0: 2, 4*ind: 4*(ind+1)] = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])

    def update(self):
        for network_operator in self._network_operator_list:
            time_occupancy = self._time_domain_occupancy[network_operator]
            occupancy = self._current_occupancy[network_operator]
            time_occupancy[:] = np.row_stack((time_occupancy[2:, :], occupancy))
            points = np.column_stack((self._grid_x[network_operator].flatten(order='F'),
                                      self._grid_y[network_operator].flatten(order='F'),
                                      time_occupancy.flatten(order='F')))
            self._mesh_dict[network_operator].points = points


class Visualization:
    def __init__(self, video_file_path, freq_channel_list, network_operator_list, show_video: bool = True,
                 resolution: Tuple = (1280, 720), frame_rate: int = 20):
        self._plotter = Plotter(window_size=resolution, off_screen=not show_video)

        self._radio_time = RadioTime(self._plotter)
        self._radio_time.set_freq_channel_list(freq_channel_list)
        self._radio_time.set_network_operator_list(network_operator_list)

        self._text = self._plotter.add_text('')
        self._text.SetMaximumFontSize(30)
        self._plotter.open_movie(video_file_path, framerate=frame_rate)
        self._times = 0
        self._rewards = 0
        self._pattern = tuple()

    def __call__(self, log):
        self._times += 1
        self._rewards += log['reward']
        self._pattern = log['pattern']
        self._text.SetText(2, f' Times: {self._times}\n Rewards: {self._rewards}\n Pattern: {self._pattern}')
        ch_info = log['channel info']
        self._radio_time.log_decode(ch_info)
        self._radio_time.update()
        self._plotter.render()
        self._plotter.write_frame()

    def close(self):
        self._plotter.close()