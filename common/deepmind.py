#This code is written by @sracaniere from DeepMind
#https://github.com/sracaniere

import numpy as np
import math

STANDARD_MAP = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


def get_random_position(map_array):
  """Gets a random available position in a binary map array.

  Args:
    map_array: numpy array of the map to search an available position on.

  Returns:
    The chosen random position.

  Raises:
    ValueError: if there is no available space in the map.
  """
  if map_array.sum() <= 0:
    raise ValueError("There is no available space in the map.")
  map_dims = len(map_array.shape)
  pos = np.zeros(map_dims, dtype=np.int32)
  while True:
    result = map_array
    for i in range(map_dims):
      pos[i] = np.random.randint(map_array.shape[i])
      result = result[pos[i]]
    if result == 0:
      break
  return pos


def update_2d_pos(array_map, pos, action, pos_result):
  posv = array_map[pos[0]][pos[1]][action - 1]
  pos_result[0] = posv[0]
  pos_result[1] = posv[1]
  return pos_result


def parse_map(map_array):
  """Parses a map when there are actions: stay, right, up, left, down.

  Args:
    map_array: 2D numpy array that contains the map.

  Returns:
    A 3D numpy array (height, width, actions) that contains the resulting state
    for a given position + action, and a 2D numpy array (height, width) with the
    walls of the map.

  Raises:
    ValueError: if the map does not contain only zeros and ones.
  """
  act_def = [[0, 0], [0, 1], [-1, 0], [0, -1], [1, 0]]
  walls = np.zeros_like(map_array)
  new_map_array = []
  for i in range(map_array.shape[0]):
    new_map_array.append([])
    for j in range(map_array.shape[1]):
      new_map_array[i].append([])
      if map_array[i, j] == 0:
        for k in range(len(act_def)):
          new_map_array[i][j].append([i + act_def[k][0], j + act_def[k][1]])
      elif map_array[i, j] == 1:
        for k in range(len(act_def)):
          new_map_array[i][j].append([i, j])
        walls[i, j] = 1
      else:
        raise ValueError("Option not understood, %d" % map_array[i, j])
      for k in range(len(new_map_array[i][j])):
        if map_array[new_map_array[i][j][k][0]][new_map_array[i][j][k][1]] == 1:
          new_map_array[i][j][k][0] = i
          new_map_array[i][j][k][1] = j
  return np.array(new_map_array), walls


def observation_as_rgb(obs):
  """Reduces the 6 channels of `obs` to 3 RGB.

  Args:
    obs: the observation as a numpy array.

  Returns:
    An RGB image in the form of a numpy array, with values between 0 and 1.
  """
  height = obs.shape[0]
  width = obs.shape[1]
  rgb = np.zeros((height, width, 3), dtype=np.float32)
  for x in range(height):
    for y in range(width):
      if obs[x, y, PillEater.PILLMAN] == 1:
        rgb[x, y] = [0, 1, 0]
      elif obs[x, y, PillEater.GHOSTS] > 0. or obs[x, y, PillEater.GHOSTS_EDIBLE] > 0.:
        g = obs[x, y, PillEater.GHOSTS]
        ge = obs[x, y, PillEater.GHOSTS_EDIBLE]
        rgb[x, y] = [g + ge, ge, 0]
      elif obs[x, y, PillEater.PILL] == 1:
        rgb[x, y] = [0, 1, 1]
      elif obs[x, y, PillEater.FOOD] == 1:
        rgb[x, y] = [0, 0, 1]
      elif obs[x, y, PillEater.WALLS] == 1:
        rgb[x, y] = [1, 1, 1]
  return rgb


class PillEater(object):

  WALLS = 0
  FOOD = 1
  PILLMAN = 2
  GHOSTS = 3
  GHOSTS_EDIBLE = 4
  PILL = 5
  NUM_ACTIONS = 5
  MODES = ('regular', 'avoid', 'hunt', 'ambush', 'rush')

  def __init__(self, mode, frame_cap=3000):
    assert mode in PillEater.MODES
    self.nghosts_init = 1
    self.ghost_speed_init = 0.5
    self.ghost_speed = self.ghost_speed_init
    self.ghost_speed_increase = 0.1
    self.end_on_collect = False
    self.npills = 2
    self.pill_duration = 20
    self.seed = 123
    self.discount = 1
    self.stochasticity = 0.05
    self.obs_is_rgb = True
    self.frame_cap = frame_cap
    self.safe_distance = 5
    map_array = STANDARD_MAP
    self.map, self.walls = parse_map(map_array)
    self.map = np.array(self.map)
    self.nactions = self.map.shape[2]
    self.height = self.map.shape[0]
    self.width = self.map.shape[1]
    self.reverse_dir = (4, 5, 2, 3)
    self.dir_vec = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]])
    self.world_state = dict(
        pillman=self._make_pillman(),
        ghosts=[],
        food=np.zeros(shape=(self.height, self.width), dtype=np.float32),
        pills=[None] * self.npills,
        power=0
    )
    self.nplanes = 6
    self.image = np.zeros(
        shape=(self.height, self.width, self.nplanes), dtype=np.float32)
    self.color_image = np.zeros(shape=(3, self.height, self.width),
                                dtype=np.float32)
    self.frame = 0
    self.reward = 0.
    self.pcontinue = 1.
    self._init_level(1)
    self._make_image()
    self.mode = mode
    self.timer = 0
    if self.mode == 'regular':
      self.step_reward = 0
      self.food_reward = 1
      self.big_pill_reward = 2
      self.ghost_hunt_reward = 5
      self.ghost_death_reward = 0
      self.all_pill_terminate = False
      self.all_ghosts_terminate = False
      self.all_food_terminate = True
      self.timer_terminate = -1
    elif self.mode == 'avoid':
      self.step_reward = 0.1
      self.food_reward = -0.1
      self.big_pill_reward = -5
      self.ghost_hunt_reward = -10
      self.ghost_death_reward = -20
      self.all_pill_terminate = False
      self.all_ghosts_terminate = False
      self.all_food_terminate = True
      self.timer_terminate = 128
    elif self.mode == 'hunt':
      self.step_reward = 0
      self.food_reward = 0
      self.big_pill_reward = 1
      self.ghost_hunt_reward = 10
      self.ghost_death_reward = -20
      self.all_pill_terminate = False
      self.all_ghosts_terminate = True
      self.all_food_terminate = False
      self.timer_terminate = -1
    elif self.mode == 'ambush':
      self.step_reward = 0
      self.food_reward = -0.1
      self.big_pill_reward = 0
      self.ghost_hunt_reward = 10
      self.ghost_death_reward = -20
      self.all_pill_terminate = False
      self.all_ghosts_terminate = True
      self.all_food_terminate = False
      self.timer_terminate = -1
    elif self.mode == 'rush':
      self.step_reward = 0
      self.food_reward = -0.1
      self.big_pill_reward = 10
      self.ghost_hunt_reward = 0
      self.ghost_death_reward = 0
      self.all_pill_terminate = True
      self.all_ghosts_terminate = False
      self.all_food_terminate = False
      self.timer_terminate = -1

  def _make_pillman(self):
    return self._make_actor(0)

  def _make_enemy(self):
    return self._make_actor(self.safe_distance)

  def _make_actor(self, safe_distance):
    """Creates an actor.

    An actor is a `ConfigDict` with a positions `pos` and a direction `dir`.
    The position is an array with two elements, the height and width. The
    direction is an integer representing the direction faced by the actor.

    Args:
      safe_distance: a `float`. The minimum distance from Pillman.

    Returns:
      A `ConfigDict`.
    """
    actor = {}
    if safe_distance > 0:
      occupied_map = np.copy(self.walls)

      from_ = (self.world_state['pillman']['pos'] - np.array(
          [self.safe_distance, self.safe_distance]))
      to = (self.world_state['pillman']['pos'] + np.array(
          [self.safe_distance, self.safe_distance]))
      from_[0] = max(from_[0], 1)
      from_[1] = max(from_[1], 1)
      to[0] = min(to[0], occupied_map.shape[0])
      to[1] = min(to[1], occupied_map.shape[1])

      occupied_map[from_[0]:to[0], from_[1]:to[1]] = 1

      actor['pos'] = get_random_position(occupied_map)
      actor['dir'] = np.random.randint(4)
    else:
      actor['pos'] = get_random_position(self.walls)
      actor['dir'] = np.random.randint(4)

    return actor

  def _make_pill(self):
    pill = dict(
        pos=get_random_position(self.walls)
    )
    return pill

  def _init_level(self, level):
    """Initialises the level."""
    self.level = level
    self._fill_food(self.walls, self.world_state['food'])
    self.world_state['pills'] = [self._make_pill() for _ in range(self.npills)]
    self.world_state['pillman']['pos'] = get_random_position(self.walls)

    self.nghosts = int(self.nghosts_init + math.floor((level - 1) / 2))
    self.world_state['ghosts'] = [self._make_enemy() for _ in range(self.nghosts)]
    self.world_state['power'] = 0

    self.ghost_speed = (
        self.ghost_speed_init + self.ghost_speed_increase * (level - 1))
    self.timer = 0

  def _fill_food(self, walls, food):
    food.fill(-1)
    food *= walls
    food += 1
    self.nfood = food.sum()

  def _get_food(self, posx, posy):
    self.reward += self.food_reward
    self.world_state['food'][posx][posy] = 0
    self.nfood -= 1
    if self.nfood == 0 and self.all_food_terminate:
      self._init_level(self.level + 1)

  def _get_pill(self, pill_index):
    self.world_state['pills'].pop(pill_index)
    self.reward += self.big_pill_reward
    self.world_state['power'] = self.pill_duration
    if (not self.world_state['pills']) and self.all_pill_terminate:
      self._init_level(self.level + 1)

  def _kill_ghost(self, ghost_index):
    self.world_state['ghosts'].pop(ghost_index)
    self.reward += self.ghost_hunt_reward
    if (not self.world_state['ghosts']) and self.all_ghosts_terminate:
      self._init_level(self.level + 1)

  def _die_by_ghost(self):
    self.reward += self.ghost_death_reward
    self.pcontinue = 0

  def _move_pillman(self, action):
    """Moves Pillman following the action in the proto `action_proto`."""
    action += 1  # our code is 1 based
    pos = self.world_state['pillman']['pos']
    pillman = self.world_state['pillman']
    update_2d_pos(self.map, pos, action, pos)
    if self.world_state['food'][pos[0]][pos[1]] == 1:
      self._get_food(pos[0], pos[1])
    for i, pill in enumerate(self.world_state['pills']):
      pos = pill['pos']
      if pos[0] == pillman['pos'][0] and pos[1] == pillman['pos'][1]:
        self._get_pill(i)
        break

  def _move_ghost(self, ghost):
    """Moves the given ghost."""
    pos = ghost['pos']
    new_pos = np.zeros(shape=(2,), dtype=np.float32)
    pillman = self.world_state['pillman']
    available = []
    for i in range(2, self.nactions + 1):
      update_2d_pos(self.map, pos, i, new_pos)
      if pos[0] != new_pos[0] or pos[1] != new_pos[1]:
        available.append(i)
    n_available = len(available)
    if n_available == 1:
      ghost['dir'] = available[0]
    elif n_available == 2:
      if ghost['dir'] not in available:
        if self.reverse_dir[ghost['dir'] - 2] == available[0]:
          ghost['dir'] = available[1]
        else:
          ghost['dir'] = available[0]
    else:
      rev_dir = self.reverse_dir[ghost['dir'] - 2]
      for i in range(n_available):
        if available[i] == rev_dir:
          available.pop(i)
          n_available -= 1
          break
      prods = np.zeros(n_available, dtype=np.float32)
      x = np.array(
          [pillman['pos'][0] - pos[0], pillman['pos'][1] - pos[1]], dtype=np.float32)
      norm = np.linalg.norm(x)
      if norm > 0:
        x *= 1. / norm
        for i in range(n_available):
          prods[i] = np.dot(x, self.dir_vec[available[i] - 2])
        if self.world_state['power'] == 0:
          if self.stochasticity > np.random.uniform():
            j = np.random.randint(n_available)
          else:
            # move towards pillman:
            j = np.argmax(prods)
        else:
          # run away from pillman:
          j = np.argmin(prods)
        ghost['dir'] = available[j]
    update_2d_pos(self.map, pos, ghost['dir'], pos)

  def _make_image(self):
    """Represents world in a `height x width x 6` `Tensor`."""
    self.image.fill(0)
    self.image[:, :, PillEater.WALLS] = self.walls
    self.image[:, :, PillEater.FOOD] = self.world_state['food']
    self.image[self.world_state['pillman']['pos'][0], self.world_state['pillman']['pos'][1],
               PillEater.PILLMAN] = 1
    for ghost in self.world_state['ghosts']:
      edibility = self.world_state['power'] / float(self.pill_duration)
      self.image[ghost['pos'][0], ghost['pos'][1], PillEater.GHOSTS] = 1. - edibility
      self.image[ghost['pos'][0], ghost['pos'][1], PillEater.GHOSTS_EDIBLE] = edibility
    for pill in self.world_state['pills']:
      self.image[pill['pos'][0], pill['pos'][1], PillEater.PILL] = 1
    return self.image

  def start(self):
    """Starts a new episode."""
    self.frame = 0
    self._init_level(1)
    self.reward = 0
    self.pcontinue = 1
    self.ghost_speed = self.ghost_speed_init
    return self._make_image(), self.reward, self.pcontinue

  def step(self, action):
    """Advances environment one time-step following the given action."""
    self.frame += 1
    pillman = self.world_state['pillman']
    self.pcontinue = self.discount
    self.reward = self.step_reward
    self.timer += 1
    # Update world state
    self.world_state['power'] = max(0, self.world_state['power']-1)

    # move pillman
    self._move_pillman(action)

    for i, ghost in enumerate(self.world_state['ghosts']):
      # first check if pillman went onto a ghost
      pos = ghost['pos']
      if pos[0] == pillman['pos'][0] and pos[1] == pillman['pos'][1]:
        if self.world_state['power'] == 0:
          self._die_by_ghost()
        else:
          self._kill_ghost(i)
          break
      # Then move ghosts
      speed = self.ghost_speed
      if self.world_state['power'] != 0:
        speed *= 0.5
      if np.random.uniform() < speed:
        self._move_ghost(ghost)
        pos = ghost['pos']
        # check if ghost went onto pillman
        if pos[0] == pillman['pos'][0] and pos[1] == pillman['pos'][1]:
          if self.world_state['power'] == 0:
            self._die_by_ghost()
          else:
            self._kill_ghost(i)
            # assume you can only eat one ghost per turn:
            break
    self._make_image()

    # Check if level over
    if self.timer == self.timer_terminate:
      self._init_level(self.level + 1)

    # Check if framecap reached
    if self.frame_cap > 0 and self.frame >= self.frame_cap:
      self.pcontinue = 0

  def observation(self, agent_id=0):
    return (self.reward,
            self.pcontinue,
            observation_as_rgb(self.image))