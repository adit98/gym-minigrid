from gym_minigrid.envs.crossing import LavaCrossingEnv
import numpy as np
from queue import Queue
import cv2

class LavaCrossingSpotRewardEnv(LavaCrossingEnv):
  def __init__(self):
    super().__init__()
    self.training = True
    self.reward_grid = None
    self.goal_pos = (0, 0)
    self.reset()

  def _gen_reward(self):
    # generate the reward values in the grid (use the numpy array reward_grid)
    # first, get shortest path to goal for every square in the grid, then take
    # the inverse of the values to get our P function (the higher it is, the
    # closer we are to the goal
    bfs_q = Queue()
    processed = np.zeros_like(self.reward_grid)

    # append array positions, neighbor's distance to goal to the queue, we will
    # process a square by setting its proper distance value, adding unprocessed
    # neighbors, modifying the # processed mask, and popping the current
    # position from the queue
    bfs_q.put((self.goal_pos, 0))
    while not bfs_q.empty():
      # parse queue item
      pos, neighbor_dist = bfs_q.get()
      r, c = pos
      neighbors = [(r, c+1), (r+1, c), (r, c-1), (r-1, c)]

      # if it is the goal, we don't change the reward value, otherwise, the
      # shortest path is 1 + neighbor's distance
      if self.reward_grid[r, c] != 2:
        self.reward_grid[r, c] = neighbor_dist + 1

      for r_n, c_n in neighbors:
        # skip positions that are out of bounds
        if r_n >= self.reward_grid.shape[0] or r_n < 0: continue
        elif c_n >= self.reward_grid.shape[1] or c_n < 0: continue
        # don't process unreachable squares (lava or wall, marked as 1
        elif self.reward_grid[r_n, c_n] == 1: continue
        # don't process squares that have already been processed
        elif processed[r_n, c_n]: continue

        # add the neighbor to the queue with the current square's distance
        bfs_q.put(((r_n, c_n), self.reward_grid[r, c]))

        # set the current neighbor as processed
        processed[r_n, c_n] = 1

    # finally, invert the values in the reward grid to get P
    # use 2 in the numerator so that reaching the goal has reward of 1
    # first need to store lava/wall spots so we can set those to have 0 reward
    unreachable_inds = (self.reward_grid == 1)
    self.reward_grid = np.max(self.reward_grid) - self.reward_grid
    # normalize to be from 0 to 1
    self.reward_grid = self.reward_grid / np.max(self.reward_grid)
    # set unreachable spots to have -1 reward
    self.reward_grid[unreachable_inds] = -1
    print(self.reward_grid)

  def _reward(self, action=None, last_pos=None):
    # hacky solution - ideally we raise an exception if last_pos isn't provided
    # at train time, however, this reward function is called when calling the
    # step method of the base class, and we cannot pass last_pos to that step
    # method
    # TODO set weighting factor, this way multiply reward on turns by 0.2
    #base_reward = np.ones(3) * 0.5
    #base_reward[-1] = 1
    if self.training:
      if last_pos is not None:
        # here, we use the spot reward
        r, c = self.agent_pos
        r_l, c_l = last_pos
        print("Last Pos", last_pos, "Cur Pos", self.agent_pos)
        # get I_sr (0 if reward has decreased from last timestep)
        I_sr = int(self.reward_grid[c, r] >= self.reward_grid[c_l, r_l])
        # get P (reward grid value)
        P = self.reward_grid[c, r]
        reward = I_sr * P
        #if action is not None:
        #  reward *= base_reward[action]

        return reward
      else:
        # we reach this condition when we call the step method of the superclass
        # while training and reach the goal - since the indicator must have been
        # positive in this scenario, we return the value of P
        # this is hacky because of the way we are extending the grid
        r, c = self.goal_pos
        # we must have moved forward to reach the goal, so base reward is 1
        return self.reward_grid[c, r]
    else:
      return super()._reward()

  def step(self, action, last_pos=None):
    if self.training:
      if last_pos is None:
        raise ValueError("Must provide last position of agent to step function \
          at train time if using progress reward")
 
      next_state, reward, done, _ = super().step(action)

      # if we have finished, the step method has already called the reward method
      if not done:
        reward = self._reward(action, last_pos)

    else:
      # run the step method of the superclass
      next_state, reward, done, _ = super().step(action)

    return next_state, reward, done

  def eval(self):
    self.training = False

  def train(self):
    self.training = True

  def reset(self):
    super().reset()
    grid = self.grid.grid.copy()
    for ind, i in enumerate(grid):
      if i is None:
        grid[ind] = 0
      elif i.type == 'lava':
        grid[ind] = 1
      elif i.type == 'wall':
        grid[ind] = 1
      elif i.type == 'goal':
        # we are at the goal here, assign it a value of 2
        grid[ind] = 2
        # calculate row and column of goal, store this info
        r = ind // self.grid.height
        c = ind - (r * self.grid.height)
        self.goal_pos = (r, c)
      else:
        raise NotImplementedError(str(i) + " is not supported")

    self.reward_grid = np.array(grid).reshape(self.grid.height, self.grid.width)
    self._gen_reward()

    # return first observation
    obs = self.gen_obs()
    return obs
