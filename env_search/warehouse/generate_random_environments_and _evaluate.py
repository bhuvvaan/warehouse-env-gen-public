# script_generate_and_repair.py

import numpy as np
from env_search.warehouse.milp_repair import repair_env
from env_search.utils import (
    kiva_env_number2str,
    format_env_str,
    kiva_obj_types,
    flip_tiles
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_random_environment(
    height,
    width,
    num_objects,
    min_n_shelf,
    max_n_shelf,
    rng,
    w_mode=False
):
    """
    Generates a random warehouse environment with specified parameters.

    Args:
        height (int): Height of the warehouse grid.
        width (int): Width of the warehouse grid.
        num_objects (int): Number of different object types.
        min_n_shelf (int): Minimum number of shelves.
        max_n_shelf (int): Maximum number of shelves.
        rng (np.random.Generator): Numpy random number generator.
        w_mode (bool): Whether to use workstation mode.

    Returns:
        np.ndarray: The generated environment as a numpy array.
    """
    solution_dim = height * width

    # Initialize the solution array (default is floor '.')
    sol = np.full(solution_dim, kiva_obj_types.index('.'), dtype=int)

    # Determine the number of shelves for this environment
    n_shelf = rng.integers(low=min_n_shelf, high=max_n_shelf + 1)

    # Randomly place shelves ('@') in the environment
    shelf_indices = rng.choice(solution_dim, size=n_shelf, replace=False)
    sol[shelf_indices] = kiva_obj_types.index('@')  # '@' represents a shelf

    # Optionally, place workstations ('w') if w_mode is True
    if w_mode:
        n_workstations = rng.integers(low=1, high=5)  # Adjust as needed
        workstation_indices = rng.choice(
            np.setdiff1d(np.arange(solution_dim), shelf_indices),
            size=n_workstations,
            replace=False
        )
        sol[workstation_indices] = kiva_obj_types.index('w')  # 'w' represents a workstation
    else:
        # Place robot starting positions ('r') along one side
        for i in range(width):
            sol[i] = kiva_obj_types.index('r')  # 'r' represents a robot start

    # Reshape to grid
    env = sol.reshape((height, width))
    return env


def repair_environment(
    env,
    agent_num,
    min_n_shelf,
    max_n_shelf,
    w_mode=False,
    add_movement=True,
    time_limit=120
):
    """
    Repairs the given warehouse environment using MILP.

    Args:
        env (np.ndarray): The environment to repair.
        agent_num (int): Number of agents.
        min_n_shelf (int): Minimum number of shelves.
        max_n_shelf (int): Maximum number of shelves.
        w_mode (bool): Whether to use workstation mode.
        add_movement (bool): Whether to consider movement in the repair.
        time_limit (int): Time limit for the MILP solver in seconds.

    Returns:
        np.ndarray: The repaired environment as a numpy array.
    """
    repaired_env = repair_env(
        env_np=env,
        agent_num=agent_num,
        min_n_shelf=min_n_shelf,
        max_n_shelf=max_n_shelf,
        w_mode=w_mode,
        add_movement=add_movement,
        time_limit=time_limit
    )
    return repaired_env


def main():
    # Parameters
    height = 9
    width = 12
    num_environments = 3
    num_objects = 2  # Typically, 0: '.', 1: '@' (floor and shelf)
    min_n_shelf = 10
    max_n_shelf = 20
    agent_num = 50
    w_mode = False  # Set to True if using workstation mode
    time_limit = 120  # Time limit for MILP solver in seconds
    seed = 42

    # Set up random number generator
    rng = np.random.default_rng(seed)

    for i in range(num_environments):
        logger.info(f"\n--- Environment {i+1} ---")

        # Generate random environment
        env = generate_random_environment(
            height=height,
            width=width,
            num_objects=num_objects,
            min_n_shelf=min_n_shelf,
            max_n_shelf=max_n_shelf,
            rng=rng,
            w_mode=w_mode
        )

        logger.info("Generated Environment:")
        env_str = kiva_env_number2str(env)
        formatted_env = format_env_str(env_str)
        print(formatted_env)

        # Repair the environment
        repaired_env = repair_environment(
            env=env,
            agent_num=agent_num,
            min_n_shelf=min_n_shelf,
            max_n_shelf=max_n_shelf,
            w_mode=w_mode,
            add_movement=True,
            time_limit=time_limit
        )

        # Check if repair was successful
        if repaired_env is not None:
            logger.info("Repaired Environment:")
            repaired_env_str = kiva_env_number2str(repaired_env)

            # If w_mode is True, flip 'r' back to 'w' for display
            if w_mode:
                repaired_env_flipped = flip_tiles(repaired_env, 'r', 'w')
                repaired_env_str = kiva_env_number2str(repaired_env_flipped)

            formatted_repaired_env = format_env_str(repaired_env_str)
            print(formatted_repaired_env)
        else:
            logger.warning("Repair failed or timed out.")


if __name__ == "__main__":
    main()