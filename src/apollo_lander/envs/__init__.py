"""
Apollo Lander Gymnasium environments.

Registers the ApolloLander-v0 environment for use with
gymnasium.make('ApolloLander-v0').
"""

from gymnasium.envs.registration import register

register(
    id="ApolloLander-v0",
    entry_point="apollo_lander.envs.apollo_lander_env:ApolloLanderEnv",
)
