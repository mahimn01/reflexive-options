"""Direct (no-RL, no-simulator-fit) empirical tests of the reflexive mechanism.

The primary redesigned hypothesis H1' is a dealer-gamma-exposure (GEX) regression:
estimate aggregate dealer gamma from an end-of-day SPX option open-interest grid,
then regress next-period realized vol-of-vol on signed GEX with controls. This
de-confounds the reflexivity claim from the Mamba+PPO+EWC RL surface tournament,
which is demoted to a secondary/exploratory result (original H1).
"""
