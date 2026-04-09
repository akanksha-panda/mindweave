#server\config\policy_config.py

ROUTER_POLICY = {
    "distortion_threshold": 7,
    "low_energy_threshold": 0,
    "sentiment_threshold": -0.8
}

REWARD_WEIGHTS = {
    "reframe": 2.0,
    "activity_good": 2.0,
    "activity_bad": -2.0,
    "empathy": 1.5,
    "toxic_penalty": -3.0
}