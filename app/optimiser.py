from typing import Any, Dict

import numpy as np


def compute_expected_costs(damage_flag: bool, channel: str, region_risk_score: float) -> float:
    """Deterministic cost model."""
    base_cost = 250.0
    damage_cost = 500.0 if damage_flag else 0.0

    channel_costs = {"dealer": 100.0, "private": 200.0, "fleet": 50.0}
    cost_channel = channel_costs.get(channel.lower(), 150.0)
    risk_multiplier = 100.0 * region_risk_score

    return base_cost + damage_cost + cost_channel + risk_multiplier


def compute_ev(
    offer: float,
    p_win: float,
    e_sale: float,
    e_costs: float,
    price_q10: float,
    risk_lambda: float = 0.5,
) -> float:
    """
    EV(offer) = P(win|offer) * (E(sale) - offer - E(costs)) - λ * tail_penalty
    """
    margin = e_sale - offer - e_costs
    tail_penalty_from_lower_quantile = max(0.0, offer - price_q10)

    ev = p_win * margin - (risk_lambda * tail_penalty_from_lower_quantile)
    return ev


def optimise_offer(
    e_sale: float,
    price_q10: float,
    e_costs: float,
    predict_p_win_func,  # function that takes offer and returns p_win
    min_margin: float = 200.0,
    risk_lambda: float = 0.5,
) -> Dict[str, Any]:
    """
    Grid search over valid offers to maximize EV.
    """
    max_offer = e_sale - e_costs - min_margin
    min_offer = max(500.0, max_offer * 0.5)

    if max_offer <= min_offer:
        # Cannot make a profitable offer
        return {
            "recommended_offer": 0.0,
            "expected_value": 0.0,
            "p_win": 0.0,
            "risk_band": "high",
            "explanation": {"reason": "Negative or zero margin"},
        }

    offers = np.linspace(min_offer, max_offer, num=50)
    best_ev = -float("inf")
    best_offer = 0.0
    best_p_win = 0.0

    for opt_offer in offers:
        p_win = predict_p_win_func(opt_offer)
        ev = compute_ev(opt_offer, p_win, e_sale, e_costs, price_q10, risk_lambda)

        if ev > best_ev:
            best_ev = ev
            best_offer = opt_offer
            best_p_win = p_win

    risk_band = "low" if (best_offer < price_q10) else "medium" if best_offer < e_sale else "high"

    return {
        "recommended_offer": best_offer,
        "expected_value": best_ev,
        "p_win": best_p_win,
        "risk_band": risk_band,
        "explanation": {
            "e_sale": e_sale,
            "e_costs": e_costs,
            "tail_penalty": max(0.0, best_offer - price_q10),
        },
    }
