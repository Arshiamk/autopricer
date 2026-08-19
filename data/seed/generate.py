import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_synthetic_data(num_vehicles=50000, offers_per_enquiry=1, seed=42):
    np.random.seed(seed)

    # 1. Generate Regions (30-100)
    num_regions = np.random.randint(30, 101)
    region_ids = [f"R{str(i).zfill(3)}" for i in range(1, num_regions + 1)]
    region_names = [f"Region {i}" for i in range(1, num_regions + 1)]
    region_risk_scores = np.random.uniform(0.1, 1.0, num_regions)

    regions_df = pd.DataFrame(
        {
            "region_id": region_ids,
            "name": region_names,
            "country": "UK",
            "risk_score": region_risk_scores,
        }
    )

    # 2. Generate Vehicles
    vehicle_ids = [f"V{str(i).zfill(6)}" for i in range(1, num_vehicles + 1)]
    # Load from json
    import json

    json_path = os.path.join(os.path.dirname(__file__), "assets", "makes_models.json")
    with open(json_path, "r") as f:
        mm_dict = json.load(f)

    makes_models = []
    for make, models in mm_dict.items():
        for model in models:
            base_val = np.random.randint(8000, 50000)
            makes_models.append((make, model, base_val))

    make_model_idx = np.random.choice(len(makes_models), num_vehicles)
    vehicle_makes = [makes_models[i][0] for i in make_model_idx]
    vehicle_models = [makes_models[i][1] for i in make_model_idx]
    base_values = np.array([makes_models[i][2] for i in make_model_idx])

    years = np.random.randint(2010, 2025, num_vehicles)
    age = 2025 - years
    age_depreciation = np.power(0.88, age)  # 12% depreciation per year (more realistic for UK)

    mileages = np.random.exponential(scale=30000, size=num_vehicles) + 5000
    mileage_depreciation = np.maximum(0.3, 1.0 - (mileages / 150000.0) * 0.5)

    fuel_types = np.random.choice(
        ["petrol", "diesel", "electric", "hybrid"], num_vehicles, p=[0.5, 0.3, 0.1, 0.1]
    )
    body_types = np.random.choice(["hatchback", "saloon", "suv", "estate"], num_vehicles)

    vehicles_df = pd.DataFrame(
        {
            "vehicle_id": vehicle_ids,
            "make": vehicle_makes,
            "model": vehicle_models,
            "year": years,
            "mileage": mileages.astype(int),
            "fuel_type": fuel_types,
            "body_type": body_types,
        }
    )

    # Calculate Latent True Market Value
    true_market_values = base_values * age_depreciation * mileage_depreciation
    # Add some noise
    true_market_values *= np.random.normal(1.0, 0.1, num_vehicles)

    # Ensure a reasonable floor for modern cars
    true_market_values = np.where(
        (2025 - years) < 10,
        np.maximum(1500, true_market_values),
        np.maximum(500, true_market_values),
    )
    true_market_values = true_market_values.round(2)

    # 3. Generate Enquiries & Sales
    channels = np.random.choice(["dealer", "private", "fleet"], num_vehicles, p=[0.6, 0.3, 0.1])
    damage_flags = np.random.binomial(1, 0.2, num_vehicles)
    damage_types = np.where(
        damage_flags == 1,
        np.random.choice(
            ["scratches", "dents", "structural", "mechanical"], num_vehicles, p=[0.5, 0.3, 0.1, 0.1]
        ),
        "none",
    )

    enquiry_regions = np.random.choice(region_ids, num_vehicles)

    # Generate dates over the last year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    random_days = np.random.randint(0, 365, num_vehicles)
    enquiry_dates = [start_date + timedelta(days=int(d)) for d in random_days]

    enquiries_list = []
    sales_list = []

    enquiry_counter = 1

    for i in range(num_vehicles):
        vid = vehicle_ids[i]
        tmv = true_market_values[i]
        region_id = enquiry_regions[i]
        channel = channels[i]
        dmg_flag = damage_flags[i]
        dmg_type = damage_types[i]
        enquiry_date = enquiry_dates[i]

        # Dealer threshold: they won't sell unless offer is close to tmv minus dealer threshold
        dealer_threshold = tmv * 0.10  # e.g. dealer expects to lose 10% max on tmv
        price_sensitivity = tmv * 0.05  # How sharp the curve is

        # Generate N counterfactual offers mapping to different win probs
        for offer_idx in range(offers_per_enquiry):
            eid = f"E{str(enquiry_counter).zfill(7)}"

            if offer_idx == 0:
                # Historical buying policy: aim for ~85% of true market value
                offer_ratio = np.random.normal(0.85, 0.1)
            else:
                # Counterfactual offers deliberately explore a wide ratio range so a
                # conversion model can learn the offer->win curve across the whole
                # region a pricing policy might search, not just around 0.85.
                offer_ratio = np.random.uniform(0.60, 1.10)
            offer_price = round(tmv * offer_ratio, 2)

            # Sigmoid P(win | offer)
            p_win = sigmoid((offer_price - (tmv - dealer_threshold)) / price_sensitivity)
            win = np.random.binomial(1, p_win) == 1

            enquiries_list.append(
                {
                    "enquiry_id": eid,
                    "vehicle_id": vid,
                    "region_id": region_id,
                    "channel": channel,
                    "damage_flag": bool(dmg_flag),
                    "damage_type": dmg_type,
                    "offer_price": offer_price,
                    "date": enquiry_date.strftime("%Y-%m-%d"),
                }
            )

            # If win, actual costs are incurred
            actual_costs = 250.0  # base fee
            if dmg_flag:
                actual_costs += 500.0

            # Add some noise to final sale price (might be slightly different from tmv)
            sale_price = round(tmv * np.random.normal(1.0, 0.05), 2)
            gross_margin = round(sale_price - offer_price - actual_costs, 2)

            sales_list.append(
                {
                    "enquiry_id": eid,
                    "true_market_value": tmv,
                    "sale_price": sale_price if win else np.nan,
                    "won": int(win),
                    "actual_costs": actual_costs if win else 0.0,
                    "gross_margin": gross_margin if win else 0.0,
                    "date": enquiry_date.strftime("%Y-%m-%d"),
                }
            )

            enquiry_counter += 1

    enquiries_df = pd.DataFrame(enquiries_list)
    sales_df = pd.DataFrame(sales_list)

    # Save to disk
    out_dir = os.path.join(os.path.dirname(__file__), "..", "raw")
    os.makedirs(out_dir, exist_ok=True)

    regions_df.to_csv(os.path.join(out_dir, "regions.csv"), index=False)
    vehicles_df.to_csv(os.path.join(out_dir, "vehicles.csv"), index=False)
    enquiries_df.to_csv(os.path.join(out_dir, "enquiries.csv"), index=False)
    sales_df.to_csv(os.path.join(out_dir, "sales.csv"), index=False)

    print(f"Generated {len(regions_df)} regions")
    print(f"Generated {len(vehicles_df)} vehicles")
    print(f"Generated {len(enquiries_df)} enquiries")
    print(f"Generated {len(sales_df)} sales")
    print(f"Saved to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic AutoPricer data")
    parser.add_argument("--num-vehicles", type=int, default=50000, help="Number of unique vehicles")
    parser.add_argument(
        "--offers-per-enquiry",
        type=int,
        default=1,
        help="Number of counterfactual offers per vehicle",
    )
    args = parser.parse_args()

    generate_synthetic_data(
        num_vehicles=args.num_vehicles, offers_per_enquiry=args.offers_per_enquiry
    )
