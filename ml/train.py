"""Train and compare churn models, explain each customer's score and pick a next-best retention action.

Steps
 1. Today's rule (the baseline the retention team uses now) vs logistic regression vs gradient boosting.
    Train on the June cohort, test on the August cohort (a later time period, after a price rise and a
    competitor promo: a fair test of whether the model holds up when the market moves).
 2. Fairness check on the test cohort by age band and gender (never model inputs).
 3. Retrain the chosen model on June + August, score this week's 2,000 customers, explain each score with
    SHAP and turn the top drivers into plain-language reasons.
 4. Next-best action: expected value of each approved offer = P(churn) x save rate x 12-month margin - cost.

Usage: python3 ml/train.py   (after ml/generate_data.py)
Writes ml/results/metrics.json, ml/results/model_card.md and data/scored_current.json (used by the app).
"""
import json
import os

import numpy as np
import pandas as pd
import shap
from scipy.stats import chi2_contingency
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "ml", "results")
SEED = 42
CAPACITY = 0.10  # retention team can call the top 10% of the base each cycle (assumption)

NUM = ["monthly_charge", "tenure_months", "contract_end_days", "lines_on_account", "device_instalment_months_left",
       "device_age_months", "data_gb_3m_avg", "data_trend_pct", "overage_3m_usd", "dropped_call_rate_pct",
       "home_download_mbps", "support_tickets_90d", "unresolved_tickets", "billing_disputes_90d", "last_nps",
       "app_logins_30d", "late_payments_12m", "roaming_trips_12m"]
BOOL = ["autopay", "price_rise_notice", "competitor_promo_area"]
CAT = ["plan", "contract_type"]
EXCLUDED = ["region", "age_band", "gender", "last_ticket_summary", "customer_id", "snapshot_date"]

CATEGORY = {  # feature -> reason category shown to agents
    "contract_type": "contract", "contract_end_days": "contract", "device_instalment_months_left": "contract",
    "device_age_months": "contract", "dropped_call_rate_pct": "network", "home_download_mbps": "network",
    "support_tickets_90d": "service", "unresolved_tickets": "service", "last_nps": "service",
    "price_rise_notice": "price", "competitor_promo_area": "price", "monthly_charge": "price",
    "overage_3m_usd": "billing", "billing_disputes_90d": "billing", "late_payments_12m": "billing",
    "autopay": "billing", "app_logins_30d": "engagement", "data_trend_pct": "engagement",
    "tenure_months": "engagement", "data_gb_3m_avg": "engagement", "lines_on_account": "engagement",
    "roaming_trips_12m": "engagement", "plan": "price",
}

# Approved retention offers. Save rates are ASSUMPTIONS until the first A/B test measures them.
OFFERS = {
    "network_fix": {"name": "Network fix: signal booster + priority engineer visit", "cost": 30,
                    "save": {"network": 0.35, "service": 0.15}},
    "loyalty_discount": {"name": "Loyalty discount: 15% off for 6 months", "cost": None,
                         "save": {"price": 0.30, "billing": 0.12}},
    "billing_review": {"name": "Billing review callback with a specialist", "cost": 8,
                       "save": {"billing": 0.22, "service": 0.10}},
    "renewal_upgrade": {"name": "Early renewal with device upgrade credit", "cost": 60,
                        "save": {"contract": 0.30}},
    "data_bonus": {"name": "10 GB data bonus for 3 months", "cost": 15,
                   "save": {"engagement": 0.15}},
}
BASE_SAVE = 0.04   # any well-made call saves a few customers whatever the offer (assumption)
MARGIN = 0.55      # gross margin on the monthly charge (assumption)
CLV_MONTHS = 24    # value horizon for a saved customer (assumption)


def load(name):
    df = pd.read_csv(os.path.join(DATA, name))
    for b in BOOL:
        df[b] = df[b].astype(int)
    return df


def features(df):
    return df[NUM + BOOL + CAT]


def pre():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), NUM),
        ("bool", "passthrough", BOOL),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])


def rule_score(df):
    """Today's rule: call customers whose contract ends within 60 days or who raised 2+ tickets in 90 days."""
    end_soon = df["contract_end_days"].fillna(999) <= 60
    return (end_soon | (df["support_tickets_90d"] >= 2)).astype(float) + 0.001 * df["support_tickets_90d"]


def top_k_metrics(y, s, frac):
    k = int(round(len(y) * frac))
    idx = np.argsort(-s, kind="stable")[:k]
    hits = y[idx].sum()
    return {"precision": round(hits / k, 4), "recall": round(hits / y.sum(), 4), "lift": round((hits / k) / y.mean(), 2)}


def evaluate(y, s, probs=True):
    out = {"roc_auc": round(roc_auc_score(y, s), 4), "pr_auc": round(average_precision_score(y, s), 4),
           "top10": top_k_metrics(y, s, CAPACITY), "top5": top_k_metrics(y, s, 0.05)}
    if probs:
        out["brier"] = round(brier_score_loss(y, s), 4)
        out["mean_pred"] = round(float(np.mean(s)), 4)
    out["actual_rate"] = round(float(np.mean(y)), 4)
    return out


def rule_eval(y, df):
    flag = rule_score(df) >= 1
    k = int(flag.sum())
    hits = int(y[flag.values].sum())
    return {"flagged_share": round(k / len(y), 4), "precision": round(hits / k, 4), "recall": round(hits / y.sum(), 4),
            "lift": round((hits / k) / y.mean(), 2), "roc_auc": round(roc_auc_score(y, rule_score(df)), 4)}


def calib_table(y, p, bins=10):
    order = np.argsort(p)
    rows = []
    for chunk in np.array_split(order, bins):
        rows.append({"predicted": round(float(p[chunk].mean()), 4), "actual": round(float(y[chunk].mean()), 4), "n": int(len(chunk))})
    return rows


def fairness(df, y, s):
    k = int(round(len(y) * CAPACITY))
    flagged = np.zeros(len(y), bool)
    flagged[np.argsort(-s, kind="stable")[:k]] = True
    out = {}
    for col in ("age_band", "gender", "region"):
        groups = {}
        for g, idx in df.groupby(col).groups.items():
            idx = np.array(list(idx))
            yy, ff = y[idx], flagged[idx]
            groups[str(g)] = {"n": int(len(idx)), "churn_rate": round(float(yy.mean()), 4),
                              "flag_rate": round(float(ff.mean()), 4),
                              "recall": round(float(ff[yy == 1].mean()), 4) if yy.sum() else None,
                              "precision": round(float(yy[ff].mean()), 4) if ff.sum() else None}
        big = [g for g, v in groups.items() if v["n"] >= 300]
        rates = [groups[g]["flag_rate"] for g in big]
        recalls = [groups[g]["recall"] for g in big if groups[g]["recall"] is not None]
        sub = df[col].isin(big).values
        # Chi-square tests: is being flagged (all customers) / caught (churners only) independent of the group?
        tab_flag = pd.crosstab(df[col].values[sub], flagged[sub])
        churn = sub & (y == 1)
        tab_recall = pd.crosstab(df[col].values[churn], flagged[churn])
        out[col] = {"groups": groups, "groups_tested": big,
                    "flag_rate_ratio": round(min(rates) / max(rates), 3),
                    "recall_gap": round(max(recalls) - min(recalls), 4),
                    "flag_rate_p_value": round(float(chi2_contingency(tab_flag)[1]), 4),
                    "recall_p_value": round(float(chi2_contingency(tab_recall)[1]), 4)}
    return out


def reason_text(feat, row):
    v = row[feat]
    t = {
        "contract_type": lambda: f"On a {row['contract_type']} contract",
        "contract_end_days": lambda: f"Contract ends in {int(v)} days" if pd.notna(v) else "No fixed contract",
        "device_instalment_months_left": lambda: f"{int(v)} device instalment months left" if v else "No device instalment keeping them",
        "device_age_months": lambda: f"Phone is {int(v)} months old",
        "dropped_call_rate_pct": lambda: f"Dropped-call rate {v:.1f}%",
        "home_download_mbps": lambda: f"Home download speed {v:.1f} Mbps",
        "support_tickets_90d": lambda: f"{int(v)} support tickets in 90 days",
        "unresolved_tickets": lambda: f"{int(v)} unresolved tickets",
        "last_nps": lambda: f"Last NPS score {int(v)}" if pd.notna(v) else "No recent NPS answer",
        "price_rise_notice": lambda: "Received the September price-rise notice" if v else "No price-rise notice",
        "competitor_promo_area": lambda: "Lives where a competitor is running a promo" if v else "No competitor promo nearby",
        "monthly_charge": lambda: f"Monthly charge ${v:.2f}",
        "overage_3m_usd": lambda: f"${v:.2f} overage charges in 3 months",
        "billing_disputes_90d": lambda: f"{int(v)} billing disputes in 90 days",
        "late_payments_12m": lambda: f"{int(v)} late payments in 12 months",
        "autopay": lambda: "Not on autopay" if not v else "On autopay",
        "app_logins_30d": lambda: f"{int(v)} app logins in 30 days",
        "data_trend_pct": lambda: f"Data use changed {int(v):+d}% vs prior 3 months",
        "tenure_months": lambda: f"Customer for {int(v)} months",
        "data_gb_3m_avg": lambda: f"Uses {v:.1f} GB a month",
        "lines_on_account": lambda: f"{int(v)} line(s) on the account",
        "roaming_trips_12m": lambda: f"{int(v)} roaming trips in 12 months",
        "plan": lambda: f"{row['plan']} plan",
    }
    return t[feat]()


def offer_cost(key, row):
    c = OFFERS[key]["cost"]
    return round(0.15 * row["monthly_charge"] * 6, 2) if c is None else c


def next_best_action(p, row, reasons):
    clv = row["monthly_charge"] * CLV_MONTHS * MARGIN
    cats = [r["category"] for r in reasons]
    weights = [0.6, 0.3, 0.1][: len(cats)]
    options = []
    for key, o in OFFERS.items():
        save = BASE_SAVE + sum(w * o["save"].get(c, 0) for w, c in zip(weights, cats))
        if key == "renewal_upgrade" and not (pd.notna(row["contract_end_days"]) and row["contract_end_days"] <= 90
                                              or row["contract_type"] == "month-to-month"):
            save = BASE_SAVE  # only meaningful near the end of a contract
        cost = offer_cost(key, row)
        ev = p * save * clv - cost
        options.append({"offer": key, "name": o["name"], "save_rate": round(save, 3), "cost": cost, "expected_value": round(ev, 2)})
    options.sort(key=lambda o: -o["expected_value"])
    best = options[0] if options[0]["expected_value"] > 0 else None
    return best, options, round(clv, 2)


def main():
    os.makedirs(RES, exist_ok=True)
    tr, ho, cur = load("train_2026-06-30.csv"), load("holdout_2026-08-31.csv"), load("current_2026-10-31.csv")
    ytr, yho = tr["churned_60d"].values, ho["churned_60d"].values

    lr = Pipeline([("pre", pre()), ("clf", LogisticRegression(C=0.5, max_iter=3000, class_weight=None))])
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                        l2_regularization=1.0, categorical_features="from_dtype", random_state=SEED)
    gb_cal = CalibratedClassifierCV(Pipeline([("pre", pre()), ("clf", gb)]), method="isotonic", cv=5)

    lr.fit(features(tr), ytr)
    gb_cal.fit(features(tr), ytr)
    p_lr = lr.predict_proba(features(ho))[:, 1]
    p_gb = gb_cal.predict_proba(features(ho))[:, 1]

    metrics = {
        "data": {"train": {"snapshot": "2026-06-30", "n": len(tr), "churn_rate": round(float(ytr.mean()), 4)},
                 "test": {"snapshot": "2026-08-31", "n": len(ho), "churn_rate": round(float(yho.mean()), 4)},
                 "note": "Time-based split: trained on June, tested on August, after a price rise and a competitor promo."},
        "capacity": CAPACITY,
        "rule_baseline": rule_eval(yho, ho),
        "logistic_regression": evaluate(yho, p_lr),
        "gradient_boosting": evaluate(yho, p_gb),
        "excluded_from_model": EXCLUDED,
    }
    choice = "gradient_boosting" if metrics["gradient_boosting"]["pr_auc"] >= metrics["logistic_regression"]["pr_auc"] + 0.01 else "logistic_regression"
    metrics["chosen"] = choice
    p_test = p_gb if choice == "gradient_boosting" else p_lr
    metrics["calibration_test"] = calib_table(yho, p_test)
    metrics["fairness"] = fairness(ho, yho, p_test)

    # Production model: retrain on June + August so it has seen the price rise.
    both = pd.concat([tr, ho], ignore_index=True)
    yb = both["churned_60d"].values
    if choice == "gradient_boosting":
        prod = Pipeline([("pre", pre()), ("clf", HistGradientBoostingClassifier(**gb.get_params()))])
        prod_cal = CalibratedClassifierCV(Pipeline([("pre", pre()), ("clf", HistGradientBoostingClassifier(**gb.get_params()))]), method="isotonic", cv=5)
        prod.fit(features(both), yb)
        prod_cal.fit(features(both), yb)
        p_cur = prod_cal.predict_proba(features(cur))[:, 1]
        X_cur = prod.named_steps["pre"].transform(features(cur))
        explainer = shap.TreeExplainer(prod.named_steps["clf"])
        sv = explainer.shap_values(X_cur)
    else:
        prod = Pipeline([("pre", pre()), ("clf", LogisticRegression(C=0.5, max_iter=3000))])
        prod.fit(features(both), yb)
        p_cur = prod.predict_proba(features(cur))[:, 1]
        X_cur = prod.named_steps["pre"].transform(features(cur))
        bg = prod.named_steps["pre"].transform(features(both))
        sv = shap.LinearExplainer(prod.named_steps["clf"], bg).shap_values(X_cur)
    names = prod.named_steps["pre"].get_feature_names_out()

    def base_feature(n):
        n = n.split("__", 1)[1]
        if n.startswith("missingindicator_"):
            return n[len("missingindicator_"):]
        for c in CAT:
            if n.startswith(c + "_"):
                return c
        return n

    bases = np.array([base_feature(n) for n in names])
    uniq = list(dict.fromkeys(bases))
    agg = np.stack([sv[:, bases == b].sum(axis=1) for b in uniq], axis=1)
    global_imp = sorted(zip(uniq, np.abs(agg).mean(axis=0)), key=lambda t: -t[1])
    metrics["global_drivers"] = [{"feature": f, "mean_abs_shap": round(float(v), 4), "category": CATEGORY[f]} for f, v in global_imp]

    rank = np.argsort(-p_cur, kind="stable")
    cutoff = float(np.sort(p_cur)[::-1][int(round(len(p_cur) * CAPACITY)) - 1])
    customers = []
    for pos, i in enumerate(rank, 1):
        row = cur.iloc[i]
        order = np.argsort(-agg[i])
        reasons = []
        for j in order:
            if agg[i, j] <= 0 or len(reasons) == 3:
                break
            f = uniq[j]
            reasons.append({"feature": f, "category": CATEGORY[f], "text": reason_text(f, row), "impact": round(float(agg[i, j]), 3)})
        best, options, clv = next_best_action(float(p_cur[i]), row, reasons)
        facts = {k: (None if pd.isna(row[k]) else (bool(row[k]) if k in BOOL else (row[k].item() if hasattr(row[k], "item") else row[k])))
                 for k in NUM + BOOL + CAT}
        customers.append({
            "customer_id": row["customer_id"], "rank": pos, "churn_probability": round(float(p_cur[i]), 4),
            "risk_band": "high" if p_cur[i] >= cutoff else ("medium" if p_cur[i] >= 0.08 else "low"),
            "in_call_list": bool(p_cur[i] >= cutoff), "reasons": reasons, "facts": facts,
            "ticket_note": row["last_ticket_summary"], "clv_24m": clv, "next_best_action": best,
            "offer_options": options,
        })
    metrics["current"] = {"snapshot": "2026-10-31", "n": len(cur), "call_list_size": sum(c["in_call_list"] for c in customers),
                          "call_list_cutoff_probability": round(cutoff, 4),
                          "expected_churners_in_call_list": round(float(sum(c["churn_probability"] for c in customers if c["in_call_list"])), 1),
                          "no_offer_in_call_list": sum(1 for c in customers if c["in_call_list"] and not c["next_best_action"])}
    json.dump(metrics, open(os.path.join(RES, "metrics.json"), "w"), indent=1)
    offers_meta = {k: {"name": v["name"], "cost": v["cost"] if v["cost"] is not None else "15% of the monthly charge for 6 months"} for k, v in OFFERS.items()}
    json.dump({"model": choice, "snapshot": "2026-10-31", "offers": offers_meta, "customers": customers},
              open(os.path.join(DATA, "scored_current.json"), "w"), separators=(",", ":"))
    # Compact copy for the browser: full detail for the 300 highest-risk customers, a summary row for the rest.
    compact = []
    for c in customers:
        row = {"id": c["customer_id"], "rank": c["rank"], "p": c["churn_probability"], "band": c["risk_band"],
               "call": c["in_call_list"], "reasons": [[r["category"], r["text"], r["impact"]] for r in c["reasons"]],
               "offer": c["next_best_action"]["offer"] if c["next_best_action"] else None,
               "ev": c["next_best_action"]["expected_value"] if c["next_best_action"] else None,
               "plan": c["facts"]["plan"], "charge": c["facts"]["monthly_charge"]}
        if c["rank"] <= 300:
            row.update({"facts": c["facts"], "clv": c["clv_24m"], "options": c["offer_options"]})
        compact.append(row)
    os.makedirs(os.path.join(ROOT, "public"), exist_ok=True)
    json.dump({"snapshot": "2026-10-31", "model": choice, "offers": offers_meta, "customers": compact},
              open(os.path.join(ROOT, "public", "customers.json"), "w"), separators=(",", ":"))
    public_metrics = {k: metrics[k] for k in ("data", "capacity", "rule_baseline", "logistic_regression", "gradient_boosting",
                                              "chosen", "calibration_test", "fairness", "global_drivers", "current", "excluded_from_model")}
    json.dump(public_metrics, open(os.path.join(ROOT, "public", "model.json"), "w"), separators=(",", ":"))

    m = metrics
    print(json.dumps({k: m[k] for k in ("rule_baseline", "logistic_regression", "gradient_boosting", "chosen")}, indent=1))
    print("fairness:", {k: {x: v[x] for x in ("flag_rate_ratio", "recall_gap", "flag_rate_p_value", "recall_p_value")} for k, v in m["fairness"].items()})
    print("top drivers:", [d["feature"] for d in m["global_drivers"][:8]])
    print("current:", m["current"])


if __name__ == "__main__":
    main()
