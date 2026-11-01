"""Blueprint steps 1.3-1.4: NLSS 2018/19 food-group shares of poor households mapped to NBS items,
and state recipient counts (poor population). Validated against official 2019 poverty rates."""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = os.path.join(ROOT, "data", "raw", "nlss", "NGA_2018_LSS_v01_M_CSV", "Household")
OUT, TAB = os.path.join(ROOT, "data", "derived"), os.path.join(ROOT, "results", "tables")
LINE = 137430  # official 2019 poverty line, NGN per person per year
STATES = ['Abia','Adamawa','Akwa Ibom','Anambra','Bauchi','Bayelsa','Benue','Borno','Cross River','Delta','Ebonyi','Edo',
          'Ekiti','Enugu','Gombe','Imo','Jigawa','Kaduna','Kano','Katsina','Kebbi','Kogi','Kwara','Lagos','Nasarawa','Niger',
          'Ogun','Ondo','Osun','Oyo','Plateau','Rivers','Sokoto','Taraba','Yobe','Zamfara','FCT Abuja']
ZONES = {1: 'North Central', 2: 'North East', 3: 'North West', 4: 'South East', 5: 'South South', 6: 'South West'}
GROUPS = {1: 'Grains (other)', 2: 'Baked food', 3: 'Starchy foods', 4: 'Pulses', 5: 'Oils', 6: 'Fruits', 7: 'Vegetables',
          8: 'Poultry and eggs', 9: 'Meat', 10: 'Fish', 11: 'Dairy', 12: 'Sugar', 13: 'Other food', 14: 'Non-alcoholic drinks',
          15: 'Alcohol', 16: 'Rice, local', 17: 'Maize', 18: 'Coffee and tea', 19: 'Rice, imported'}
ITEMS = {  # NBS 2017-2024 basket items priced for each food group
 1: ['Wheat flour: prepacked (golden penny 2kg)'],
 2: ['Bread sliced 500g', 'Bread unsliced 500g'],
 3: ['Yam tuber', 'Gari white,sold loose', 'Gari yellow,sold loose', 'Irish potato', 'Sweet potato', 'Plantain(ripe)', 'Plantain(unripe)'],
 4: ['Beans brown,sold loose', 'Beans:white black eye. sold loose'],
 5: ['Groundnut oil: 1 bottle, specify bottle', 'Palm oil: 1 bottle,specify bottle', 'Vegetable oil:1 bottle,specify bottle'],
 7: ['Tomato', 'Onion bulb'],
 8: ['Frozen chicken', 'Chicken Feet', 'Chicken Wings', 'Agric eggs medium size'],
 9: ['Beef Bone in', 'Beef,boneless'],
 10: ['Catfish (obokun) fresh', 'Catfish :dried', 'Catfish Smoked', 'Dried Fish Sardine', 'Iced Sardine', 'Mackerel : frozen',
      'Mudfish (aro) fresh', 'Mudfish : dried', 'Tilapia fish (epiya) fresh', 'Titus:frozen'],
 11: ['Evaporated tinned milk carnation 170g', 'Evaporated tinned milk(peak), 170g'],
 16: ['Rice local sold loose', 'Rice agric sold loose', 'Broken Rice (Ofada)'],
 17: ['Maize grain white sold loose', 'Maize grain yellow sold loose'],
 19: ['Rice,imported high quality sold loose', 'Rice Medium Grained'],
}
t = pd.read_csv(os.path.join(H, "totcons.csv"))
t["state_name"] = t.state.map(dict(enumerate(STATES, 1))); t["zone_name"] = t.zone.map(ZONES)
t["poor"] = t.totcons_adj < LINE
for g in GROUPS: t[f"g{g}"] = t[f"food_own{g}"].fillna(0) + t[f"food_purch{g}"].fillna(0)
gcols = [f"g{g}" for g in GROUPS]

# validation against official 2019 figures
val = pd.DataFrame({"national_excl_borno": [np.average(t[t.state != 8].poor, weights=t[t.state != 8].popw)]})
print("poverty headcount excl. Borno:", round(val.iloc[0, 0], 4), "(official 2019: 0.401)")

# food-group shares: aggregate (population-weighted) food value of the group of households
def shares(d):
    tot = (d[gcols].multiply(d.wt_final, axis=0)).sum()
    return tot / tot.sum()
pop_sets = {"poor": t[t.poor], "all": t}
q = t.totcons_adj.rank(pct=True)  # approx; use weighted quantile below
w = t.sort_values("totcons_adj"); cw = w.popw.cumsum() / w.popw.sum(); bottom40 = w[cw <= 0.40]
pop_sets["bottom40"] = bottom40
rows = []
for pname, d in pop_sets.items():
    for scope, dd in [("National", d)] + [(z, d[d.zone_name == z]) for z in ZONES.values()]:
        sh = shares(dd)
        for g, name in GROUPS.items():
            rows.append(dict(population=pname, scope=scope, group_code=g, group=name, share=sh[f"g{g}"],
                             priced_items=len(ITEMS.get(g, []))))
W = pd.DataFrame(rows)
W["covered"] = W.priced_items > 0
cov = W[W.covered].groupby(["population", "scope"]).share.sum().unstack(0).round(3)
print("share of food spending covered by priced NBS items:\n", cov.to_string())
W.to_csv(os.path.join(TAB, "nlss_food_group_shares.csv"), index=False)

# item weights: group share renormalised over covered groups, split equally across the group's items
iw = []
for (pname, scope), d in W[W.covered].groupby(["population", "scope"]):
    tot = d.share.sum()
    for _, r in d.iterrows():
        for it in ITEMS[r.group_code]:
            iw.append(dict(population=pname, scope=scope, group=r.group, item=it, weight=r.share / tot / len(ITEMS[r.group_code])))
IW = pd.DataFrame(iw); IW.to_csv(os.path.join(OUT, "basket_item_weights.csv"), index=False)
print(IW[(IW.population == "poor") & (IW.scope == "National")].groupby("group").weight.sum().sort_values(ascending=False).round(3).to_string())

# recipients: poor population by state (2018/19 survey weights)
R = t.groupby(["state_name", "zone_name"]).apply(lambda d: pd.Series({
        "poverty_headcount": np.average(d.poor, weights=d.popw), "population_2019": d.popw.sum(),
        "poor_population_2019": (d.popw * d.poor).sum(), "households_sampled": len(d)}), include_groups=False).reset_index()
R = R.rename(columns={"state_name": "state", "zone_name": "zone"})
R["recipient_share"] = R.poor_population_2019 / R.poor_population_2019.sum()
R.to_csv(os.path.join(OUT, "state_recipients.csv"), index=False)
print(R.sort_values("recipient_share", ascending=False).head(6)[["state", "poverty_headcount", "poor_population_2019", "recipient_share"]].round(3).to_string(index=False))
